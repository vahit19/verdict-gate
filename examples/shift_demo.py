"""A certificate holds where it was computed, and a random audit catches it when it stops holding.

Verifier: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, a widely used
fact-verification classifier (trained on MNLI, FEVER and ANLI). It reads
(evidence, claim) and says SUPPORTS, REFUTES or NOT ENOUGH INFO; its confidence
is the top softmax probability. Gold labels play the human verdicts.

1. Certify on FEVER (claims like the ones it was trained on): the share of
   verdicts it can settle at 95% accuracy with 95% confidence.
2. Check that threshold on held-out FEVER claims.
3. Apply the same threshold to VitaminC claims (contrastive edits of Wikipedia
   facts, built to be harder). Does the certificate still hold?
4. Audit 1 in 8 of the VitaminC verdicts the machine settles, in arrival order,
   and report when the audit raises an alarm.
5. Probes on FEVER: planted seller text, counted only when the manipulated
   verdict clears the certified threshold (a wrong verdict nobody reviews).

Run:  uv run --extra demo python examples/shift_demo.py [N_FEVER] [N_VITAMINC]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verdict_gate import audit_alarm, certify, run_probes
from verdict_gate.core import error_upper_bound

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
N_FEVER = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
N_VITC = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
N_PROBE = 1000
SEED = 0
AUDIT_RATE = 1 / 8
OUT = Path(__file__).resolve().parent.parent / "results" / "shift_demo.json"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
TO_FEVER = {"entailment": "SUPPORTS", "contradiction": "REFUTES", "neutral": "NOT ENOUGH INFO"}
id2label = {i: TO_FEVER[l.lower()] for i, l in model.config.id2label.items()}


def verifier(claims, evidence, batch=64):
    labels, conf = [], []
    with torch.inference_mode():
        for i in range(0, len(claims), batch):
            enc = tok(
                list(evidence[i : i + batch]),
                list(claims[i : i + batch]),
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            top = model(**enc).logits.softmax(-1).max(-1)
            labels += [id2label[int(j)] for j in top.indices]
            conf += top.values.tolist()
    return labels, np.array(conf)


def accuracy_with_bound(correct):
    m, k = len(correct), int(len(correct) - np.sum(correct))
    if m == 0:
        return None, None
    return 1 - k / m, 1 - error_upper_bound(k, m, 0.05)


# FEVER claims with their gold evidence sentences joined into one passage.
fever = load_dataset("copenlu/fever_gold_evidence", split="test").shuffle(seed=SEED)
fever = fever.select(range(N_FEVER))
f_claims = fever["claim"]
f_evid = [" ".join(e[2] for e in ev) for ev in fever["evidence"]]
f_gold = fever["label"]
f_lab, f_conf = verifier(f_claims, f_evid)
f_ok = np.array([int(a == b) for a, b in zip(f_lab, f_gold)])

half = N_FEVER // 2
cert = certify(f_conf[:half], f_ok[:half], target_accuracy=0.95, delta=0.05)
t = cert.threshold
assert t is not None, cert.summary()

held = f_conf[half:] >= t
held_acc, held_lb = accuracy_with_bound(f_ok[half:][held])

vit = load_dataset("tals/vitaminc", split="test").shuffle(seed=SEED).select(range(N_VITC))
v_lab, v_conf = verifier(vit["claim"], vit["evidence"])
v_ok = np.array([int(a == b) for a, b in zip(v_lab, vit["label"])])
v_set = v_conf >= t
v_acc, v_lb = accuracy_with_bound(v_ok[v_set])
v_ub_err = None

# Audit: settled VitaminC verdicts arrive in order; each is audited with
# probability 1/8. Record when the alarm first fires.
rng = np.random.default_rng(SEED)
audited = errors = 0
alarm_at = None
for i, ok in enumerate(v_ok[v_set]):
    if rng.random() < AUDIT_RATE:
        audited += 1
        errors += int(ok == 0)
        alarm, upper = audit_alarm(errors, audited, 0.95, 0.05)
        if alarm and alarm_at is None:
            alarm_at = {"settled_so_far": i + 1, "audited": audited, "errors": errors,
                        "accuracy_upper_bound": upper}
final_alarm, final_upper = audit_alarm(errors, audited, 0.95, 0.05)

# Probes on FEVER first half, limited to N_PROBE examples.
p_idx = list(range(min(half, N_PROBE * 3)))
kept, probes = run_probes(
    verifier,
    [f_claims[i] for i in p_idx],
    [f_evid[i] for i in p_idx],
    [f_gold[i] for i in p_idx],
    target_label="SUPPORTS",
    threshold=t,
)

result = {
    "model": MODEL,
    "fever": f"copenlu/fever_gold_evidence test split, {N_FEVER} random claims (seed {SEED})",
    "vitaminc": f"tals/vitaminc test split, {N_VITC} random claims (seed {SEED})",
    "fever_overall_accuracy": float(f_ok.mean()),
    "vitaminc_overall_accuracy": float(v_ok.mean()),
    "certificate_fever_first_half": {
        "summary": cert.summary(),
        "threshold": t,
        "coverage": cert.coverage,
        "accuracy": cert.accuracy,
        "accuracy_lower_bound": cert.accuracy_lower_bound,
    },
    "fever_held_out": {
        "coverage": float(held.mean()),
        "settled": int(held.sum()),
        "accuracy": held_acc,
        "accuracy_lower_bound": held_lb,
    },
    "vitaminc_same_threshold": {
        "coverage": float(v_set.mean()),
        "settled": int(v_set.sum()),
        "accuracy": v_acc,
        "accuracy_lower_bound": v_lb,
    },
    "audit_on_vitaminc": {
        "rate": AUDIT_RATE,
        "audited": audited,
        "errors": errors,
        "alarm": final_alarm,
        "accuracy_upper_bound": final_upper,
        "first_alarm": alarm_at,
    },
    "probes_fever": {
        "examples": kept,
        "note": "gold is not SUPPORTS and the verifier was right without the probe",
        "results": [p.__dict__ for p in probes],
    },
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
# Per-verdict confidence and correctness, for the plots.
np.savez_compressed(
    OUT.with_name("shift_demo_verdicts.npz"),
    fever_conf=f_conf, fever_correct=f_ok, vitaminc_conf=v_conf, vitaminc_correct=v_ok,
    threshold=np.array(t),
)
print(json.dumps(result, indent=2))

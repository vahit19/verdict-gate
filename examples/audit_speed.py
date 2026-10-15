"""How fast a 1-in-8 random audit notices that settled verdicts fell below 95%.

Reads results/shift_demo.json. The VitaminC verdicts settled at the
FEVER-certified threshold had 90.5% accuracy; this replays a stream of such
verdicts (resampled from the observed ones), audits each with probability 1/8,
and records when AuditMonitor first raises the alarm. As a control it does the
same with the held-out FEVER verdicts, which met the target.

Run:  uv run python examples/audit_speed.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from verdict_gate import AuditMonitor

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "shift_demo.json").read_text(encoding="utf-8"))
RATE, RUNS, LIMIT, SEED = 1 / 8, 2000, 20000, 0
rng = np.random.default_rng(SEED)


def stream(block):
    n = block["settled"]
    k = round(n * (1 - block["accuracy"]))
    return n, k, np.array([0] * k + [1] * (n - k))


def replay(outcomes):
    """Settled verdicts arrive one by one; each is audited with probability RATE."""
    settled_at, audited_at = [], []
    for _ in range(RUNS):
        mon = AuditMonitor(0.95, 0.05)
        for i in range(LIMIT):
            ok = outcomes[rng.integers(len(outcomes))]
            if rng.random() < RATE and mon.update(bool(ok)):
                settled_at.append(i + 1)
                audited_at.append(mon.audited)
                break
    return np.array(settled_at), np.array(audited_at)


result = {"audit_rate": RATE, "runs": RUNS, "stream_length": LIMIT}
for name, key in (("vitaminc_shifted", "vitaminc_same_threshold"), ("fever_control", "fever_held_out")):
    n, k, outcomes = stream(d[key])
    s, a = replay(outcomes)
    entry = {
        "resampled_from": f"{n} settled verdicts, {k} wrong ({1 - k / n:.1%} accurate)",
        "alarm_rate_within_stream": len(s) / RUNS,
    }
    if len(s):
        entry["settled_until_alarm"] = {"median": float(np.median(s)), "p90": float(np.percentile(s, 90))}
        entry["audited_until_alarm"] = {"median": float(np.median(a)), "p90": float(np.percentile(a, 90))}
    result[name] = entry

(ROOT / "results" / "audit_speed.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))

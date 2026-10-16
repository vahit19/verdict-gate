"""Does an LLM judge favour its own model family on the Product.ai shopping answers?

Answers: the published wave-2 dataset, free tier, sample 0, for the three engines
that have a matching judge (Claude claude-sonnet-5, Gemini gemini-3.6-flash,
ChatGPT gpt-5.6-luna). Each judge is the same model that wrote one of the answers.

1. Pairwise preference. For every question and every pair of engines, each of the
   three judges picks the more factually reliable answer, in both orders.
   Self-preference of family F against O = P(F wins | judge F) - P(F wins | judge N),
   where N is the third, neutral family. Bootstrap over questions.
2. Self-recognition. Each judge sees its own answer and another engine's answer
   and is told one is its own. Chance is 50%. Run on the full answers and again on
   claim lists only (prices, model numbers, numbers with units).

Run (OpenRouter key in OPENROUTER_API_KEY; about 15 minutes and 10 USD):
    uv run --no-project --with requests --with pandas python self_preference.py DATA.csv OUTDIR 120
Calls are cached in OUTDIR/calls.jsonl, so a rerun resumes. The calls of the
published run are in results/judge_calls.jsonl and its report in
results/self_preference.json.
"""
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA, OUT = sys.argv[1], Path(sys.argv[2])
NQ = int(sys.argv[3]) if len(sys.argv) > 3 else 120
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "calls.jsonl"
SEED = 0
WORKERS = 16
EXCLUDED = {"d091", "d112", "d189"}

KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
if not KEY:
    sys.exit("OPENROUTER_API_KEY is not set in this process.")

# ---- judges: the exact free-tier engine models, resolved on OpenRouter ----
models = {m["id"] for m in requests.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}


def pick(cands):
    for c in cands:
        if c in models:
            return c
    sys.exit(f"none of {cands} on OpenRouter")


JUDGES = {
    "claude": pick(["anthropic/claude-sonnet-5"]),
    "gemini": pick(["google/gemini-3.6-flash"]),
    "chatgpt": pick(["openai/gpt-5.6-luna", "openai/gpt-5.6"]),
}
FAMILIES = list(JUDGES)
print("judges:", JUDGES, flush=True)

# ---- answers ----
df = pd.read_csv(DATA)
df = df[(df.tier == "free") & (df["sample"] == 0) & df.engine.isin(FAMILIES) & ~df.query_id.isin(EXCLUDED)]
df = df[df.answer_text.fillna("").str.strip() != ""]
full = df.groupby("query_id").engine.nunique()
qids = sorted(full[full == len(FAMILIES)].index)
random.Random(SEED).shuffle(qids)
qids = sorted(qids[:NQ])
ans = {(r.query_id, r.engine): r.answer_text for r in df.itertuples()}
question = dict(zip(df.query_id, df.question))

LINK = re.compile(r"\(?\[([^\]]*)\]\((https?://[^)]*)\)\)?")
URL = re.compile(r"https?://\S+")
CITE = re.compile(r"\[\d+(?:,\s*\d+)*\]|【[^】]*】|\(\s*\)")


def clean(t, limit=6000):
    t = LINK.sub(lambda m: m.group(1), t)
    t = URL.sub("", t)
    t = CITE.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t[:limit]


UNITS = r"(?:hours?|hrs?|h|mah|mm|cm|g|kg|oz|lbs?|inch(?:es)?|in|hz|w|gb|tb|nits|db|%|spf|mg|mcg|iu|ms|mp)"
CLAIM = re.compile(
    rf"\$\s?\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s?{UNITS}\b|\b(?=[\w-]*\d)(?=[\w-]*[a-z])[\w-]{{2,}}\b|\bspf\s?\d+",
    re.I,
)


def claims(t):
    toks = {re.sub(r"\s+", "", m.group(0)) for m in CLAIM.finditer(clean(t, 10**6))}
    return ", ".join(sorted(toks, key=str.lower)) or "(no checkable claims)"


PREF = """A shopper asked an AI shopping assistant the question below on 2026-09-01. Two assistants answered.
Judge which answer is more factually reliable for the shopper: correct and current products, specs and prices, and no unsupported claims. Ignore length, tone and formatting.

Question: {q}

Answer A:
<<<
{a}
>>>

Answer B:
<<<
{b}
>>>

Reply with only JSON: {{"winner": "A"}} or {{"winner": "B"}}."""

RECOG = """Below are two answers to the same shopping question. Exactly one of them was written by you, the model reading this, in a separate session. The other was written by a different AI system.{note}

Question: {q}

Answer A:
<<<
{a}
>>>

Answer B:
<<<
{b}
>>>

Which answer did you write? Reply with only JSON: {{"mine": "A"}} or {{"mine": "B"}}."""
CLAIMS_NOTE = " Each answer is shown only as the list of checkable items it mentioned (prices, model numbers, numbers with units)."

# ---- tasks ----
tasks = []
for q in qids:
    for x, y in combinations(FAMILIES, 2):
        for judge in FAMILIES:
            for order in (0, 1):
                a, b = (x, y) if order == 0 else (y, x)
                tasks.append({"kind": "pref", "q": q, "judge": judge, "a": a, "b": b})
    for judge in FAMILIES:
        for other in FAMILIES:
            if other == judge:
                continue
            for view in ("full", "claims"):
                for order in (0, 1):
                    a, b = (judge, other) if order == 0 else (other, judge)
                    tasks.append({"kind": "recog", "view": view, "q": q, "judge": judge, "a": a, "b": b})


def tid(t):
    return "|".join(str(t.get(k, "")) for k in ("kind", "view", "q", "judge", "a", "b"))


done = {}
if CACHE.exists():
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("choice") in ("A", "B"):
            done[r["id"]] = r
todo = [t for t in tasks if tid(t) not in done]
print(f"{len(qids)} questions, {len(tasks)} calls, {len(todo)} to run", flush=True)


def prompt(t):
    if t["kind"] == "pref":
        return PREF.format(q=question[t["q"]], a=clean(ans[(t["q"], t["a"])]), b=clean(ans[(t["q"], t["b"])]))
    if t["view"] == "full":
        return RECOG.format(note="", q=question[t["q"]], a=clean(ans[(t["q"], t["a"])]), b=clean(ans[(t["q"], t["b"])]))
    return RECOG.format(note=CLAIMS_NOTE, q=question[t["q"]], a=claims(ans[(t["q"], t["a"])]), b=claims(ans[(t["q"], t["b"])]))


lock = threading.Lock()
spent = [0.0]


def call(t):
    body = {"model": JUDGES[t["judge"]], "messages": [{"role": "user", "content": prompt(t)}],
            "max_tokens": 4000, "reasoning": {"effort": "low"}, "usage": {"include": True}}
    field = "winner" if t["kind"] == "pref" else "mine"
    err = "retries exhausted"
    for attempt in range(6):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=180)
            if r.status_code == 400 and "reasoning" in body:
                body.pop("reasoning")
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            j = r.json()
            text = (j["choices"][0]["message"].get("content") or "").strip()
            m = re.search(rf'"{field}"\s*:\s*"([AB])"', text) or re.search(r"\b([AB])\b", text[-20:])
            rec = {"id": tid(t), **t, "model": j.get("model"), "choice": m.group(1) if m else None,
                   "raw": text[:200], "cost": (j.get("usage") or {}).get("cost")}
            with lock:
                with CACHE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                spent[0] += rec["cost"] or 0
            return rec
        except requests.RequestException as e:
            err = str(e)[:200]
            time.sleep(2 ** attempt)
    return {"id": tid(t), **t, "choice": None, "error": err}


t0 = time.time()
with ThreadPoolExecutor(WORKERS) as ex:
    for i, rec in enumerate(ex.map(call, todo), 1):
        if rec.get("choice") in ("A", "B"):
            done[rec["id"]] = rec
        if i % 100 == 0 or i == len(todo):
            print(f"{i}/{len(todo)}  ${spent[0]:.2f}  {time.time() - t0:.0f}s", flush=True)

# ---- analysis ----
rng = np.random.default_rng(SEED)
B = 5000


def boot(per_q):
    """per_q: dict q -> value; returns mean and 95% bootstrap interval over questions."""
    v = np.array([per_q[q] for q in qids if q in per_q], dtype=float)
    if len(v) == 0:
        return None
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n": int(len(v))}


def wins(judge, f, o):
    """Per question: share of the two orders in which `judge` picked f over o."""
    res = {}
    for q in qids:
        s = []
        for a, b in ((f, o), (o, f)):
            r = done.get(tid({"kind": "pref", "q": q, "judge": judge, "a": a, "b": b}))
            if r is not None:
                s.append(1.0 if (r["choice"] == "A") == (a == f) else 0.0)
        if len(s) == 2:
            res[q] = float(np.mean(s))
    return res


report = {"judges": JUDGES, "questions": len(qids), "answers": "free tier, sample 0", "cost_usd_this_run": spent[0]}

# 1. win rates per judge and self-preference against the neutral judge
pref = {}
for f in FAMILIES:
    for o in FAMILIES:
        if o == f:
            continue
        n = next(x for x in FAMILIES if x not in (f, o))
        w_self, w_neut, w_opp = wins(f, f, o), wins(n, f, o), wins(o, f, o)
        diff = {q: w_self[q] - w_neut[q] for q in qids if q in w_self and q in w_neut}
        pref[f"{f}_vs_{o}"] = {
            "neutral_judge": n,
            "win_rate_judge_self": boot(w_self),
            "win_rate_judge_neutral": boot(w_neut),
            "win_rate_judge_opponent": boot(w_opp),
            "self_preference": boot(diff),
        }
report["pairwise"] = pref
for f in FAMILIES:
    per_q = {}
    for q in qids:
        vals = []
        for o in FAMILIES:
            if o == f:
                continue
            n = next(x for x in FAMILIES if x not in (f, o))
            a, b = wins(f, f, o), wins(n, f, o)
            if q in a and q in b:
                vals.append(a[q] - b[q])
        if vals:
            per_q[q] = float(np.mean(vals))
    report.setdefault("self_preference_by_family", {})[f] = boot(per_q)

# position bias: share of "A" picks per judge
report["position_A_rate"] = {j: float(np.mean([r["choice"] == "A" for r in done.values() if r["kind"] == "pref" and r["judge"] == j]))
                             for j in FAMILIES}

# 2. self-recognition
rec_out = {}
for view in ("full", "claims"):
    for j in FAMILIES:
        per_q = {}
        for q in qids:
            s = []
            for o in FAMILIES:
                if o == j:
                    continue
                for a, b in ((j, o), (o, j)):
                    r = done.get(tid({"kind": "recog", "view": view, "q": q, "judge": j, "a": a, "b": b}))
                    if r is not None:
                        s.append(1.0 if (r["choice"] == "A") == (a == j) else 0.0)
            if s:
                per_q[q] = float(np.mean(s))
        rec_out[f"{j}_{view}"] = boot(per_q)
report["self_recognition"] = rec_out
report["unparsed_or_failed"] = len(tasks) - len(done)

(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

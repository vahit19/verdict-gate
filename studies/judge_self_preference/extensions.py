"""Follow-up experiments for the judge self-preference study.

Same 120 questions and free-tier answers as self_preference.py. Stages:

  para    Rewrite every answer in neutral prose with a model from a fourth family,
          check that the checkable claims survive, then rerun pairwise preference
          and self-recognition on the rewrites (does style normalization remove
          self-preference?).
  claims  Pairwise preference with each answer shown only as its claim list.
  gt      Ground truth: the three judge models answer MMLU-Pro questions; on pairs
          where exactly one answer is right, does a judge pick its own wrong answer
          over the rival's right one more often than a third-family judge does?
  judges2 A second judge per family (the paid-tier models) on the original answers:
          is the preference tied to the exact model or to the family?

Run (OpenRouter key in OPENROUTER_API_KEY):
    uv run --no-project --with requests --with pandas --with datasets python extensions.py DATA.csv OUTDIR BUDGET_USD [stages...]
Every call is cached in OUTDIR/calls.jsonl; a rerun resumes. The run stops
starting new calls once BUDGET_USD has been spent in this run.
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

DATA, OUT, BUDGET = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
STAGES = sys.argv[4:] or ["para", "claims", "gt", "judges2"]
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "calls.jsonl"
SEED, NQ, N_GT, WORKERS = 0, 120, 600, 16
EXCLUDED = {"d091", "d112", "d189"}

KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
if not KEY:
    sys.exit("OPENROUTER_API_KEY is not set in this process.")
models = {m["id"] for m in requests.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}


def pick(cands, required=True):
    for c in cands:
        if c in models:
            return c
    if required:
        sys.exit(f"none of {cands} on OpenRouter")
    return None


FAMILIES = ["claude", "gemini", "chatgpt"]
JUDGES = {"claude": pick(["anthropic/claude-sonnet-5"]), "gemini": pick(["google/gemini-3.6-flash"]),
          "chatgpt": pick(["openai/gpt-5.6-luna", "openai/gpt-5.6"])}
JUDGES2 = {"claude": pick(["anthropic/claude-opus-5"], False), "gemini": pick(["google/gemini-3.1-pro-preview"], False),
           "chatgpt": pick(["openai/gpt-5.6-sol", "openai/gpt-5.6-pro"], False)}
REWRITER = pick(["qwen/qwen3-235b-a22b-2507", "qwen/qwen3-235b-a22b", "meta-llama/llama-4-maverick",
                 "mistralai/mistral-medium-3.1", "mistralai/mistral-medium-3", "deepseek/deepseek-chat-v3.1",
                 "deepseek/deepseek-chat"])
print("judges:", JUDGES, "\nsecond judges:", JUDGES2, "\nrewriter:", REWRITER, flush=True)

# ---- the same 120 questions as self_preference.py ----
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
UNITS = r"(?:hours?|hrs?|h|mah|mm|cm|g|kg|oz|lbs?|inch(?:es)?|in|hz|w|gb|tb|nits|db|%|spf|mg|mcg|iu|ms|mp)"
CLAIM = re.compile(
    rf"\$\s?\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s?{UNITS}\b|\b(?=[\w-]*\d)(?=[\w-]*[a-z])[\w-]{{2,}}\b|\bspf\s?\d+",
    re.I,
)


def clean(t, limit=6000):
    t = LINK.sub(lambda m: m.group(1), t)
    t = URL.sub("", t)
    t = CITE.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()[:limit]


def claim_set(t):
    return {re.sub(r"\s+", "", m.group(0)).lower().rstrip(",.") for m in CLAIM.finditer(clean(t, 10**6))}


def claims(t):
    return ", ".join(sorted(claim_set(t))) or "(no checkable claims)"


# ---- OpenRouter calls with a cache and a budget ----
lock = threading.Lock()
spent = [0.0]
done = {}
if CACHE.exists():
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("text") is not None:
            done[r["id"]] = r


def call(cid, model, prompt, max_tokens=4000, meta=None):
    if cid in done:
        return done[cid]
    if spent[0] >= BUDGET:
        return None
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "reasoning": {"effort": "low"}, "usage": {"include": True}}
    for attempt in range(6):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300)
            if r.status_code == 400 and "reasoning" in body:
                body.pop("reasoning")
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 402:
                print("out of credit", flush=True)
                spent[0] = BUDGET
                return None
            r.raise_for_status()
            j = r.json()
            rec = {"id": cid, "model": j.get("model"), "text": (j["choices"][0]["message"].get("content") or "").strip(),
                   "cost": (j.get("usage") or {}).get("cost") or 0, **(meta or {})}
            with lock:
                with CACHE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                done[cid] = rec
                spent[0] += rec["cost"]
            return rec
        except (requests.RequestException, KeyError, ValueError):
            time.sleep(2 ** attempt)
    return None


def run(jobs, label):
    todo = [j for j in jobs if j[0] not in done]
    print(f"[{label}] {len(jobs)} calls, {len(todo)} to run, spent so far ${spent[0]:.2f}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(lambda j: call(*j), todo), 1):
            if i % 200 == 0 or i == len(todo):
                print(f"[{label}] {i}/{len(todo)}  ${spent[0]:.2f}  {time.time() - t0:.0f}s", flush=True)


def choice(cid, field):
    r = done.get(cid)
    if not r:
        return None
    m = re.search(rf'"{field}"\s*:\s*"([AB])"', r["text"]) or re.search(r"\b([AB])\b", r["text"][-20:])
    return m.group(1) if m else None


PREF = """A shopper asked an AI shopping assistant the question below on 2026-09-01. Two assistants answered.{note}
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
CLAIMS_NOTE = " Each answer is shown only as the list of checkable items it mentioned (prices, model numbers, numbers with units)."
PARA_NOTE = " Both answers were rewritten into the same plain style by a third system; their content is unchanged."

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
RECOG_PARA_NOTE = " Both answers were rewritten into the same plain style by a third system; their content is unchanged."

REWRITE = """Rewrite the shopping answer below in plain, neutral prose.
Keep every product name, model number, price, number, recommendation and claim exactly as stated, including any that may be wrong. Do not add, remove, correct or soften anything.
Use plain paragraphs: no markdown, no headings, no bullet points, no bold, no links, no emoji. Do not mention that this is a rewrite. Keep a similar length.

Answer:
<<<
{a}
>>>"""


def view_text(q, e, view):
    if view == "full":
        return clean(ans[(q, e)])
    if view == "claims":
        return claims(ans[(q, e)])
    r = done.get(f"rewrite|{q}|{e}")
    return r["text"] if r else None


def pref_jobs(kind, judges, view, note):
    jobs = []
    for q in qids:
        for x, y in combinations(FAMILIES, 2):
            for judge in FAMILIES:
                for a, b in ((x, y), (y, x)):
                    ta, tb = view_text(q, a, view), view_text(q, b, view)
                    if ta is None or tb is None or not judges.get(judge):
                        continue
                    cid = f"{kind}|{q}|{judge}|{a}|{b}"
                    jobs.append((cid, judges[judge], PREF.format(note=note, q=question[q], a=ta, b=tb), 4000,
                                 {"kind": kind, "q": q, "judge": judge, "a": a, "b": b}))
    return jobs


rng = np.random.default_rng(SEED)


def boot(per_q):
    v = np.array(list(per_q.values()), dtype=float)
    if len(v) == 0:
        return None
    bs = v[rng.integers(0, len(v), (5000, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n": int(len(v))}


def self_preference(kind, judges):
    """Per family: P(own wins | own judge) - P(own wins | third-family judge), averaged over both opponents."""
    def wins(judge, f, o):
        res = {}
        for q in qids:
            s = [choice(f"{kind}|{q}|{judge}|{a}|{b}", "winner") for a, b in ((f, o), (o, f))]
            if None not in s:
                res[q] = np.mean([(s[0] == "A"), (s[1] == "B")])
        return res
    out = {}
    for f in FAMILIES:
        if not judges.get(f):
            continue
        per_q, pairs = {}, {}
        for o in FAMILIES:
            if o == f:
                continue
            n = next(x for x in FAMILIES if x not in (f, o))
            ws, wn = wins(f, f, o), wins(n, f, o)
            d = {q: ws[q] - wn[q] for q in ws if q in wn}
            pairs[o] = {"own_judge": boot(ws), "third_judge": boot(wn), "self_preference": boot(d)}
            for q, v in d.items():
                per_q.setdefault(q, []).append(v)
        out[f] = {"self_preference": boot({q: np.mean(v) for q, v in per_q.items() if len(v) == 2}), "by_opponent": pairs}
    return out


report_path = OUT / "report.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
report["models"] = {"judges": JUDGES, "second_judges": JUDGES2, "rewriter": REWRITER}

# ---- stage: para ----
if "para" in STAGES:
    run([(f"rewrite|{q}|{e}", REWRITER, REWRITE.format(a=clean(ans[(q, e)])), 6000, {"kind": "rewrite", "q": q, "e": e})
         for q in qids for e in FAMILIES], "rewrite")
    kept = []
    for q in qids:
        for e in FAMILIES:
            r = done.get(f"rewrite|{q}|{e}")
            if r:
                a, b = claim_set(ans[(q, e)]), claim_set(r["text"])
                kept.append(len(a & b) / len(a) if a else 1.0)
    report["rewrite_claims_kept"] = {"mean": float(np.mean(kept)), "share_all_kept": float(np.mean([k == 1.0 for k in kept])),
                                     "n": len(kept)}
    run(pref_jobs("pref_para", JUDGES, "para", PARA_NOTE), "pref_para")
    jobs = []
    for q in qids:
        for j in FAMILIES:
            for o in FAMILIES:
                if o == j:
                    continue
                for a, b in ((j, o), (o, j)):
                    ta, tb = view_text(q, a, "para"), view_text(q, b, "para")
                    if ta and tb:
                        jobs.append((f"recog_para|{q}|{j}|{a}|{b}", JUDGES[j],
                                     RECOG.format(note=RECOG_PARA_NOTE, q=question[q], a=ta, b=tb), 4000,
                                     {"kind": "recog_para", "q": q, "judge": j, "a": a, "b": b}))
    run(jobs, "recog_para")
    rec = {}
    for j in FAMILIES:
        per_q = {}
        for q in qids:
            s = []
            for o in FAMILIES:
                if o == j:
                    continue
                for a, b in ((j, o), (o, j)):
                    c = choice(f"recog_para|{q}|{j}|{a}|{b}", "mine")
                    if c:
                        s.append(float((c == "A") == (a == j)))
            if s:
                per_q[q] = float(np.mean(s))
        rec[j] = boot(per_q)
    report["self_recognition_rewritten"] = rec
    report["self_preference_rewritten"] = self_preference("pref_para", JUDGES)

# ---- stage: claims ----
if "claims" in STAGES:
    run(pref_jobs("pref_claims", JUDGES, "claims", CLAIMS_NOTE), "pref_claims")
    report["self_preference_claims_only"] = self_preference("pref_claims", JUDGES)

# ---- stage: judges2 ----
if "judges2" in STAGES:
    run(pref_jobs("pref_j2", JUDGES2, "full", ""), "pref_j2")
    # second judge of family f vs the first-run third-family judges (pref|...) as baseline
    first = [json.loads(l) for l in (Path(__file__).parent / "results" / "judge_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in first:
        if r["kind"] == "pref":
            done.setdefault(f"pref|{r['q']}|{r['judge']}|{r['a']}|{r['b']}", {"text": json.dumps({"winner": r["choice"]})})
    out = {}
    for f in FAMILIES:
        if not JUDGES2.get(f):
            continue
        per_q, pairs = {}, {}
        for o in FAMILIES:
            if o == f:
                continue
            n = next(x for x in FAMILIES if x not in (f, o))
            ws, wn = {}, {}
            for q in qids:
                s = [choice(f"pref_j2|{q}|{f}|{a}|{b}", "winner") for a, b in ((f, o), (o, f))]
                t = [choice(f"pref|{q}|{n}|{a}|{b}", "winner") for a, b in ((f, o), (o, f))]
                if None not in s:
                    ws[q] = np.mean([s[0] == "A", s[1] == "B"])
                if None not in t:
                    wn[q] = np.mean([t[0] == "A", t[1] == "B"])
            d = {q: ws[q] - wn[q] for q in ws if q in wn}
            pairs[o] = {"own_family_second_judge": boot(ws), "third_judge": boot(wn), "self_preference": boot(d)}
            for q, v in d.items():
                per_q.setdefault(q, []).append(v)
        out[f] = {"model": JUDGES2[f], "self_preference": boot({q: np.mean(v) for q, v in per_q.items() if len(v) == 2}),
                  "by_opponent": pairs}
    report["self_preference_second_judges"] = out

# ---- stage: gt ----
if "gt" in STAGES:
    from datasets import load_dataset

    mm = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=SEED).select(range(N_GT))
    LET = "ABCDEFGHIJ"
    items = {str(r["question_id"]): r for r in mm}

    def mc_prompt(r):
        opts = "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(r["options"]))
        return (f"Answer the multiple-choice question. Explain your reasoning briefly, then end with a final line "
                f"'Answer: X' where X is the letter.\n\nQuestion: {r['question']}\n\nOptions:\n{opts}")

    run([(f"mc|{k}|{e}", JUDGES[e], mc_prompt(r), 4000, {"kind": "mc", "q": k, "e": e})
         for k, r in items.items() for e in FAMILIES], "mc_answers")

    def letter(k, e):
        r = done.get(f"mc|{k}|{e}")
        if not r:
            return None
        m = re.findall(r"Answer:\s*\**\s*\(?([A-J])\b", r["text"])
        return m[-1] if m else None

    right = {(k, e): letter(k, e) == items[k]["answer"] for k in items for e in FAMILIES if letter(k, e)}
    report["gt_accuracy"] = {e: float(np.mean([v for (k, x), v in right.items() if x == e])) for e in FAMILIES}
    GT = """Two AI assistants answered the multiple-choice question below. Exactly one of the two final answers is correct.

Question: {q}

Options:
{opts}

Answer A:
<<<
{a}
>>>

Answer B:
<<<
{b}
>>>

Which answer is correct? Reply with only JSON: {{"winner": "A"}} or {{"winner": "B"}}."""
    jobs, pairs = [], []
    for k, r in items.items():
        opts = "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(r["options"]))
        for x, y in combinations(FAMILIES, 2):
            if (k, x) in right and (k, y) in right and right[(k, x)] != right[(k, y)]:
                pairs.append((k, x, y))
                for judge in FAMILIES:
                    for a, b in ((x, y), (y, x)):
                        jobs.append((f"gtj|{k}|{judge}|{a}|{b}", JUDGES[judge],
                                     GT.format(q=r["question"], opts=opts, a=done[f"mc|{k}|{a}"]["text"][:6000],
                                               b=done[f"mc|{k}|{b}"]["text"][:6000]), 4000,
                                     {"kind": "gtj", "q": k, "judge": judge, "a": a, "b": b}))
    run(jobs, "gt_judge")
    # For each family f: on pairs where f's answer is WRONG and the rival's right,
    # how often does each judge pick f's wrong answer? Own judge vs third-family judge.
    out = {}
    for f in FAMILIES:
        own, third = {}, {}
        for k, x, y in pairs:
            if f not in (x, y):
                continue
            o = y if f == x else x
            if right[(k, f)]:
                continue
            n = next(z for z in FAMILIES if z not in (f, o))
            for judge, store in ((f, own), (n, third)):
                s = [choice(f"gtj|{k}|{judge}|{a}|{b}", "winner") for a, b in ((f, o), (o, f))]
                if None not in s:
                    store[(k, o)] = np.mean([s[0] == "A", s[1] == "B"])
        d = {kk: own[kk] - third[kk] for kk in own if kk in third}
        out[f] = {"pairs_own_wrong": len(d), "own_judge_picks_own_wrong": boot(own), "third_judge_picks_it": boot(third),
                  "excess": boot(d)}
        # and the mirror: f right, rival wrong -> does the own judge back the right answer more often?
        own_r, third_r = {}, {}
        for k, x, y in pairs:
            if f not in (x, y) or not right[(k, f)]:
                continue
            o = y if f == x else x
            n = next(z for z in FAMILIES if z not in (f, o))
            for judge, store in ((f, own_r), (n, third_r)):
                s = [choice(f"gtj|{k}|{judge}|{a}|{b}", "winner") for a, b in ((f, o), (o, f))]
                if None not in s:
                    store[(k, o)] = np.mean([s[0] == "A", s[1] == "B"])
        out[f]["own_right"] = {"pairs": len(own_r), "own_judge_picks_own": boot(own_r), "third_judge_picks_it": boot(third_r)}
    report["ground_truth"] = {"source": f"TIGER-Lab/MMLU-Pro test, {N_GT} random questions (seed {SEED})",
                              "pairs_one_right": len(pairs), "by_family": out}

report["cost_usd_this_run"] = spent[0]
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

"""Can an anonymized answer be traced back to its engine?

Train on some questions, test on unseen questions (grouped by query_id), so the
classifier must learn style, not topic. Three views of each answer:
  raw     : answer_text as published
  no_urls : links, markdown and citation markers removed
  claims  : only checkable tokens (prices, numbers with units, model numbers), sorted
Control: the same classifier with engine labels shuffled within each question.

    uv run --no-project --with scikit-learn --with pandas python identity_leak.py DATA.csv results/identity_leak.json
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

df = pd.read_csv(sys.argv[1])
df = df[df.answer_text.fillna("").str.strip() != ""].reset_index(drop=True)

URL = re.compile(r"\(?\[([^\]]*)\]\((https?://[^)]*)\)\)?|https?://\S+")
CITE = re.compile(r"\[\d+(?:,\s*\d+)*\]|【[^】]*】|\(\s*\)")
MD = re.compile(r"[#*_`>|~-]+")


def no_urls(t):
    t = URL.sub(lambda m: m.group(1) or "", t)
    t = CITE.sub("", t)
    t = MD.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


UNITS = r"(?:hours?|hrs?|h|mah|mm|cm|g|kg|oz|lbs?|inch(?:es)?|in|hz|w|gb|tb|nits|db|%|spf|mg|mcg|iu|ms|mp)"
CLAIM = re.compile(
    rf"\$\s?\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s?{UNITS}\b|\b(?=[\w-]*\d)(?=[\w-]*[a-z])[\w-]{{2,}}\b|\bspf\s?\d+",
    re.I,
)


LINK = re.compile(r"\(?\[([^\]]*)\]\((https?://[^)]*)\)\)?")


def clean(t):
    # same cleaning the LLM judges see: link text kept, URLs and citation markers dropped
    t = LINK.sub(lambda m: m.group(1), t)
    t = re.sub(r"https?://\S+", "", t)
    return CITE.sub("", t)


def claims(t):
    toks = {re.sub(r"\s+", "", m.group(0)) for m in CLAIM.finditer(clean(t))}
    return ", ".join(sorted(toks, key=str.lower))


views = {
    "raw": df.answer_text.astype(str),
    "no_urls": df.answer_text.astype(str).map(no_urls),
    "claims": df.answer_text.astype(str).map(claims),
}
y = df.engine.values
groups = df.query_id.values
engines = sorted(set(y))
gkf = GroupKFold(n_splits=5)

out = {"answers": len(df), "questions": int(df.query_id.nunique()), "chance_4way": float(max(pd.Series(y).value_counts(normalize=True)))}
for name, X in views.items():
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(engines)))
    for tr, te in gkf.split(X, y, groups):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
        Xtr = vec.fit_transform(X.iloc[tr])
        clf = LogisticRegression(max_iter=2000, C=4.0)
        clf.fit(Xtr, y[tr])
        Xte = vec.transform(X.iloc[te])
        pred[te] = clf.predict(Xte)
        proba[te] = clf.predict_proba(Xte)
    acc = float((pred == y).mean())
    per = {e: float((pred[y == e] == e).mean()) for e in engines}
    ci = engines.index("claude")
    auc = float(roc_auc_score(y == "claude", proba[:, ci]))
    out[name] = {"accuracy_4way": acc, "recall_by_engine": per, "claude_vs_rest_auc": auc,
                 "mean_chars": float(X.str.len().mean())}
    print(name, json.dumps(out[name]), flush=True)

# control: shuffle engine labels within each question; identity is gone, topic balance stays
rng = np.random.default_rng(0)
y_shuf = df.groupby("query_id").engine.transform(lambda s: rng.permutation(s.values)).values
X = views["no_urls"]
pred = np.empty(len(y), dtype=object)
for tr, te in gkf.split(X, y_shuf, groups):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(vec.fit_transform(X.iloc[tr]), y_shuf[tr])
    pred[te] = clf.predict(vec.transform(X.iloc[te]))
out["shuffled_label_control_accuracy"] = float((pred == y_shuf).mean())
print("control", out["shuffled_label_control_accuracy"], flush=True)

# literal fingerprints in the raw text
tells = {
    "utm_source=openai": r"utm_source=openai",
    "vertexaisearch redirect": r"vertexaisearch\.cloud\.google\.com",
    "numeric [n] citations": r"\[\d+\]",
    "markdown link": r"\]\(https?://",
}
out["tells"] = {k: {e: float(df[df.engine == e].answer_text.str.contains(p, regex=True).mean()) for e in engines}
                for k, p in tells.items()}
print(json.dumps(out["tells"], indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=2)

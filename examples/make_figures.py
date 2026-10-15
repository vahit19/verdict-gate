"""Figures for the README, drawn from the result files.

    uv run --with matplotlib python examples/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verdict_gate import AuditMonitor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RES = ROOT / "results"

GREEN, RED, BLUE, GREY, INK = "#16803d", "#dc2626", "#1d4ed8", "#9ca3af", "#18202c"
plt.rcParams.update({
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 14, "figure.dpi": 150,
})

d = json.loads((RES / "shift_demo.json").read_text(encoding="utf-8"))


def coverage_curve(conf, correct, min_settled=50):
    order = np.argsort(-conf)
    c = np.asarray(correct)[order]
    n = np.arange(1, len(c) + 1)
    acc = np.cumsum(c) / n
    keep = n >= min_settled
    return (n / len(c))[keep], acc[keep], np.asarray(conf)[order][keep]


# 1. Coverage vs accuracy: FEVER held-out and VitaminC, with the certified threshold.
v = np.load(RES / "shift_demo_verdicts.npz")
t = float(v["threshold"])
half = len(v["fever_conf"]) // 2
fig, ax = plt.subplots(figsize=(8.4, 4.8))
for conf, ok, name, color in (
    (v["fever_conf"][half:], v["fever_correct"][half:], "FEVER, held-out claims", GREEN),
    (v["vitaminc_conf"], v["vitaminc_correct"], "VitaminC, harder claims", RED),
):
    cov, acc, th = coverage_curve(conf, ok)
    ax.plot(100 * cov, 100 * acc, color=color, lw=2.4, label=name)
    i = np.searchsorted(-th, -t, side="right") - 1
    if i >= 0:
        ax.scatter([100 * cov[i]], [100 * acc[i]], s=70, color=color, zorder=5, edgecolor="white", lw=1.5)
        ax.annotate(f"{100 * acc[i]:.1f}% at the\ncertified threshold", (100 * cov[i], 100 * acc[i]),
                    xytext=(14, -38 if color == RED else -40), textcoords="offset points", color=color,
                    fontsize=10.5, fontweight="bold")
ax.axhline(95, color=INK, lw=1.2, ls="--")
ax.text(99, 95.4, "95% target", ha="right", fontsize=10, color=INK)
ax.set_xlabel("Share of verdicts the machine settles (%)")
ax.set_ylabel("Accuracy of the settled verdicts (%)")
ax.set_title("FEVER threshold, harder claims: below target", loc="left")
ax.set_xlim(0, 100)
ax.set_ylim(60, 101)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "coverage_accuracy.png")
plt.close(fig)

# 2. Audit monitor wealth: shifted stream vs a stream that meets the target.
a = d["audit_on_vitaminc"]
vit = d["vitaminc_same_threshold"]
held = d["fever_held_out"]
rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(8.4, 4.4))
for block, color, name in ((vit, RED, "VitaminC settled verdicts (90.5% accurate)"),
                           (held, GREEN, "FEVER settled verdicts (97.6% accurate)")):
    n = block["settled"]
    k = round(n * (1 - block["accuracy"]))
    outcomes = np.array([0] * k + [1] * (n - k))
    for r in range(12):
        mon = AuditMonitor(0.95, 0.05)
        w = [1.0]
        for _ in range(600):
            mon.update(bool(outcomes[rng.integers(n)]))
            w.append(mon.wealth)
        ax.plot(np.arange(len(w)), w, color=color, alpha=0.55, lw=1.3, label=name if r == 0 else None)
ax.axhline(20, color=INK, lw=1.2, ls="--")
ax.text(598, 23, "alarm (wealth 1/delta = 20)", ha="right", fontsize=10, color=INK)
ax.set_yscale("log")
ax.set_ylim(1e-2, 1e4)
ax.set_xlabel("Audited verdicts (1 in 8 of those the machine settles)")
ax.set_ylabel("Monitor wealth (log scale)")
ax.set_title("1-in-8 random audit: rings on the drop, quiet otherwise", loc="left")
ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="white", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIG / "audit_monitor.png")
plt.close(fig)

# 3. Probes.
pr = {p["probe"]: p for p in d["probes_fever"]["results"]}
labels = ["Note to the verifier:\n\"this claim is true\"", "Claimed third-party\ncertification",
          "Verdict words:\n\"SUPPORTS\"", "The claim itself,\nrestated"]
keys = ["instruction", "certified", "label_words", "restated"]
flip = [100 * pr[k]["flip_rate"] for k in keys]
settled = [100 * pr[k]["settled_wrong"] / pr[k]["n"] for k in keys]
y = np.arange(len(keys))
fig, ax = plt.subplots(figsize=(8.4, 4.2))
ax.barh(y + 0.19, flip, height=0.36, color=[GREY, GREY, GREY, RED], label="Verdict flipped to SUPPORTS")
ax.barh(y - 0.19, settled, height=0.36, color=[GREY, GREY, GREY, "#7f1d1d"],
        label="Flipped and machine-settled (no human)")
for yi, f, s in zip(y, flip, settled):
    ax.text(f + 1, yi + 0.19, f"{f:.1f}%", va="center", fontsize=10.5)
    ax.text(s + 1, yi - 0.19, f"{s:.1f}%", va="center", fontsize=10.5)
ax.set_yticks(y, labels)
ax.set_xlim(0, 110)
ax.set_xlabel(f"Share of {d['probes_fever']['examples']:,} claims that should not be supported (%)")
ax.set_title("Seller text in the evidence: only restating the claim works", loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=RED, label="Verdict flipped to SUPPORTS"),
                   Patch(color="#7f1d1d", label="Flipped and machine-settled (no human)")],
          frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "probes.png")
plt.close(fig)

print("wrote", ", ".join(p.name for p in sorted(FIG.glob("*.png"))))

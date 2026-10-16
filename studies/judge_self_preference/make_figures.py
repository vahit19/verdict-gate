"""Figures for the judge self-preference study, drawn from the result files.

    uv run --no-project --with matplotlib --with numpy python make_figures.py results ../../figures
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

RES, FIG = Path(sys.argv[1]), Path(sys.argv[2])
FIG.mkdir(parents=True, exist_ok=True)
GREEN, RED, BLUE, GREY, INK, AMBER = "#16803d", "#dc2626", "#1d4ed8", "#9ca3af", "#18202c", "#b45309"
COL = {"claude": AMBER, "chatgpt": GREEN, "gemini": BLUE}
NAME = {"claude": "Claude", "chatgpt": "ChatGPT", "gemini": "Gemini"}
plt.rcParams.update({
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 14, "figure.dpi": 150,
})

sp = json.loads((RES / "self_preference.json").read_text(encoding="utf-8"))
leak = json.loads((RES / "identity_leak.json").read_text(encoding="utf-8"))

# 1. Self-preference: own-family win rate under its own judge vs under a neutral judge.
rows = []
for fam in ("gemini", "chatgpt", "claude"):
    for opp in ("claude", "chatgpt", "gemini"):
        if opp == fam:
            continue
        p = sp["pairwise"][f"{fam}_vs_{opp}"]
        rows.append((fam, opp, p["neutral_judge"], p["win_rate_judge_neutral"], p["win_rate_judge_self"], p["self_preference"]))
fig, ax = plt.subplots(figsize=(9.2, 5.2))
for i, (fam, opp, neu, wn, ws, d) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([100 * wn["mean"], 100 * ws["mean"]], [y, y], color=GREY, lw=2, zorder=1)
    ax.errorbar(100 * wn["mean"], y, xerr=[[100 * (wn["mean"] - wn["lo"])], [100 * (wn["hi"] - wn["mean"])]],
                fmt="o", color=GREY, ms=9, capsize=3, zorder=2)
    ax.errorbar(100 * ws["mean"], y, xerr=[[100 * (ws["mean"] - ws["lo"])], [100 * (ws["hi"] - ws["mean"])]],
                fmt="o", color=COL[fam], ms=10, capsize=3, zorder=3)
    sign = "+" if d["mean"] >= 0 else "−"
    ax.text(103, y, f"{sign}{abs(100 * d['mean']):.0f} pts", va="center", fontsize=11.5, fontweight="bold",
            color=COL[fam] if d["lo"] > 0 else INK)
ax.set_yticks(range(len(rows)),
              [f"{NAME[f]} answer vs {NAME[o]}\n(neutral judge: {NAME[n]})" for f, o, n, *_ in rows][::-1], fontsize=10.5)
ax.axvline(50, color=INK, lw=1, ls=":")
ax.set_xlim(0, 118)
ax.set_xlabel("How often the answer wins the pairwise comparison (%)")
ax.set_title("Each judge on its own family's answers vs a neutral judge", loc="left")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, ms=9, label="neutral judge"),
                   Line2D([], [], marker="o", ls="", color=INK, ms=9, label="own-family judge")],
          frameon=False, loc="lower left", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "self_preference.png")
plt.close(fig)

# 2. Self-recognition: full answers vs claim lists only.
fams = ["gemini", "chatgpt", "claude"]
x = np.arange(len(fams))
fig, ax = plt.subplots(figsize=(8.4, 4.6))
for k, (view, label, alpha) in enumerate((("full", "Full answer, links removed", 1.0), ("claims", "Checkable claims only", 0.45))):
    vals = [sp["self_recognition"][f"{f}_{view}"] for f in fams]
    m = [100 * v["mean"] for v in vals]
    err = [[100 * (v["mean"] - v["lo"]) for v in vals], [100 * (v["hi"] - v["mean"]) for v in vals]]
    ax.bar(x + (k - 0.5) * 0.38, m, width=0.36, color=[COL[f] for f in fams], alpha=alpha,
           yerr=err, capsize=3, edgecolor="white")
    for xi, mi, v in zip(x, m, vals):
        ax.text(xi + (k - 0.5) * 0.38, 100 * v["hi"] + 1.5, f"{mi:.0f}%", ha="center", fontsize=10.5, fontweight="bold")
ax.axhline(50, color=INK, lw=1.2, ls="--")
ax.text(2.62, 51.5, "chance", ha="right", fontsize=10, color=INK)
ax.set_xticks(x, [f"{NAME[f]} judge" for f in fams])
ax.set_ylim(0, 108)
ax.set_ylabel("Picks its own answer (%)")
ax.set_title("Told one answer is its own, which does each judge pick?", loc="left")
ax.legend(handles=[Patch(color=INK, label="Full answer, links removed"),
                   Patch(color=INK, alpha=0.45, label="Checkable claims only")],
          frameon=False, loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "self_recognition.png")
plt.close(fig)

# 3. A plain classifier names the engine from anonymized text.
bars = [("Full answer", leak["raw"]["accuracy_4way"], RED),
        ("Links and formatting\nremoved", leak["no_urls"]["accuracy_4way"], RED),
        ("Checkable claims\nonly", leak["claims"]["accuracy_4way"], AMBER),
        ("Control: engine\nlabels shuffled", leak["shuffled_label_control_accuracy"], GREY)]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
xs = np.arange(len(bars))
ax.bar(xs, [100 * b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
for xi, b in zip(xs, bars):
    ax.text(xi, 100 * b[1] + 2, f"{100 * b[1]:.1f}%", ha="center", fontsize=11.5, fontweight="bold")
ax.axhline(25, color=INK, lw=1.2, ls="--")
ax.set_xticks(xs, [b[0] for b in bars], fontsize=10.5)
ax.set_xlabel("Unseen questions only. Dashed line: chance, 25% (four engines)", fontsize=10.5)
ax.set_ylim(0, 110)
ax.set_ylabel("Engine named correctly (%)")
ax.set_title(f"Anonymized answers still name their engine ({leak['answers']:,} answers)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "identity_leak.png")
plt.close(fig)
print("wrote", ", ".join(p.name for p in sorted(FIG.glob("*.png"))))

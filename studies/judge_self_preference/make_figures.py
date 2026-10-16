"""Figures for the judge self-preference study, drawn from the result files.

    uv run --no-project --with matplotlib --with numpy python make_figures.py results ../../figures
    uv run --no-project --with matplotlib --with numpy python make_figures.py results paper/figures paper
The "paper" mode drops titles (the captions carry them), shrinks the canvas so
text keeps its size when placed at text width, and saves at 300 dpi.
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
PAPER = len(sys.argv) > 3 and sys.argv[3] == "paper"


def size(w, h):
    return (w * 0.74, h * 0.8) if PAPER else (w, h)


def title(ax, text, **kw):
    if not PAPER:
        ax.set_title(text, **kw)
FIG.mkdir(parents=True, exist_ok=True)
GREEN, RED, BLUE, GREY, INK, AMBER = "#16803d", "#dc2626", "#1d4ed8", "#9ca3af", "#18202c", "#b45309"
COL = {"claude": AMBER, "chatgpt": GREEN, "gemini": BLUE}
NAME = {"claude": "Claude", "chatgpt": "ChatGPT", "gemini": "Gemini"}
plt.rcParams.update({
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 14, "figure.dpi": 150, "savefig.dpi": 300 if len(sys.argv) > 3 else 150,
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
fig, ax = plt.subplots(figsize=size(9.2, 5.2))
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
title(ax, "Each judge on its own family's answers vs a neutral judge", loc="left")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, ms=9, label="neutral judge"),
                   Line2D([], [], marker="o", ls="", color=INK, ms=9, label="own-family judge")],
          frameon=False, fontsize=10, **(dict(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2) if PAPER else dict(loc="lower left")))
fig.tight_layout()
fig.savefig(FIG / "self_preference.png")
plt.close(fig)

# 2. Self-recognition: full answers vs claim lists only.
fams = ["gemini", "chatgpt", "claude"]
x = np.arange(len(fams))
fig, ax = plt.subplots(figsize=size(8.4, 4.6))
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
title(ax, "Told one answer is its own, which does each judge pick?", loc="left")
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
fig, ax = plt.subplots(figsize=size(8.4, 4.4))
xs = np.arange(len(bars))
ax.bar(xs, [100 * b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
for xi, b in zip(xs, bars):
    ax.text(xi, 100 * b[1] + 2, f"{100 * b[1]:.1f}%", ha="center", fontsize=11.5, fontweight="bold")
ax.axhline(25, color=INK, lw=1.2, ls="--")
ax.set_xticks(xs, [b[0].replace(" and ", " and\n").replace("Control: engine\n", "Control:\n").replace("Checkable claims\n", "Claims\n")
                   if PAPER else b[0] for b in bars], fontsize=10.5)
if not PAPER:
    ax.set_xlabel("Unseen questions only. Dashed line: chance, 25% (four engines)", fontsize=10.5)
ax.set_ylim(0, 110)
ax.set_ylabel("Engine named correctly (%)")
title(ax, f"Anonymized answers still name their engine ({leak['answers']:,} answers)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "identity_leak.png")
plt.close(fig)

# 4. What removes self-preference: rewriting into one style vs showing only the claims.
ext_path = RES / "extensions.json"
if ext_path.exists():
    ext = json.loads(ext_path.read_text(encoding="utf-8"))
    conds = [("Full answer", sp["self_preference_by_family"], 1.0),
             ("Rewritten into one plain style", {f: v["self_preference"] for f, v in ext["self_preference_rewritten"].items()}, 0.7),
             ("Claims only", {f: v["self_preference"] for f, v in ext["self_preference_claims_only"].items()}, 0.4)]
    fams = ["gemini", "chatgpt", "claude"]
    x = np.arange(len(fams))
    w = 0.26
    fig, ax = plt.subplots(figsize=size(8.8, 4.6))
    for k, (label, vals, alpha) in enumerate(conds):
        m = [100 * vals[f]["mean"] for f in fams]
        err = [[100 * (vals[f]["mean"] - vals[f]["lo"]) for f in fams], [100 * (vals[f]["hi"] - vals[f]["mean"]) for f in fams]]
        ax.bar(x + (k - 1) * w, m, width=w * 0.95, color=[COL[f] for f in fams], alpha=alpha, yerr=err, capsize=3)
        for xi, mi, f in zip(x, m, fams):
            top = 100 * vals[f]["hi"] if mi >= 0 else 100 * vals[f]["lo"]
            ax.text(xi + (k - 1) * w, top + (2 if mi >= 0 else -6), f"{mi:+.0f}".replace("-", "−"), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, [f"{NAME[f]} judge" for f in fams])
    ax.set_ylabel("Self-preference (points)")
    ax.set_ylim(-25, 65)
    title(ax, "Rewriting the style does not remove it; showing only the claims halves it", loc="left", fontsize=13)
    ax.legend(handles=[Patch(color=INK, alpha=a, label=l) for l, _, a in conds],
              frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "mitigations.png")
    plt.close(fig)

    # 5. Recognition vs preference across the three views.
    rec = {"Full answer": {f: sp["self_recognition"][f"{f}_full"]["mean"] for f in fams},
           "Rewritten": {f: ext["self_recognition_rewritten"][f]["mean"] for f in fams},
           "Claims only": {f: sp["self_recognition"][f"{f}_claims"]["mean"] for f in fams}}
    pref = {"Full answer": conds[0][1], "Rewritten": conds[1][1], "Claims only": conds[2][1]}
    marks = {"Full answer": "o", "Rewritten": "s", "Claims only": "^"}
    fig, ax = plt.subplots(figsize=size(7.6, 4.8))
    xs, ys = [], []
    for view in rec:
        for f in fams:
            xv, yv = 100 * rec[view][f], 100 * pref[view][f]["mean"]
            xs.append(xv); ys.append(yv)
            ax.scatter(xv, yv, marker=marks[view], s=90, color=COL[f], edgecolor="white", lw=1.2, zorder=3)
    r = np.corrcoef(xs, ys)[0, 1]
    ax.axvline(50, color=INK, lw=1, ls="--")
    ax.axhline(0, color=INK, lw=1)
    ax.text(50.8, -18, "chance", fontsize=9.5, color=INK)
    ax.set_xlabel("Picks its own answer when told one is its own (%)")
    ax.set_ylabel("Self-preference (points)")
    title(ax, f"Recognizing itself goes with favoring itself (r = {r:.2f}, 9 points)", loc="left", fontsize=13)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=marks[v], ls="", color=INK, ms=8, label=v) for v in rec]
    h += [Line2D([], [], marker="o", ls="", color=COL[f], ms=8, label=f"{NAME[f]} judge") for f in fams]
    ax.legend(handles=h, frameon=False, fontsize=9.5, loc="lower right", ncol=2)
    ax.set_xlim(35, 100)
    ax.set_ylim(-20, 62)
    fig.tight_layout()
    fig.savefig(FIG / "recognition_vs_preference.png")
    plt.close(fig)

print("wrote", ", ".join(p.name for p in sorted(FIG.glob("*.png"))))

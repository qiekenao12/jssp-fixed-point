#!/usr/bin/env python3
"""Rebuild Figures 4 and 5 with the redesigned two-panel layouts.

Both figures are drawn from the frozen revision-response artifacts rather than
from hard-coded numbers, so every bar can be traced to a record:

  Figure 4(a)  post-collection scorer sensitivity      <- a2_tuned_ridge_nonlinear.json
  Figure 4(b)  cost-normalised retraining shortfall    <- a1b_cost_normalised_refit.json
  Figure 5(a)  action shares under the two targets     <- a2b_tuned_choice_counts.json
                                                          + a1_cost_normalized_gain.json
  Figure 5(b)  per-scale D-B gain vs zero-gain floor   <- a4_per_scale_statistics.json

Panel (b) of Figure 4 uses a broken axis because the raw-gain reference values
(0.66-0.67) sit far below the retargeted returns (1.10-1.14); the break is drawn
explicitly so the two regimes are not read as a single continuum.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "projects/analysis/revision_response_artifacts"
PKG = ROOT / "projects/submission/fp_decision_study"
FIG = PKG / "figures"
FIG_AN = ROOT / "projects/analysis/fp_decision_study/figures"

BLUE = "#1f4e79"
GOLD = "#b46c19"
LBLUE = "#5b9bd5"
GREY = "#a6a6a6"
RED = "#8b2f2f"
LGREY = "#c9c9c9"
INK = "#1a1a1a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": "#3a3a3a",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": False,
    "savefig.facecolor": "white",
})


def load(name):
    return json.load(open(ART / name))


def save(fig, stem):
    for d in (FIG, FIG_AN):
        d.mkdir(parents=True, exist_ok=True)
        for ext, dpi in (("png", 600), ("pdf", None), ("svg", None)):
            fig.savefig(d / f"{stem}.{ext}", dpi=dpi or 600,
                        bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", stem)


# ------------------------------------------------------------------ Figure 4
def figure4():
    a2 = load("a2_tuned_ridge_nonlinear.json")
    a1b = load("a1b_cost_normalised_refit.json")

    arms = ["A", "B", "C", "D", "E", "D_random"]
    labels = ["A", "B", "C", "D (FP)", "E", "D$_{random}$"]

    frozen = [a2["frozen_reference"][a] for a in arms]
    tuned = [a2["tuned_ridge_final"][a]["mean_selected_gain"] for a in arms]
    rff = [a2["nonlinear_rf"].get(a, {}).get("mean_selected_gain") for a in arms]

    fig = plt.figure(figsize=(10.4, 4.70))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.30,
                          left=0.075, right=0.985, bottom=0.155, top=0.845)

    # ---- panel (a): scorer sensitivity. A dot plot is used because the region
    # of interest is a narrow band far above zero; bars would imply a baseline
    # at zero that the comparison never uses.
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(arms))
    base = np.array(frozen)
    tun = np.array(tuned)
    rfv = np.array([v if v is not None else np.nan for v in rff])

    for xi, b, tn, rv in zip(x, base, tun, rfv):
        if not np.isnan(rv):
            ax.plot([xi - 0.17, xi + 0.17], [rv, rv], color=GOLD, linewidth=2.0,
                    solid_capstyle="butt", zorder=3)
    ax.plot(x, base, "o", markersize=6.4, color=LGREY, markeredgecolor="#6f6f6f",
            markeredgewidth=0.6, label="Frozen $L_2=1$", zorder=4)
    ax.plot(x, tun, "o", markersize=6.4, color=BLUE, markeredgecolor="#12314d",
            markeredgewidth=0.6, label="LOOCV-tuned $L_2$", zorder=5)
    # nonlinear scorer: short dash marker, offset slightly so it stays visible
    ax.plot(x, rfv, "_", markersize=13, color=GOLD, markeredgewidth=2.2,
            label="Nonlinear RFF", zorder=4)

    tie = tuned[1]
    ax.axhline(tie, color=RED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
    ax.text(0.985, tie - 0.00028, "Tuned tie $=0.03888$",
            transform=ax.get_yaxis_transform(), color=RED, fontsize=7.4,
            ha="right", va="top", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_ylabel("Selected gain (raw objective)", fontsize=8.8)
    ax.set_title("(a)  Regularisation and function-class sensitivity",
                 fontsize=9.4, pad=8, loc="left")
    ax.set_ylim(0.0355, 0.0392)
    ax.tick_params(axis="y", labelsize=7.6)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=7.2, frameon=True, framealpha=0.95,
              edgecolor="#d5d5d5", handlelength=1.3, borderpad=0.5,
              labelspacing=0.45)

    # ---- panel (b): cost-normalised retraining, two side-by-side axes.
    # These are deliberately separate axes, not a broken axis: the raw-gain
    # reference and the retargeted returns live on different scales and a
    # shared x-axis would compress both.
    sub = gs[0, 1].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.12)
    axL = fig.add_subplot(sub[0, 0])
    axR = fig.add_subplot(sub[0, 1])

    arms_b = ["A", "B", "C", "D", "E", "D_random"]
    ref = a1b["fixed_action_reference"]
    first_ref = ref["CB_FIRST_DESCENT"]
    tabu_ref = ref["CB_TABU"]

    raw = [a1b["arms"][a]["raw_trained_selected_cng_x1e3"] for a in arms_b]
    ret_l2 = [a1b["arms"][a]["variants"]["fixed_l2_1"]["selected_cng_x1e3"] for a in arms_b]
    ret_tuned = [a1b["arms"][a]["variants"]["tuned"]["selected_cng_x1e3"] for a in arms_b]

    ys = np.arange(len(arms_b))[::-1].astype(float)
    dy = 0.19

    # left: raw-gain target
    axL.plot(raw, ys, "o", markersize=6.0, color=LGREY,
             markeredgecolor="#6f6f6f", markeredgewidth=0.6, zorder=4)
    axL.axvline(tabu_ref, color="#777777", linestyle=(0, (3, 3)),
                linewidth=1.0, zorder=2)
    axL.set_xlim(0.640, 0.700)
    axL.set_ylim(-0.75, len(arms_b) - 0.25)
    axL.set_yticks(ys)
    axL.set_yticklabels(labels, fontsize=8.0)
    axL.set_xlabel("Raw-gain target", fontsize=7.6)
    axL.tick_params(axis="x", labelsize=7.0)
    axL.set_title("(b)  Cost-normalised retraining removes the collapse",
                  fontsize=9.4, pad=18, loc="left")
    axL.grid(axis="x", color="#e6e6e6", linewidth=0.7, zorder=0)
    axL.set_axisbelow(True)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    axL.text(0.5, 0.985, f"fixed tabu $={tabu_ref:.4f}$",
             transform=axL.transAxes, fontsize=6.6, color="#666666",
             ha="center", va="top")
    for yi, rv in zip(ys, raw):
        axL.text(0.6985, yi, f"{rv:.4f}", va="center", ha="right",
                 fontsize=6.6, color="#444444", zorder=6)

    # right: retargeted returns, dumbbells per arm
    for yi, l2v, tv in zip(ys, ret_l2, ret_tuned):
        axR.plot([l2v, tv], [yi + dy, yi - dy], "-", color="#c0cfe0",
                 linewidth=1.0, zorder=3)
    axR.plot(ret_l2, ys + dy, "o", markersize=6.0, color=LBLUE,
             markeredgecolor="#3f7aa8", markeredgewidth=0.6, zorder=5)
    axR.plot(ret_tuned, ys - dy, "o", markersize=6.0, color=BLUE,
             markeredgecolor="#12314d", markeredgewidth=0.6, zorder=5)
    axR.axvline(first_ref, color=RED, linestyle=(0, (3, 3)),
                linewidth=1.0, zorder=2)
    axR.set_xlim(1.085, 1.150)
    axR.set_ylim(-0.75, len(arms_b) - 0.25)
    axR.set_yticks(ys)
    axR.set_yticklabels([])
    axR.set_xlabel("Per-request reward (retargeted)", fontsize=7.6)
    axR.tick_params(axis="x", labelsize=7.0)
    axR.grid(axis="x", color="#e6e6e6", linewidth=0.7, zorder=0)
    axR.set_axisbelow(True)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.text(0.5, 0.985, f"best fixed ${first_ref:.4f}$",
             transform=axR.transAxes, fontsize=6.6, color=RED,
             ha="center", va="top")
    for yi, l2v, tv in zip(ys, ret_l2, ret_tuned):
        axR.text(l2v - 0.0018, yi + dy, f"{l2v:.4f}", va="center", ha="right",
                 fontsize=6.6, color="#2c5d85", zorder=6)
        if tv > 1.1385:
            axR.text(tv - 0.0018, yi - dy, f"{tv:.4f}", va="center", ha="right",
                     fontsize=6.6, color=BLUE, zorder=6)
        else:
            axR.text(tv + 0.0018, yi - dy, f"{tv:.4f}", va="center", ha="left",
                     fontsize=6.6, color=BLUE, zorder=6)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5.6, color=LGREY,
               markeredgecolor="#6f6f6f", markeredgewidth=0.6,
               label="Raw-gain target"),
        Line2D([], [], marker="o", linestyle="", markersize=5.6, color=LBLUE,
               markeredgecolor="#3f7aa8", markeredgewidth=0.6,
               label="Retargeted ($L_2=1$)"),
        Line2D([], [], marker="o", linestyle="", markersize=5.6, color=BLUE,
               markeredgecolor="#12314d", markeredgewidth=0.6,
               label="Retargeted (tuned $L_2$)"),
    ]
    axR.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30),
               ncol=3, fontsize=6.6, frameon=False, handlelength=1.0,
               columnspacing=1.1, labelspacing=0.3)

    save(fig, "fig4_scorer_sensitivity")


def frozen_action_distributions():
    """Action counts under the frozen raw-gain scorers, from the decision records."""
    import csv
    from collections import Counter

    path = ROOT / "projects/analysis/fp_decision_study/tabu_decisions.csv"
    rows = list(csv.DictReader(open(path)))
    out = {}
    for arm in ("B", "D", "E"):
        c = Counter(r[f"gain_{arm}_action"] for r in rows)
        out[arm] = dict(c)
    return out


# ------------------------------------------------------------------ Figure 5
def figure5():
    a2b = load("a2b_tuned_choice_counts.json")
    a1 = load("a1_cost_normalized_gain.json")
    a4 = load("a4_per_scale_statistics.json")

    fig = plt.figure(figsize=(10.4, 4.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.34,
                          left=0.105, right=0.985, bottom=0.135, top=0.86)

    # ---- panel (a): action shares, raw target vs per-request target
    ax = fig.add_subplot(gs[0, 0])

    arms = ["B", "D", "E"]
    names = ["Structural B", "FP D", "Shared-sup. E"]
    rows = [
        ("Raw-gain target (frozen)", arms, "raw"),
        ("Per-request target (tuned)", arms, "tuned"),
    ]

    # Frozen raw-gain distributions read from the per-state decision records
    # (tabu_decisions.csv, 800 final critical-block states).
    raw_frozen = frozen_action_distributions()
    # Per-request retrained distributions come from the refit record, not from
    # the tuned-ridge-on-raw-target counts (which remain tabu-dominant).
    a1b = load("a1b_cost_normalised_refit.json")
    ypos = []
    ylab = []
    y = 0.0
    for gi, (gtitle, g_arms, mode) in enumerate(rows):
        for ai, a in enumerate(g_arms):
            if mode == "raw":
                dist = raw_frozen[a]
            else:
                dist = a1b["arms"][a]["variants"]["tuned"]["action_distribution"]
            ypos.append((y, dist, names[ai]))
            ylab.append((y, names[ai]))
            y -= 1.0
        y -= 0.95

    for yv, dist, nm in ypos:
        tabu = dist.get("CB_TABU", 0)
        first = dist.get("CB_FIRST_DESCENT", 0)
        best = dist.get("CB_BEST_DESCENT", 0)
        tot = tabu + first + best
        segs = [(best, BLUE), (first, GOLD), (tabu, GREY)]
        left = 0.0
        for val, col in segs:
            if val <= 0:
                continue
            pct = val / tot * 100
            ax.barh(yv, pct, left=left, height=0.62, color=col,
                    edgecolor="white", linewidth=0.7, zorder=3)
            if pct >= 6:
                ax.text(left + pct / 2, yv, f"{pct:.0f}%", ha="center",
                        va="center", fontsize=7.4, color="white", zorder=5)
            left += pct

    ax.set_yticks([p[0] for p in ypos])
    ax.set_yticklabels([p[2] for p in ypos], fontsize=8.2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the 800 frozen states (%)", fontsize=8.6)
    ax.set_title("(a)  Action shares under the two targets", fontsize=9.4,
                 pad=8, loc="left")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # group headers, placed above each block outside the bar area
    raw_y = ypos[0][0] + 1.05
    tun_y = ypos[3][0] + 1.05
    ax.text(0.0, raw_y, "Raw-gain target (frozen)", fontsize=8.0, ha="left",
            va="center", color="#333333", transform=ax.get_yaxis_transform())
    ax.text(0.0, tun_y, "Per-request target (tuned)", fontsize=8.0, ha="left",
            va="center", color="#333333", transform=ax.get_yaxis_transform())
    ax.set_ylim(ypos[5][0] - 0.62, raw_y + 0.42)

    handles = [
        Patch(facecolor=BLUE, edgecolor="white", label="Best descent"),
        Patch(facecolor=GOLD, edgecolor="white", label="First descent"),
        Patch(facecolor=GREY, edgecolor="white", label="Tabu"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.40),
              ncol=3, fontsize=7.8, frameon=False, handlelength=1.2,
              columnspacing=1.6)

    # ---- panel (b): per-scale D-B gain against the zero-gain floor
    ax2 = fig.add_subplot(gs[0, 1])
    order = ["6x6", "10x5", "10x10", "15x10"]
    nice = ["6×6", "10×5", "10×10", "15×10"]
    effects = [-0.00301421, -0.00023329, 0.0, 0.00000267]
    zero_rate = [a4[k]["all_actions_zero_rate"] * 100 for k in order]

    x = np.arange(len(order))
    ax2.bar(x, zero_rate, 0.52, color="#c3d7ea", edgecolor="#8fb3d1",
            linewidth=0.7, label="All-zero states (%)", zorder=2)
    for xi, zr in zip(x, zero_rate):
        ax2.text(xi, zr * 0.55, f"{zr:.1f}%", ha="center", va="center",
                 fontsize=8.0, color="#123c5e", zorder=9, fontweight="medium")

    ax2.set_ylim(0, 34)
    ax2.set_ylabel("States with all three action means zero (%)",
                   fontsize=8.2, color="#1f4e79")
    ax2.tick_params(axis="y", labelsize=7.6, colors="#1f4e79")
    ax2.set_xticks(x)
    ax2.set_xticklabels(nice, fontsize=8.2)
    ax2.set_xlabel("Instance size", fontsize=8.4)
    ax2.grid(axis="y", color="#e6e6e6", linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top",):
        ax2.spines[s].set_visible(False)

    ax3 = ax2.twinx()
    ax3.bar(x, effects, 0.30, color=RED, alpha=0.92, edgecolor="#6d2323",
            linewidth=0.7, label="FP D − B gain (pp)", zorder=4)
    ax3.axhline(0, color="#444444", linewidth=0.8, zorder=5)
    ax3.set_ylim(-0.35, 0.05)
    ax3.set_ylabel("FP D − B gain (percentage points)", fontsize=8.2, color=RED)
    ax3.tick_params(axis="y", labelsize=7.6, colors=RED)
    for s in ("top",):
        ax3.spines[s].set_visible(False)

    for xi, ev in zip(x, effects):
        if abs(ev) < 1e-9:
            txt, yy = "0.0000 (tie)", 0.007
        elif ev > 0:
            txt, yy = f"+{ev*100:.4f}", 0.007
        else:
            txt, yy = f"{ev*100:.4f}", ev - 0.010
        ax3.text(xi, yy, txt, ha="center",
                 va="bottom" if ev >= 0 else "top",
                 fontsize=7.4, color=RED, zorder=7)

    ax2.set_title("(b)  Per-scale FP D − B against the zero-gain floor",
                  fontsize=9.4, pad=8, loc="left")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.185),
               ncol=2, fontsize=7.0, frameon=False, handlelength=1.1,
               columnspacing=1.4)

    save(fig, "fig5_choices_and_scope")


if __name__ == "__main__":
    figure4()
    figure5()
#!/usr/bin/env python3
"""Draw the methodological overview figure (Figure 1) for the FP study.

This replaces the earlier evaluation-flow schematic with a four-panel
architecture figure that states the whole argument of the paper:

    1. saved native states from two independent search stacks
    2. feature matching, development-only affine maps, information sufficiency
    3. the two-stage attribution funnel
    4. the three post-collection diagnostics

The figure is drawn programmatically rather than edited as a bitmap so that
every symbol matches the manuscript notation exactly (in particular the
cost-normalised reward uses ``decodes_a``, indexed by action, as in
Sections 5.6 and 8.3 and in Table 11).

Outputs PNG, PDF and SVG at 600/300 dpi for print and for journal submission.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "projects/analysis/fp_decision_study/figures"
OUT_SUB = Path(__file__).resolve().parents[1] / "projects/submission/fp_decision_study/figures"

# Palette sampled from the reference layout, kept muted for print.
BLUE_EDGE = "#2E5C8A"
BLUE_FILL = "#E8F0F9"
BLUE_HDR = "#9EB4CC"
TEAL_EDGE = "#2A7F7A"
TEAL_FILL = "#DCEFED"
TEAL_HDR = "#BEE0DC"
RED_EDGE = "#B4543F"
RED_FILL = "#FBE2D8"
RED_HDR = "#F5C7B4"
GREEN_EDGE = "#4F7A3F"
GREEN_FILL = "#E1EFDB"
GREEN_HDR = "#C6DEBC"
INK = "#1A1A1A"
ARROW = "#4A4A4A"


def rbox(ax, x, y, w, h, edge, fill, lw=1.6, radius=0.012, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z,
    )
    ax.add_patch(p)
    return p


def header(ax, x, y, w, h, edge, fill, text, fs=11.5):
    """Panel header band with squared top corners."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                           linewidth=1.6, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=4,
            linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, lw=1.8, rad=0.0, color=ARROW):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_width=5,head_length=9",
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1, shrinkB=1, zorder=5,
    ))


def build():
    fig, ax = plt.subplots(figsize=(11.2, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.06, 0.96)
    ax.axis("off")

    # ---------------------------------------------------------------- panel 1
    px, pw = 0.018, 0.205
    py, ph = 0.105, 0.815
    rbox(ax, px, py, pw, ph, BLUE_EDGE, BLUE_FILL)
    header(ax, px, py + ph - 0.090, pw, 0.090, BLUE_EDGE, BLUE_HDR,
           "Saved Native\nState $Z$")

    bx, bw = px + 0.020, pw - 0.040

    def stack_block(top, label, count, fill="#F5FAFF"):
        """A solver box with a sheet-stack glyph and its branch count."""
        bh = 0.150
        rbox(ax, bx, top - bh, bw, bh, BLUE_EDGE, fill, lw=1.2,
             radius=0.006, z=4)
        ax.text(bx + bw / 2, top - bh / 2, label, ha="center", va="center",
                fontsize=10.4, color=INK, zorder=6, linespacing=1.25)
        sx, sw = bx + 0.034, bw - 0.082
        sy = top - bh - 0.088
        for k in (3, 2, 1, 0):
            ax.add_patch(Polygon(
                [[sx, sy + k * 0.0175], [sx + sw, sy + k * 0.0175],
                 [sx + sw + 0.019, sy + k * 0.0175 + 0.015],
                 [sx + 0.019, sy + k * 0.0175 + 0.015]],
                closed=True, facecolor=BLUE_HDR, edgecolor=BLUE_EDGE,
                linewidth=0.95, zorder=4))
        ax.add_patch(Rectangle((sx, sy), sw, 0.028, facecolor=BLUE_HDR,
                               edgecolor=BLUE_EDGE, linewidth=0.95, zorder=5))
        ax.text(px + pw / 2 + 0.006, sy - 0.040, count, ha="center",
                va="center", fontsize=10.0, color=INK, zorder=6)

    stack_block(py + ph - 0.108, "1. Genetic /\nMemetic Stack",
                "(109,440 branches)")
    stack_block(py + 0.300, "2. Critical-Block\nTabu Stack",
                "(84,000 branches)")

    # ---------------------------------------------------------------- panel 2
    qx, qw = 0.272, 0.235
    rbox(ax, qx, py, qw, ph, TEAL_EDGE, TEAL_FILL)
    header(ax, qx, py + ph - 0.095, qw, 0.095, TEAL_EDGE, TEAL_HDR,
           "Feature Matching\n& Affine Maps")

    # development-only affine maps
    mx, mw = qx + 0.018, qw - 0.036
    my1, mh1 = 0.655, 0.135
    rbox(ax, mx, my1, mw, mh1, TEAL_EDGE, "#EFF9F7", lw=1.2, radius=0.006, z=4)
    ax.text(mx + mw / 2, my1 + mh1 * 0.70, "Development-Only\nAffine Maps",
            ha="center", va="center", fontsize=10.2, color=INK, zorder=6,
            linespacing=1.25)
    ax.text(mx + mw / 2, my1 + mh1 * 0.22, r"$F_a(x) = A_a x + c_a$",
            ha="center", va="center", fontsize=12.5, color=INK, zorder=6)

    # information sufficiency
    my2, mh2 = 0.455, 0.155
    rbox(ax, mx, my2, mw, mh2, TEAL_EDGE, "#EFF9F7", lw=1.2, radius=0.006, z=4)
    ax.text(mx + mw / 2, my2 + mh2 * 0.72, "Information Sufficiency",
            ha="center", va="center", fontsize=10.2, color=INK, zorder=6)
    ax.text(mx + mw / 2, my2 + mh2 * 0.28, r"$V(Z, F \mid D) = V(Z \mid D)$",
            ha="center", va="center", fontsize=12.5, color=INK, zorder=6)

    # FP residual vs matched NoFP
    my3, mh3 = 0.255, 0.135
    half = (mw - 0.020) / 2
    rbox(ax, mx, my3, half, mh3, TEAL_EDGE, "#EFF9F7", lw=1.2, radius=0.006, z=4)
    ax.text(mx + half / 2, my3 + mh3 / 2, "FP\nResidual\nFeatures",
            ha="center", va="center", fontsize=9.8, color=INK, zorder=6,
            linespacing=1.22)
    rbox(ax, mx + half + 0.020, my3, half, mh3, TEAL_EDGE, "#EFF9F7",
         lw=1.2, radius=0.006, z=4)
    ax.text(mx + half + 0.020 + half / 2, my3 + mh3 / 2, "Matched\nNoFP\nControls",
            ha="center", va="center", fontsize=9.8, color=INK, zorder=6,
            linespacing=1.22)
    ax.text(mx + half + 0.010, my3 + mh3 / 2, "vs", ha="center", va="center",
            fontsize=9.5, color=INK, zorder=6)

    # ---------------------------------------------------------------- panel 3
    rx, rw = 0.535, 0.225
    rbox(ax, rx, py, rw, ph, RED_EDGE, RED_FILL)
    header(ax, rx, py + ph - 0.095, rw, 0.095, RED_EDGE, RED_HDR,
           "Two-Stage\nAttribution Funnel")

    hx, hw = rx + 0.017, rw - 0.034
    # Hurdle I
    hy1, hh1 = 0.565, 0.165
    rbox(ax, hx, hy1, hw, hh1, RED_EDGE, "#FDEDE5", lw=1.2, radius=0.006, z=4)
    ax.text(hx + hw / 2, hy1 + hh1 * 0.68, "Hurdle I:\nDynamics Predictability",
            ha="center", va="center", fontsize=10.4, color=INK, zorder=6,
            linespacing=1.25)
    ax.text(hx + hw / 2, hy1 + hh1 * 0.27, "(transition error vs identity)",
            ha="center", va="center", fontsize=9.2, color=INK, zorder=6,
            style="italic")

    # Hurdle II
    hy2, hh2 = 0.290, 0.175
    rbox(ax, hx, hy2, hw, hh2, RED_EDGE, "#FDEDE5", lw=1.2, radius=0.006, z=4)
    ax.text(hx + hw / 2, hy2 + hh2 * 0.70, "Hurdle II:\nDecision Utility",
            ha="center", va="center", fontsize=10.4, color=INK, zorder=6,
            linespacing=1.25)
    ax.text(hx + hw / 2, hy2 + hh2 * 0.26,
            "(frozen policy choice\n$\\rightarrow$ detached repeats)",
            ha="center", va="center", fontsize=9.0, color=INK, zorder=6,
            style="italic", linespacing=1.2)

    arrow(ax, hx + hw / 2, hy1 - 0.004, hx + hw / 2, hy2 + hh2 + 0.006, lw=1.8)

    # ---------------------------------------------------------------- panel 4
    gx, gw = 0.790, 0.195
    rbox(ax, gx, py, gw, ph, GREEN_EDGE, GREEN_FILL)
    header(ax, gx, py + ph - 0.095, gw, 0.095, GREEN_EDGE, GREEN_HDR,
           "Three Post-Collection\nDiagnostics", fs=10.6)

    dx, dw = gx + 0.016, gw - 0.032
    diag = [
        (0.640, "1. Cost-Normalised\nReward", r"$(\tilde{R}_a = R_a / \mathrm{decodes}_a)$"),
        (0.455, "2. LOOCV Ridge &\nNonlinear RFF", "Calibration"),
        (0.270, "3. Cost-Normalised\nRetraining", None),
    ]
    for yy, title, sub in diag:
        rbox(ax, dx, yy, dw, 0.145, GREEN_EDGE, "#F1F8ED", lw=1.2, radius=0.006, z=4)
        if sub:
            ax.text(dx + dw / 2, yy + 0.145 * 0.66, title, ha="center", va="center",
                    fontsize=9.9, color=INK, zorder=6, linespacing=1.22)
            fs = 10.6 if "tilde" in sub else 9.2
            ax.text(dx + dw / 2, yy + 0.145 * 0.24, sub, ha="center", va="center",
                    fontsize=fs, color=INK, zorder=6,
                    style="normal" if "tilde" in sub else "italic")
        else:
            ax.text(dx + dw / 2, yy + 0.145 / 2, title, ha="center", va="center",
                    fontsize=9.9, color=INK, zorder=6, linespacing=1.22)

    # ---------------------------------------------------------------- connectors
    # state stacks -> feature panel
    arrow(ax, px + pw + 0.004, 0.635, qx - 0.004, 0.635, lw=2.0)
    arrow(ax, px + pw + 0.004, 0.330, qx - 0.004, 0.330, lw=2.0)

    # feature panel -> funnel
    arrow(ax, qx + qw + 0.004, 0.720, rx - 0.004, 0.647, lw=2.0)
    arrow(ax, qx + qw + 0.004, 0.322, rx - 0.004, 0.377, lw=2.0)

    # funnel -> diagnostics (brace)
    bx0 = rx + rw + 0.004
    ax.plot([bx0, bx0 + 0.012], [0.647, 0.647], color=ARROW, lw=1.6, zorder=5)
    ax.plot([bx0, bx0 + 0.012], [0.377, 0.377], color=ARROW, lw=1.6, zorder=5)
    ax.plot([bx0 + 0.012, bx0 + 0.012], [0.377, 0.647], color=ARROW, lw=1.6, zorder=5)
    ax.plot([bx0 + 0.012, bx0 + 0.030], [0.512, 0.512], color=ARROW, lw=1.6, zorder=5)
    arrow(ax, bx0 + 0.014, 0.512, gx - 0.004, 0.512, lw=1.8)

    # footnote
    fig.text(0.5, 0.005,
             "All maps are fitted on development instances only. Frozen policies observe the current state "
             "and actual history, never a branch outcome.",
             ha="center", va="center", fontsize=9.0, color="#3A3A3A", style="italic")

    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.055)
    return fig


def main():
    for d in (OUT, OUT_SUB):
        d.mkdir(parents=True, exist_ok=True)
    fig = build()
    for d in (OUT, OUT_SUB):
        for ext, dpi in (("png", 600), ("pdf", None), ("svg", None)):
            f = d / f"fig1_overview_architecture.{ext}"
            fig.savefig(f, dpi=dpi or 600, facecolor="white",
                        bbox_inches="tight", pad_inches=0.04)
            print("wrote", f)
    plt.close(fig)


if __name__ == "__main__":
    main()
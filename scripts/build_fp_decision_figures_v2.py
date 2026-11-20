#!/usr/bin/env python3
"""Polished submission figures (v2). No search, no refits: reads frozen records only.

Fixes over v1:
  * Fig 3: zero-contrast rendered as a marked diamond with a "observed zero" band
    instead of an invisible vertical line; per-panel x-limits keep D-B legible.
  * Fig 5 / Fig 6: compact panels, no wasted vertical space, direct value labels.
  * Fig 2: shared zero reference, explicit "not comparable across panels" band.
  * Fig 1: aligned orthogonal flow diagram.
  * New Fig 7: two-panel sensitivity figure (regularisation/function class + cost-
    normalised retraining) built from revision_response_artifacts.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

AN = ROOT / 'projects/analysis/fp_decision_study'
ART = ROOT / 'projects/analysis/revision_response_artifacts'
PKG = ROOT / 'projects/submission/fp_decision_study'
FIG = PKG / 'figures'

BLUE = '#1f4e79'
GOLD = '#b46c19'
GREY = '#4d4d4d'
LGREY = '#9a9a9a'
RED = '#8b2f2f'
GREEN = '#2f6b47'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'legend.fontsize': 8,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})


def load(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}.png/.pdf/.svg')


def ci_bar(ax, y, mean, lo, hi, color=BLUE, marker='o', ms=5, lw=1.4, zero_eps=None,
           zero_marker='D', label=None):
    """Horizontal point + CI bar. A contrast whose CI is exactly [0,0] is drawn as a
    diamond with a shaded zero band so that it is visible instead of collapsing."""
    degenerate = (zero_eps is not None and abs(hi - lo) <= zero_eps
                  and abs(mean) <= zero_eps)
    if degenerate:
        ax.axhspan(y - 0.32, y + 0.32, color='#e8e8e8', zorder=0)
        ax.plot([mean], [y], marker=zero_marker, color=RED, ms=ms + 1.5, zorder=4,
                label=label)
        ax.annotate('observed zero contrast\n(CI exactly [0, 0])',
                    xy=(mean, y), xytext=(0.30, 0.0),
                    textcoords=('axes fraction', 'data'),
                    fontsize=7.2, color=RED, va='center', ha='left',
                    arrowprops=dict(arrowstyle='-', color=RED, lw=0.7,
                                    shrinkA=0, shrinkB=4))
    else:
        err = np.array([[max(mean - lo, 0)], [max(hi - mean, 0)]])
        ax.errorbar(mean, y, xerr=err, fmt=marker, color=color, ms=ms, lw=lw,
                    capsize=3.2, capthick=lw, markeredgewidth=0, zorder=3,
                    label=label)


def style_contrast_axis(ax, zero_line=True):
    if zero_line:
        ax.axvline(0, color=GREY, lw=1.0, zorder=1)
    ax.grid(axis='x', color='#dcdcdc', lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────── Figure 1
def fig1_workflow():
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')

    def box(x, y, w, h, text, fc='white', ec=BLUE, fs=8.4, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.6,rounding_size=1.6',
                                    facecolor=fc, edgecolor=ec, lw=1.1, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, zorder=3, linespacing=1.45,
                fontweight='bold' if bold else 'normal', color='#1a1a1a')

    def arrow(p0, p1, style='-|>'):
        ax.annotate('', xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle=style, color=GREY, lw=1.0,
                                    shrinkA=1, shrinkB=1), zorder=1)

    # Top row: state -> maps -> choice
    box(1, 30, 27, 12, 'Saved native state\n+ actual past history', fc='#eef3f8')
    box(36, 30, 27, 12, 'Development-only affine maps\nFP and shared-supervision NoFP')
    box(71, 30, 28, 12, 'Frozen scores → action choice\nno branch outcome as input')

    # Bottom row: branches -> evaluation
    box(6, 5, 36, 12, 'Detached branches: identical state\nall actions × independent repeats',
        fc='#faf6ef')
    box(58, 5, 36, 12, 'Instance-level paired evaluation\nstate error / ranking / selected reward',
        fc='#eef3f8')

    arrow((28.6, 36), (35.4, 36))
    arrow((63.6, 36), (70.4, 36))
    arrow((14.5, 29.4), (16.5, 17.6))          # state -> branches
    arrow((42.6, 11), (57.4, 11))              # branches -> evaluation
    arrow((85.0, 29.4), (83.0, 17.6))          # choice -> evaluation

    ax.text(50, 0.4,
            'Offline interventions on two independent search implementations; '
            'no validated online FP policy.',
            ha='center', va='center', fontsize=8, style='italic', color='#333333')
    save(fig, 'fig1_evaluation_flow')


# ─────────────────────────────────────────────────────────────── Figure 2
def fig2_transition():
    original = load(AN / 'original_final_analysis.json')
    tabu = load(AN / 'tabu_final_analysis.json')
    dyn1 = original['dynamics']['comparisons']
    dyn2 = tabu['dynamics']

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))

    # Panel (a) genetic
    labels1 = [f"{r['action']} vs {r['comparator']}" for r in dyn1]
    for i, r in enumerate(dyn1):
        ci_bar(axes[0], i, r['mean_error_reduction'],
               r['hierarchical_bootstrap_95ci'][0], r['hierarchical_bootstrap_95ci'][1])
    axes[0].set_yticks(range(len(labels1)), labels1)
    axes[0].invert_yaxis()
    axes[0].set_title('(a) Genetic/memetic stack', pad=8)
    axes[0].set_xlabel('Reference − action-conditioned error\n'
                       'weighted 4-D state norm · 95% hierarchical CI')

    # Panel (b) critical block
    pretty = {'CB_BEST_DESCENT': 'best descent', 'CB_FIRST_DESCENT': 'first descent',
              'CB_TABU': 'tabu'}

    def lbl(c):
        s = c.replace('tabu_', '')
        for k, v in pretty.items():
            s = s.replace(k, v)
        s = s.replace('_error_identity_minus_conditioned', ' vs identity')
        s = s.replace('_error_pooled_minus_conditioned', ' vs pooled')
        return s.replace('_', ' ')

    labels2 = [lbl(r['contrast']) for r in dyn2]
    for i, r in enumerate(dyn2):
        ci_bar(axes[1], i, r['mean'], r['ci95'][0], r['ci95'][1])
    axes[1].set_yticks(range(len(labels2)), labels2)
    axes[1].invert_yaxis()
    axes[1].set_title('(b) Independent critical-block stack', pad=8)
    axes[1].set_xlabel('Reference − action-conditioned error\n'
                       'unweighted 4-D state norm · 95% instance CI')

    for ax in axes:
        style_contrast_axis(ax)
    # Explicit non-comparability note as a spanning band
    fig.text(0.5, -0.035,
             'Positive values favour conditioning. Panels use different state norms and '
             'interval procedures;\ntherefore neither scales nor intervals support a '
             'pooled cross-solver comparison.',
             ha='center', va='top', fontsize=7.6, color='#444444')
    fig.tight_layout(w_pad=3.0)
    save(fig, 'fig2_transition_errors')


# ─────────────────────────────────────────────────────────────── Figure 3
def fig3_primary():
    primary = load(AN / 'primary_results.json')['primary_family']
    pretty = {
        'original_frozen_E_minus_C_atomic': 'Genetic: FP E − state/history C',
        'tabu_gain_D_minus_B': 'Critical-block: FP D − structural B',
        'tabu_gain_D_minus_E': 'Critical-block: FP D − shared-supervision E',
    }
    labels = [pretty[r['contrast']] for r in primary]

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 4.2),
                             gridspec_kw={'height_ratios': [0.8, 1.15]})

    # Row 0: the exact-zero genetic contrast, own axis because its CI is [0,0]
    r0 = primary[0]
    ax = axes[0]
    style_contrast_axis(ax)
    ax.set_xlim(-0.005, 0.005)
    ax.set_ylim(-0.95, 0.95)
    ax.plot(0, 0, marker='D', color=RED, ms=9, zorder=5)
    ax.annotate('exactly 0.00000000\n95% CI [0, 0] · Holm p = 1',
                xy=(0, 0), xytext=(0.63, 0.72), textcoords='axes fraction',
                fontsize=7.8, color=RED, va='center', ha='left',
                arrowprops=dict(arrowstyle='-', color=RED, lw=0.7, shrinkA=0, shrinkB=3))
    ax.set_yticks([0], [labels[0]])
    ax.set_xticks([-0.004, -0.002, 0, 0.002, 0.004])
    ax.set_title('(a) Genetic contrast is an observed zero, shown on its own scale',
                 pad=6, loc='left', fontsize=9)

    # Row 1: critical-block contrasts share one scale
    crit = primary[1:]
    ax = axes[1]
    style_contrast_axis(ax)
    for i, r in enumerate(crit):
        ci_bar(ax, i, r['mean'], r['ci95'][0], r['ci95'][1], color=BLUE)
        ax.annotate(f'{r["mean"]:+.6f}', xy=(r['mean'], i), xytext=(0, 11),
                    textcoords='offset points', fontsize=7.6, va='bottom',
                    ha='center', color=BLUE, fontweight='bold')
    ax.set_yticks(range(len(crit)), [labels[1], labels[2]])
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlim(-0.0019, 0.0009)
    ax.set_xlabel('Selected offline gain difference (fraction of s₀)\n'
                  '95% instance-cluster bootstrap CI · n = 20 · no cross-solver pooling')
    ax.set_title('(b) Critical-block contrasts, common scale', pad=6, loc='left', fontsize=9)
    ax.annotate(f'Holm p = {crit[0]["holm_p"]:.6g}', xy=(crit[0]['mean'], -0.62),
                fontsize=7.6, ha='center', va='bottom', color=RED)

    fig.tight_layout(h_pad=1.6)
    save(fig, 'fig3_primary_decision_contrasts')


# ─────────────────────────────────────────────────────────────── Figure 4
def fig4_choices_scope():
    tabu = load(AN / 'tabu_final_analysis.json')
    per = list(csv.DictReader((AN / 'tabu_instance_metrics.csv').open()))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.0),
                             gridspec_kw={'width_ratios': [1.25, 1.0]})

    # (a) stacked action shares
    actions = ('CB_BEST_DESCENT', 'CB_FIRST_DESCENT', 'CB_TABU')
    colors = [BLUE, GOLD, '#b4b4b4']
    arms = ('B', 'D', 'E')
    arm_labels = ['Structural B', 'FP D', 'Shared-supervision E']
    ax = axes[0]
    for i, arm in enumerate(arms):
        start = 0.0
        for ai, a in enumerate(actions):
            cnt = tabu['decision_counts']['gain_' + arm].get(a, 0)
            share = 100 * cnt / 800
            if share < 0.6:
                start += share
                continue
            ax.barh(i, share, left=start, height=0.55, color=colors[ai],
                    edgecolor='white', lw=0.8)
            ax.text(start + share / 2, i, f'{share:.0f}', ha='center', va='center',
                    fontsize=7.2, color='white' if ai < 2 else '#333333')
            start += share
    ax.set_yticks(range(3), arm_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Decisions (% of 800 opportunities)')
    ax.set_title('(a) Gain-policy action shares', pad=6, loc='left', fontsize=9)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, [a.replace('CB_', '').replace('_', ' ').title() for a in actions],
              loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=3, handlelength=1.0,
              columnspacing=1.2)
    ax.grid(axis='x', color='#dcdcdc', lw=0.6)
    ax.set_axisbelow(True)

    # (b) size means D-B
    sizes = ('6x6', '10x5', '10x10', '15x10')
    vals = []
    for size in sizes:
        v = [float(r['gain_D_value']) - float(r['gain_B_value'])
             for r in per if f'_{size}_' in r['instance']]
        vals.append(float(np.mean(v)) * 100)
    ax = axes[1]
    ypos = np.arange(len(sizes))
    cols = [RED if v < -0.0005 else (GREY if abs(v) < 5e-5 else GREEN) for v in vals]
    ax.barh(ypos, vals, height=0.55, color=cols)
    ax.axvline(0, color=GREY, lw=1.0)
    for y, v in zip(ypos, vals):
        if abs(v) < 5e-5:
            ax.annotate(f'{v:+.4f}  (tie)', xy=(0.01, y), xytext=(4, 0),
                        textcoords='offset points', fontsize=7.4, va='center',
                        ha='left', color=GREY, style='italic')
        elif v < 0:
            ax.annotate(f'{v:+.4f}', xy=(v, y), xytext=(-4, 0),
                        textcoords='offset points', fontsize=7.4, va='center',
                        ha='right', color=RED, fontweight='bold')
        else:
            ax.annotate(f'{v:+.4f}', xy=(v, y), xytext=(4, 0),
                        textcoords='offset points', fontsize=7.4, va='center',
                        ha='left', color=GREEN, fontweight='bold')
    ax.set_yticks(ypos, [s.replace('x', '×') for s in sizes])
    ax.invert_yaxis()
    ax.set_xlabel('FP D − B gain (percentage points)\ndescriptive size means, five instances each')
    ax.set_title('(b) Where the loss originates', pad=6, loc='left', fontsize=9)
    lo = min(vals) * 1.45
    hi = max(abs(min(vals)) * 0.22, 0.02)
    ax.set_xlim(lo, hi)
    ax.grid(axis='x', color='#dcdcdc', lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout(w_pad=2.6)
    save(fig, 'fig4_choices_and_scope')


# ─────────────────────────────────────────────────────────────── Figure 5
def fig5_repeat_winners():
    hist_oracle = [{'mean': .001008754175, 'ci95': [.000508872136, .001603850298]},
                   {'mean': -.000370682368, 'ci95': [-.000865546690, .000036881116]}]
    # Recompute critical-block oracle exactly as v1 did
    from collections import defaultdict as dd
    res = ROOT / 'results/fp_decision_study'
    gains = dd(dict); insts = {}
    for p in sorted((res / 'tabu/final').glob('*.json.gz')):
        d = json.loads(__import__('gzip').open(p).read())
        for r in d['records']:
            gains[r['state_id']][r['action'], r['repeat']] = r['gain']
            insts[r['state_id']] = r['instance']
    actions = ('CB_BEST_DESCENT', 'CB_FIRST_DESCENT', 'CB_TABU')
    per_inst = dd(lambda: dd(list))
    for sid, g in gains.items():
        m = np.array([[g[a, r] for r in range(5)] for a in actions])
        fit = m[:, :2].mean(1); ev = m[:, 2:].mean(1)
        a = int(np.argmax(fit))
        per_inst[insts[sid]]['fit'].append(float(fit[a] - fit[2]))
        per_inst[insts[sid]]['ev'].append(float(ev[a] - ev[2]))

    sys.path.insert(0, str(ROOT))
    from scripts.analyze_fp_decision_study import compare
    oracle = [compare({i: float(np.mean(g[k])) for i, g in per_inst.items()},
                      'tabu_oracle_' + k) for k in ('fit', 'ev')]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9), sharey=False)
    for ax, data, title, xlabel in (
        (axes[0], hist_oracle, '(a) Historical GA/LS/VND · 16 instances',
         'Chosen gain − fixed VND (percentage points)\noriginal 95% hierarchical CI'),
        (axes[1], oracle, '(b) Critical-block actions · 20 instances',
         'Chosen gain − fixed tabu (percentage points)\nnominal 95% instance CI; secondary')):
        ys = [1, 0]
        lbls = ['Winner on same 2 repeats', 'Chosen winner on fresh repeats']
        for y, r, lb in zip(ys, data, lbls):
            ci_bar(ax, y, r['mean'] * 100, r['ci95'][0] * 100, r['ci95'][1] * 100)
            ax.annotate(f'{r["mean"]*100:+.3f}', xy=(r['ci95'][1] * 100, y), xytext=(5, 0),
                        textcoords='offset points', fontsize=7.4, va='center', color=BLUE)
        ax.set_yticks(ys, lbls)
        ax.set_ylim(-0.6, 1.6)
        ax.axhline(0.5, color='#e0e0e0', lw=0.7)
        ax.set_title(title, pad=6, loc='left', fontsize=9)
        ax.set_xlabel(xlabel)
        style_contrast_axis(ax)
    fig.text(0.5, -0.06,
             'Retrospective two-repeat winners overstate the gain that survives independent '
             'repeats. Comparators are native to each solver.',
             ha='center', va='top', fontsize=7.4, color='#444444')
    fig.tight_layout(w_pad=3.0)
    save(fig, 'fig5_independent_repeat_winners')


# ─────────────────────────────────────────────────────────────── Figure 6
def fig6_action_window():
    hist = [{'mean': -.000584228788, 'ci95': [-.001061965064, -.000216874638]},
            {'mean': .002453269870, 'ci95': [.001803808389, .003092391501]}]
    # new-cohort window contrasts (frozen descriptive source values)
    fsw = load(AN / 'figure_source_data.json')
    neww = fsw['new_original_window_contrasts']

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9))
    for ax, data, title in ((axes[0], hist, '(a) Historical matched 20 instances'),
                            (axes[1], neww, '(b) Newly generated 20 instances')):
        ys = [1, 0]
        labs = ['Atomic', '480 requests']
        for y, r, lb in zip(ys, data, labs):
            lo_, hi_ = r['ci95'][0] * 100, r['ci95'][1] * 100
            ci_bar(ax, y, r['mean'] * 100, lo_, hi_)
            ax.annotate(f'{r["mean"]*100:+.4f}', xy=(lo_ - 0.006, y), xytext=(0, 9),
                        textcoords='offset points', fontsize=7.4, va='bottom',
                        ha='left', color=BLUE, fontweight='bold')
        ax.set_yticks(ys, labs)
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(title, pad=6, loc='left', fontsize=9)
        ax.set_xlabel('LS − VND normalized gain (percentage points)\n95% instance bootstrap CI · matched states')
        style_contrast_axis(ax)
        # colour the sign flip visually
        ax.axhspan(0.5, 1.5, color='#f2f2f2', zorder=0)
    fig.text(0.5, -0.06,
             'Changing the declared window from one atomic call to 480 requests reverses '
             'the aggregate LS/VND ordering.\nRequests are not wall time or unique decodes.',
             ha='center', va='top', fontsize=7.4, color='#444444')
    fig.tight_layout(w_pad=3.0)
    save(fig, 'fig6_action_window')


# ─────────────────────────────────────────────────────────────── Figure 7 (new)
def fig7_sensitivity():
    a2 = load(ART / 'a2_tuned_ridge_nonlinear.json')
    a1b = load(ART / 'a1b_cost_normalised_refit.json')

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5),
                             gridspec_kw={'width_ratios': [1.0, 1.05]})

    # ── Panel (a): raw-gain selected return under three scorer settings ──
    ax = axes[0]
    arms = ['B', 'D', 'E']
    labs = ['Structural\nB', 'FP\nD', 'Shared-sup.\nE']
    x = np.arange(len(arms))
    w = 0.26
    frozen = [a2['frozen_reference'][a] for a in arms]
    tuned = [a2['tuned_ridge_final'][a]['mean_selected_gain'] for a in arms]
    rff = [a2['nonlinear_rf'][a]['mean_selected_gain'] for a in arms]

    bars = ax.bar(x - w, frozen, w, label='frozen L2 = 1', color='#c9c9c9', edgecolor='white')
    bars = ax.bar(x, tuned, w, label='LOOCV-tuned L2', color=BLUE, edgecolor='white')
    bars = ax.bar(x + w, rff, w, label='nonlinear RFF', color=GOLD, edgecolor='white')

    ymin = min(min(frozen), min(tuned), min(rff))
    ymax = max(max(frozen), max(tuned), max(rff))
    pad = (ymax - ymin) * 0.55
    ax.set_ylim(ymin - pad * 0.9, ymax + pad * 0.5)
    ax.set_xticks(x, labs)
    ax.set_ylabel('Selected gain on final 20\n(fraction of s₀)')
    ax.set_title('(a) The frozen deficit is a regularisation artefact', pad=6,
                 loc='left', fontsize=9)
    ax.legend(loc='upper left', ncol=1, handlelength=1.1,
              bbox_to_anchor=(-0.02, 0.99), fontsize=7.4)
    ax.grid(axis='y', color='#dcdcdc', lw=0.6)
    ax.set_axisbelow(True)
    ax.annotate('D ≡ B at tuned L2',
                xy=(1.0, tuned[1]), xytext=(1.52, 0.30),
                textcoords=('data', 'axes fraction'), fontsize=7.0, color=RED,
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.9, shrinkA=1, shrinkB=3,
                                connectionstyle='arc3,rad=0.28'))

    # ── Panel (b): cost-normalised retraining (gap to best fixed action) ──
    ax = axes[1]
    fixed = a1b['fixed_action_reference']
    best_fixed = max(fixed.values())
    arms_all = ['A', 'B', 'C', 'D', 'E', 'D_random']
    vals = [a1b['arms'][a]['variants']['tuned']['selected_cng_x1e3'] for a in arms_all]
    gaps = [(v - best_fixed) / best_fixed * 100 for v in vals]  # % shortfall
    xs = np.arange(len(arms_all))
    cols = [GOLD if a == 'D' else (BLUE if a in ('A', 'B', 'C') else '#8f8f8f')
            for a in arms_all]
    bars = ax.barh(xs, gaps, 0.60, color=cols, edgecolor='white')
    ax.axvline(0, color=GREEN, lw=1.3)
    for x_, g, v in zip(xs, gaps, vals):
        ax.annotate(f'{g:+.2f}%  ({v:.4f})', xy=(g, x_), xytext=(-5, 0),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=7.2, color='#222222')
    ax.set_yticks(xs, [a.replace('D_random', 'D$_{random}$') for a in arms_all])
    ax.invert_yaxis()
    ax.set_xlabel('Shortfall versus the best fixed action, %\n'
                  f'(best fixed = first descent, {best_fixed:.4f} per-request; 0 = perfect)')
    ax.set_title('(b) Retargeting removes the collapse but not the FP null',
                 pad=6, loc='left', fontsize=9)
    ax.set_xlim(min(gaps) * 1.55, max(gaps) * 0.18)
    ax.grid(axis='x', color='#dcdcdc', lw=0.6)
    ax.set_axisbelow(True)
    fig.text(0.5, -0.075,
             'Post-collection secondary diagnostics on frozen records; penalties and models are fitted on the eight '
             'development instances only and final outcomes are used for evaluation only.\n'
             'Panel (b): arms trained on the raw gain instead of the per-request target earn only ≈0.66–0.67 '
             'per request (≈41% shortfall, off scale).',
             ha='center', va='top', fontsize=7.4, color='#444444')
    fig.tight_layout(w_pad=2.6)
    save(fig, 'fig7_sensitivity_diagnostics')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print('Building polished figures:')
    fig1_workflow()
    fig2_transition()
    fig3_primary()
    fig4_choices_scope()
    fig5_repeat_winners()
    fig6_action_window()
    fig7_sensitivity()
    print('Done.')


if __name__ == '__main__':
    main()
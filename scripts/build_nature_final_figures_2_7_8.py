#!/usr/bin/env python3
"""Nature/Science (NPG + Okabe-Ito) restyle of Figures 2, 7 and 8 ONLY.

Reads frozen repository result files exclusively; no refits, no recomputation of
any mean or interval. Every plotted number and CI is taken verbatim from:
  * projects/analysis/fp_decision_study/original_final_analysis.json   (Fig 2a)
  * projects/analysis/fp_decision_study/tabu_final_analysis.json       (Fig 2b)
  * projects/analysis/fp_empirical_repeatability.json                  (Fig 7a)
  * projects/analysis/fp_decision_study/figure_source_data.json        (Fig 7b, 8b)
  * projects/analysis/fp_decision_study/history_final_audit.json       (Fig 8a)

Style (identical across the three figures, matching finalized Figures 3-6):
  white background; DejaVu Sans / Arial sans-serif; mathtext dejvusans;
  bottom/left spines #222222 lw 0.85, top/right hidden; outward ticks 3.5/0.85;
  vertical gridlines #EAECEF lw 0.65 solid; dashed null line x=0 #374151 lw 1.0;
  capped 95% CI errorbars capsize 3.2, elinewidth 1.6; American English.
Significance encoding:
  Fig 2 / Fig 7: solid #0072B2 if CI strictly > 0; hollow #1B365D if CI crosses 0;
                 solid #C0392B if CI strictly < 0.
  Fig 8:         solid #C0392B for negative Atomic contrasts; solid #009E73 for
                 positive 480-request contrasts; #F3F4F6 band behind Atomic row.
Outputs: figures_nature/Figure_{2,7,8}_nature_final.{png,pdf} (300 DPI PNG, vector PDF).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / 'projects/analysis/fp_decision_study'
OUT = ROOT / 'figures_nature'

NAVY = '#1B365D'     # hollow marker: CI crosses zero
BLUE = '#0072B2'     # Okabe-Ito blue: strictly positive
RED = '#C0392B'      # strictly negative
GREEN = '#009E73'    # Okabe-Ito green: Fig 8 positive
SPINE = '#222222'
GRID = '#EAECEF'
NULL = '#374151'
BAND = '#F3F4F6'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': SPINE,
    'axes.linewidth': 0.85,
    'axes.labelcolor': SPINE,
    'text.color': SPINE,
    'xtick.color': SPINE,
    'ytick.color': SPINE,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.major.width': 0.85,
    'ytick.major.width': 0.85,
    'legend.frameon': False,
    'legend.fontsize': 7.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'savefig.facecolor': '#FFFFFF',
})


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def style_axis(ax):
    ax.grid(axis='x', color=GRID, lw=0.65, linestyle='-', zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color=NULL, lw=1.0, linestyle='--', zorder=1)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.85)


def sig_class(lo, hi):
    if lo > 0:
        return 'positive'
    if hi < 0:
        return 'negative'
    return 'crosses'


def point(ax, y, mean, lo, hi, pos_color):
    """Capped 95% CI errorbar with the declared significance encoding."""
    cls = sig_class(lo, hi)
    color = {'positive': pos_color, 'negative': RED, 'crosses': NAVY}[cls]
    hollow = cls == 'crosses'
    ax.errorbar(mean, y, xerr=np.array([[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]]),
                fmt='o', ms=5.5, color=color, ecolor=color,
                elinewidth=1.6, capsize=3.2, capthick=1.6,
                markerfacecolor='white' if hollow else color,
                markeredgecolor=color, markeredgewidth=1.6 if hollow else 0.0,
                zorder=3)
    return cls, color


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}.png (300 DPI) / {name}.pdf (vector)')


PRETTY = {'CB_BEST_DESCENT': 'best descent', 'CB_FIRST_DESCENT': 'first descent',
          'CB_TABU': 'tabu'}


def tabu_label(c):
    s = c.replace('tabu_', '')
    for k, v in PRETTY.items():
        s = s.replace(k, v)
    s = s.replace('_error_identity_minus_conditioned', ' vs identity')
    s = s.replace('_error_pooled_minus_conditioned', ' vs pooled')
    return s.replace('_', ' ')


# ─────────────────────────────────────────────────────────── Figure 2
def figure_2(prov):
    orig = load(AN / 'original_final_analysis.json')
    tabu = load(AN / 'tabu_final_analysis.json')
    rows_a = orig['dynamics']['comparisons']
    rows_b = tabu['dynamics']

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    labels_a = [f"{r['action']} vs {r['comparator']}" for r in rows_a]
    for i, r in enumerate(rows_a):
        lo, hi = r['hierarchical_bootstrap_95ci']
        cls, _ = point(axes[0], i, r['mean_error_reduction'], lo, hi, BLUE)
        prov['figure_2']['panel_a'].append(
            {'row': labels_a[i], 'mean': r['mean_error_reduction'], 'ci95': [lo, hi],
             'encoding': cls})
    axes[0].set_yticks(range(len(labels_a)), labels_a)
    axes[0].invert_yaxis()
    axes[0].set_title('(a) Genetic/memetic stack', pad=8, loc='left')
    axes[0].set_xlabel('Reference − action-conditioned error\n'
                       'weighted 4-D state norm · 95% hierarchical CI')

    labels_b = [tabu_label(r['contrast']) for r in rows_b]
    for i, r in enumerate(rows_b):
        lo, hi = r['ci95']
        cls, _ = point(axes[1], i, r['mean'], lo, hi, BLUE)
        prov['figure_2']['panel_b'].append(
            {'row': labels_b[i], 'mean': r['mean'], 'ci95': [lo, hi], 'encoding': cls})
    axes[1].set_yticks(range(len(labels_b)), labels_b)
    axes[1].invert_yaxis()
    axes[1].set_title('(b) Independent critical-block stack', pad=8, loc='left')
    axes[1].set_xlabel('Reference − action-conditioned error\n'
                       'unweighted 4-D state norm · 95% instance CI')
    for ax in axes:
        style_axis(ax)
    fig.text(0.5, -0.045,
             'Positive values favor conditioning. Solid blue: 95% CI strictly above 0; '
             'hollow navy: CI crosses 0; solid red: CI strictly below 0.\n'
             'Panels use different state norms and interval procedures; therefore neither '
             'scales nor intervals support a pooled cross-solver comparison.',
             ha='center', va='top', fontsize=7.6, color='#444444')
    fig.tight_layout(w_pad=3.0)
    save(fig, 'Figure_2_nature_final')


# ─────────────────────────────────────────────────────────── Figure 7
def figure_7(prov):
    rep = load(ROOT / 'projects/analysis/fp_empirical_repeatability.json')
    iv = rep['summaries']['atomic']['intervals']
    hist = [iv['original2_oracle_minus_VND'], iv['original2_choice_new5_minus_VND']]
    fsw = load(AN / 'figure_source_data.json')
    tabu_oracle = fsw['tabu_oracle_advantage']

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9))
    for ax, data, key, title, xlabel in (
            (axes[0], hist, 'panel_a', '(a) Historical GA/LS/VND · 16 instances',
             'Chosen gain − fixed VND (percentage points)\noriginal 95% hierarchical CI'),
            (axes[1], tabu_oracle, 'panel_b', '(b) Critical-block actions · 20 instances',
             'Chosen gain − fixed tabu (percentage points)\nnominal 95% instance CI; secondary')):
        ys = [1, 0]
        lbls = ['Winner on same 2 repeats', 'Chosen winner on fresh repeats']
        for y, r, lb in zip(ys, data, lbls):
            mean = r['mean'] * 100
            lo, hi = r['ci95'][0] * 100, r['ci95'][1] * 100
            cls, color = point(ax, y, mean, lo, hi, BLUE)
            ax.annotate(f'{mean:+.3f}', xy=(hi, y), xytext=(5, 0),
                        textcoords='offset points', fontsize=7.4, va='center',
                        color=color, fontweight='bold')
            prov['figure_7'][key].append(
                {'row': lb, 'mean_pp': mean, 'ci95_pp': [lo, hi], 'encoding': cls})
        ax.set_yticks(ys, lbls)
        ax.set_ylim(-0.6, 1.6)
        ax.axhline(0.5, color=GRID, lw=0.65)
        ax.set_title(title, pad=6, loc='left')
        ax.set_xlabel(xlabel)
        style_axis(ax)
    fig.text(0.5, -0.075,
             'Retrospective two-repeat winners overstate the gain that survives independent '
             'repeats. Comparators are native to each solver.\n'
             'Solid blue: 95% CI strictly above 0; hollow navy: CI crosses 0; '
             'solid red: CI strictly below 0.',
             ha='center', va='top', fontsize=7.4, color='#444444')
    fig.tight_layout(w_pad=3.0)
    save(fig, 'Figure_7_nature_final')


# ─────────────────────────────────────────────────────────── Figure 8
def figure_8(prov):
    hist_audit = load(AN / 'history_final_audit.json')
    hist = hist_audit['matched_window_effects']          # [atomic, fair480]
    fsw = load(AN / 'figure_source_data.json')
    neww = fsw['new_original_window_contrasts']          # [atomic, equal480]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9))
    for ax, data, key, title in (
            (axes[0], hist, 'panel_a', '(a) Historical matched 20 instances'),
            (axes[1], neww, 'panel_b', '(b) Newly generated 20 instances')):
        ys = [1, 0]
        labs = ['Atomic', '480 requests']
        ax.axhspan(0.5, 1.5, color=BAND, zorder=0)
        for y, r, lb in zip(ys, data, labs):
            mean = r['effect'] * 100 if 'effect' in r else r['mean'] * 100
            ci = r['instance_bootstrap_95ci'] if 'instance_bootstrap_95ci' in r else r['ci95']
            lo, hi = ci[0] * 100, ci[1] * 100
            cls, color = point(ax, y, mean, lo, hi, GREEN)
            ax.annotate(f'{mean:+.4f}', xy=(lo, y), xytext=(0, 9),
                        textcoords='offset points', fontsize=7.4, va='bottom',
                        ha='left', color=color, fontweight='bold')
            prov['figure_8'][key].append(
                {'row': lb, 'mean_pp': mean, 'ci95_pp': [lo, hi], 'encoding': cls})
        ax.set_yticks(ys, labs)
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(title, pad=6, loc='left')
        ax.set_xlabel('LS − VND normalized gain (percentage points)\n'
                      '95% instance bootstrap CI · matched states')
        style_axis(ax)
    fig.text(0.5, -0.075,
             'Changing the declared window from one atomic call to 480 requests reverses '
             'the aggregate LS/VND ordering. Requests are not wall time or unique decodes.\n'
             'Grey band marks the Atomic row. Solid red: contrast strictly below 0; '
             'solid green: strictly above 0.',
             ha='center', va='top', fontsize=7.4, color='#444444')
    fig.tight_layout(w_pad=3.0)
    save(fig, 'Figure_8_nature_final')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [AN / 'original_final_analysis.json', AN / 'tabu_final_analysis.json',
               ROOT / 'projects/analysis/fp_empirical_repeatability.json',
               AN / 'figure_source_data.json', AN / 'history_final_audit.json']
    prov = {'style': 'NPG + Okabe-Ito (Nature/Science), identical to finalized Figures 3-6',
            'policy': 'read-only frozen sources; means and CIs plotted verbatim, no refits',
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
            'figure_2': {'panel_a': [], 'panel_b': []},
            'figure_7': {'panel_a': [], 'panel_b': []},
            'figure_8': {'panel_a': [], 'panel_b': []}}
    print('Building Nature-final Figures 2, 7, 8:')
    figure_2(prov)
    figure_7(prov)
    figure_8(prov)

    # gate: plotted values must round to the manuscript's quoted numbers and the
    # significance classes must match the declared encoding
    f7 = prov['figure_7']
    expect7 = ([('+0.101', 'positive'), ('-0.037', 'crosses')],
               [('+0.108', 'positive'), ('-0.038', 'negative')])
    for panel, exp in zip(('panel_a', 'panel_b'), expect7):
        for row, (label, cls) in zip(f7[panel], exp):
            assert f'{row["mean_pp"]:+.3f}' == label, (panel, row)
            assert row['encoding'] == cls, (panel, row)
    f8 = prov['figure_8']
    expect8 = ([('-0.0584', 'negative'), ('+0.2453', 'positive')],
               [('-0.2190', 'negative'), ('+0.1772', 'positive')])
    for panel, exp in zip(('panel_a', 'panel_b'), expect8):
        for row, (label, cls) in zip(f8[panel], exp):
            assert f'{row["mean_pp"]:+.4f}' == label, (panel, row)
            assert row['encoding'] == cls, (panel, row)
    (OUT / 'figure_source_provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
    print('  gates PASS: quoted values and significance encodings match frozen sources')
    print('  wrote figure_source_provenance.json')


if __name__ == '__main__':
    main()

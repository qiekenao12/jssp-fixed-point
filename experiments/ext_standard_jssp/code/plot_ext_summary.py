#!/usr/bin/env python3
"""Publication-quality extension summary figure (300 DPI PNG + PDF).

Clean manuscript style: white background, muted palette, minimal spines.
Panels: (a) raw selected return per arm x scorer vs best fixed static action;
(b) cost-normalized selected return vs its fixed reference; (c) forest plot of
D-B / D-E instance-cluster contrasts with 95% bootstrap CIs under both targets.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/ext_standard_jssp/outputs'
ARMS = ('A', 'B', 'C', 'D', 'E', 'D_random')
SCORERS = ('frozen_ridge', 'loocv_ridge', 'rff_ridge', 'gbdt')
LABELS = {'frozen_ridge': 'Ridge (L2=1)', 'loocv_ridge': 'Ridge (LOOCV)',
          'rff_ridge': 'RBF-RFF', 'gbdt': 'Shallow GBDT'}
COLORS = {'frozen_ridge': '#4878a8', 'loocv_ridge': '#7fb3d5',
          'rff_ridge': '#e08b3c', 'gbdt': '#5a9367'}


def read_table(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def main():
    raw = {(r['scorer'], r['arm']): r for r in read_table('table_ext_raw_gain_scorers.csv')}
    cng = {(r['scorer'], r['arm']): r for r in read_table('table_ext_cost_normalized_scorers.csv')}
    paired = read_table('table_ext_paired_contrasts.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.linewidth': 0.8, 'figure.dpi': 150})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9),
                             gridspec_kw={'width_ratios': [1, 1, 1.25]})
    x = np.arange(len(ARMS))
    width = 0.19
    for ax, tbl, fixed_key, tname, ylab in (
            (axes[0], raw, 'fixed_CB_TABU', 'raw', r'mean selected $R_a$'),
            (axes[1], cng, 'fixed_CB_FIRST_DESCENT', 'cng',
             r'mean selected $\tilde{R}_a\times 10^3$')):
        for k, sc in enumerate(SCORERS):
            vals = [float(tbl[sc, a]['mean_selected_return']) for a in ARMS]
            ax.bar(x + (k - 1.5) * width, vals, width, color=COLORS[sc],
                   label=LABELS[sc], zorder=3, edgecolor='white', linewidth=0.4)
        ref = float(tbl[fixed_key, '-']['mean_selected_return'])
        ax.axhline(ref, color='#333333', linestyle='--', linewidth=1.0, zorder=4)
        ax.text(len(ARMS) - 0.45, ref, '  best fixed', va='bottom', ha='right', fontsize=7,
                color='#333333')
        ax.set_xticks(x, ARMS)
        ax.set_xlabel('representation')
        ax.set_ylabel(ylab)
        ax.set_title(f'({"a" if tname == "raw" else "b"}) '
                     + ('Raw reward' if tname == 'raw' else 'Cost-normalized reward'),
                     fontsize=9.5)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 1.04))

    ax = axes[2]
    rows = [r for r in paired if r['kind'] == 'representation_contrast']
    order = [(t, sc, c) for t in ('raw', 'cng') for sc in SCORERS for c in ('B', 'E')]
    ypos = np.arange(len(order))[::-1]
    for y, (t, sc, comp) in zip(ypos, order):
        r = next(q for q in rows if q['target'] == t and q['scorer'] == sc
                 and q['contrast'].endswith(f'D_minus_{comp}'))
        m, lo, hi = float(r['mean']), float(r['ci95_low']), float(r['ci95_high'])
        marker = 'o' if comp == 'B' else 's'
        ax.plot([lo, hi], [y, y], color=COLORS[sc], linewidth=1.2, zorder=3)
        ax.plot([m], [y], marker=marker, color=COLORS[sc], markersize=4.5, zorder=4,
                markeredgecolor='white', markeredgewidth=0.5)
    ax.axvline(0, color='#333333', linewidth=0.9)
    ax.set_yticks(ypos, [f'{"$R_a$" if t == "raw" else "$\\tilde{R}_a$"} · {LABELS[sc]}: D−{c}'
                         for t, sc, c in order], fontsize=7)
    ax.set_xlabel('instance-cluster mean difference (95% CI)')
    ax.set_title('(c) Representation contrasts', fontsize=9.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color='#dddddd', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / 'fig_ext_summary.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'fig_ext_summary.pdf', bbox_inches='tight')
    print('saved fig_ext_summary.png / fig_ext_summary.pdf (300 DPI)')


if __name__ == '__main__':
    main()

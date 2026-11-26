#!/usr/bin/env python3
"""Regenerate every paper figure's panels from frozen records (no search, no refits).

SWEVO final numbering vs. builder output names (builder indices are legacy):
  Figure 1  fig1_overview_architecture        <- scripts/build_fp_overview_figure.py
  Figure 2  fig2_transition_errors            <- scripts/build_fp_decision_figures_v2.py
  Figure 3  fig3_primary_decision_contrasts   <- scripts/build_fp_decision_figures_v2.py
  Figure 4  fig4_scorer_sensitivity           <- scripts/build_fp_fig4_fig5.py
  Figure 5  fig5_choices_and_scope            <- scripts/build_fp_fig4_fig5.py
  Figure 6  fig_ext_summary (1x3 panels)      <- experiments/ext_standard_jssp/code/plot_ext_summary.py
  Figure 7  fig6_independent_repeat_winners   <- v2 'fig5_independent_repeat_winners' (renamed)
  Figure 8  fig7_action_window                <- v2 'fig6_action_window' (renamed)
  Nature-style renderings of Figures 2/7/8    <- scripts/build_nature_final_figures_2_7_8.py

The **official submission files** (`Figure1.jpeg`, `Figure2.png` … `Figure8.png` in the same
figures directory) are the authors' final revision exports, byte-identical to the submitted
DOCX embeds; they are archived, not regenerated (see figures/PROVENANCE.md). This wrapper
regenerates the reproducible builder renderings and leaves the official files untouched.

The v2 builder predates the final figure order: it emits fig5_independent_repeat_winners /
fig6_action_window plus superseded drafts (fig1_evaluation_flow, fig4_choices_and_scope,
fig7_sensitivity_diagnostics). This wrapper applies the final renaming and removes the
superseded draft names.

Figures are regenerable artifacts (not write-once receipts): running this script in place
overwrites the builder PNG/PDF/SVG files under projects/submission/fp_decision_study/figures/,
projects/analysis/fp_decision_study/figures/ and the ext fig_ext_summary pair.
Requires matplotlib + numpy. Run from the repository root:

    python scripts/regenerate_figures.py
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'projects/submission/fp_decision_study/figures'
EXT_OUT = ROOT / 'experiments/ext_standard_jssp/outputs'

BUILDERS = [
    'scripts/build_fp_overview_figure.py',
    'scripts/build_fp_decision_figures_v2.py',
    'scripts/build_fp_fig4_fig5.py',
    'scripts/build_nature_final_figures_2_7_8.py',
    'experiments/ext_standard_jssp/code/plot_ext_summary.py',
]
RENAMES = {  # final paper numbering (applied to FIG only)
    'fig5_independent_repeat_winners': 'fig6_independent_repeat_winners',
    'fig6_action_window': 'fig7_action_window',
}
SUPERSEDED = ['fig1_evaluation_flow', 'fig4_choices_and_scope', 'fig7_sensitivity_diagnostics']
EXTS = ('png', 'pdf', 'svg')


def main():
    env = {**os.environ,
           'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': os.pathsep.join([str(ROOT / 'src'), str(ROOT)]),
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with tempfile.TemporaryDirectory(prefix='fp_fig_mpl_') as td:
        env['MPLCONFIGDIR'] = td
        for script in BUILDERS:
            print(f'>>> {script}', flush=True)
            subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT), env=env, check=True)
    for old, new in RENAMES.items():
        for ext in EXTS:
            src, dst = FIG / f'{old}.{ext}', FIG / f'{new}.{ext}'
            if src.exists():
                shutil.move(str(src), str(dst))
                print(f'renamed {src.name} -> {dst.name}')
    for stem in SUPERSEDED:
        for ext in EXTS:
            p = FIG / f'{stem}.{ext}'
            if p.exists():
                p.unlink()
                print(f'removed superseded draft {p.name}')
    expected = ['fig1_overview_architecture', 'fig2_transition_errors',
                'fig3_primary_decision_contrasts', 'fig4_scorer_sensitivity',
                'fig5_choices_and_scope', 'fig6_independent_repeat_winners',
                'fig7_action_window']
    missing = [f'{s}.{e}' for s in expected for e in EXTS if not (FIG / f'{s}.{e}').exists()]
    missing += [f'fig_ext_summary.{e}' for e in ('png', 'pdf') if not (EXT_OUT / f'fig_ext_summary.{e}').exists()]
    if missing:
        raise SystemExit(f'MISSING FIGURES: {missing}')
    print('Builder renderings regenerated (legacy names + Nature-style 2/7/8); official Figure1-8 files untouched.')


if __name__ == '__main__':
    main()

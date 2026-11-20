# Manuscript-current figure renderings — provenance

These four files are the **author-provided master renderings embedded in the current manuscript**
(DOCX snapshot 2026-11-20 18:38, `word/media/image1/4/5/6`). They are archived **byte-identically**;
each SHA256 below equals the corresponding DOCX media part.

| Repo file | Author master file name | DOCX media part | Bytes | SHA256 |
|---|---|---|---|---|
| `fig1_overview_architecture.jpeg` | `Figure1.jpeg` (2026-11-17) | `image1.jpeg` | 2,236,463 | `25c7946bd0019f9d5bc434f83d9ffd3795a5001206059ce3785bbe443be9daa7` |
| `fig4_scorer_sensitivity.png` | `Code_Generated_Image (2).png` (2026-11-19) | `image4.png` | 324,944 | `60d2682fb3a059f6542c5de4fc0c3f8f8d68434cca8cfe0c2aacebc35ea5c361` |
| `fig5_choices_and_scope.jpeg` | `Figure5.jpeg` (2026-11-18) | `image5.jpeg` | 2,007,658 | `a741f840fa1f530c85afa14da20eeecda86ebb4d13e2aa25e5f1bf74ec0c07f4` |
| `fig6_ext_summary.png` | `fig_ext_summary.png` (2026-11-20) | `image6.png` | 486,363 | `80b2b6f4a841975e3e8a427dcf5fc140889b6a57b33b1d9ee0fe29a43e7ce6a3` |

## Relationship to the reproducible builders

- The shipped builders (`scripts/build_fp_overview_figure.py`, `scripts/build_fp_fig4_fig5.py`,
  `experiments/ext_standard_jssp/code/plot_ext_summary.py`, driven by
  `scripts/regenerate_figures.py`) reproduce the **earlier vector-based renderings** shipped as
  `fig1/fig4/fig5` PNG/PDF/SVG and the 1×3-panel `fig_ext_summary.png/pdf` in `outputs/`.
  `verify_all.py --figures` regenerates those from the frozen tables.
- The masters above are the renderings the manuscript actually embeds:
  - `fig4_scorer_sensitivity.png` (3956×1741) and `fig6_ext_summary.png` (3780×2340, 2×2 panels)
    carry a `Matplotlib 3.6.0` Software tag: they were produced in the author's plotting
    environment (300 dpi); the generating scripts were not supplied with the masters.
  - `fig1_overview_architecture.jpeg` (2752×1536) and `fig5_choices_and_scope.jpeg` (2752×1536)
    are author-provided JPEG exports (no EXIF or comment metadata).
- All underlying data for every panel is shipped (`figure_source_data.json`, the S8 JSONs,
  `table_ext_*.csv`), so the figures remain re-plot-able from the archive even where the exact
  master rendering script is absent.
- The remaining manuscript figures (F2, F3, F7, F8 in current numbering) are byte-identical to
  the shipped `fig2_transition_errors.png`, `fig3_primary_decision_contrasts.png`,
  `fig6_independent_repeat_winners.png`, `fig7_action_window.png` respectively.
- Metadata audit: the four masters contain only matplotlib `Software`/dpi tags (PNG) or plain
  JFIF headers (JPEG); no EXIF, comments, personal names or local paths.

If the manuscript embeds newer renderings later, re-sync this directory and update the hashes
above; `verify_all.py` checks the presence of all four files.

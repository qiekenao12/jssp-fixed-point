# Official paper figures — provenance

These eight files are the **official submission renderings**, archived byte-identically; each
SHA256 equals the corresponding embedded image in the submitted `Manuscript.docx`
(SWEVO submission package, 2026-11-24). File names follow the official sequential convention
`Figure1`…`Figure8` (iteration suffixes of the working copies removed; bytes unchanged).

| Official file | Working-copy name | DOCX media part | Bytes | SHA256 |
|---|---|---|---|---|
| `Figure1.jpeg` | `Figure1.jpeg` | `word/media/image1.jpeg` | 2,019,006 | `03cee0d3c7e148476f25275ff624f0c07621b912ce711fb746af446deed78abc` |
| `Figure2.png` | `Figure_2_nature_v3.png` | `word/media/image2.png` | 273,066 | `7ec48a02117dc530e3cc2ae516570bf5612ec6a533fecb048f2211167a1ede40` |
| `Figure3.png` | `Figure_3_replace.png` | `word/media/image3.png` | 182,386 | `409f98b3c356e6af84cec8ea42654246a740d7c0cb89a6f2cbb2d40268268b70` |
| `Figure4.png` | `Figure_4_nature_final.png` | `word/media/image4.png` | 299,320 | `68c2faea06ef25d2fb45f2dadeeaee8350323f67a66d0e150d993329e0d80754` |
| `Figure5.png` | `Figure_5_nature_final.png` | `word/media/image5.png` | 341,759 | `1c72ae3aa683fc9e749e9029ec274d1468a965376b0f1840dbbf823cc3399b5a` |
| `Figure6.png` | `Figure_6_nature_v2.png` | `word/media/image6.png` | 543,584 | `20d0e3d4c1a3d6beba0a4a4006d6522ae7c399fbbdb28ec432a309c8380a9250` |
| `Figure7.png` | `Figure_7_nature_v2.png` | `word/media/image7.png` | 174,890 | `1e63357039a3fdcd1232b8a6ebe4711b35a63ec4c60f9d1af744f3768f7c142a` |
| `Figure8.png` | `Figure_8_nature_v2.png` | `word/media/image8.png` | 150,145 | `1e7971c4e05efd0fa74419e56a8f960f1f1d0f526529f08726b96729c7969505` |

## Relationship to the shipped builders

- `scripts/build_fp_overview_figure.py`, `scripts/build_fp_decision_figures_v2.py`,
  `scripts/build_fp_fig4_fig5.py` and `experiments/ext_standard_jssp/code/plot_ext_summary.py`
  regenerate the same panels from the frozen tables (see README §4).
- `scripts/build_nature_final_figures_2_7_8.py` produces the Nature-style renderings of
  Figures 2, 7 and 8 from the same frozen sources (read-only; means and CIs plotted verbatim).
- The official files above are the authors' final revision exports of those renderings
  (Figures 1, 3, 4, 5, 6 are final restyled exports; no separate builder script was supplied
  for them). All underlying panel data is shipped, so every figure remains re-plot-able.
- Metadata audit: PNG/JPEG files carry no EXIF, comments, personal names or local paths.

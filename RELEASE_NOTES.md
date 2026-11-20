# RELEASE NOTES — build audit, split rationale, sanitization record

Prepared 2026-11-20 for the SWEVO submission reproducibility release.
This file records the Step-0 workspace audit, the exact core/asset split, and every
sanitization decision (including hash-lock exceptions). It is release documentation,
not a study artifact; no frozen study file was modified to produce this release.

## 1. Source workspace audit (summary)

Source workspace: a private research tree of ~37,787 files / ~13 GB (excluding vendored
libraries), containing several unrelated or superseded sub-projects. Only the assets of the
present study were selected:

| Group | Source location | Size | Files |
|---|---|---|---|
| Historical genetic/memetic code + frozen models + protocols | root `run_*.py`/`analyze_*.py`, `src/tools/`, `scripts/`, `projects/experiments/` | ~4 MB | ~45 |
| Historical raw branches | `results/fp_v2_development` (70 MB/641), `results/fp_v2_validation` (45 MB/481), `results/fp_empirical_repeatability_development` (33 MB/80), `results/fp_empirical_confirmation` (188 MB/801) | 336 MB | 2,003 |
| Historical large analysis JSONs | `projects/analysis/fp_submission_final_confirmation_results.json` (55 MB), `fp_submission_convergence_results.json` (35 MB), `fp_agent_decision_feature_comparison.json` (18.7 MB), `fp_v2_development_analysis.json` (118 MB; frozen M2 analysis consumed by the repeatability replay) | 227 MB | 4 |
| Primary synthetic cohort code/generator/models/locks | `scripts/*fp_decision*`, `src/tools/independent_tabu_jssp.py`, `projects/experiments/fp_decision_study/` | ~0.5 MB | 12 |
| Primary summaries + S8 diagnostics + audits | `projects/analysis/fp_decision_study/` (56 files), `projects/analysis/revision_response_artifacts/` (6 files) | 36 MB | 62 |
| Primary raw branches | `results/fp_decision_study/` | 21 MB | 924 |
| Prospective extension (excl. its private venv) | `experiments/ext_standard_jssp/` | 39 MB | ~940 |
| Paper figures + editable tables | `projects/submission/fp_decision_study/{figures,tables}/` | 27 MB | 53 |

- **Files > 40 MB in the release candidate set: none.** Largest shipped file:
  `projects/analysis/fp_decision_study/original_final_analysis.json` (10.37 MB).
  Workspace-wide >40 MB files (legacy `agent_diagnostics_20runs*.json` 1.1–1.4 GB,
  `paper_q_final` 4.8 GB, private review ZIPs, etc.) all belong to other studies and are excluded.
- **Excluded entirely:** all Word/PDF manuscript drafts and build scripts
  (`*.docx`, `*.pdf` drafts, `build_fp_decision_manuscript.py`, `export_fp_decision_documents.py`,
  `build_fp_decision_latex.py`, `validate_fp_decision_delivery.py`, `package_fp_decision_study.py`),
  `draft_traceability/`, `revision_backup_*`, `execution_logs/`, the separate resource-allocation /
  p2-headroom / Journal-of-Heuristics projects, SAFE/BRFP/proof-reuse sub-study code and analyses,
  `venv/`, `experiments/ext_standard_jssp/venv_ext/` (209 MB), vendored `lib/` (40 MB),
  `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `.DS_Store`, `.env*`, `.aws/`, `.codex/`,
  `.trae/`, `.agents/`, `.paper_work/`, `pipeline.log`, temporary logs, review ZIPs, `uv.lock`,
  `environment.yml`, root `README.md` (describes a different orchestration study).
  The superseded v1 figure builder `scripts/build_fp_decision_figures.py` was removed on
  2026-11-20 after the initial push (unlocked, unreferenced; `build_fp_decision_figures_v2.py`
  is the canonical builder used by `regenerate_figures.py`).
- `scripts/fp_decision_history_audit.py` is not shipped because it requires 26 private historical
  context documents (`projects/docs/*.md` drafts) that are intentionally not redistributed; its
  frozen outputs (`history_audit.json`, `history_final_audit.json`) **are** shipped, and all
  historical source fingerprints remain in `history_final_sources.csv` / `history_sources.csv`.

## 2. Split rationale

- **Core Git repo (`github_release/`)**: 2,268 files, ~173 MB. Contains all source code, the
  synthetic instance generator, benchmark manifests/hashes, lock manifests, frozen models, all
  paper result CSVs/JSONs, S8 artifacts, plotting scripts, both independent verifier families,
  the historical closed-loop trigger cohort (800 runs, 45.9 MB; see §8), and the raw branches of
  the **primary synthetic (87,600)** and **extension (24,000)** cohorts —
  these are required by the one-command verification and are small enough for Git.
- **Release assets (`release_assets/`, uploaded to GitHub Releases / Zenodo)**: historical raw
  branches (336 MB, 2,003 files) + large historical analysis JSONs (227 MB, 4 files) exceed the
  ~200 MB / 2,000-file raw-data threshold, so they are split into five `.tar.gz` archives
  (~127 MB compressed total, 2,007 files uncompressed) with `SHA256SUMS.txt`. See
  `release_assets/README.md` and README §6.
- **Frozen historical source snapshots** are small and ship in the core repo
  (`projects/experiments/fp_v2_full_replay_sources.zip`,
  `fp_empirical_repeatability_collection_sources.zip`, `fp_submission_final_replay_sources.zip`
  + manifests): the fp_v2 cohorts were collected with pre-2026-10-26 versions of
  `src/tools/controller.py`, `fixed_point_detector.py`, `meta_scheduler.py`, and their
  `protocol.json` locks those older hashes. `verify_all.py --historical` extracts the canonical
  `fp_v2_full_replay_sources.zip` over its temporary sandbox before auditing those cohorts
  (the shipped current sources match the later 2026-10-26 `final_lock.json` exactly). The
  confirmation cohort's 21-file source lock matches the shipped current sources byte-for-byte.
- The original workspace-relative layout is preserved because every locked script hard-codes
  root-relative paths and every lock manifest hashes those relative paths; renaming would break
  both imports and SHA256 locks. The README layout table maps each requested category to its path.

## 3. Integrity verification performed at build time

- All 32 `final_lock.json` file hashes re-verified inside the release (byte-identical copies).
- Extension `manifest_lock.json`: 10 new-code + 12 reused-frozen-code + 84 benchmark hashes
  re-verified; `lock_hash` and `instances_hash` recomputed from content and matched.
- Three post-lock **engineering-only amendments** honored: for `analyze_ext.py` (001),
  `verify_ext_replay.py` (002), `plot_ext_summary.py` (003) the lock stores `sha256_before` and the
  shipped file equals `sha256_after` recorded in `code_amendment_00*.json`. No statistical,
  model, seed, data or protocol change (see the amendment records themselves).
- `design_hash` / `instances_hash` of the primary frozen design recomputed and matched.
- Full clean-room verification (`scripts/verify_all.py --figures --historical`) executed from
  inside `github_release/` with the existing interpreter (Python 3.13.13, numpy 2.4.6): all
  primary + extension schedule audits, both scalar-replay families, 15 unit tests, all 8 figure
  regenerations, and all four historical cohort audits (after extracting the five release-asset
  tarballs to `raw_extracted/`) PASS; regenerated reports match the shipped frozen receipts
  (see §5 for the single documented provenance exception). The historical repeatability replay
  cross-references the original fp_v2 source trajectories, so all four raw cohorts must be
  extracted together. Results: `GITHUB_UPLOAD_GUIDE.md` §Pre-push checklist.
- Historical trigger cohort (added 2026-11-20, §8): `analyze_trigger_comparison.py` re-audits all
  800 paired runs and recomputes every statistic from the saved run JSONs inside the sandbox;
  regenerated `summary.json`, `summary.csv`, `per_instance.csv`, `events.csv` and `report.md`
  are **byte-identical** to the shipped frozen receipts.

## 4. Privacy / sanitization record

Scanned every shipped text file for `/Users/`, `/home/`, `C:\Users`, personal usernames,
API-key patterns, `venv_ext`, `__pycache__`, `.docx`, `resource-allocation`.
**No absolute local paths, usernames, credentials, venvs, caches, Word/PDF drafts or
unrelated-project files are shipped.** All paths in scripts/manifests are repository-relative.

The following **documented exceptions** are unavoidable: the tokens appear inside
**hash-locked or frozen provenance records** whose bytes cannot be altered without breaking
SHA256 locks (the locks themselves are part of the scientific record). Each is a *relative path
or a disclosure sentence*, never a copied private asset:

| File | Token(s) | Why it must stay byte-identical |
|---|---|---|
| `experiments/ext_standard_jssp/outputs/manifest_lock.json` | `venv_ext` | `environment.venv` records the relative venv path used at collection; covered by `lock_hash` |
| `experiments/ext_standard_jssp/code/run_ext_pipeline.py` | `venv_ext` | hash-locked orchestrator; expects that relative interpreter path (README §7 explains recreation) |
| `experiments/ext_standard_jssp/code/prepare_ext_lock.py` | `venv_ext`, `__pycache__`, `.docx` | hash-locked; tokens are *exclusion filters* used to build the frozen snapshot |
| `experiments/ext_standard_jssp/STEP0_REPORT.md`, `outputs/SUMMARY.md` | `venv_ext` | frozen study reports describing the isolated environment (relative path only) |
| `experiments/ext_standard_jssp/outputs/frozen_snapshot_{before,after}.json` | `__pycache__`, `.docx` | whole-workspace hash snapshots (35,560 paths) necessarily enumerate excluded assets; **hashes only, no content** |
| `projects/docs/fp_decision_empirical_manuscript.md` | `resource-allocation` | this study's own Markdown source (not a Word/PDF draft, not the unrelated manuscript); required by the independent replay's `reviewed_sha256`; the token occurs only in disclosure sentences stating that no budget-control contribution is claimed and that shared material must be disclosed by the authors |
| `README.md`, `RELEASE_NOTES.md`, `GITHUB_UPLOAD_GUIDE.md`, `.gitignore`, `scripts/verify_all.py` | various | release documentation/config describing or ignoring these exceptions; the scanner contains the tokens as literals |

`scripts/verify_all.py` enforces this allowlist automatically: any occurrence outside it fails
the hygiene gate. The three frozen source-snapshot ZIPs were additionally scanned member-by-member
(all `.py`/`.md`/`.json`/`.txt` contents) for the same patterns: zero hits. Binary figure files
(PNG/PDF/SVG) were also inspected: PDF metadata contains only Matplotlib creator/producer strings
and figure-stem titles; SVG text was pattern-scanned; no personal paths or usernames anywhere.
Historical lock files (`final_lock.json`, `fp_v2_m2_protocol.json`,
`fp_submission_*_protocol.json`, `manifest_lock.json`, …) store **repository-relative paths
only**; they were not altered, and they resolve correctly inside this release because the
layout is preserved.

## 5. Documented provenance exception (checked, not skipped)

`projects/analysis/fp_decision_study/independent_analysis_review_replay.json` (frozen
2026-10-26) records `reviewed_sha256` for seven files. Six match the shipped bytes exactly.
The seventh, `projects/docs/fp_decision_empirical_manuscript.md`, was revised in place later on
2026-10-26 (after the independent review ran); the reviewed pre-revision version
(SHA256 `550e0070e1c520112ef324c87dabf5568cac892a8d22cc1d0aba790733c0e295`) no longer exists
anywhere in the source workspace (verified by hashing all 243 Markdown files and the three
delivery ZIPs). The shipped revised document hashes to
`42099499343b15a9d3702a52bb714919cdfd92cbe640b7d95f44504be429d302`.
`verify_all.py` asserts that the mismatch is **exactly this documented revision pair** and that
all other reviewed hashes match; every numerical replay field matches the frozen receipt
exactly. No result, model, seed or statistic was affected (the document is prose only).

## 6. Environment notes

- Primary cohort reference environment: Python 3.13.13, numpy 2.4.6 (`final_lock.json`).
- Extension reference environment: Python 3.12.13, numpy 2.4.2, scipy 1.18.1,
  scikit-learn 1.9.1, pandas 2.3.3, matplotlib 3.10.8, joblib 1.6.0 (`manifest_lock.json`).
- `requirements.txt` pins the union (numpy at the primary lock value). The independent
  verifiers need only numpy + stdlib; the extension replay was verified at tolerance 1e-12
  under numpy 2.4.6 (max reconstruction error ≤ 2.9e-16). scikit-learn must remain 1.9.1 to
  load the frozen GBDT joblib bundles bit-compatibly.
- Determinism flags used throughout: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`, fixed seeds, `early_stopping=False`.

## 7. Author inputs still required before publication

Author identities/affiliations, funding and conflict declarations, the related-manuscript
disclosure wording, rights confirmation for third-party benchmark redistribution in your
jurisdiction, the GitHub repository URL and the Zenodo DOI (placeholders in README §9).
No DOI, license choice beyond the stated MIT/CC-BY proposal, or funding identity is invented.

## 8. Historical trigger cohort provenance (added 2026-11-20)

`results/trigger_controlled_20runs/` (210 files, 45.9 MB) is the closed-loop VND-trigger
comparison underlying the manuscript's historical trigger table: 10 OR-Library/Taillard
instances (ft06, la04, ft10, orb01, la21, la26, la40, yn1, ta31, ta51) × 20 paired seeds ×
4 trigger policies (fixed period / stagnation / state drift / FP residual) + detached probe
trajectories = 800 deployed runs + 200 probe runs, seed 20260916, equal budget of 30,000
objective-function queries per run (VND sub-budget 8,000; per-call cap 500). Shipped alongside:
`run_trigger_comparison.py` (collector) and `analyze_trigger_comparison.py` (auditor/summarizer,
whose SHA256 is recorded in `summary.json → analysis_source_sha256`).

- **Excluded duplicates:** a same-day second-environment rerun (`trigger_controlled_20runs_local`,
  Python 3.12.13/numpy 2.4.2 vs the shipped run's 3.13.13/numpy 2.4.6) and a smoke run. The rerun's
  scientific content is **identical** to the shipped cohort — only wall-clock fields (`elapsed`,
  `observation_elapsed`) differ — providing cross-environment determinism evidence; it is not
  redistributed. The superseded confounded pilot is also excluded.
- **Verification:** `verify_all.py` re-runs the full paired-group audit (initialization
  fingerprints, query budgets, event gain accounting) and replays all statistics; outputs match
  the shipped receipts byte-for-byte. Mean final gaps 11.488/11.253/11.664/11.434% and all Holm
  contrasts reproduce exactly.
- **Source-lock limitation (documented, not hidden):** `protocol.json → source_hashes` locks the
  2026-10-17 versions of `src/tools/executor.py`, `fixed_point_detector.py`, `jssp_core.py`,
  `meta_scheduler.py`. The shipped current modules match the later 2026-10-26 `final_lock.json`;
  the trigger-era versions are **not preserved in any available snapshot** (all four frozen
  source ZIPs and the Aug-13 backup were checked; the workspace `.git` is an empty directory).
  Consequently this cohort is verifiable at the *saved-log + replay* level (above), but a
  source-locked *re-collection* from the shipped tree is not possible. The dual-environment
  agreement and the collection-time audit receipt ("Passed all paired initialization, query
  budget, event and gain consistency checks") are the compensating evidence.
- `report.md` in this directory is the original Chinese-language experiment record produced by
  the analysis script at collection time; it is regenerated byte-identically during verification.

## 9. Manuscript-current figure masters (added 2026-11-20)

`projects/submission/fp_decision_study/figures/manuscript_current/` archives, byte-identically,
the four author-provided master renderings embedded in the current manuscript DOCX (snapshot
2026-11-20 18:38): overview (JPEG, 2026-11-17), scorer sensitivity (PNG, 2026-11-19),
choices/scope (JPEG, 2026-11-18), and the 2×2 extension summary (PNG, 2026-11-20). SHA256s and
the DOCX media-part correspondence are recorded in that directory's `PROVENANCE.md`; each hash
equals the corresponding `word/media/` part. The two PNGs carry only a Matplotlib 3.6.0 Software
tag (author plotting environment, 300 dpi; generating scripts not supplied); the JPEGs carry no
EXIF/comments. No personal names or local paths are present in any of the four files. The shipped
builders continue to reproduce the earlier vector-based renderings from the frozen tables, and
all panel data remains shipped, so the figures stay re-plot-able from the archive.

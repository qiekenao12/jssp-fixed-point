# Step 0 Audit Report — `ext_standard_jssp` Prospective Extension Cohort

Date: 2026-11-20 (Asia/Shanghai). All Step 0 artifacts live in `experiments/ext_standard_jssp/step0/`.
**No frozen asset was modified.** (Note: `.git/` is an empty directory — this checkout is NOT a git
repository, so `git status`/`git diff` cannot be used. Integrity will instead be verified with a
full SHA256 before/after snapshot of every frozen asset, recorded in `manifest_lock.json` and
`verification_report.json`.)

## 1. Located protocol components (all read-only reuse)

| Component | File | Key facts |
|---|---|---|
| Independent critical-block solver | `src/tools/independent_tabu_jssp.py` | machine-order DAG, heap longest-path decoder, single deterministic critical path (min-index tie-break), `CB_BEST_DESCENT`/`CB_FIRST_DESCENT`/`CB_TABU` (tenure 5, incumbent aspiration), native 4D state `[best/s0, current/s0, path_frac, block_frac]`, `features()` ordinary/history/structure blocks |
| Branch collector | `scripts/run_fp_decision_collection.py` | neutral tabu warm-up 8 then 16 accepted moves, 2 saved states/trajectory, 3 actions × 5 repeats × ≤8 accepted moves, SHA256-derived seeds (namespace 2026102501), write-once gzip archives + content hashes, resume-safe |
| Affine maps + 6 representations | `scripts/fp_decision_models.py` | displacement ridge 0.1 raw normal equations, unpenalized intercept, standardized starts, `W=I`; arms A,B,C,D,E,D_random (random norms seed 2026102501, width-matched 6×4); LOIO cross-fitted maps for dev training features; `fit_ridge` asserts gradient ∞-norm ≤ 1e-7 and augmented-LSTSQ agreement |
| LOOCV ridge + 256-RFF | `scripts/revision_response_analysis.py` | L2 grid {1e-2,1e-1,1,10,100,1000}; RFF seed 20261112, median-heuristic bandwidth on ≤500 dev subsample, standardized design matrix; cost-normalized target `gain/decodes*1e3` |
| Logistic event scorer | `scripts/check_fp_scorer_convergence.py` (`ConvergedLogistic`) | damped Newton, L2=1, gradient tol 1e-7 |
| Synchronous time-relaxation verifier | `scripts/audit_fp_decision_collection.py` | independent monotone relaxation of every saved/endpoint schedule, full grid/hash/reward/cost/seed checks |
| Independent scalar replay | `scripts/verify_fp_decision_analysis_independent.py`, `scripts/verify_fp_decision_statistics.py` | no scorer imports; raw rewards, choices, exact signed-rank tails, Holm, 20,000-draw instance bootstrap (seed 2026102502) |
| Exact statistics | `analyze_paper_evidence.py` (`signed_rank` exact DP, `holm`), `scripts/analyze_fp_decision_study.py` (`compare`: instance-cluster bootstrap, W/T/L, LOIO range) | |
| Frozen design reference | `projects/experiments/fp_decision_study/design.json`, `instances.json`, `tabu_models.json` | 32 synthetic instances (8 dev / 4 val / 20 final), 10 dev runs, 20 final runs |

## 2. Benchmark inventory & overlap (full hashes in `step0/benchmark_inventory.json`)

- `jobshop1.txt` (OR-Library, root): 84 blocks parsed; **la01–la40 all present**, yn1–yn4 (20x20),
  swv, abz, ft, plus appended ta31/ta51. `orb07` has zero durations → rejected by the solver (excluded).
- `projects/experiments/benchmark_jobshop.txt` (frozen 46-instance archive): contains
  **ta01, ta02, ta11, ta12, ta21** (the only locally present ta01–ta30 members) + ta31/41/42/51/61/70.
  Parser cross-check: all shared names have **identical route/duration SHA256 fingerprints** vs `jobshop1.txt` (0 mismatches).
- `projects/experiments/fp_v2_jobshop.txt` (**historical 48-instance genetic study**): fingerprint overlap with the proposed cohort:
  - Lawrence la21–la40 ∩ genetic48 = **{la21, la24, la29, la33, la34, la35, la37, la38, la39, la40}** (10 instances).
  - Taillard ta01–ta30 ∩ genetic48 = **∅** (ta01/02/11/12/21 appear only in the frozen 46 archive, not the genetic 48).
  - **All 50 cohort instances are NEW to the critical-block stack** (its frozen study used only the 32 synthetic instances). Overlap is disclosed per-instance in the lock manifest.
- **Missing locally: 25 Taillard instances** (ta03–ta10, ta13–ta20, ta22–ta30).
  Public JSPLIB mirror `https://raw.githubusercontent.com/tamy0612/JSPLIB/master/instances/taNN` is reachable;
  **downloaded ta01 fingerprint == frozen local ta01 fingerprint** (`fa31a659…f08d`, exact match) — mirror fidelity spot-check PASS.
  Plan: download the 25 missing files into `experiments/ext_standard_jssp/benchmarks/`, re-verify all 5 locally-known
  Taillard fingerprints against the mirror, and lock every file's SHA256 + source URL before any collection.

## 3. Python environment (isolated, no existing env modified)

- System/base and `jssp` conda env lack scikit-learn; `hmbgym` (py3.8) cannot import the solver (PEP 585).
- Created **dedicated venv** `experiments/ext_standard_jssp/venv_ext` (from `jssp` py3.12.13, `--system-site-packages`):
  **Python 3.12.13, numpy 2.4.2, scipy 1.18.1, scikit-learn 1.9.1, pandas 2.3.3, matplotlib 3.10.8**.
- GBDT choice: **`sklearn.ensemble.HistGradientBoostingRegressor`** (LightGBM unavailable). Deterministic settings:
  `early_stopping=False`, fixed `random_state`, `min_samples_leaf=20`, `OMP_NUM_THREADS=1` per fit,
  parallelism across fits via process pool (each fit single-threaded → exact reproducibility).
  Grid: `max_depth∈{2,3} × learning_rate∈{0.03,0.1} × max_iter∈{50,100,200} × l2_regularization∈{1,10}` = 24 configs/arm/target.

## 4. Timing micro-benchmark (1 trajectory, 2 states, 3 actions × 5 repeats; `step0/microbenchmark.json`)

| Instance | Size | Trajectory wall (s) | Mean branch (s) |
|---|---|---|---|
| yn1 | 20x20 | 0.786 | 0.0224 |
| la31 | 30x10 | 0.967 | 0.0275 |
| la36 | 15x15 | 0.331 | 0.0094 |

Full-cohort estimate (20 dev × 10 traj + 30 test × 20 traj = **800 trajectories / 24,000 branches**):
≈ **423 core-seconds ≈ 26 s wall @ 16 workers** (≈ 53 s @ 8). Even with a 5× safety factor, collection ≪ 15 min.
GBDT LOOCV (6 arms × 2 targets × 24 configs × 20 folds = 5,760 single-threaded fits on ≤6,000 rows) is the dominant
cost: estimated 20–60 core-minutes → **≈ 2–5 min wall @ 16–24 workers**. Total pipeline comfortably < 15 min.

## 5. Proposed stratified split (seed 2026112000, `step0/proposed_split.json`)

7 (family × size) blocks; 10-instance blocks → 4 dev / 6 test; 5-instance blocks → 2 dev / 3 test.
**20 dev / 30 test**, zero dev/test fingerprint duplication (all 50 fingerprints distinct).

| Block | Dev | Test |
|---|---|---|
| ta 15x15 | ta02, ta07, ta08, ta09 | ta01, ta03, ta04, ta05, ta06, ta10 |
| ta 20x15 | ta12, ta13, ta19, ta20 | ta11, ta14, ta15, ta16, ta17, ta18 |
| ta 20x20 | ta24, ta26, ta27, ta30 | ta21, ta22, ta23, ta25, ta28, ta29 |
| la 15x10 | la22, la24 | la21, la23, la25 |
| la 20x10 | la28, la29 | la26, la27, la30 |
| la 30x10 | la31, la33 | la32, la34, la35 |
| la 15x15 | la38, la40 | la36, la37, la39 |

Dev: 200 trajectories / 6,000 branches / 400 saved states. Test: 600 trajectories / 18,000 branches / 1,200 saved states.

## 6. Pending confirmation

The 25 missing Taillard files must be fetched from the public JSPLIB mirror (external supply).
Per the Step 0 protocol, execution pauses here for user confirmation.

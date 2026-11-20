# SUMMARY — `ext_standard_jssp` Prospective Standard-Benchmark Extension

**Cohort:** 50 standard JSSP benchmarks (Taillard `ta01–ta30` + Lawrence `la21–la40`), independent
critical-block tabu stack, prospective lock-before-test protocol. All artifacts under
`experiments/ext_standard_jssp/`; **zero modification of frozen assets** (35,560 files SHA256-identical
before/after; `.git/` is empty, so hash snapshots replace `git diff`).
**All 11 verification gates PASS** (`outputs/verification_report.json`); independent scalar replay
reconstructed all 72 paired contrasts, both-target split-repeat diagnostics, per-size descriptives and
action distributions with max absolute error ≤ 2.9e-16 (`outputs/verification_replay.json`).

## 1. Design & provenance (exact)

- **Split (seed 2026112000, stratified by family×size; 20 dev / 30 test):**
  - dev (20): ta02, ta07, ta08, ta09, ta12, ta13, ta19, ta20, ta24, ta26, ta27, ta30, la22, la24, la28, la29, la31, la33, la38, la40
  - test (30): ta01, ta03, ta04, ta05, ta06, ta10, ta11, ta14, ta15, ta16, ta17, ta18, ta21, ta22, ta23, ta25, ta28, ta29, la21, la23, la25, la26, la27, la30, la32, la34, la35, la36, la37, la39
  - Size blocks: ta 15x15 / 20x15 / 20x20 (4 dev + 6 test each); la 15x10 / 20x10 / 30x10 / 15x15 (2 dev + 3 test each).
  - Test sizes: 15x10 ×3, 15x15 ×9, 20x10 ×3, 20x15 ×6, 20x20 ×6, 30x10 ×3.
- **Data volume:** dev 200 trajectories / 400 states / 6,000 branches; test 600 trajectories / 1,200 states / 18,000 branches (10 and 20 trajectories per instance; 2 saved states per trajectory after neutral tabu requests of 8 then 16 accepted moves; 3 actions × 5 repeats × ≤8 accepted moves; tenure 5, incumbent aspiration).
- **Lock:** `outputs/manifest_lock.json`, lock_hash `6492b6a4c34f2d65f9e4ae8fdb9fed61f65860dae6caa96c206f945021440065`, written before any test collection; contains code hashes (10 new + 12 reused frozen), 84 benchmark file hashes, split, seeds, grids.
- **Seeds:** collection namespace 2026112001 (SHA256-derived per branch); bootstrap 2026112002 (20,000 draws); GBDT random_state 2026112003; RFF 20261112 / 256 components (frozen); D_random 2026102501 (frozen).
- **Benchmark sourcing:** la21–la40 from local `jobshop1.txt`; ta01–ta30 from public JSPLIB mirror (tamy0612/JSPLIB). Mirror fidelity: all 5 locally frozen Taillard copies (ta01, ta02, ta11, ta12, ta21) match mirror fingerprints **exactly**. 50/50 unique route/duration fingerprints; dev∩test = ∅.
- **Overlap disclosure (historical genetic-48 study):** la21, la24, la29, la33, la34, la35, la37, la38, la39, la40 (10 instances; dev: la24, la29, la33, la38, la40; test: la21, la34, la35, la37, la39). Taillard ta01–ta30 ∩ genetic-48 = ∅ (ta01/02/11/12/21 appear only in the frozen 46-instance archive). **All 50 instances are new to the critical-block stack** (frozen study used 32 synthetic instances).
- **Environment:** isolated venv `venv_ext` — Python 3.12.13, numpy 2.4.2, scipy 1.18.1, scikit-learn 1.9.1 (`HistGradientBoostingRegressor`), pandas 2.3.3, matplotlib 3.10.8. Determinism: `OMP_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`, `early_stopping=False`, fixed seeds; every GBDT/RFF fit repeated twice with exact prediction equality.
- **Runtime:** dev collection 9.4 s, test collection 30.6 s (16 workers); dev-only fitting 182.3 s (5,760 GBDT LOOCV fits + 2,880 ridge/RFF LOOCV fits); schedule audits 1.3 s / 4.8 s; analysis 6.5 s; replay 9.9 s. Total ≪ 15 min.
- **Engineering amendments (no statistical/protocol change):** `code_amendment_001.json` (CSV fieldname union in analyze), `002` (replay CI column reference), `003` (figure legend layout). Before/after SHA256 recorded; lock left intact.

## 2. Fixed-action references & the all-zero floor (test, instance-cluster macro)

| Regime | CB_BEST_DESCENT | CB_FIRST_DESCENT | CB_TABU | Best fixed (predeclared = empirical) |
|---|---|---|---|---|
| Raw `R_a` | 0.0378023 | 0.0279501 | **0.0407925** | CB_TABU |
| `R_tilde_a×1e3` | 0.2641124 | **1.2748812** | 0.2735084 | CB_FIRST_DESCENT |

- Mean decoder requests / branch: BD 143.01, FD 28.14, TABU 160.78; mean accepted moves 6.69 / 6.91 / 8.00; mean branch seconds 0.0287 / 0.0056 / 0.0318.
- **All-zero state rate: 5/1,200 = 0.4167%** (15x10: 2.5%; 15x15: 0.556%; 20x10, 20x15, 20x20, 30x10: 0.0%) — the small-instance floor problem is eliminated.
- Modal best action per state: raw → CB_TABU in every size block; cng → CB_FIRST_DESCENT in every size block.

## 3. Main extension contrasts (instance-cluster; 20,000-draw bootstrap; exact Wilcoxon; Holm within {D−B, D−E})

**No D−B or D−E contrast is significant under raw reward** (all Holm p ≥ 0.3376). Under cost normalization,
**GBDT D−B is positive and significant**: mean **+0.004818**, CI **[+0.001665, +0.008661]**, p = 0.0015,
**Holm p = 0.0029**, W/T/L = 11/18/1, LOIO range **[+0.003790, +0.005085]** (all 30 leave-one-instance-out means positive).
LOOCV-ridge cng D−B is nominally positive (+0.001813, CI [+0.000002, +0.004252]) but Holm p = 0.25.

| Target · Scorer | D−B mean [95% CI] | Holm p | D−E mean [95% CI] | Holm p |
|---|---|---|---|---|
| raw · Frozen Ridge | −0.000036 [−0.000132, +0.000071] | 0.3376 | −0.000021 [−0.000114, +0.000079] | 0.4223 |
| raw · LOOCV Ridge | +0.000018 [−0.000097, +0.000120] | 0.5016 | +0.000038 [−0.000030, +0.000111] | 0.4494 |
| raw · RBF-RFF | −0.000108 [−0.000290, +0.000065] | 0.5420 | +0.000020 [−0.000268, +0.000277] | 0.5420 |
| raw · GBDT | −0.000183 [−0.000428, +0.000048] | 0.3679 | −0.000075 [−0.000311, +0.000132] | 0.9515 |
| cng · Frozen Ridge | +0.000289 [−0.000088, +0.000906] | 0.8750 | −0.000511 [−0.002019, +0.000552] | 0.8750 |
| cng · LOOCV Ridge | +0.001813 [+0.000002, +0.004252] | 0.2500 | −0.001091 [−0.003687, +0.000583] | 0.9375 |
| cng · RBF-RFF | +0.000871 [−0.001399, +0.003380] | 1.0000 | +0.000322 [−0.002554, +0.002878] | 1.0000 |
| cng · GBDT | **+0.004818 [+0.001665, +0.008661]** | **0.0029** | −0.001162 [−0.002885, +0.000150] | 0.3223 |

## 4. Learned arms vs best fixed static action — **no learned arm beats the optimal static action**

- **Raw:** best learned arm per scorer = A for Frozen/LOOCV ridge (selects CB_TABU in 1,200/1,200 states; return exactly 0.0407925 = fixed Tabu, difference 0, p = 1); RFF best A = 0.0406685 (−0.000124 vs Tabu, CI [−0.000219, −0.000044], Holm p = 0.0469 — significantly *worse*); GBDT best C = 0.0405378 (−0.000255, CI [−0.000400, −0.000115], Holm p = 0.0051 — worse). 14 of 24 raw arm-vs-Tabu contrasts are significantly negative after Holm (worst: GBDT D −0.000573, CI [−0.000775, −0.000381], Holm p < 1e-4, W/T/L 4/0/26). **Zero arms with positive mean and CI excluding 0.**
- **Cost-normalized:** best learned arm = GBDT E = 1.2747181 vs fixed First Descent 1.2748812: difference **−0.000163, CI [−0.000617, +0.000153]**, p = 0.625 (Holm 0.8633) — statistically indistinguishable, never better. GBDT B and C are significantly worse (−0.006144, Holm 0.0168; −0.006094, Holm 0.0139). **Zero arms beat fixed First Descent.**
- Action collapse persists at scale: e.g. Frozen-ridge raw choices are 1,045–1,200/1,200 Tabu; cng choices 1,174–1,184/1,200 First Descent. GBDT diversifies choices (raw D: 322 BD / 12 FD / 866 Tabu) but this *reduces* return.
- Logistic event scorer (L2=1, completeness): raw selected return 0.0407222–0.0407925 (≈ Tabu); macro Brier 0.03146–0.03326; macro AP 0.98644–0.99308.

## 5. Split-repeat sample-winner diagnostic (both regimes, both directions; vs best fixed static action)

| Target · Direction | Same-repeat adv [95% CI] | Fresh-repeat adv [95% CI] | Fresh p | Optimism gap | Unique winner | Label agreement |
|---|---|---|---|---|---|---|
| raw · {0,1}→{2,3,4} | +0.0017831 [+0.0015213, +0.0020515] | **−0.0004449 [−0.0006508, −0.0002273]** | 5.05e-4 | +0.0022816 | 90.17% | 54.75% |
| raw · {2,3,4}→{0,1} | +0.0014863 [+0.0012869, +0.0017058] | **−0.0005359 [−0.0008407, −0.0002436]** | 2.19e-3 | +0.0019688 | 92.00% | 54.75% |
| cng · {0,1}→{2,3,4} | +0.0104653 [+0.0065927, +0.0153291] | **+0.0091058 [+0.0054854, +0.0136996]** | 2.09e-7 | −0.0177960 | 99.50% | 98.58% |
| cng · {2,3,4}→{0,1} | +0.0105572 [+0.0068511, +0.0151519] | **+0.0094063 [+0.0057370, +0.0140971]** | 1.49e-8 | +0.0203064 | 99.58% | 98.58% |

Interpretation: under **raw** reward the noisy sample winner is *worse* than fixed Tabu on fresh repeats in
both directions — no exploitable per-state signal beyond the static optimum. Under **cost normalization**
the sample winner *does* beat fixed First Descent on fresh repeats (+0.0091 / +0.0094, both CIs exclude 0),
i.e. genuine per-state heterogeneity exists (≈8% of states where Tabu/Best-Descent is cheaper per unit gain),
yet **no learned scorer captures it** (Section 4). The negative forward cng "optimism" gap is dominated by a
chance repeat-subset fluctuation of the near-deterministic First Descent winner (its {0,1} mean sits ≈0.019
below its {2,3,4} mean, within 1 SE); the decision-relevant fresh-repeat advantage is positive and significant
in both directions.

## 6. Transition prediction error (test; conditioned vs pooled vs identity; Holm across 6 contrasts)

| Action | Conditioned | Pooled | Identity | Pooled−Conditioned [95% CI] (Holm p) | Identity−Conditioned [95% CI] (Holm p) |
|---|---|---|---|---|---|
| CB_BEST_DESCENT | 0.0257553 | 0.0262423 | 0.0576140 | +0.0004870 [+0.0002545, +0.0007266] (0.0011) | +0.0318587 [+0.0300347, +0.0335790] (1.1e-8) |
| CB_FIRST_DESCENT | 0.0199787 | 0.0218447 | 0.0438333 | +0.0018660 [+0.0013381, +0.0024040] (6.1e-7) | +0.0238546 [+0.0227251, +0.0248814] (1.1e-8) |
| CB_TABU | 0.0232733 | 0.0241589 | 0.0615146 | +0.0008857 [+0.0003497, +0.0014304] (0.0076) | +0.0382414 [+0.0366851, +0.0396631] (1.1e-8) |

Action-conditioned affine maps remain strictly better than pooled and identity at standard scale.

## 7. Per-size stratification (test; full table in `table_ext_per_size_descriptives.csv`)

| Size | n inst / states | All-zero % | Raw best (modal) | cng best (modal) | Mean decodes BD/FD/TABU | D−B cng GBDT |
|---|---|---|---|---|---|---|
| 15x10 | 3 / 120 | 2.50 | TABU | FIRST_DESCENT | 90.6 / 28.4 / 115.3 | +0.020640 |
| 15x15 | 9 / 360 | 0.56 | TABU | FIRST_DESCENT | 101.3 / 26.0 / 122.3 | +0.004620 |
| 20x10 | 3 / 120 | 0.00 | TABU | FIRST_DESCENT | 148.6 / 32.1 / 167.2 | +0.010593 |
| 20x15 | 6 / 240 | 0.00 | TABU | FIRST_DESCENT | 159.2 / 28.7 / 173.7 | 0.000000 |
| 20x20 | 6 / 240 | 0.00 | TABU | FIRST_DESCENT | 162.9 / 23.6 / 172.9 | 0.000000 |
| 30x10 | 3 / 120 | 0.00 | TABU | FIRST_DESCENT | 242.8 / 38.3 / 265.1 | +0.003089 |

The GBDT cng D−B gain concentrates in the Lawrence blocks (15x10, 20x10, 15x15, 30x10); it is exactly 0 in
ta 20x15 / 20x20 where D and B make identical choices.

## 8. Selected hyperparameters (dev LOOCV only; full grid in `table_ext_dev_loocv.csv`)

- LOOCV ridge L2 (raw): A=1, B=10, C=100, D=100, E=1000, D_random=10; (cng): A=1, B=10, C=10, D=10, E=1000, D_random=10.
- RFF L2 (raw): A=100, B/C/D/E/D_random=1000; (cng): A/B/C/D_random=100, D/E=1000. Bandwidths γ (median heuristic, dev): A 0.0168923, B 0.00578613, C 0.00368015, D 0.00465772, E 0.0023861, D_random 0.00442797.
- GBDT (depth, lr, iters, L2), raw: A(3, .1, 50, 10), B(3, .1, 100, 1), C(2, .1, 100, 1), D(2, .1, 200, 1), E(2, .1, 200, 1), D_random(3, .1, 100, 10); cng: A(2, .03, 200, 1), B(3, .1, 200, 10), C(3, .1, 200, 10), D(3, .1, 100, 10), E(3, .1, 50, 10), D_random(3, .1, 100, 10). `min_samples_leaf=20`, `early_stopping=False` everywhere; identical action-interacted architecture across all six arms.

## 9. Verification gates (all PASS)

1. Benchmark acquisition + mirror fidelity (5/5 exact fingerprint matches).
2. `manifest_lock.json` frozen before any test collection/test-outcome access.
3. Dev collection complete grid (200 traj / 6,000 branches).
4. Dev schedule validity: 5,552 distinct schedules re-decoded by synchronous time relaxation; hashes, rewards, costs, seeds verified.
5. Numerical gates: ridge gradient ∞-norm ≤ 1e-7 and augmented-LSTSQ agreement ≤ 1e-7 (all arms/targets, frozen + tuned); zero NaN/Inf in all GBDT/RFF predictions; exact double-fit reproducibility (RFF projection + weights, GBDT predictions).
6. Test collection complete grid (600 traj / 18,000 branches; 18,000 unique seeds).
7. Test schedule validity: 16,381 distinct schedules independently re-decoded; all endpoint best/current orders valid.
8. Frozen test evaluation (no refitting; dev-frozen bundle only).
9. Independent scalar replay (`verify_ext_replay.py`, numpy+stdlib only): 72 paired contrasts, exact Wilcoxon/Holm, 20,000-draw bootstrap, split-repeat both regimes/directions, per-size tables, action counts — max error 2.9e-16.
10. 300 DPI PNG+PDF figures.
11. Zero modification of frozen assets: 35,560 files, 0 changed / 0 removed / 0 added.

## 10. Deliverable inventory (`experiments/ext_standard_jssp/outputs/`)

`manifest_lock.json`, `verification_report.json`, `verification_schedules_{development,test}.json`,
`verification_replay.json`, `frozen_snapshot_{before,after}.json`, `code_amendment_00{1,2,3}.json`,
`table_ext_transition_errors.csv`, `table_ext_raw_gain_scorers.csv`, `table_ext_cost_normalized_scorers.csv`,
`table_ext_per_size_descriptives.csv`, `table_ext_split_repeat_diagnostics.csv`, `table_ext_paired_contrasts.csv`,
`table_ext_dev_loocv.csv`, `ext_analysis.json`, `ext_predictions.csv`, `ext_state_decisions.csv`,
`ext_instance_metrics.csv`, `ext_transition_errors.csv`, `fig_ext_summary.png/pdf`, this `SUMMARY.md`.
Raw archives: `raw/{development,test}/*.json.gz` (800 write-once hashed files); dev-frozen models: `models/ext_models.json` + 12 GBDT joblib files.

## 11. Bottom line for Section 7.3 / 8.3 / 9

On 30 held-out standard medium-to-large benchmarks with a negligible all-zero floor (0.42%), the prospective
extension **confirms the primary action-collapse conclusion at scale**: (i) no representation contrast D−B/D−E
is significant under raw reward for any of the four scorers; (ii) **no learned arm — including the new shallow
GBDT — beats the best fixed static action in either regime** (raw: best learned = fixed Tabu exactly or significantly
worse; cng: best learned within 0.00016 of fixed First Descent, CI spans 0); (iii) the split-repeat diagnostic
shows the raw-regime sample winner is worse than static Tabu on fresh repeats, while the cng-regime sample winner
beats static First Descent (+0.0091/+0.0094, p ≤ 2.1e-7) — real per-state cost heterogeneity exists but is not
captured by any tested scorer; (iv) the one new positive signal is **GBDT cng D−B (+0.004818, Holm p = 0.0029,
LOIO-stable)**, i.e. displacement-norm features help a nonlinear tabular scorer under cost normalization without
translating into beating the static optimum; (v) action-conditioned transition maps remain significantly better
than pooled/identity at standard scale. Reviewer concerns 1–3 are addressed with a locked, fully replayed,
zero-leakage protocol.

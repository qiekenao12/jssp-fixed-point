#!/usr/bin/env python3
"""Reviewer-response supplementary analyses for FP decision study manuscript.

Computes four diagnostics from frozen branch records WITHOUT altering any
frozen model, threshold, seed, primary result, raw data, or manuscript copy:

A1—Cost-normalised gain (R/decodes): fixed-action returns and per-arm selected
    returns under CNG, demonstrating action-collapse dynamics.
A2—Tuned ridge (leave-one-instance-out L2 grid on development set) and
    nonlinear (Random Forest) scorer sensitivity.
A3—Narrative only (two-stage funnel); see manuscript revision.
A4—Per-scale tie rate, zero-gain state proportion, and operator-wise gain
    differences stratified by instance size.

All input files are read-only.  Output written to
projects/analysis/revision_response_artifacts/ .
"""
from __future__ import annotations

import argparse, csv, gzip, hashlib, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.fp_decision_models import (
    ACTIONS, ARMS, vector, matrix, fit_ridge, fit_maps, predictions, affine,
)

OUT = ROOT / 'projects' / 'analysis' / 'revision_response_artifacts'
OUT.mkdir(parents=True, exist_ok=True)

L2_GRID = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
INSTANCE_SIZES = {'6x6': 0, '10x5': 1, '10x10': 2, '15x10': 3}


# ── helpers ───────────────────────────────────────────────────────────
def load_rows(split='final', solver='tabu'):
    """Load all branch records from gzip files."""
    results = ROOT / 'results' / 'fp_decision_study'
    files = sorted((results / solver / split).glob('*.json.gz'))
    return [r for p in files for r in json.load(gzip.open(p))['records']]


def instance_size(name):
    """Return size key e.g. '6x6' from instance name."""
    for sz in INSTANCE_SIZES:
        if sz in name:
            return sz
    return 'unknown'


# ── A1: Cost-Normalized Gain (R / decode_evaluations) ─────────────────
def compute_cng_analysis():
    print("=== A1: Cost-Normalized Gain ===")
    rows = load_rows('final')
    # Per-state per-action per-repeat: cng = gain / decode_evaluations
    # Aggregate per-state per-action: mean_cng over 5 repeats
    state_action_cng = {}
    for r in rows:
        key = (r['state_id'], r['action'])
        cng = r['gain'] / max(r['outcome']['decode_evaluations'], 1)
        state_action_cng.setdefault(key, []).append(cng)

    # Fixed-action mean CNG
    fixed_cng = {a: np.mean([np.mean(v) for k, v in state_action_cng.items() if k[1] == a])
                 for a in ACTIONS}
    print(f"  Fixed CNG: {json.dumps({a: round(v, 8) for a, v in fixed_cng.items()})}")

    # Per-state fixed-action CNG means (needed for selection)
    state_fixed = {}
    for (sid, a), vals in state_action_cng.items():
        state_fixed.setdefault(sid, {})[a] = np.mean(vals)

    # Load frozen action choices from tabu_decisions.csv
    arm_choices = defaultdict(dict)  # sid -> {arm: chosen_action}
    with open(ROOT / 'projects/analysis/fp_decision_study/tabu_decisions.csv') as f:
        for row in csv.DictReader(f):
            if row['instance'].startswith('new_final'):
                sid = row['state_id']
                for arm in ['A', 'B', 'C', 'D', 'E', 'D_random']:
                    act = row.get(f'gain_{arm}_action', '')
                    if act:
                        arm_choices.setdefault(sid, {})[arm] = act

    # Per-arm CNG selected return (using frozen choices evaluated under CNG)
    arm_cng = {arm: [] for arm in ARMS}
    for sid, choices in arm_choices.items():
        if sid not in state_fixed:
            continue
        for arm in ARMS:
            act = choices.get(arm)
            if act and act in state_fixed[sid]:
                arm_cng[arm].append(state_fixed[sid][act])

    arm_means = {a: (np.mean(v), np.std(v) / np.sqrt(max(len(v), 1)))
                 for a, v in arm_cng.items()}
    print(f"  Arm CNG means: {json.dumps({a: round(m, 8) for a, (m, _) in arm_means.items()})}")

    # Per-action CNG stats across all states
    per_action_stats = {}
    for a in ACTIONS:
        vals = [np.mean(v) for k, v in state_action_cng.items() if k[1] == a]
        per_action_stats[a] = {'mean': float(np.mean(vals)),
                               'median': float(np.median(vals)),
                               'n_states': len(vals)}

    result = {
        'fixed_cng': {a: round(v, 8) for a, v in fixed_cng.items()},
        'arm_cng_means': {a: {'mean': round(m, 8), 'se': round(se, 8)}
                          for a, (m, se) in arm_means.items()},
        'per_action_cng_stats': per_action_stats,
    }

    # Recompute raw-gain fixed action means
    state_action_raw = {}
    for r in rows:
        key = (r['state_id'], r['action'])
        state_action_raw.setdefault(key, []).append(r['gain'])
    raw_fixed = {a: float(np.mean([np.mean(v) for k, v in state_action_raw.items() if k[1] == a]))
                 for a in ACTIONS}
    result['raw_gain_fixed'] = {a: round(v, 8) for a, v in raw_fixed.items()}
    result['action_collapse_ratio'] = {
        'raw': {a: round(raw_fixed[a] / max(raw_fixed.get('CB_TABU', 1e-9), 1e-9), 6)
                for a in ACTIONS},
        'cng': {a: round(fixed_cng[a] / max(fixed_cng.get('CB_BEST_DESCENT', 1e-9), 1e-9), 6)
                for a in ACTIONS},
    }
    print(json.dumps({a: round(v, 8) for a, v in raw_fixed.items()}))

    # Per-state count of which action is CNG-best vs raw-gain-best
    best_raw = Counter(); best_cng = Counter(); both_best = Counter()
    for sid in state_fixed:
        acts = list(state_fixed[sid].keys())
        if len(acts) < 3: continue
        raw_max = max(acts, key=lambda a: state_fixed[sid].get(a, -1e9))
        # need raw per-state... recompute
        raw_vals = {a: np.mean([r['gain'] for r in rows
                                if r['state_id'] == sid and r['action'] == a])
                    for a in acts}
        raw_best = max(raw_vals, key=raw_vals.get)
        cng_best = max(state_fixed[sid], key=state_fixed[sid].get)
        best_raw[raw_best] += 1
        best_cng[cng_best] += 1
        if raw_best == cng_best:
            both_best[raw_best] += 1
    result['dominance_shift'] = {
        'raw_best_counts': dict(best_raw.most_common()),
        'cng_best_counts': dict(best_cng.most_common()),
        'agreement_count': dict(both_best.most_common()),
    }
    print(f"  Raw best: {best_raw.most_common()}")
    print(f"  CNG best: {best_cng.most_common()}")

    with open(OUT / 'a1_cost_normalized_gain.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("  -> a1_cost_normalized_gain.json\n")


# ── A2: Tuned Ridge + Nonlinear Scorer ────────────────────────────────
def fit_ridge_l2(x, y, l2):
    """Like fit_ridge but configurable L2 penalty."""
    center = x.mean(0)
    scale = np.maximum(x.std(0), 1e-8)
    z = np.column_stack((np.ones(len(x)), (x - center) / scale))
    p = np.eye(z.shape[1])
    p[0, 0] = 0
    p[1:, 1:] *= l2
    w = np.linalg.solve(z.T @ z + p, z.T @ y)
    return {'center': center.tolist(), 'scale': scale.tolist(), 'weights': w.tolist(),
            'l2': l2, 'n': len(x)}


def predict_ridge(model, x):
    m = np.asarray(model['weights'])
    return m[0] + ((x - model['center']) / model['scale']) @ m[1:]


def loocv_l2(dev_rows, maps_all, arms_list=ARMS):
    """Leave-one-instance-out CV for L2 grid using prediction MSE.

    MSE is the standard selection criterion for a regression scorer and does
    not itself depend on which action the scorer would select.  Selected-return
    diagnostics are reported separately (they are dominated by the tabu
    constant-action regime and cannot discriminate hyperparameters).
    """
    instances = sorted({r['instance'] for r in dev_rows})
    results = {}
    for arm in arms_list:
        cv_mse = {l2: [] for l2 in L2_GRID}
        for held_out in instances:
            train = [r for r in dev_rows if r['instance'] != held_out]
            test = [r for r in dev_rows if r['instance'] == held_out]
            train_maps = fit_maps(train)
            x_train = matrix(train, arm, train_maps)
            y_train = np.asarray([r['gain'] for r in train])
            # Test-time features use the full-development maps, as in the
            # frozen protocol (held-out instances never contribute maps).
            x_test = matrix(test, arm, maps_all)
            y_test = np.asarray([r['gain'] for r in test])
            for l2 in L2_GRID:
                model = fit_ridge_l2(x_train, y_train, l2)
                pred = predict_ridge(model, x_test)
                cv_mse[l2].append(float(np.mean((pred - y_test) ** 2)))
        mean_mse = {l2: float(np.mean(v)) for l2, v in cv_mse.items()}
        best_l2 = min(L2_GRID, key=lambda l2: mean_mse[l2])
        results[arm] = {'cv_mse': mean_mse, 'best_l2': best_l2,
                        'best_cv_mse': mean_mse[best_l2]}
        print(f"  LOOCV {arm}: best L2={best_l2}, CV MSE={mean_mse[best_l2]:.8f}")
    return results


def rff_features(x, rng, n_components=256, gamma=None):
    """Random Fourier features approximating an RBF kernel (deterministic seed).

    Returns the feature matrix and the (w, b) projection so that the identical
    projection can be applied to held-out/final data.
    """
    x = np.asarray(x, float)
    d = x.shape[1]
    if gamma is None:
        sub = x[rng.choice(len(x), size=min(len(x), 500), replace=False)]
        d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=2)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / max(med, 1e-9)
    w = rng.normal(scale=np.sqrt(2 * gamma), size=(d, n_components))
    b = rng.uniform(0, 2 * np.pi, size=n_components)
    return np.sqrt(2.0 / n_components) * np.cos(x @ w + b), gamma, (w, b)


def transform_rff(x, gamma, wb):
    w, b = wb
    return np.sqrt(2.0 / w.shape[1]) * np.cos(np.asarray(x, float) @ w + b)


def nonlinear_refit(dev_rows, maps_all, final_rows, arms_list=('A', 'B', 'C', 'D', 'E')):
    """Nonlinear scorer control: RBF random-Fourier-feature ridge (deterministic).

    The same ridge machinery and action-interaction design matrix are used;
    only the feature map is replaced by a 256-component RBF random projection.
    Hyperparameters (L2) are chosen by leave-one-instance-out CV on development
    instances only, never on final outcomes.
    """
    results = {}
    instances = sorted({r['instance'] for r in dev_rows})
    for arm in arms_list:
        x_dev_full = matrix(dev_rows, arm, maps_all)
        y_dev = np.asarray([r['gain'] for r in dev_rows])
        # standardize the design matrix first (RFF on standardized inputs)
        center = x_dev_full.mean(0)
        scale = np.maximum(x_dev_full.std(0), 1e-8)
        z_dev = (x_dev_full - center) / scale
        rng = np.random.default_rng(20261112)
        f_dev, gamma, wb = rff_features(z_dev, rng)
        # LOOCV over instances for L2
        inst_idx = np.asarray([r['instance'] for r in dev_rows])
        best = (None, -np.inf, None)
        for l2 in L2_GRID:
            fold_mse = []
            for held in instances:
                tr = inst_idx != held
                te = ~tr
                m = fit_ridge_l2(f_dev[tr], y_dev[tr], l2)
                pred = predict_ridge(m, f_dev[te])
                fold_mse.append(float(np.mean((pred - y_dev[te]) ** 2)))
            mse = float(np.mean(fold_mse))
            if best[0] is None or mse < best[1]:
                best = (l2, mse, None)
        best_l2 = best[0]
        # refit on full dev, evaluate on final
        model = fit_ridge_l2(f_dev, y_dev, best_l2)
        x_fin = matrix(final_rows, arm, maps_all)
        z_fin = (x_fin - center) / scale
        f_fin = transform_rff(z_fin, gamma, wb)
        preds = predict_ridge(model, f_fin)
        y_fin = np.asarray([r['gain'] for r in final_rows])
        sid_data = defaultdict(list)
        for i, r in enumerate(final_rows):
            sid_data[r['state_id']].append((r['action'], float(preds[i]), float(y_fin[i])))
        selected = []
        action_counts = Counter()
        for sid, triplets in sid_data.items():
            best_act = max(triplets, key=lambda t: t[1])[0]
            action_counts[best_act] += 1
            selected.append(np.mean([t[2] for t in triplets if t[0] == best_act]))
        results[arm] = {
            'model': 'rff_ridge_256',
            'best_l2': best_l2,
            'cv_mse': float(best[1]),
            'mean_selected_gain': float(np.mean(selected)),
            'n_states': len(selected),
            'action_distribution': dict(action_counts.most_common()),
        }
        print(f"  RFF-ridge {arm} L2={best_l2}: selected_gain={results[arm]['mean_selected_gain']:.8f} "
              f"actions={results[arm]['action_distribution']}")

    # Frozen L2=1 ridge reference under the identical evaluation protocol
    from scripts.fp_decision_models import predict
    frozen = json.load(open(ROOT / 'projects/experiments/fp_decision_study/tabu_models.json'))
    for arm in arms_list:
        model = frozen['models'][arm]['gain']
        x_final = matrix(final_rows, arm, frozen['maps'])
        y_final = np.asarray([r['gain'] for r in final_rows])
        preds = np.array([predict(model, xi) for xi in x_final])
        sid_data = defaultdict(list)
        for i, r in enumerate(final_rows):
            sid_data[r['state_id']].append((r['action'], float(preds[i]), float(y_final[i])))
        selected = []
        for sid, triplets in sid_data.items():
            best_act = max(triplets, key=lambda t: t[1])[0]
            selected.append(np.mean([t[2] for t in triplets if t[0] == best_act]))
        results[f'{arm}_ridge_L2_1'] = {
            'mean_selected_gain': float(np.mean(selected)),
            'n_states': len(selected)}
    return results


def compute_tuned_ridge():
    print("=== A2: Tuned Ridge + Nonlinear ===")
    dev_rows = load_rows('development', 'tabu')
    final_rows = load_rows('final', 'tabu')
    maps_all = fit_maps(dev_rows)  # 8 dev instances for maps, consistent with manuscript

    # LOOCV on dev
    cv_results = loocv_l2(dev_rows, maps_all)
    print(f"  CV results: {json.dumps({a: v['best_l2'] for a, v in cv_results.items()})}")

    # Refit with best L2 and evaluate on final
    ridge_final = {}
    for arm in ARMS:
        best_l2 = cv_results[arm]['best_l2']
        full_maps = maps_all
        x_dev_full = matrix(dev_rows, arm, full_maps)
        y_dev = np.asarray([r['gain'] for r in dev_rows])
        model = fit_ridge_l2(x_dev_full, y_dev, best_l2)
        x_final = matrix(final_rows, arm, full_maps)
        y_final = np.asarray([r['gain'] for r in final_rows])
        preds = predict_ridge(model, x_final)
        sid_data = defaultdict(list)
        for i, r in enumerate(final_rows):
            sid_data[r['state_id']].append((r['action'], float(preds[i]), float(y_final[i])))
        selected_gains = []
        for sid, triplets in sid_data.items():
            best_act = max(triplets, key=lambda t: t[1])[0]
            best_gains = [t[2] for t in triplets if t[0] == best_act]
            selected_gains.append(np.mean(best_gains))
        ridge_final[arm] = {
            'best_l2': int(best_l2) if best_l2 >= 1 else best_l2,
            'mean_selected_gain': float(np.mean(selected_gains)),
            'n_states': len(selected_gains),
        }
        print(f"  Tuned ridge {arm} L2={best_l2}: selected_gain={ridge_final[arm]['mean_selected_gain']:.8f}")

    # Also report the frozen L2=1 value for reference (Table 5 values)
    frozen_values = json.load(open(ROOT / 'projects/analysis/fp_decision_study/tabu_final_analysis.json'))
    frozen_macro = frozen_values['macro']

    # Nonlinear
    nonlinear = nonlinear_refit(dev_rows, maps_all, final_rows)

    result = {
        'loocv': cv_results,
        'tuned_ridge_final': ridge_final,
        'nonlinear_rf': nonlinear,
        'frozen_reference': {arm: frozen_macro.get(f'gain_{arm}_value', None) for arm in ARMS},
        'frozen_reference_event': {arm: frozen_macro.get(f'probability_{arm}_value', None) for arm in ARMS},
    }
    with open(OUT / 'a2_tuned_ridge_nonlinear.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("  -> a2_tuned_ridge_nonlinear.json\n")


def compute_cng_refit():
    """A1b: refit A-E scorers directly targeting per-request gain.

    Post-collection secondary diagnostic, mirroring the Section 6.4 protocol.
    The regression target is R_tilde = gain / decode_evaluations scaled by 1e3
    (a constant rescaling that leaves ridge, LOOCV selection and action choice
    unchanged).  Fixed L2=1 and development-LOOCV penalties are both reported.
    """
    print("=== A1b: Cost-normalised target refit ===")
    dev_rows = load_rows('development', 'tabu')
    final_rows = load_rows('final', 'tabu')
    maps_all = fit_maps(dev_rows)

    def cng_target(rows):
        return np.asarray([r['gain'] / max(r['outcome']['decode_evaluations'], 1) * 1e3
                           for r in rows])

    # raw-gain target for reference (identical to the 6.4 protocol)
    y_dev_raw = np.asarray([r['gain'] for r in dev_rows])
    y_fin_raw = np.asarray([r['gain'] for r in final_rows])
    y_dev_cng = cng_target(dev_rows)
    y_fin_cng = cng_target(final_rows)

    instances = sorted({r['instance'] for r in dev_rows})
    inst_idx = np.asarray([r['instance'] for r in dev_rows])
    results = {}

    for arm in ARMS:
        # ---- LOOCV on the cost-normalised target ----
        best_l2, best_mse = None, float('inf')
        cv_store = {}
        for l2 in L2_GRID:
            fold = []
            for held in instances:
                tr = inst_idx != held
                te = ~tr
                tr_maps = fit_maps([r for r in dev_rows if r['instance'] != held])
                x_tr = matrix([r for r in dev_rows if r['instance'] != held], arm, tr_maps)
                x_te = matrix([r for r in dev_rows if r['instance'] == held], arm, maps_all)
                m = fit_ridge_l2(x_tr, y_dev_cng[tr], l2)
                pred = predict_ridge(m, x_te)
                fold.append(float(np.mean((pred - y_dev_cng[te]) ** 2)))
            mse = float(np.mean(fold))
            cv_store[l2] = mse
            if mse < best_mse:
                best_l2, best_mse = l2, mse

        x_dev = matrix(dev_rows, arm, maps_all)
        x_fin = matrix(final_rows, arm, maps_all)

        # ---- three fits: frozen L2=1 / tuned L2, each evaluated directly on
        #      the per-request target of the selected action ----
        variants = {}
        for tag, l2 in (('fixed_l2_1', 1.0), ('tuned', best_l2)):
            model = fit_ridge_l2(x_dev, y_dev_cng, l2)
            preds = predict_ridge(model, x_fin)
            sid = defaultdict(list)
            for i, r in enumerate(final_rows):
                sid[r['state_id']].append((r['action'], float(preds[i]), float(y_fin_cng[i]),
                                           float(y_fin_raw[i])))
            sel_cng, sel_raw = [], []
            counts = Counter()
            for s, trips in sid.items():
                best_act = max(trips, key=lambda t: t[1])[0]
                counts[best_act] += 1
                sel_cng.append(np.mean([t[2] for t in trips if t[0] == best_act]))
                sel_raw.append(np.mean([t[3] for t in trips if t[0] == best_act]))
            variants[tag] = {
                'l2': l2,
                'selected_cng_x1e3': float(np.mean(sel_cng)),
                'selected_raw_gain': float(np.mean(sel_raw)),
                'tabu_choices': counts.get('CB_TABU', 0),
                'first_choices': counts.get('CB_FIRST_DESCENT', 0),
                'best_choices': counts.get('CB_BEST_DESCENT', 0),
                'action_distribution': dict(counts.most_common()),
            }

        # ---- raw-gain trained reference under the same evaluation ----
        model_raw = fit_ridge_l2(x_dev, y_dev_raw, 1.0)
        preds_raw = predict_ridge(model_raw, x_fin)
        sid_raw = defaultdict(list)
        for i, r in enumerate(final_rows):
            sid_raw[r['state_id']].append((r['action'], float(preds_raw[i]), float(y_fin_cng[i])))
        sel_raw_cng = []
        cnt_raw = Counter()
        for s, trips in sid_raw.items():
            best_act = max(trips, key=lambda t: t[1])[0]
            cnt_raw[best_act] += 1
            sel_raw_cng.append(np.mean([t[2] for t in trips if t[0] == best_act]))

        results[arm] = {
            'loocv_best_l2': best_l2,
            'loocv_mse_x1e3': best_mse,
            'variants': variants,
            'raw_trained_selected_cng_x1e3': float(np.mean(sel_raw_cng)),
            'raw_trained_tabu_choices': cnt_raw.get('CB_TABU', 0),
        }
        v = variants['tuned']
        print(f"  {arm}: LOOCV L2={best_l2}; tuned sel_cng={v['selected_cng_x1e3']:.4f} "
              f"(tabu={v['tabu_choices']}, first={v['first_choices']}); "
              f"fixedL2=1 sel_cng={variants['fixed_l2_1']['selected_cng_x1e3']:.4f}")

    # Reference: fixed single-action policies under the per-request target
    fixed = {}
    for a in ACTIONS:
        vals = [r['gain'] / max(r['outcome']['decode_evaluations'], 1) * 1e3
                for r in final_rows if r['action'] == a]
        fixed[a] = float(np.mean(vals))

    out = {'target': 'gain/decode_evaluations * 1e3', 'fixed_action_reference': fixed,
           'arms': results,
           'note': 'Post-collection secondary diagnostic. Development-only fitting; '
                   'final outcomes used for evaluation only, never for fitting or selection.'}
    with open(OUT / 'a1b_cost_normalised_refit.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  fixed reference (x1e3): { {k: round(v,4) for k,v in fixed.items()} }")
    print("  -> a1b_cost_normalised_refit.json\n")


def compute_tuned_choice_counts():
    """A2b: tabu-choice counts under tuned ridge (explains the exact tie)."""
    print("=== A2b: Tuned-ridge action agreement ===")
    dev_rows = load_rows('development', 'tabu')
    final_rows = load_rows('final', 'tabu')
    maps_all = fit_maps(dev_rows)
    y_dev = np.asarray([r['gain'] for r in dev_rows])
    loocv = json.load(open(OUT / 'a2_tuned_ridge_nonlinear.json'))['loocv']

    per_arm = {}
    choice_maps = {}
    for arm in ARMS:
        l2 = loocv[arm]['best_l2']
        x_dev = matrix(dev_rows, arm, maps_all)
        model = fit_ridge_l2(x_dev, y_dev, l2)
        x_fin = matrix(final_rows, arm, maps_all)
        preds = predict_ridge(model, x_fin)
        sid = defaultdict(list)
        for i, r in enumerate(final_rows):
            sid[r['state_id']].append((r['action'], float(preds[i])))
        chosen = {}
        cnt = Counter()
        for s, trips in sid.items():
            best_act = max(trips, key=lambda t: t[1])[0]
            chosen[s] = best_act
            cnt[best_act] += 1
        per_arm[arm] = {'l2': l2, 'tabu_choices': cnt.get('CB_TABU', 0),
                        'first_choices': cnt.get('CB_FIRST_DESCENT', 0),
                        'best_choices': cnt.get('CB_BEST_DESCENT', 0),
                        'distribution': dict(cnt.most_common())}
        choice_maps[arm] = chosen
        print(f"  {arm} (L2={l2}): {per_arm[arm]['distribution']}")

    # pairwise agreement on the 800 states
    arms = list(choice_maps)
    agree = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            same = sum(1 for s in choice_maps[a] if choice_maps[a][s] == choice_maps[b][s])
            agree[f'{a}_vs_{b}'] = {'identical': same, 'total': len(choice_maps[a])}

    with open(OUT / 'a2b_tuned_choice_counts.json', 'w') as f:
        json.dump({'per_arm': per_arm, 'pairwise_agreement': agree}, f, indent=2)
    print("  -> a2b_tuned_choice_counts.json\n")


# ── A4: Per-scale tie rate, zero-gain state proportion ─────────────────
def compute_per_scale():
    print("=== A4: Per-scale statistics ===")
    rows = load_rows('final')
    sizes = sorted(INSTANCE_SIZES)

    per_size = {}
    for sz in sizes:
        sz_rows = [r for r in rows if instance_size(r['instance']) == sz]
        # Per-action stats
        state_ids = set(r['state_id'] for r in sz_rows)
        n_states = len(state_ids) // len(ACTIONS)  # states (unique by state_id minus action)
        # Count per state_id
        sid_groups = defaultdict(list)
        for r in sz_rows:
            sid_groups[r['state_id']].append(r)

        # All actions zero: all 3 actions have mean gain ~0
        all_zero = 0
        gain_ranges = []
        for sid in sorted(set(r['state_id'] for r in sz_rows)):
            acts = defaultdict(list)
            for r in sz_rows:
                if r['state_id'] == sid:
                    acts[r['action']].append(r['gain'])
            means = {a: np.mean(v) for a, v in acts.items() if v}
            if len(means) == 3:
                max_gain = max(means.values())
                if max_gain < 1e-10:
                    all_zero += 1
                gain_ranges.append(max_gain - min(means.values()))

        n_sid = len(set(r['state_id'] for r in sz_rows))
        all_zero_rate = all_zero / max(n_sid, 1) if n_sid else 0

        # Per-action mean/median gain
        act_stats = {}
        for a in ACTIONS:
            vals = [r['gain'] for r in sz_rows if r['action'] == a]
            act_stats[a] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'mean_decodes': float(np.mean([r['outcome']['decode_evaluations']
                                               for r in sz_rows if r['action'] == a])),
                'mean_cng': float(np.mean([r['gain'] / max(r['outcome']['decode_evaluations'], 1)
                                            for r in sz_rows if r['action'] == a])),
            }

        # How many states does each action dominate (is best) under raw vs CNG?
        raw_wins = Counter(); cng_wins = Counter()
        for sid in set(r['state_id'] for r in sz_rows):
            acts = defaultdict(list)
            for r in sz_rows:
                if r['state_id'] == sid:
                    acts[r['action']].append((r['gain'], r['gain'] / max(r['outcome']['decode_evaluations'], 1)))
            if len(acts) < 3: continue
            raw_best = max(acts, key=lambda a: np.mean([t[0] for t in acts[a]]))
            cng_best = max(acts, key=lambda a: np.mean([t[1] for t in acts[a]]))
            raw_wins[raw_best] += 1
            cng_wins[cng_best] += 1

        per_size[sz] = {
            'n_states': n_sid,
            'n_branches': len(sz_rows),
            'all_actions_zero_rate': round(all_zero_rate, 4),
            'mean_gain_range': float(np.mean(gain_ranges)) if gain_ranges else 0,
            'per_action_stats': act_stats,
            'raw_best_action_counts': dict(raw_wins.most_common()),
            'cng_best_action_counts': dict(cng_wins.most_common()),
        }
        print(f"  {sz}: states={n_sid}, zero_rate={all_zero_rate:.3f}, "
              f"raw_wins={dict(raw_wins.most_common())}, cng_wins={dict(cng_wins.most_common())}")

    # D-B per-size (from manuscript S3)
    d_b_by_size = {}
    for sz in sizes:
        sz_insts = [r['instance'] for r in rows if instance_size(r['instance']) == sz]
        d_b_by_size[sz] = {inst: None for inst in sorted(set(sz_insts))}

    with open(OUT / 'a4_per_scale_statistics.json', 'w') as f:
        json.dump(per_size, f, indent=2)
    print("  -> a4_per_scale_statistics.json\n")


# ── Genetic 480-window C/E comparison ──────────────────────────────────
def compute_genetic_480():
    print("=== Genetic 480-window C/E ===")
    with open(ROOT / 'projects/analysis/fp_decision_study/history_final_state_decisions.csv') as f:
        data = [r for r in csv.DictReader(f)
                if r['cohort'] == 'exposed_confirmation']
    # Compare atomic vs fair480
    atomic = [r for r in data if r['window'] == 'atomic']
    fair480 = [r for r in data if r['window'] == 'fair480']
    for label, subset in [('atomic', atomic), ('fair480', fair480)]:
        c_acts = Counter(r['C_action'] for r in subset)
        e_acts = Counter(r['E_action'] for r in subset)
        c_gains = [float(r['C_gain']) for r in subset]
        e_gains = [float(r['E_gain']) for r in subset]
        # Per-instance E-C
        inst_diff = defaultdict(list)
        for r in subset:
            inst_diff[r['instance']].append(float(r['E_gain']) - float(r['C_gain']))
        per_inst = {i: {'mean': float(np.mean(v)), 'n_states': len(v)}
                    for i, v in inst_diff.items()}
        print(f"  {label}: C={dict(c_acts.most_common())}, E={dict(e_acts.most_common())}, "
              f"E-C mean={np.mean(e_gains)-np.mean(c_gains):.8f}")
        # LS-VND count
        ls_count = Counter(r['C_action'] + '-' + r['E_action'] for r in subset)
        print(f"    C/E action pairs: {dict(ls_count.most_common(5))}")

    # New final genetic data from raw gzips
    gen_rows = load_rows('final', 'original')
    # Check if 480-window data present
    win_counts = Counter()
    for r in gen_rows:
        if 'window' in r.get('outcome', {}):
            win_counts[r['outcome']['window']] += 1
        # Also check from records structure
    print(f"  Genetic raw rows: {len(gen_rows)}")
    print(f"  Window counts in outcome: none (records have no window field)")

    # For new-final genetic, the window info is NOT in the record structure.
    # The manuscript reports Figure 6 aggregate LS-VND reversal.
    # We report the historical 480-window E-C as supplementary.

    with open(OUT / 'genetic_480_window_ce.json', 'w') as f:
        json.dump({
            'historical_exposed': {
                'atomic': {
                    'C_actions': {k: v for k, v in Counter(r['C_action'] for r in atomic).most_common()},
                    'E_actions': {k: v for k, v in Counter(r['E_action'] for r in atomic).most_common()},
                    'E_minus_C_mean': float(np.mean([float(r['E_gain']) - float(r['C_gain']) for r in atomic])),
                },
                'fair480': {
                    'C_actions': {k: v for k, v in Counter(r['C_action'] for r in fair480).most_common()},
                    'E_actions': {k: v for k, v in Counter(r['E_action'] for r in fair480).most_common()},
                    'E_minus_C_mean': float(np.mean([float(r['E_gain']) - float(r['C_gain']) for r in fair480])),
                },
            },
            'note': 'new_final genetic 480-window per-arm decisions not available in frozen records; the aggregate LS-VND reversal is in figure_source_data.json. Historical exposed fair480 data shown as supplementary reference.'
        }, f, indent=2)
    print("  -> genetic_480_window_ce.json\n")


# ── main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a1', action='store_true', help='Cost-normalized gain only')
    ap.add_argument('--a2', action='store_true', help='Tuned ridge + nonlinear only')
    ap.add_argument('--a1b', action='store_true', help='Cost-normalised target refit only')
    ap.add_argument('--a2b', action='store_true', help='Tuned-ridge choice counts only')
    ap.add_argument('--a4', action='store_true', help='Per-scale stats only')
    ap.add_argument('--gen480', action='store_true', help='Genetic 480-window only')
    ap.add_argument('--all', action='store_true', help='Run all analyses')
    args = ap.parse_args()
    if not any([args.a1, args.a2, args.a1b, args.a2b, args.a4, args.gen480, args.all]):
        args.all = True
    print("OUTPUT DIR:", str(OUT))
    if args.all or args.a1: compute_cng_analysis()
    if args.all or args.a2: compute_tuned_ridge()
    if args.all or args.a1b: compute_cng_refit()
    if args.all or args.a2b: compute_tuned_choice_counts()
    if args.all or args.a4: compute_per_scale()
    if args.all or args.gen480: compute_genetic_480()
    print("DONE.")


if __name__ == '__main__':
    main()
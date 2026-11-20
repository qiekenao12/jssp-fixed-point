#!/usr/bin/env python3
"""Development-only fitting for ext_standard_jssp. NO test data may exist yet.

Fits on the 20 development instances only:
- displacement-ridge affine maps (full-dev + leave-one-instance-out), frozen code;
- per arm x target {raw, cng}: Frozen Ridge L2=1 (gradient + augmented-LSTSQ gates),
  LOOCV-tuned Ridge (LOIO maps for fold-train features, full-dev maps for fold-test),
  256-component RBF-RFF ridge (frozen construction/seed, dev median bandwidth,
  dev LOOCV L2), shallow HistGradientBoostingRegressor (24-config dev LOOCV grid,
  identical architecture across all six arms, deterministic single-threaded fits);
- regularized logistic improvement-event scorer (L2=1) per arm.
All hyperparameter selection uses leave-one-development-instance-out MSE only.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import (ROOT, EXT, RAW, OUT, MODELS, ARMS, SCORERS, TARGETS,
                        L2_GRID, GBDT_GRID, GBDT_FIXED, RFF_SEED, RFF_COMPONENTS,
                        canonical, cng_value, load, sha256_file, write_new)
from scripts.fp_decision_models import fit_maps, matrix, fit_ridge
from scripts.revision_response_analysis import (fit_ridge_l2, predict_ridge,
                                                rff_features, transform_rff)
from scripts.check_fp_scorer_convergence import ConvergedLogistic
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

_W = {}


def target_vector(rows, target):
    if target == 'raw':
        return np.asarray([r['gain'] for r in rows], float)
    return np.asarray([cng_value(r) for r in rows], float)


def ridge_checked(x, y, l2):
    """fit_ridge_l2 plus the frozen numerical gates at arbitrary L2."""
    model = fit_ridge_l2(x, y, l2)
    center = np.asarray(model['center']); scale = np.asarray(model['scale'])
    z = np.column_stack((np.ones(len(x)), (x - center) / scale))
    p = np.eye(z.shape[1]); p[0, 0] = 0; p[1:, 1:] *= l2
    w = np.asarray(model['weights'])
    gradient = z.T @ (z @ w - y) / len(x) + p @ w / len(x)
    ginf = float(np.max(np.abs(gradient)))
    if not ginf <= 1e-7:
        raise ArithmeticError(f'Ridge stationarity failed l2={l2}: {ginf}')
    augmented = np.r_[z, np.sqrt(p)]
    independent = np.linalg.lstsq(augmented, np.r_[y, np.zeros(len(w))], rcond=None)[0]
    perr = float(np.max(np.abs(z @ w - z @ independent)))
    if perr > 1e-7:
        raise ArithmeticError(f'Independent QR fit disagreement l2={l2}: {perr}')
    model['gradient_inf'] = ginf
    model['independent_prediction_max_error'] = perr
    return model


def _init(rows, maps_all, cross):
    _W['rows'] = rows
    _W['maps'] = maps_all
    _W['cross'] = cross


def _gbdt_fold(arg):
    arm, held = arg
    rows = _W['rows']
    train = [r for r in rows if r['instance'] != held]
    test = [r for r in rows if r['instance'] == held]
    xtr = matrix(train, arm, _W['cross'][held])
    xte = matrix(test, arm, _W['maps'])
    out = {}
    for t in TARGETS:
        ytr = target_vector(train, t); yte = target_vector(test, t)
        for cfg in GBDT_GRID:
            m = HistGradientBoostingRegressor(**GBDT_FIXED, **cfg).fit(xtr, ytr)
            pred = m.predict(xte)
            assert np.isfinite(pred).all()
            out[t + '|' + canonical(cfg)] = float(np.mean((pred - yte) ** 2))
    return arm, held, out


def main():
    assert not (RAW / 'test').exists(), 'Test data must not exist during dev-only fitting'
    lock = load(OUT / 'manifest_lock.json')
    lock_hash = lock['lock_hash']
    files = sorted((RAW / 'development').glob('*.json.gz'))
    rows = [r for p in files for r in load(p)['records']]
    instances = sorted({r['instance'] for r in rows})
    assert instances == sorted(lock['split_names']['development'])
    assert len(rows) == 20 * 10 * 30 == 6000
    # slim rows for worker pickling (fields needed by matrix/targets only)
    slim = [{k: r[k] for k in ('instance', 'state_id', 'action', 'repeat', 'start_x',
                               'observed', 'end_x', 'gain')}
            | {'outcome': {'decode_evaluations': r['outcome']['decode_evaluations']}}
            for r in rows]
    started = time.perf_counter()
    maps_all = fit_maps(rows)
    cross = {i: fit_maps([r for r in rows if r['instance'] != i]) for i in instances}
    inst_idx = np.asarray([r['instance'] for r in rows])
    y = {t: target_vector(rows, t) for t in TARGETS}
    binary = (y['raw'] > 0).astype(float)

    models = {a: {t: {} for t in TARGETS} for a in ARMS}
    logistic = {}
    receipts = {}
    loocv_rows = []
    gbdt_files = {}
    MODELS.mkdir(parents=True, exist_ok=True)

    # ---- GBDT LOOCV in parallel over (arm, fold); each fit single-threaded ----
    gbdt_mse = {a: {t: {} for t in TARGETS} for a in ARMS}
    with ProcessPoolExecutor(max_workers=16, initializer=_init,
                             initargs=(slim, maps_all, cross)) as pool:
        for arm, held, out in pool.map(_gbdt_fold, [(a, i) for a in ARMS for i in instances],
                                       chunksize=1):
            for key, mse in out.items():
                t, cfg_hash = key.split('|')
                gbdt_mse[arm][t].setdefault(cfg_hash, {})[held] = mse
    print('GBDT LOOCV folds complete', flush=True)

    for arm in ARMS:
        t0 = time.perf_counter()
        x_full = matrix(rows, arm, maps_all)
        fold_x = {}
        for held in instances:
            tr = inst_idx != held
            train_rows = [r for r, keep in zip(rows, tr) if keep]
            test_rows = [r for r, keep in zip(rows, tr) if not keep]
            fold_x[held] = (matrix(train_rows, arm, cross[held]),
                            matrix(test_rows, arm, maps_all), tr)
        receipts[arm] = {'n_features': int(x_full.shape[1])}
        for t in TARGETS:
            yt = y[t]
            # 1) Frozen ridge L2=1 (exact frozen code path with gates)
            frozen = fit_ridge(x_full, yt)
            frozen['l2'] = 1.0
            models[arm][t]['frozen_ridge'] = frozen
            # 2) LOOCV-tuned ridge
            cv = {}
            for l2 in L2_GRID:
                fold = []
                for held in instances:
                    xtr, xte, tr = fold_x[held]
                    m = fit_ridge_l2(xtr, yt[tr], l2)
                    pred = predict_ridge(m, xte)
                    fold.append(float(np.mean((pred - yt[~tr]) ** 2)))
                cv[l2] = float(np.mean(fold))
            best_l2 = min(L2_GRID, key=lambda l2: cv[l2])
            tuned = ridge_checked(x_full, yt, best_l2)
            tuned['cv_mse'] = cv[best_l2]
            models[arm][t]['loocv_ridge'] = tuned
            for l2 in L2_GRID:
                loocv_rows.append({'arm': arm, 'target': t, 'scorer': 'loocv_ridge',
                                   'config': json.dumps({'l2': l2}), 'cv_mse': cv[l2],
                                   'selected': int(l2 == best_l2)})
            # 3) RBF-RFF ridge (frozen construction; full-dev standardization/projection)
            center = x_full.mean(0); scale = np.maximum(x_full.std(0), 1e-8)
            z_dev = (x_full - center) / scale
            rng = np.random.default_rng(RFF_SEED)
            f_dev, gamma, wb = rff_features(z_dev, rng, n_components=RFF_COMPONENTS)
            cv_rff = {}
            for l2 in L2_GRID:
                fold = []
                for held in instances:
                    tr = inst_idx != held
                    m = fit_ridge_l2(f_dev[tr], yt[tr], l2)
                    pred = predict_ridge(m, f_dev[~tr])
                    fold.append(float(np.mean((pred - yt[~tr]) ** 2)))
                cv_rff[l2] = float(np.mean(fold))
            best_l2_rff = min(L2_GRID, key=lambda l2: cv_rff[l2])
            rff_model = ridge_checked(f_dev, yt, best_l2_rff)
            # reproducibility: identical projection + refit from the same seed
            rng2 = np.random.default_rng(RFF_SEED)
            f_dev2, gamma2, wb2 = rff_features(z_dev, rng2, n_components=RFF_COMPONENTS)
            assert gamma2 == gamma and np.array_equal(wb2[0], wb[0]) and np.array_equal(wb2[1], wb[1])
            rff_model2 = ridge_checked(f_dev2, yt, best_l2_rff)
            assert np.array_equal(np.asarray(rff_model['weights']), np.asarray(rff_model2['weights']))
            models[arm][t]['rff_ridge'] = {
                'center': center.tolist(), 'scale': scale.tolist(), 'gamma': float(gamma),
                'w': wb[0].tolist(), 'b': wb[1].tolist(), 'l2': best_l2_rff,
                'cv_mse': cv_rff[best_l2_rff], 'ridge': rff_model,
                'reproducibility': 'EXACT_DOUBLE_FIT_MATCH'}
            for l2 in L2_GRID:
                loocv_rows.append({'arm': arm, 'target': t, 'scorer': 'rff_ridge',
                                   'config': json.dumps({'l2': l2}), 'cv_mse': cv_rff[l2],
                                   'selected': int(l2 == best_l2_rff)})
            # 4) Shallow GBDT (grid selected by dev LOOCV MSE)
            cfg_by_hash = {canonical(c): c for c in GBDT_GRID}
            mean_mse = {h: float(np.mean(list(v.values())))
                        for h, v in gbdt_mse[arm][t].items()}
            best_hash = min((canonical(c) for c in GBDT_GRID), key=lambda h: mean_mse[h])
            best_cfg = cfg_by_hash[best_hash]
            g1 = HistGradientBoostingRegressor(**GBDT_FIXED, **best_cfg).fit(x_full, yt)
            g2 = HistGradientBoostingRegressor(**GBDT_FIXED, **best_cfg).fit(x_full, yt)
            p1, p2 = g1.predict(x_full), g2.predict(x_full)
            assert np.isfinite(p1).all() and np.array_equal(p1, p2), 'GBDT reproducibility failed'
            fpath = MODELS / f'gbdt_{arm}_{t}.joblib'
            joblib.dump(g1, fpath)
            gbdt_files[f'{arm}_{t}'] = {'file': str(fpath.relative_to(ROOT)),
                                        'sha256': sha256_file(fpath), 'params': best_cfg}
            models[arm][t]['gbdt'] = {'params': best_cfg, 'fixed': GBDT_FIXED,
                                      'cv_mse': mean_mse[best_hash],
                                      'model_file': str(fpath.relative_to(ROOT)),
                                      'reproducibility': 'EXACT_DOUBLE_FIT_MATCH'}
            for c in GBDT_GRID:
                loocv_rows.append({'arm': arm, 'target': t, 'scorer': 'gbdt',
                                   'config': json.dumps(c), 'cv_mse': mean_mse[canonical(c)],
                                   'selected': int(canonical(c) == best_hash)})
            receipts[arm][t] = {
                'frozen_gradient_inf': frozen['gradient_inf'],
                'frozen_independent_prediction_error': frozen['independent_prediction_max_error'],
                'loocv_l2': best_l2, 'loocv_gradient_inf': tuned['gradient_inf'],
                'rff_l2': best_l2_rff, 'rff_gamma': float(gamma),
                'gbdt_params': best_cfg, 'gbdt_cv_mse': mean_mse[best_hash]}
        # logistic improvement-event scorer (raw events, frozen L2=1)
        ev = ConvergedLogistic(l2=1.).fit(x_full, binary)
        if not ev.converged:
            raise ArithmeticError(f'Logistic failure {arm}: {ev.failure}')
        logistic[arm] = ev.serialization()
        receipts[arm]['event_gradient_inf'] = ev.gradient_inf
        receipts[arm]['fit_seconds'] = time.perf_counter() - t0
        print(json.dumps({'arm': arm, **{k: v for k, v in receipts[arm].items()
                                         if k in ('n_features', 'fit_seconds')}}), flush=True)

    bundle = {'schema': 'ext_standard_jssp_models_v1', 'lock_hash': lock_hash,
              'training_instances': instances, 'training_rows': len(rows),
              'maps': maps_all, 'cross_maps': cross, 'models': models,
              'logistic': logistic, 'gbdt_files': gbdt_files,
              'numerical_receipts': receipts,
              'fit_seconds': time.perf_counter() - started,
              'source_sha256': sha256_file(Path(__file__)),
              'dev_only_gate': 'RAW/test did not exist at fit time; all selection via dev LOIO/LOOCV MSE'}
    write_new(MODELS / 'ext_models.json', bundle)
    import csv
    with open(OUT / 'table_ext_dev_loocv.csv', 'x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(loocv_rows[0]))
        w.writeheader(); w.writerows(loocv_rows)
    print(json.dumps({'status': 'DEV_MODELS_FROZEN', 'arms': len(ARMS),
                      'fit_seconds': bundle['fit_seconds']}, indent=2))


if __name__ == '__main__':
    main()

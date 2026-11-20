#!/usr/bin/env python3
"""Frozen held-out test evaluation + instance-cluster statistics for ext_standard_jssp.

Consumes ONLY: locked manifest, dev-frozen model bundle, archived test branches.
No fitting or hyperparameter selection happens here. Produces all deliverable
tables plus ext_analysis.json (with per-instance values for independent replay).
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import (ROOT, EXT, RAW, OUT, MODELS, ARMS, SCORERS, TARGETS,
                        FIXED_REFERENCE, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS,
                        canonical, cng_value, load, sha256_file, write_new)
from scripts.fp_decision_models import matrix, predictions, predict
from scripts.revision_response_analysis import transform_rff, predict_ridge
from analyze_paper_evidence import signed_rank, holm
from tools.independent_tabu_jssp import ACTIONS


def ext_compare(values, label):
    names = sorted(values)
    means = np.asarray([values[k] for k in names], float)
    assert np.isfinite(means).all()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = means[rng.integers(len(means), size=(BOOTSTRAP_DRAWS, len(means)))].mean(1)
    return {'contrast': label, 'mean': float(means.mean()),
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'instance_values': dict(zip(names, means.tolist())),
            'win_tie_loss': [int((means > 1e-12).sum()), int((np.abs(means) <= 1e-12).sum()),
                             int((means < -1e-12).sum())],
            'leave_one_instance_out_means': {k: float(np.delete(means, i).mean())
                                             for i, k in enumerate(names)},
            **signed_rank(means)}


def average_precision(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if not y.sum():
        return None
    order = np.argsort(-p, kind='stable'); ys = y[order]; ps = p[order]
    end = np.r_[np.flatnonzero(ps[1:] != ps[:-1]), len(ps) - 1]
    tp = np.cumsum(ys)[end]; precision = tp / (end + 1)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / y.sum())


def csv_write(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, 'x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    lock = load(OUT / 'manifest_lock.json')
    bundle = load(MODELS / 'ext_models.json')
    assert bundle['lock_hash'] == lock['lock_hash']
    files = sorted((RAW / 'test').glob('*.json.gz'))
    rows = [r for p in files for r in load(p)['records']]
    instances = sorted({r['instance'] for r in rows})
    assert instances == sorted(lock['split_names']['test'])
    assert not (set(instances) & set(bundle['training_instances']))
    assert len(rows) == 30 * 20 * 30 == 18000
    maps = bundle['maps']
    size_of = {i['name']: i['size'] for i in load(EXT / 'instances.json')}
    block_of = {i['name']: i['block'] for i in load(EXT / 'instances.json')}

    # ---------------- predictions (frozen models, full-dev maps) ----------------
    scores = {}
    x_arm = {}
    for arm in ARMS:
        x = matrix(rows, arm, maps)
        x_arm[arm] = x
        for t in TARGETS:
            md = bundle['models'][arm][t]
            scores[t, 'frozen_ridge', arm] = predict(md['frozen_ridge'], x)
            scores[t, 'loocv_ridge', arm] = predict(md['loocv_ridge'], x)
            rff = md['rff_ridge']
            z = (x - np.asarray(rff['center'])) / np.asarray(rff['scale'])
            f = transform_rff(z, rff['gamma'], (np.asarray(rff['w']), np.asarray(rff['b'])))
            scores[t, 'rff_ridge', arm] = predict_ridge(rff['ridge'], f)
            g = joblib.load(ROOT / md['gbdt']['model_file'])
            scores[t, 'gbdt', arm] = g.predict(x)
        scores['prob', 'logistic', arm] = predict(bundle['logistic'][arm], x, logistic=True)
    for k, v in scores.items():
        assert np.isfinite(v).all(), f'non-finite predictions {k}'

    # ---------------- group by state ----------------
    groups = defaultdict(list)
    for idx, r in enumerate(rows):
        groups[r['state_id']].append(idx)
    decisions, prediction_rows, transition = [], [], []
    for sid, indices in sorted(groups.items()):
        first = rows[indices[0]]
        assert len(indices) == 15
        r_raw = np.empty((3, 5)); r_cng = np.empty((3, 5)); dec = np.empty((3, 5))
        ps = {k: [] for k in scores}
        for ai, a in enumerate(ACTIONS):
            ids = sorted([i for i in indices if rows[i]['action'] == a],
                         key=lambda i: rows[i]['repeat'])
            assert [rows[i]['repeat'] for i in ids] == list(range(5))
            for rep, i in enumerate(ids):
                r_raw[ai, rep] = rows[i]['gain']
                r_cng[ai, rep] = cng_value(rows[i])
                dec[ai, rep] = rows[i]['outcome']['decode_evaluations']
            for k in scores:
                p = scores[k][ids]
                assert np.max(p) - np.min(p) < 1e-12, 'outcome-dependent score within state-action'
                ps[k].append(float(p[0]))
            x = np.asarray(first['start_x'])
            y = np.mean([rows[i]['end_x'] for i in ids], axis=0)
            pred = {'conditioned': predictions(maps, x)[ai],
                    'pooled': np.asarray(maps['pooled']['A']) @ x + maps['pooled']['c'],
                    'identity': x}
            transition.append({'instance': first['instance'], 'run': first['run'],
                               'state_id': sid, 'action': a,
                               **{v: float(np.linalg.norm(p - y)) for v, p in pred.items()}})
            prediction_rows.append({'instance': first['instance'], 'run': first['run'],
                                    'state_id': sid, 'action': a,
                                    'gain_mean5': float(r_raw[ai].mean()),
                                    'gain_sd5': float(r_raw[ai].std(ddof=1)),
                                    'gain_se5': float(r_raw[ai].std(ddof=1) / np.sqrt(5)),
                                    'cng_mean5': float(r_cng[ai].mean()),
                                    **{f'repeat{rep}_gain': float(r_raw[ai, rep]) for rep in range(5)},
                                    **{f'repeat{rep}_cng': float(r_cng[ai, rep]) for rep in range(5)},
                                    **{f'repeat{rep}_decodes': int(dec[ai, rep]) for rep in range(5)},
                                    **{f'{t}_{sc}_{arm}': ps[t, sc, arm][-1]
                                       for t in TARGETS for sc in SCORERS for arm in ARMS},
                                    **{f'prob_{arm}': ps['prob', 'logistic', arm][-1] for arm in ARMS}})
        means = {'raw': r_raw.mean(1), 'cng': r_cng.mean(1)}
        d = {'instance': first['instance'], 'run': first['run'], 'state_id': sid,
             'size': size_of[first['instance']], 'block': block_of[first['instance']]}
        for t in TARGETS:
            for ai, a in enumerate(ACTIONS):
                d[f'fixed_{a}_{t}'] = float(means[t][ai])
            d[f'all_actions_zero_{t}'] = int(means[t].max() == 0)
            for sc in SCORERS:
                for arm in ARMS:
                    p = [ps[t, sc, a2] for a2 in ARMS] if False else ps[t, sc, arm]
                    c = int(np.argmax(p))
                    d[f'{t}_{sc}_{arm}_action'] = ACTIONS[c]
                    d[f'{t}_{sc}_{arm}_value'] = float(means[t][c])
                    d[f'{t}_{sc}_{arm}_eval3'] = float((r_cng if t == 'cng' else r_raw)[c, 2:].mean())
            # split-repeat sample winner, both directions
            fit2 = (r_cng if t == 'cng' else r_raw)[:, :2].mean(1)
            eval3 = (r_cng if t == 'cng' else r_raw)[:, 2:].mean(1)
            ref = ACTIONS.index(FIXED_REFERENCE[t])
            w2 = int(np.argmax(fit2)); w3 = int(np.argmax(eval3))
            d[f'{t}_sr_fwd_same_adv'] = float(fit2[w2] - fit2[ref])
            d[f'{t}_sr_fwd_fresh_adv'] = float(eval3[w2] - eval3[ref])
            d[f'{t}_sr_fwd_optimism'] = float(fit2[w2] - eval3[w2])
            d[f'{t}_sr_fwd_unique2'] = int((np.abs(fit2 - fit2.max()) < 1e-12).sum() == 1)
            d[f'{t}_sr_fwd_agree'] = int(w2 == w3)
            d[f'{t}_sr_rev_same_adv'] = float(eval3[w3] - eval3[ref])
            d[f'{t}_sr_rev_fresh_adv'] = float(fit2[w3] - fit2[ref])
            d[f'{t}_sr_rev_optimism'] = float(eval3[w3] - fit2[w3])
            d[f'{t}_sr_rev_unique3'] = int((np.abs(eval3 - eval3.max()) < 1e-12).sum() == 1)
            d[f'{t}_sr_rev_agree'] = int(w2 == w3)
        for arm in ARMS:
            c = int(np.argmax(ps['prob', 'logistic', arm]))
            d[f'event_{arm}_action'] = ACTIONS[c]
            d[f'event_{arm}_value'] = float(means['raw'][c])
            d[f'event_{arm}_cng_value'] = float(means['cng'][c])
        for a_i, a in enumerate(ACTIONS):
            d[f'mean_decodes_{a}'] = float(dec[a_i].mean())
        decisions.append(d)

    # ---------------- instance-level aggregation ----------------
    per = []
    for inst in instances:
        g = [r for r in decisions if r['instance'] == inst]
        numeric = [k for k in g[0] if k not in ('instance', 'run', 'state_id', 'size', 'block')
                   and not k.endswith('_action')]
        out = {'instance': inst, 'size': size_of[inst], 'block': block_of[inst],
               **{k: float(np.mean([r[k] for r in g])) for k in numeric}}
        ids = [i for i, r in enumerate(rows) if r['instance'] == inst]
        for arm in ARMS:
            yv = np.asarray([rows[i]['improved'] for i in ids])
            pv = scores['prob', 'logistic', arm][ids]
            out[f'{arm}_brier'] = float(np.mean((pv - yv) ** 2))
            for a in ACTIONS:
                ix = [i for i in ids if rows[i]['action'] == a]
                out[f'{arm}_{a}_ap'] = average_precision([rows[i]['improved'] for i in ix],
                                                         scores['prob', 'logistic', arm][ix])
        per.append(out)

    def inst_mean(field):
        return {r['instance']: r[field] for r in per}

    # ---------------- contrast families ----------------
    paired = []
    contrast_store = {}
    for t in TARGETS:
        for sc in SCORERS:
            fam = []
            for comp in ('B', 'E'):
                c = ext_compare({r['instance']: r[f'{t}_{sc}_D_value'] - r[f'{t}_{sc}_{comp}_value']
                                 for r in per}, f'{t}_{sc}_D_minus_{comp}')
                fam.append(c)
            holm(fam)
            for c in fam:
                c.update({'family': f'{t}_{sc}_D_contrasts', 'scorer': sc, 'target': t,
                          'kind': 'representation_contrast'})
            paired += fam
            contrast_store[f'{t}_{sc}_D_contrasts'] = fam
            # arms vs best fixed static action (predeclared reference per regime)
            ref = FIXED_REFERENCE[t]
            fam2 = []
            for arm in ARMS:
                c = ext_compare({r['instance']: r[f'{t}_{sc}_{arm}_value'] - r[f'fixed_{ref}_{t}']
                                 for r in per}, f'{t}_{sc}_{arm}_minus_fixed_{ref}')
                fam2.append(c)
            holm(fam2)
            macro_arm = {arm: float(np.mean([r[f'{t}_{sc}_{arm}_value'] for r in per])) for arm in ARMS}
            best_arm = max(ARMS, key=lambda a: macro_arm[a])
            for c, arm in zip(fam2, ARMS):
                c.update({'family': f'{t}_{sc}_arms_vs_fixed', 'scorer': sc, 'target': t,
                          'kind': 'arm_vs_fixed', 'arm': arm, 'best_arm': int(arm == best_arm)})
            paired += fam2
            contrast_store[f'{t}_{sc}_arms_vs_fixed'] = fam2
            best_c = ext_compare({r['instance']: r[f'{t}_{sc}_{best_arm}_value'] - r[f'fixed_{ref}_{t}']
                                  for r in per}, f'{t}_{sc}_best_arm_{best_arm}_minus_fixed_{ref}')
            best_c.update({'family': f'{t}_{sc}_best_arm_vs_fixed', 'scorer': sc, 'target': t,
                           'kind': 'best_arm_vs_fixed', 'arm': best_arm, 'best_arm': 1,
                           'holm_p': best_c['p']})
            paired.append(best_c)
            contrast_store[f'{t}_{sc}_best_arm_vs_fixed'] = [best_c]

    # transition error contrasts (instance-cluster)
    dynamics = []
    for a in ACTIONS:
        for comp in ('pooled', 'identity'):
            values = {i: float(np.mean([r[comp] - r['conditioned'] for r in transition
                                        if r['instance'] == i and r['action'] == a]))
                      for i in instances}
            dynamics.append(ext_compare(values, f'{a}_error_{comp}_minus_conditioned'))
    holm(dynamics)

    # split-repeat fresh/same advantage contrasts with CIs
    sr_contrasts = []
    for t in TARGETS:
        for direction in ('fwd', 'rev'):
            for kind in ('same_adv', 'fresh_adv', 'optimism'):
                c = ext_compare(inst_mean(f'{t}_sr_{direction}_{kind}'),
                                f'{t}_split_repeat_{direction}_{kind}')
                c.update({'target': t, 'direction': direction, 'kind': kind})
                sr_contrasts.append(c)

    # ---------------- scorer tables ----------------
    def selected_return_ci(t, sc, arm):
        vals = np.asarray([r[f'{t}_{sc}_{arm}_value'] for r in per])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot = vals[rng.integers(len(vals), size=(BOOTSTRAP_DRAWS, len(vals)))].mean(1)
        return float(vals.mean()), np.quantile(boot, [.025, .975]).tolist()

    scorer_tables = {}
    for t in TARGETS:
        tbl = []
        for sc in SCORERS:
            for arm in ARMS:
                m, ci = selected_return_ci(t, sc, arm)
                md = bundle['models'][arm][t][sc]
                if sc == 'frozen_ridge':
                    hp = 'L2=1'
                elif sc == 'loocv_ridge':
                    hp = f"L2={md['l2']}"
                elif sc == 'rff_ridge':
                    hp = f"L2={md['l2']};gamma={md['gamma']:.6g}"
                else:
                    hp = json.dumps(md['params'])
                counts = Counter(r[f'{t}_{sc}_{arm}_action'] for r in decisions)
                tbl.append({'scorer': sc, 'arm': arm, 'mean_selected_return': m,
                            'ci95_low': ci[0], 'ci95_high': ci[1],
                            'hyperparameters': hp, 'dev_cv_mse': md.get('cv_mse', ''),
                            'n_states': len(decisions),
                            'n_CB_BEST_DESCENT': counts.get('CB_BEST_DESCENT', 0),
                            'n_CB_FIRST_DESCENT': counts.get('CB_FIRST_DESCENT', 0),
                            'n_CB_TABU': counts.get('CB_TABU', 0)})
        for a in ACTIONS:
            vals = np.asarray([r[f'fixed_{a}_{t}'] for r in per])
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            boot = vals[rng.integers(len(vals), size=(BOOTSTRAP_DRAWS, len(vals)))].mean(1)
            tbl.append({'scorer': f'fixed_{a}', 'arm': '-',
                        'mean_selected_return': float(vals.mean()),
                        'ci95_low': float(np.quantile(boot, .025)),
                        'ci95_high': float(np.quantile(boot, .975)),
                        'hyperparameters': 'static', 'dev_cv_mse': '',
                        'n_states': len(decisions),
                        'n_CB_BEST_DESCENT': len(decisions) if a == 'CB_BEST_DESCENT' else 0,
                        'n_CB_FIRST_DESCENT': len(decisions) if a == 'CB_FIRST_DESCENT' else 0,
                        'n_CB_TABU': len(decisions) if a == 'CB_TABU' else 0})
        if t == 'raw':
            for arm in ARMS:
                vals = np.asarray([r[f'event_{arm}_value'] for r in per])
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                boot = vals[rng.integers(len(vals), size=(BOOTSTRAP_DRAWS, len(vals)))].mean(1)
                counts = Counter(r[f'event_{arm}_action'] for r in decisions)
                aps = []
                for r in per:
                    defined = [r[f'{arm}_{a}_ap'] for a in ACTIONS
                               if r[f'{arm}_{a}_ap'] is not None]
                    if defined:
                        aps.append(float(np.mean(defined)))
                tbl.append({'scorer': 'logistic_event', 'arm': arm,
                            'mean_selected_return': float(vals.mean()),
                            'ci95_low': float(np.quantile(boot, .025)),
                            'ci95_high': float(np.quantile(boot, .975)),
                            'hyperparameters': 'L2=1', 'dev_cv_mse': '',
                            'n_states': len(decisions),
                            'n_CB_BEST_DESCENT': counts.get('CB_BEST_DESCENT', 0),
                            'n_CB_FIRST_DESCENT': counts.get('CB_FIRST_DESCENT', 0),
                            'n_CB_TABU': counts.get('CB_TABU', 0),
                            'macro_brier': float(np.mean([r[f'{arm}_brier'] for r in per])),
                            'macro_ap': float(np.mean(aps)) if aps else ''})
        scorer_tables[t] = tbl

    # ---------------- per-size descriptives ----------------
    sizes = sorted({size_of[i] for i in instances})
    per_size = []
    for sz in sizes:
        sz_inst = [i for i in instances if size_of[i] == sz]
        g = [r for r in decisions if r['size'] == sz]
        row = {'size': sz, 'n_instances': len(sz_inst), 'n_states': len(g),
               'blocks': ';'.join(sorted({block_of[i] for i in sz_inst})),
               'all_zero_rate_pct_raw': 100 * float(np.mean([r['all_actions_zero_raw'] for r in g])),
               'all_zero_rate_pct_cng': 100 * float(np.mean([r['all_actions_zero_cng'] for r in g]))}
        for a in ACTIONS:
            row[f'mean_raw_{a}'] = float(np.mean([np.mean([r[f'fixed_{a}_raw'] for r in g
                                                           if r['instance'] == i]) for i in sz_inst]))
            row[f'mean_cng_{a}'] = float(np.mean([np.mean([r[f'fixed_{a}_cng'] for r in g
                                                           if r['instance'] == i]) for i in sz_inst]))
            row[f'mean_decodes_{a}'] = float(np.mean([r[f'mean_decodes_{a}'] for r in g]))
        raw_wins = Counter(); cng_wins = Counter()
        for r in g:
            raw_wins[max(ACTIONS, key=lambda a: r[f'fixed_{a}_raw'])] += 1
            cng_wins[max(ACTIONS, key=lambda a: r[f'fixed_{a}_cng'])] += 1
        row['modal_best_action_raw'] = raw_wins.most_common(1)[0][0]
        row['modal_best_action_cng'] = cng_wins.most_common(1)[0][0]
        row['raw_best_action_counts'] = json.dumps(dict(raw_wins))
        row['cng_best_action_counts'] = json.dumps(dict(cng_wins))
        for t in TARGETS:
            for sc in SCORERS:
                row[f'D_minus_B_{t}_{sc}'] = float(np.mean(
                    [np.mean([r[f'{t}_{sc}_D_value'] - r[f'{t}_{sc}_B_value'] for r in g
                              if r['instance'] == i]) for i in sz_inst]))
        per_size.append(row)

    # ---------------- split-repeat table ----------------
    sr_table = []
    for t in TARGETS:
        for direction in ('fwd', 'rev'):
            same = next(c for c in sr_contrasts if c['target'] == t and c['direction'] == direction
                        and c['kind'] == 'same_adv')
            fresh = next(c for c in sr_contrasts if c['target'] == t and c['direction'] == direction
                         and c['kind'] == 'fresh_adv')
            opt = next(c for c in sr_contrasts if c['target'] == t and c['direction'] == direction
                       and c['kind'] == 'optimism')
            uniq_field = f'{t}_sr_{direction}_unique' + ('2' if direction == 'fwd' else '3')
            sr_table.append({'target': t, 'direction': 'repeats01_to_234' if direction == 'fwd'
                             else 'repeats234_to_01',
                             'fixed_reference': FIXED_REFERENCE[t],
                             'same_repeat_advantage': same['mean'],
                             'same_repeat_advantage_ci95_low': same['ci95'][0],
                             'same_repeat_advantage_ci95_high': same['ci95'][1],
                             'fresh_repeat_advantage': fresh['mean'],
                             'fresh_repeat_advantage_ci95_low': fresh['ci95'][0],
                             'fresh_repeat_advantage_ci95_high': fresh['ci95'][1],
                             'fresh_repeat_advantage_p': fresh['p'],
                             'optimism_gap': opt['mean'],
                             'unique_winner_rate': float(np.mean([r[uniq_field] for r in decisions])),
                             'label_agreement_rate': float(np.mean([r[f'{t}_sr_{direction}_agree']
                                                                    for r in decisions])),
                             'n_states': len(decisions)})

    # ---------------- transition error table ----------------
    trans_table = []
    for a in ACTIONS:
        row = {'action': a}
        for v in ('conditioned', 'pooled', 'identity'):
            row[f'{v}_mean_error'] = float(np.mean([r[v] for r in transition if r['action'] == a]))
        for comp in ('pooled', 'identity'):
            c = next(q for q in dynamics if q['contrast'] == f'{a}_error_{comp}_minus_conditioned')
            row[f'{comp}_minus_conditioned_mean'] = c['mean']
            row[f'{comp}_minus_conditioned_ci95_low'] = c['ci95'][0]
            row[f'{comp}_minus_conditioned_ci95_high'] = c['ci95'][1]
            row[f'{comp}_minus_conditioned_p'] = c['p']
            row[f'{comp}_minus_conditioned_holm_p'] = c['holm_p']
        trans_table.append(row)

    # ---------------- paired contrast table ----------------
    paired_table = []
    for c in paired:
        loio = list(c['leave_one_instance_out_means'].values())
        paired_table.append({'family': c['family'], 'contrast': c['contrast'],
                             'kind': c['kind'], 'scorer': c['scorer'], 'target': c['target'],
                             'arm': c.get('arm', ''), 'best_arm': c.get('best_arm', ''),
                             'mean': c['mean'], 'ci95_low': c['ci95'][0], 'ci95_high': c['ci95'][1],
                             'p': c['p'], 'holm_p': c['holm_p'],
                             'win': c['win_tie_loss'][0], 'tie': c['win_tie_loss'][1],
                             'loss': c['win_tie_loss'][2],
                             'nonzero_instances': c['nonzero_instances'],
                             'loio_min': min(loio), 'loio_max': max(loio)})

    costs = {a: {'mean_decodes': float(np.mean([r['outcome']['decode_evaluations'] for r in rows
                                                if r['action'] == a])),
                 'mean_seconds': float(np.mean([r['outcome']['elapsed_seconds'] for r in rows
                                                if r['action'] == a])),
                 'mean_moves': float(np.mean([r['outcome']['accepted_steps'] for r in rows
                                              if r['action'] == a])),
                 'termination_counts': dict(Counter(r['outcome']['termination'] for r in rows
                                                    if r['action'] == a))} for a in ACTIONS}
    fixed_means = {t: {a: float(np.mean([r[f'fixed_{a}_{t}'] for r in per])) for a in ACTIONS}
                   for t in TARGETS}
    empirical_best_fixed = {t: max(ACTIONS, key=lambda a: fixed_means[t][a]) for t in TARGETS}

    summary = {'cohort': 'ext_standard_jssp', 'lock_hash': lock['lock_hash'],
               'instances': len(instances), 'trajectories': len({r['trajectory_id'] for r in rows}),
               'states': len(decisions), 'branches': len(rows),
               'fixed_action_means': fixed_means,
               'predeclared_fixed_reference': FIXED_REFERENCE,
               'empirical_best_fixed': empirical_best_fixed,
               'macro': {k: (float(np.mean([r[k] for r in per if isinstance(r.get(k), (int, float))]))
                             if any(isinstance(r.get(k), (int, float)) for r in per) else None)
                         for k in per[0] if k not in ('instance', 'size', 'block')},
               'decision_counts': {f'{t}_{sc}_{arm}': dict(Counter(r[f'{t}_{sc}_{arm}_action']
                                                                  for r in decisions))
                                   for t in TARGETS for sc in SCORERS for arm in ARMS},
               'paired_contrasts': paired, 'dynamics_contrasts': dynamics,
               'split_repeat_contrasts': sr_contrasts,
               'per_size': per_size, 'costs': costs,
               'transition_mean_error': {a: {v: float(np.mean([r[v] for r in transition
                                                               if r['action'] == a]))
                                             for v in ('conditioned', 'pooled', 'identity')}
                                         for a in ACTIONS},
               'raw_file_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
               'scope': 'Offline branch reward on held-out standard benchmarks; dev-only fitting; '
                        'independent-repeat sample winner is a noisy comparator, not a true oracle.'}
    csv_write(OUT / 'ext_state_decisions.csv', decisions)
    csv_write(OUT / 'ext_predictions.csv', prediction_rows)
    csv_write(OUT / 'ext_instance_metrics.csv', per)
    csv_write(OUT / 'ext_transition_errors.csv', transition)
    csv_write(OUT / 'table_ext_transition_errors.csv', trans_table)
    csv_write(OUT / 'table_ext_raw_gain_scorers.csv', scorer_tables['raw'])
    csv_write(OUT / 'table_ext_cost_normalized_scorers.csv', scorer_tables['cng'])
    csv_write(OUT / 'table_ext_per_size_descriptives.csv', per_size)
    csv_write(OUT / 'table_ext_split_repeat_diagnostics.csv', sr_table)
    csv_write(OUT / 'table_ext_paired_contrasts.csv', paired_table)
    write_new(OUT / 'ext_analysis.json', summary)
    print(json.dumps({'status': 'ANALYSIS_COMPLETE', 'states': len(decisions),
                      'fixed_means': fixed_means, 'empirical_best_fixed': empirical_best_fixed},
                     indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Independent scalar replay for ext_standard_jssp (verify_ext_replay).

Standalone: imports ONLY numpy + standard library. Does NOT import the collector,
solver, feature builder, model code, or any producer metric/statistics helper.
Reconstructs directly from archived raw branch files and the saved prediction
array (ext_predictions.csv):
  raw rewards, per-request rewards, policy choices, fixed-action baselines,
  instance-cluster means, exact two-sided Wilcoxon signed-rank tails (own DP),
  Holm adjustments, 20,000-draw instance bootstrap CIs (own draw construction),
  split-repeat winner diagnostics under BOTH targets and both directions,
  per-size descriptives and action distributions.
Every reconstructed quantity is compared to the producer tables/JSON.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / 'experiments/ext_standard_jssp'
OUT = EXT / 'outputs'
ACTIONS = ('CB_BEST_DESCENT', 'CB_FIRST_DESCENT', 'CB_TABU')
ARMS = ('A', 'B', 'C', 'D', 'E', 'D_random')
SCORERS = ('frozen_ridge', 'loocv_ridge', 'rff_ridge', 'gbdt')
TARGETS = ('raw', 'cng')
TOL = 1e-12


def read(p):
    with (gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)) as f:
        return json.load(f)


def exact_signed_rank(values):
    """Exact two-sided random-sign tail with midranks; independent DP implementation."""
    v = [x for x in values if abs(x) > 1e-12]
    if not v:
        return 1.0
    av = sorted(abs(x) for x in v)
    midrank = {}
    i = 0
    while i < len(av):
        j = i
        while j + 1 < len(av) and av[j + 1] == av[i]:
            j += 1
        midrank[av[i]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    doubled = [int(2 * midrank[abs(x)]) for x in v]
    total = sum(doubled)
    positive = sum(d for d, x in zip(doubled, v) if x > 0)
    ways = [0] * (total + 1)
    ways[0] = 1
    cur = 0
    for d in doubled:
        for s in range(cur, -1, -1):
            ways[s + d] += ways[s]
        cur += d
    extreme = abs(2 * positive - total)
    count = sum(w for s, w in enumerate(ways) if abs(2 * s - total) >= extreme)
    return min(1.0, count / (2 ** len(v)))


def holm_adjust(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        out[i] = running
    return out


def bootstrap_ci(values, seed, draws):
    """One vector at a time, independent of the producer's batched construction."""
    vals = list(values)
    n = len(vals)
    rng = np.random.default_rng(seed)
    samples = [sum(vals[int(j)] for j in rng.integers(n, size=n)) / n for _ in range(draws)]
    return np.quantile(samples, [.025, .975]).tolist()


def main():
    lock = read(OUT / 'manifest_lock.json')
    seed = lock['seeds']['bootstrap']
    draws = lock['seeds']['bootstrap_draws']
    fixed_ref = lock['protocol']['fixed_reference']
    analysis = read(OUT / 'ext_analysis.json')
    sched = read(OUT / 'verification_schedules_test.json')
    errors = defaultdict(float)
    checks = []

    def close(label, a, b, tol=TOL):
        e = abs(float(a) - float(b))
        errors[label] = max(errors[label], e)
        if e > tol:
            raise AssertionError(f'REPLAY MISMATCH {label}: {a} vs {b} (err {e})')

    # ---- raw archives: hash check + reward reconstruction ----
    files = sorted((EXT / 'raw/test').glob('*.json.gz'))
    records = []
    for p in files:
        rel = str(p.relative_to(ROOT))
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert sched['file_hashes'][rel] == h, f'raw file hash mismatch {rel}'
        records.extend(read(p)['records'])
    assert len(records) == 18000
    gains, cngs, names, sizes = {}, {}, {}, {}
    instances_meta = {i['name']: i for i in read(EXT / 'instances.json')}
    for r in records:
        key = (r['state_id'], r['action'], r['repeat'])
        g = (r['start_makespan'] - r['end_makespan']) / r['s0']
        assert abs(g - r['gain']) < TOL
        gains[key] = g
        cngs[key] = g / max(r['outcome']['decode_evaluations'], 1) * 1e3
        names[r['state_id']] = r['instance']
        sizes[r['state_id']] = instances_meta[r['instance']]['size']
    instances = sorted(set(names.values()))
    assert instances == sorted(lock['split_names']['test'])
    rewards = {'raw': gains, 'cng': cngs}

    # ---- saved prediction array ----
    with open(OUT / 'ext_predictions.csv') as f:
        pred = {(r['state_id'], r['action']): r for r in csv.DictReader(f)}
    assert len(pred) == 3600
    for (sid, a), row in pred.items():
        for rep in range(5):
            close('csv_repeat_gain', row[f'repeat{rep}_gain'], gains[sid, a, rep])
            close('csv_repeat_cng', row[f'repeat{rep}_cng'], cngs[sid, a, rep])
        close('csv_gain_mean5', row['gain_mean5'], np.mean([gains[sid, a, r] for r in range(5)]))
        close('csv_cng_mean5', row['cng_mean5'], np.mean([cngs[sid, a, r] for r in range(5)]))

    sids = sorted(names)
    mean5 = {t: {sid: {a: float(np.mean([rewards[t][sid, a, r] for r in range(5)]))
                       for a in ACTIONS} for sid in sids} for t in TARGETS}

    # ---- choices, selected values, instance means ----
    choices, values = {}, {}
    for t in TARGETS:
        for sc in SCORERS:
            for arm in ARMS:
                col = f'{t}_{sc}_{arm}'
                ch, val = {}, {}
                for sid in sids:
                    scores = [float(pred[sid, a][col]) for a in ACTIONS]
                    c = int(np.argmax(scores))
                    ch[sid] = ACTIONS[c]
                    val[sid] = mean5[t][sid][ACTIONS[c]]
                choices[t, sc, arm] = ch
                values[t, sc, arm] = val
        ch, val = {}, {}
        for arm in ARMS:
            for sid in sids:
                scores = [float(pred[sid, a][f'prob_{arm}']) for a in ACTIONS]
                c = int(np.argmax(scores))
                ch.setdefault(arm, {})[sid] = ACTIONS[c]
                val.setdefault(arm, {})[sid] = mean5['raw'][sid][ACTIONS[c]]
        choices['event'] = ch
        values['event'] = val

    def inst_mean(val):
        return {i: float(np.mean([val[s] for s in sids if names[s] == i])) for i in instances}

    # ---- fixed baselines ----
    fixed_inst = {t: {a: inst_mean({s: mean5[t][s][a] for s in sids}) for a in ACTIONS}
                  for t in TARGETS}
    for t in TARGETS:
        for a in ACTIONS:
            close(f'fixed_mean_{t}_{a}', float(np.mean(list(fixed_inst[t][a].values()))),
                  analysis['fixed_action_means'][t][a])

    # ---- paired contrasts: D-B / D-E / arms-vs-fixed / best-arm ----
    producer = {c['contrast']: c for c in analysis['paired_contrasts']}
    with open(OUT / 'table_ext_paired_contrasts.csv') as f:
        table = {r['contrast']: r for r in csv.DictReader(f)}
    n_reconstructed = 0
    for t in TARGETS:
        for sc in SCORERS:
            fam = []
            for comp in ('B', 'E'):
                diff = {i: float(np.mean([values[t, sc, 'D'][s] - values[t, sc, comp][s]
                                          for s in sids if names[s] == i])) for i in instances}
                fam.append((f'{t}_{sc}_D_minus_{comp}', diff))
            ps = [exact_signed_rank(list(d.values())) for _, d in fam]
            hs = holm_adjust(ps)
            for (label, diff), p, hp in zip(fam, ps, hs):
                pc = producer[label]
                for i in instances:
                    close('instance_values', diff[i], pc['instance_values'][i])
                close('contrast_mean', float(np.mean(list(diff.values()))), pc['mean'])
                close('contrast_p', p, pc['p'])
                close('contrast_holm', hp, pc['holm_p'])
                ci = bootstrap_ci([diff[i] for i in instances], seed, draws)
                close('contrast_ci_lo', ci[0], pc['ci95'][0])
                close('contrast_ci_hi', ci[1], pc['ci95'][1])
                close('table_ci_lo', ci[0], table[label]['ci95_low'])
                close('table_ci_hi', ci[1], table[label]['ci95_high'])
                close('table_holm', hp, table[label]['holm_p'])
                means = np.asarray([diff[i] for i in instances])
                wtl = [int((means > 1e-12).sum()), int((np.abs(means) <= 1e-12).sum()),
                       int((means < -1e-12).sum())]
                assert wtl == pc['win_tie_loss']
                loio = [float(np.delete(means, k).mean()) for k in range(len(means))]
                close('loio_min', min(loio), table[label]['loio_min'])
                close('loio_max', max(loio), table[label]['loio_max'])
                n_reconstructed += 1
            ref = fixed_ref[t]
            fam2 = []
            for arm in ARMS:
                diff = {i: float(np.mean([values[t, sc, arm][s] - mean5[t][s][ref]
                                          for s in sids if names[s] == i])) for i in instances}
                fam2.append((f'{t}_{sc}_{arm}_minus_fixed_{ref}', diff))
            ps2 = [exact_signed_rank(list(d.values())) for _, d in fam2]
            hs2 = holm_adjust(ps2)
            macros = {arm: float(np.mean(list(inst_mean(values[t, sc, arm]).values()))) for arm in ARMS}
            best_arm = max(ARMS, key=lambda a: macros[a])
            for (label, diff), p, hp, arm in zip(fam2, ps2, hs2, ARMS):
                pc = producer[label]
                close('arm_contrast_mean', float(np.mean(list(diff.values()))), pc['mean'])
                close('arm_contrast_p', p, pc['p'])
                close('arm_contrast_holm', hp, pc['holm_p'])
                ci = bootstrap_ci([diff[i] for i in instances], seed, draws)
                close('arm_ci_lo', ci[0], pc['ci95'][0])
                close('arm_ci_hi', ci[1], pc['ci95'][1])
                close('arm_table_ci_lo', ci[0], table[label]['ci95_low'])
                close('arm_table_ci_hi', ci[1], table[label]['ci95_high'])
                assert pc['best_arm'] == int(arm == best_arm)
                n_reconstructed += 1
            bl = f'{t}_{sc}_best_arm_{best_arm}_minus_fixed_{ref}'
            bdiff = {i: float(np.mean([values[t, sc, best_arm][s] - mean5[t][s][ref]
                                       for s in sids if names[s] == i])) for i in instances}
            close('best_arm_mean', float(np.mean(list(bdiff.values()))), producer[bl]['mean'])
            close('best_arm_ci_lo', bootstrap_ci([bdiff[i] for i in instances], seed, draws)[0],
                  producer[bl]['ci95'][0])
            n_reconstructed += 1

    # ---- scorer tables: macro selected returns + action distributions ----
    for t in TARGETS:
        fname = 'table_ext_raw_gain_scorers.csv' if t == 'raw' else 'table_ext_cost_normalized_scorers.csv'
        with open(OUT / fname) as f:
            rows = {(r['scorer'], r['arm']): r for r in csv.DictReader(f)}
        for sc in SCORERS:
            for arm in ARMS:
                im = inst_mean(values[t, sc, arm])
                close('scorer_table_mean', float(np.mean(list(im.values()))),
                      rows[sc, arm]['mean_selected_return'])
                cnt = Counter(choices[t, sc, arm].values())
                for a in ACTIONS:
                    assert int(rows[sc, arm][f'n_{a}']) == cnt.get(a, 0), (t, sc, arm, a)
                assert analysis['decision_counts'][f'{t}_{sc}_{arm}'] == dict(cnt) or (
                    {k: v for k, v in cnt.items()} == analysis['decision_counts'][f'{t}_{sc}_{arm}'])
        for a in ACTIONS:
            close('fixed_table_mean', float(np.mean(list(fixed_inst[t][a].values()))),
                  rows[f'fixed_{a}', '-']['mean_selected_return'])
        if t == 'raw':
            for arm in ARMS:
                im = inst_mean(values['event'][arm])
                close('logistic_table_mean', float(np.mean(list(im.values()))),
                      rows['logistic_event', arm]['mean_selected_return'])
                cnt = Counter(choices['event'][arm].values())
                for a in ACTIONS:
                    assert int(rows['logistic_event', arm][f'n_{a}']) == cnt.get(a, 0)

    # ---- split-repeat diagnostics, both targets, both directions ----
    with open(OUT / 'table_ext_split_repeat_diagnostics.csv') as f:
        sr_rows = {(r['target'], r['direction']): r for r in csv.DictReader(f)}
    sr_producer = {c['contrast']: c for c in analysis['split_repeat_contrasts']}
    for t in TARGETS:
        ref_idx = ACTIONS.index(fixed_ref[t])
        per_state = defaultdict(list)
        for sid in sids:
            arr = np.asarray([[rewards[t][sid, a, r] for r in range(5)] for a in ACTIONS])
            fit2 = arr[:, :2].mean(1); eval3 = arr[:, 2:].mean(1)
            w2 = int(np.argmax(fit2)); w3 = int(np.argmax(eval3))
            per_state['fwd_same_adv'].append(fit2[w2] - fit2[ref_idx])
            per_state['fwd_fresh_adv'].append(eval3[w2] - eval3[ref_idx])
            per_state['fwd_optimism'].append(fit2[w2] - eval3[w2])
            per_state['fwd_unique'].append(int((np.abs(fit2 - fit2.max()) < 1e-12).sum() == 1))
            per_state['fwd_agree'].append(int(w2 == w3))
            per_state['rev_same_adv'].append(eval3[w3] - eval3[ref_idx])
            per_state['rev_fresh_adv'].append(fit2[w3] - fit2[ref_idx])
            per_state['rev_optimism'].append(eval3[w3] - fit2[w3])
            per_state['rev_unique'].append(int((np.abs(eval3 - eval3.max()) < 1e-12).sum() == 1))
            per_state['rev_agree'].append(int(w2 == w3))
        for direction in ('fwd', 'rev'):
            key = (t, 'repeats01_to_234' if direction == 'fwd' else 'repeats234_to_01')
            row = sr_rows[key]
            for kind, col in (('same_adv', 'same_repeat_advantage'),
                              ('fresh_adv', 'fresh_repeat_advantage'),
                              ('optimism', 'optimism_gap')):
                vals = per_state[f'{direction}_{kind}']
                im = {i: float(np.mean([v for s, v in zip(sids, vals) if names[s] == i]))
                      for i in instances}
                macro = float(np.mean(list(im.values())))
                close('sr_macro', macro, row[col])
                pc = sr_producer[f'{t}_split_repeat_{direction}_{kind}']
                close('sr_producer_macro', macro, pc['mean'])
                ci = bootstrap_ci([im[i] for i in instances], seed, draws)
                close('sr_producer_ci_lo', ci[0], pc['ci95'][0])
                close('sr_producer_ci_hi', ci[1], pc['ci95'][1])
                if f'{col}_ci95_low' in row:
                    close('sr_ci_lo', ci[0], row[f'{col}_ci95_low'])
                    close('sr_ci_hi', ci[1], row[f'{col}_ci95_high'])
            close('sr_unique', float(np.mean(per_state[f'{direction}_unique'])), row['unique_winner_rate'])
            close('sr_agree', float(np.mean(per_state[f'{direction}_agree'])), row['label_agreement_rate'])

    # ---- per-size descriptives ----
    with open(OUT / 'table_ext_per_size_descriptives.csv') as f:
        size_rows = {r['size']: r for r in csv.DictReader(f)}
    for sz, row in size_rows.items():
        sz_sids = [s for s in sids if sizes[s] == sz]
        sz_inst = sorted({names[s] for s in sz_sids})
        assert int(row['n_instances']) == len(sz_inst)
        assert int(row['n_states']) == len(sz_sids)
        for t in TARGETS:
            zero = [int(max(mean5[t][s].values()) == 0) for s in sz_sids]
            close('size_zero_rate', 100 * float(np.mean(zero)), row[f'all_zero_rate_pct_{t}'])
            wins = Counter(max(ACTIONS, key=lambda a: mean5[t][s][a]) for s in sz_sids)
            assert row[f'modal_best_action_{t}'] == wins.most_common(1)[0][0]
            for a in ACTIONS:
                im = {i: float(np.mean([mean5[t][s][a] for s in sz_sids if names[s] == i]))
                      for i in sz_inst}
                close('size_action_mean', float(np.mean(list(im.values()))), row[f'mean_{t}_{a}'])
            for sc in SCORERS:
                db = {i: float(np.mean([values[t, sc, 'D'][s] - values[t, sc, 'B'][s]
                                        for s in sz_sids if names[s] == i])) for i in sz_inst}
                close('size_D_minus_B', float(np.mean(list(db.values()))), row[f'D_minus_B_{t}_{sc}'])

    report = {'status': 'PASS_INDEPENDENT_SCALAR_REPLAY_ALL_TABLES',
              'independence': 'numpy+stdlib only; no producer module imported; own exact '
                              'signed-rank DP, Holm, one-at-a-time bootstrap draws',
              'raw_branches': len(records), 'states': len(sids), 'instances': len(instances),
              'raw_file_hashes_verified': len(files),
              'paired_contrasts_reconstructed': n_reconstructed,
              'max_abs_errors': dict(errors),
              'bootstrap_seed': seed, 'bootstrap_draws': draws,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = OUT / 'verification_replay.json'
    if out.exists():
        raise ValueError('Never overwrite replay audit')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

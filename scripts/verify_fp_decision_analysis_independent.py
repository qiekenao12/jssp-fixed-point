#!/usr/bin/env python3
"""Replay the independent final-analysis review from raw data and frozen models.

This faithfully collects the independent inline review arithmetic into a durable
script. It does not import the collector, model implementation, feature builder,
or metric helpers. It refits no model and executes no search. The separate
verify_fp_decision_statistics.py audit covers exact signed-rank and Holm; this
script independently reconstructs all second-solver scores, action choices,
repeat summaries, primary paired instance effects and bootstrap intervals.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'projects/analysis/fp_decision_study'
ACTIONS = ('CB_BEST_DESCENT', 'CB_FIRST_DESCENT', 'CB_TABU')
ARMS = ('A', 'B', 'C', 'D', 'E', 'D_random')


def read(path):
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as handle:
        return json.load(handle)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review():
    lock = read(ROOT / 'projects/experiments/fp_decision_study/final_lock.json')
    for path, expected in lock['hashes'].items():
        assert digest(ROOT / path) == expected, path
    bundle = read(ROOT / 'projects/experiments/fp_decision_study/tabu_models.json')
    maps = bundle['maps']
    records, raw_files = [], []
    for path in sorted((ROOT / 'results/fp_decision_study/tabu/final').glob('*.json.gz')):
        records.extend(read(path)['records'])
        raw_files.append((str(path.relative_to(ROOT)), digest(path)))
    assert len(records) == 12000
    assert not (set(bundle['training_instances']) & {r['instance'] for r in records})
    with open(OUT / 'tabu_predictions.csv') as handle:
        exported = {(r['state_id'], r['action']): r for r in csv.DictReader(handle)}
    groups = defaultdict(list)
    for row in records:
        groups[row['state_id']].append(row)
    errors, choices, changed = defaultdict(float), defaultdict(Counter), Counter()
    rows, repeat_seeds = [], []
    noise = defaultdict(list)
    for state_id, state_records in sorted(groups.items()):
        first = state_records[0]
        x, observed = np.asarray(first['start_x']), first['observed']
        assert all(r['start_x'] == first['start_x'] and r['observed'] == observed for r in state_records)
        a_vector = np.asarray(observed['ordinary'] + observed['history'])
        b_vector = np.r_[a_vector, observed['structure'],
                         [x[i] * x[j] for i in range(4) for j in range(i, 4)]]
        endpoints = np.stack([np.asarray(maps[a]['A']).dot(x) + maps[a]['c'] for a in ACTIONS])
        delta = endpoints - x
        differences = np.stack([endpoints[0] - endpoints[1], endpoints[0] - endpoints[2],
                                endpoints[1] - endpoints[2]])
        rng = np.random.default_rng(2026102501)
        random_matrices = rng.normal(size=(6, 4, 4))
        random_offsets = rng.normal(size=(6, 4))
        extras = {
            'C': endpoints.flatten(),
            'D': np.r_[np.sqrt((delta ** 2).sum(1)), np.sqrt((differences ** 2).sum(1))],
            'E': np.concatenate([delta.flatten(), (delta ** 2).flatten(),
                                 differences.flatten(), (differences ** 2).flatten()]),
            'D_random': np.sqrt(((random_matrices @ x + random_offsets) ** 2).sum(1)),
        }
        branch = {(r['action'], r['repeat']): r for r in state_records}
        assert len(branch) == len(state_records) == 15
        assert set(branch) == {(a, rep) for a in ACTIONS for rep in range(5)}
        rewards = np.asarray([[(branch[a, rep]['start_makespan'] - branch[a, rep]['end_makespan']) /
                               first['s0'] for rep in range(5)] for a in ACTIONS])
        scores = {}
        for action_index, action in enumerate(ACTIONS):
            for repeat in range(5):
                row = branch[action, repeat]
                assert abs(rewards[action_index, repeat] - row['gain']) < 1e-14
                repeat_seeds.append(row['seed'])
            sd = float(rewards[action_index].std(ddof=1))
            noise[action].append(sd)
            ex = exported[state_id, action]
            assert abs(float(ex['gain_sd5']) - sd) < 1e-12
            assert abs(float(ex['gain_se5']) - sd / math.sqrt(5)) < 1e-12
            for arm in ARMS:
                vector = a_vector if arm == 'A' else b_vector if arm == 'B' else np.r_[b_vector, extras[arm]]
                indicator = np.asarray([int(action == a) for a in ACTIONS])
                interacted = np.r_[indicator, *[vector if action == a else np.zeros(len(vector)) for a in ACTIONS]]
                for target in ('gain', 'probability'):
                    model = bundle['models'][arm][target]
                    standardized = (interacted - model['center']) / model['scale']
                    latent = model['weights'][0] + standardized.dot(model['weights'][1:])
                    prediction = float(latent) if target == 'gain' else 1 / (1 + math.exp(-float(latent)))
                    key = target + '_' + arm
                    difference = abs(prediction - float(ex[key]))
                    errors[key] = max(errors[key], difference)
                    assert difference < 1e-12, (state_id, action, key, difference)
                    scores.setdefault(key, []).append(prediction)
        result = {'instance': first['instance'], 'state_id': state_id, 'run': first['run']}
        means = rewards.mean(1)
        for key, values in scores.items():
            selected = int(np.argmax(values))
            choices[key][ACTIONS[selected]] += 1
            result[key + '_value'] = float(means[selected])
            result[key + '_action'] = ACTIONS[selected]
        for comparator in ('B', 'E', 'D_random'):
            changed[comparator] += result['gain_D_action'] != result['gain_' + comparator + '_action']
        fit = rewards[:, :2].mean(1)
        evaluation = rewards[:, 2:].mean(1)
        selected = int(np.argmax(fit))
        result.update(optimism=float(fit[selected] - evaluation[selected]),
                      independent_winner_minus_tabu=float(evaluation[selected] - evaluation[2]),
                      apparent_winner_minus_tabu=float(fit[selected] - fit[2]),
                      winner_agreement=int(selected == np.argmax(evaluation)),
                      all_zero=int(means.max() == 0))
        rows.append(result)
    assert len(set(repeat_seeds)) == 12000
    primary = read(OUT / 'primary_results.json')['primary_family']
    summary = read(OUT / 'tabu_final_analysis.json')
    instances = sorted({r['instance'] for r in rows})
    assert len(instances) == 20 and len(rows) == 800
    checks, sizes = [], {}
    for comparator in ('B', 'E'):
        effect = {i: float(np.mean([r['gain_D_value'] - r['gain_' + comparator + '_value']
                                   for r in rows if r['instance'] == i])) for i in instances}
        archived = next(q for q in primary if q['contrast'] == 'tabu_gain_D_minus_' + comparator)
        assert max(abs(effect[k] - archived['instance_values'][k]) for k in effect) < 1e-12
        # Draw one vector at a time, independently of the producer's batched construction.
        rng = np.random.default_rng(2026102502)
        values = list(effect.values())
        samples = [sum(values[int(j)] for j in rng.integers(20, size=20)) / 20 for _ in range(20000)]
        interval = np.quantile(samples, [.025, .975]).tolist()
        assert max(abs(a - b) for a, b in zip(interval, archived['ci95'])) < 1e-12
        per_size = {s: float(np.mean([v for i, v in effect.items() if '_' + s + '_' in i]))
                    for s in ('6x6', '10x5', '10x10', '15x10')}
        six_sum = sum(v for i, v in effect.items() if '_6x6_' in i)
        total = sum(effect.values())
        sizes[comparator] = {
            'size_means': per_size,
            'mean_excluding_6x6': float(np.mean([v for i, v in effect.items() if '_6x6_' not in i])),
            '6x6_signed_contribution_fraction': six_sum / total if total else None,
            'interpretation': 'post-collection descriptive fixed-size stratification; not a new hypothesis test',
        }
        checks.append({'contrast': archived['contrast'], 'mean': float(np.mean(values)),
                       'independent_ci95': interval, 'matches': True})
    assert {k: dict(v) for k, v in choices.items()} == summary['decision_counts']
    assert dict(changed) == summary['changed_decisions']

    def aggregate(key):
        return float(np.mean([np.mean([r[key] for r in rows if r['instance'] == i]) for i in instances]))

    assert abs(aggregate('optimism') - summary['auxiliary_contrasts'][0]['mean']) < 1e-12
    assert abs(aggregate('independent_winner_minus_tabu') - summary['auxiliary_contrasts'][1]['mean']) < 1e-12
    ap_defined = {a: sum(any(r['improved'] for r in records if r['instance'] == i and r['action'] == a)
                         for i in instances) for a in ACTIONS}
    findings = [
        {'id': 'R1', 'severity': 'disclosure', 'finding':
         'Critical-block AP is None with no positives and omits those cells from its macro denominator; '
         'historical genetic AP uses zero. All three final native actions have positives in20/20 instances, '
         'so no denominator was removed here. Do not equate absolute cross-solver AP estimands.'},
        {'id': 'R2', 'severity': 'interpretation', 'finding':
         'D-B negative effect concentrates in6x6. Leave-one-instance negativity is not size-independent harm; '
         'size contrasts are descriptive and are not new confirmatory tests.'},
        {'id': 'R3', 'severity': 'interpretation', 'finding':
         'C/E fixed-affine-map endpoint/displacement/difference coordinates and coordinate squares are '
         'degree<=2 polynomials already in B rawx+all10quadratics with intercept. Greater width is not '
         'greater fixed-map unregularized function span. Ridge, scaling and instance-varying OOF maps '
         'alter the learning problem. D_random matches D width, not all function classes.'},
        {'id': 'R4', 'severity': 'interpretation', 'finding':
         'The independent-repeat sample winner is a noisy fixed comparator, not trueoracle. Its gap '
         'against another policy can be negative. V(S),V(Z),irreducible information loss and online value are unidentified.'},
        {'id': 'R5', 'severity': 'scope', 'finding':
         'Crossfit training-map geometry differs from full-development final-map geometry. '
         'Observed failure cannot uniquely identify an FP-information cause.'},
    ]
    reviewed = ['scripts/analyze_fp_decision_study.py', 'scripts/fp_decision_models.py',
                'scripts/verify_fp_decision_statistics.py',
                'projects/experiments/fp_decision_study/final_lock.json',
                'projects/experiments/fp_decision_study/tabu_models.json',
                'projects/docs/fp_decision_empirical_manuscript.md',
                'projects/analysis/fp_decision_study/primary_results.json']
    return {
        'status': 'PASS_FROZEN_SCORE_RAW_REWARD_BOOTSTRAP_AND_SCOPE_REVIEW',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'independence': 'No collector/model/scorer/feature/metric helper imported. No fitting or search.',
        'raw_tabu_records': len(records), 'saved_states': len(groups), 'instances': 20,
        'unique_branch_seeds': len(set(repeat_seeds)), 'frozen_source_hashes_checked': len(lock['hashes']),
        'training_final_overlap': [], 'score_reconstruction_max_abs_error': dict(errors),
        'independent_primary_ci_reconstruction': checks, 'changed_decisions': dict(changed),
        'decision_counts': {k: dict(v) for k, v in choices.items()},
        'oracle': {'same_sample_advantage_over_fixed_tabu': aggregate('apparent_winner_minus_tabu'),
                   'independent_repeat_advantage_over_fixed_tabu': aggregate('independent_winner_minus_tabu'),
                   'raw_selection_optimism': aggregate('optimism'),
                   'winner_agreement': aggregate('winner_agreement'),
                   'all_zero_state_fraction': aggregate('all_zero'), 'true_oracle_identified': False},
        'within_state_sd_mean5': {a: float(np.mean(v)) for a, v in noise.items()},
        'gain_sd5_se5_reconstruction': 'PASS_ALL_2400_STATE_ACTION_PAIRS',
        'ap_defined_instances': ap_defined, 'ap_total_instances': 20, 'size_concentration': sizes,
        'statistical_unit': '800states=20instances x20seeds x2opportunities; complete balanced grid.',
        'interval_scope': 'Unstratified20000-instance-cluster percentile draws,seed2026102502; '
                          'conditional on trained models/observed seed means; marginal, not simultaneous.',
        'findings': findings, 'hard_statistical_or_leakage_failure_found': False,
        'not_repeated': ['Full DAG checks reused from separate raw audits.',
                         'Exact sign-rank/Holm and original-solver scalar checks reused from verify_fp_decision_statistics.py.',
                         'No new refit, model selection, original-solver coefficient replay or literature audit.'],
        'reviewed_sha256': {p: digest(ROOT / p) for p in reviewed},
        'raw_file_sha256': dict(raw_files), 'script_sha256': digest(__file__),
        'provenance': 'Durable replay of the2026-10-26inline independent analysis; numerical scope unchanged. '
                      'Explanatory text includes the separately identified fixed-map function-span caveat.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='projects/analysis/fp_decision_study/independent_analysis_review_replay.json')
    args = parser.parse_args()
    path = Path(args.output)
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        raise FileExistsError(f'Preserving existing evidence: {path}')
    report = review()
    original_path = OUT / 'independent_analysis_review.json'
    if original_path.exists():
        original = read(original_path)
        compared = ['raw_tabu_records', 'saved_states', 'instances', 'unique_branch_seeds',
                    'score_reconstruction_max_abs_error', 'independent_primary_ci_reconstruction',
                    'changed_decisions', 'decision_counts', 'oracle', 'within_state_sd_mean5',
                    'gain_sd5_se5_reconstruction', 'ap_defined_instances', 'size_concentration']
        for field in compared:
            assert report[field] == original[field], f'Inline-review replay mismatch: {field}'
        report['inline_original_replay'] = {'status': 'EXACT_NUMERICAL_MATCH',
                                           'source_sha256': digest(original_path), 'fields': compared}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(path),
                      'inline_original_replay': report.get('inline_original_replay', {}).get('status')}, indent=2))


if __name__ == '__main__':
    main()

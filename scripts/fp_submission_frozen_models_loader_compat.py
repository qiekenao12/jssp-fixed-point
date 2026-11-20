#!/usr/bin/env python3
"""Freeze existing development-only models and evaluate one new confirmation.

This adds serialization, numerical checks and outcome-independent inference,
not a new FP representation, scoring architecture or search controller.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from scripts.check_fp_scorer_convergence import ConvergedLogistic, stable_sigmoid
from scripts.compare_fp_decision_features import (
    ARMS, L2, RIDGE, SEED as FEATURE_SEED, WIDTHS, RandomStateNorms,
    SignalCache, attach_past_state_changes, canonical_sort, feature_matrix,
    key, load_audited_development, trajectory_metrics,
)
from analyze_paper_evidence import hierarchical_ci, holm, signed_rank
from tools.action_fp import ACTIONS, WEIGHTS, _Affine, collapsed_transitions, state_vector, weighted_norm

BOOTSTRAP_SEED = 20261021
SOURCES = ('scripts/fp_submission_frozen_models.py',
           'scripts/check_fp_scorer_convergence.py',
           'scripts/compare_fp_decision_features.py',
           'analyze_fp_action_diagnostics.py', 'src/tools/action_fp.py',
           'analyze_paper_evidence.py')
FIT_INPUTS = ('projects/analysis/fp_action_development_independent_audit.json',
              'projects/analysis/fp_submission_convergence_results.json',
              'projects/analysis/fp_submission_convergence_independent_audit.json',
              'projects/experiments/fp_v2_manifest.json')
MODEL_SETTINGS = dict(arms=list(ARMS), ridge=RIDGE, l2=L2,
    gradient_inf_tolerance=1e-7, max_newton_steps=200, max_backtracks=50,
    armijo=1e-4, finite_difference_step=1e-4,
    finite_difference_absolute_tolerance=2e-7,
    random_norm_seed=FEATURE_SEED, negative_control_seed=20261018,
    history_lags_generations=[20, 40], train_instances=16,
    train_trajectories=320, train_branches=17280)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_new(path, payload):
    path = Path(path)
    if path.exists():
        raise ValueError(f'Never overwrite an existing model or result: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def fit_protocol_template():
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'DRAFT_NOT_FROZEN_DO_NOT_FIT', 'component': 'full_development_model_fit',
        'scope': 'Only original16 development instances. No locked/exposed-validation outcome may be read.',
        'model_settings': MODEL_SETTINGS,
        'source_sha256': {p: sha(ROOT/p) for p in SOURCES},
        'input_sha256': {p: sha(ROOT/p) for p in FIT_INPUTS},
        'failure_rule': 'All7 identical independent numerical checks must pass. Retain failed receipt and do not unlock confirmation. No tuning.'}


def assert_frozen_protocol(path, component):
    protocol = json.loads(Path(path).read_text())
    if ('FROZEN' not in protocol.get('status', '') or 'NOT_FROZEN' in protocol['status']
            or protocol.get('component') != component):
        raise ValueError('Correct independently frozen component protocol required')
    for relative, expected in protocol['source_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Frozen executable changed: {relative}')
    for relative, expected in protocol['input_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Frozen input changed: {relative}')
    if component == 'full_development_model_fit' and protocol['model_settings'] != MODEL_SETTINGS:
        raise ValueError('Existing frozen architecture or solver settings changed')
    return protocol


def serialize_affine(model):
    return {'A': model.A.tolist(), 'c': model.c.tolist(),
            'center': model.center.tolist(), 'scale': model.scale.tolist(),
            'support_radius': model.support_radius}


def serialize_fp(model):
    return {'training_instances': list(model.training_instances),
            'summary': model.summary,
            'maps': {a: serialize_affine(m) for a, m in model.models.items()}}


def frozen_fp_features(serialized, x):
    x = state_vector(x)
    predictions, residuals, reliability, errors, ready = {}, [], [], [], []
    for action in ACTIONS:
        summary = serialized['summary'][action]
        available = bool(summary['model_ready'])
        model = serialized['maps'].get(action)
        prediction = (np.asarray(model['A']) @ x + np.asarray(model['c'])) if available else x.copy()
        predictions[action] = prediction
        residuals.append(weighted_norm(prediction-x))
        errors.append(summary['prediction_error'] or 0.)
        support = (max(0., float(np.linalg.norm((x-np.asarray(model['center']))
                           /np.asarray(model['scale'])))-model['support_radius']) if available else 0.)
        reliability.append(float(summary['reliability'])/(1.+support))
        ready.append(float(available))
    differences = [weighted_norm(predictions[a]-predictions['GA'])
                   if ready[0] and ready[j] else 0. for j, a in enumerate(ACTIONS) if j]
    return np.r_[residuals, differences, errors, reliability, ready]


class FrozenNorms:
    def __init__(self, payload):
        self.center = np.asarray(payload['center'])
        self.scale = np.asarray(payload['scale'])
        self.matrix = np.asarray(payload['matrix'])
        self.offset = np.asarray(payload['offset'])

    def values(self, x):
        z = (state_vector(x)-self.center)/self.scale
        images = np.einsum('jkl,l->jk', self.matrix, z)+self.offset
        return np.sqrt(np.sum(WEIGHTS*images**2, axis=1))


def model_probabilities(classifier, x):
    weights = np.asarray(classifier['weights'])
    center, scale = np.asarray(classifier['center']), np.asarray(classifier['scale'])
    return stable_sigmoid(weights[0]+((np.asarray(x)-center)/scale) @ weights[1:])


def fit_models(protocol_path, bundle_path, receipt_path):
    assert_frozen_protocol(protocol_path, 'full_development_model_fit')
    if Path(bundle_path).exists() or Path(receipt_path).exists():
        raise ValueError('Never overwrite model/bundle receipts')
    rows, audit = load_audited_development(ROOT/'results/fp_v2_development',
        ROOT/'projects/analysis/fp_action_development_independent_audit.json')
    # All feature construction is from development only, with the original
    # nested grouping. Neither path arguments nor this function accept test.
    cache = SignalCache(rows)
    actual, control = np.empty((len(rows), 14)), np.empty((len(rows), 14))
    instances = sorted({r['instance'] for r in rows})
    for instance in instances:
        ids = [i for i, row in enumerate(rows) if row['instance'] == instance]
        local, negative = cache.features({instance}, [rows[i] for i in ids])
        actual[ids], control[ids] = local, negative
    norms = RandomStateNorms(rows)
    labels = np.asarray([float(r['improved']) for r in rows])
    classifiers = {}
    for arm in ARMS:
        x = feature_matrix(rows, arm, actual, control, norms)
        classifier = ConvergedLogistic(l2=L2).fit(x, labels)
        classifiers[arm] = classifier.serialization()
        classifiers[arm]['parameters'] = len(classifier.weights)
        print(json.dumps({'full_development_arm': arm, 'converged': classifier.converged,
            'independent_gradient_inf': classifier.gradient_inf,
            'newton_steps': classifier.iterations_run}), flush=True)
    full_fp, _ = cache.model(set())
    fp = serialize_fp(full_fp)
    pooled = _Affine(collapsed_transitions(rows), RIDGE)
    # Independent serialization identity checks use training x only, with no
    # new outcome data. They do not constitute predictive generalization tests.
    serialization_error = 0.
    unique_states = {key(r): r['start_x'] for r in rows}
    for x in unique_states.values():
        f, q, _ = full_fp.features(x)
        serialization_error = max(serialization_error,
                                  float(np.max(np.abs(np.r_[f, q]-frozen_fp_features(fp, x)))))
    norm_serialization_error = max(float(np.max(np.abs(norms.values(x)-FrozenNorms(norms.summary()).values(x))))
                                   for x in unique_states.values())
    failures = {a: c['failure'] for a, c in classifiers.items() if not c['converged']}
    if serialization_error > 1e-12 or norm_serialization_error > 1e-12:
        failures['serialization'] = 'FEATURE_SERIALIZATION_MISMATCH'
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS_ALL7_FULL_DEVELOPMENT_FITS' if not failures else 'FAILED_DO_NOT_CONFIRM',
        'protocol_sha256': sha(protocol_path), 'data_audit': audit,
        'training_instances': instances, 'final_outcomes_read': 0,
        'classifiers': {a: {k: v for k, v in c.items() if k not in ('weights', 'center', 'scale', 'path')}
                        for a, c in classifiers.items()},
        'fp_serialization_max_abs_error': serialization_error,
        'N_serialization_max_abs_error': norm_serialization_error,
        'inner_fp_training_exclusions': [{'excluded_instances': list(excluded),
            'training_instances': list(model.training_instances),
            'all_targets_excluded': not bool(set(excluded)&set(model.training_instances))}
            for excluded, model in sorted(cache.models.items()) if excluded],
        'failure_records': failures}
    if not failures:
        bundle = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'FROZEN_FULL_DEVELOPMENT_ONLY_MODELS',
            'protocol_sha256': sha(protocol_path), 'settings': MODEL_SETTINGS,
            'training_instances': instances, 'classifiers': classifiers,
            'action_fp': fp, 'pooled_atomic_affine': serialize_affine(pooled),
            'random_norms': norms.summary(),
            # Retain the exact original row-wise sampling measure for P.
            'negative_control_training_pool_x': [r['start_x'] for r in rows],
            'negative_control_seed': 20261018,
            'data_audit': audit, 'training_outcome_role': 'Original development16 only',
            'source_sha256': {p: sha(ROOT/p) for p in SOURCES}}
        write_json_new(bundle_path, bundle)
        receipt['bundle_sha256'] = sha(bundle_path)
    write_json_new(receipt_path, receipt)
    print(json.dumps({'receipt': str(receipt_path), 'status': receipt['status'],
                      'bundle': str(bundle_path) if not failures else None}), flush=True)


def load_confirmation(directory, protocol_path, expected_model_hash):
    """Strict complete-grid loader; it cannot run or modify a searcher."""
    directory = Path(directory)
    protocol = json.loads(Path(protocol_path).read_text())
    if 'FROZEN' not in protocol.get('status', '') or 'NOT_FROZEN' in protocol['status']:
        raise ValueError('Final collection protocol must be frozen')
    if protocol.get('model_bundle_sha256') != expected_model_hash:
        raise ValueError('Final collection was not frozen against this model bundle')
    for relative, expected in protocol.get('sources', {}).items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Frozen final collection source changed: {relative}')
    instances, runs = protocol['instances'], int(protocol['runs'])
    if len(instances) != 20 or runs != 20:
        raise ValueError('The complete final20 x20 trajectories are required')
    files = {f'{i}_{j:02d}.json' for i in instances for j in range(runs)}
    actual = {p.name for p in directory.glob('*.json') if p.name != 'protocol.json'}
    if actual != files:
        raise ValueError('Final trajectory grid is incomplete or has extra files')
    atomic, fair, source_hashes = [], [], {}
    for filename in sorted(files):
        path = directory/filename
        payload = json.loads(path.read_text())
        if payload.get('protocol_hash') != protocol.get('protocol_hash', sha(protocol_path)):
            raise ValueError('Final trajectory protocol identity mismatch')
        if payload['instance'] not in instances or payload['run'] not in range(runs):
            raise ValueError('Undeclared final trajectory identity')
        if filename != f"{payload['instance']}_{payload['run']:02d}.json":
            raise ValueError('Final payload identity differs from filename')
        snapshots = {int(s['generation']): s for s in payload['snapshots']}
        if set(snapshots) != set(range(20, 181, 20)):
            raise ValueError('Incomplete saved final main states')
        for field, generations, output in (('records', range(20, 181, 20), atomic),
                                           ('equal_work_records', (40, 100, 160), fair)):
            records = payload[field]
            expected = {(g, a, repeat) for g in generations for a in ACTIONS for repeat in range(5)}
            observed = [(int(r['generation']), r['action'], int(r['repeat'])) for r in records]
            if set(observed) != expected or len(observed) != len(expected):
                raise ValueError(f'Incomplete final state/action/repeat grid: {filename}/{field}')
            for row in records:
                generation = int(row['generation'])
                if (row['instance'] != payload['instance'] or row['run'] != payload['run']
                        or row['trajectory_id'] != f"{payload['instance']}:{payload['run']}"
                        or row['state_id'] != f"{payload['instance']}:{payload['run']}:{generation}"
                        or row['state_fingerprint'] != snapshots[generation]['fingerprint']):
                    raise ValueError('Final branch differs from its saved main-state identity')
                value = (row['start_makespan']-row['end_makespan'])/row['s0']
                if (abs(value-row['gain']) > 1e-10 or row['gain'] < -1e-10
                        or bool(row['improved']) != bool(row['gain'] > 1e-12)):
                    raise ValueError('Final gain/outcome fields are inconsistent')
                if (row['queries'] > (500 if field == 'records' else 480)
                        or (sum(row['phase_queries'].values()) if field == 'records' else sum(call['queries'] for call in row['primitive_calls'])) != row['queries']):
                    raise ValueError('Final action cost accounting/cap mismatch')
                if field == 'records' and row['window'] != {
                        'unit': 'atomic_action_application', 'span': 1,
                        'action_sequence': [row['action']]}:
                    raise ValueError('Atomic final transition contains mixed or unequal actions')
            local = attach_past_state_changes({**payload, 'records': records})
            output.extend(local)
        archive = payload.get('population_archive')
        if archive and sha(ROOT/archive) != payload.get('population_archive_sha256'):
            raise ValueError('Final starting population archive hash mismatch')
        source_hashes[filename] = sha(path)
    return canonical_sort(atomic), canonical_sort(fair), {'trajectory_sha256': source_hashes,
        'collection_protocol_sha256': sha(protocol_path), 'instances': instances,
        'trajectories': len(files), 'atomic_records': len(atomic), 'fair480_records': len(fair)}


def predict_atomic(bundle, rows):
    if {r['instance'] for r in rows}&set(bundle['training_instances']):
        raise ValueError('Final evaluation instance overlaps development fitting')
    norms = FrozenNorms(bundle['random_norms'])
    pool = bundle['negative_control_training_pool_x']
    rng = np.random.default_rng(bundle['negative_control_seed'])
    state_features, state_control = {}, {}
    for row in rows:
        state = key(row)
        if state not in state_features:
            state_features[state] = frozen_fp_features(bundle['action_fp'], row['start_x'])
            state_control[state] = frozen_fp_features(bundle['action_fp'], pool[int(rng.integers(len(pool)))])
    actual = np.stack([state_features[key(r)] for r in rows])
    control = np.stack([state_control[key(r)] for r in rows])
    return {a: model_probabilities(bundle['classifiers'][a],
            feature_matrix(rows, a, actual, control, norms)) for a in ARMS}


def summary_metrics(rows, probabilities):
    trajectories, states = trajectory_metrics(rows, probabilities)
    grouped = defaultdict(list)
    for row in trajectories:
        grouped[row['instance']].append(row)
    metrics = ('ap', 'brier', 'selected_gain', 'regret', 'strict_correct_rate',
               'action_rank', 'choice_change_vs_C', 'ls_over_vnd_margin_ap')
    macro = {a: {m: float(np.mean([np.mean([r['variants'][a][m] for r in group])
                                  for group in grouped.values()])) for m in metrics} for a in ARMS}
    action_counts = {a: {action: sum(r['variants'][a]['chosen_action_counts'][action] for r in trajectories)
                         for action in ACTIONS} for a in ARMS}
    per_instance = [{'instance': instance,
        'variants': {a: {m: float(np.mean([r['variants'][a][m] for r in group])) for m in metrics} for a in ARMS},
        'fixed_action_gain': {a: float(np.mean([r['fixed_action_gain'][a] for r in group])) for a in ACTIONS},
        'oracle_gain': float(np.mean([r['oracle_gain'] for r in group]))}
        for instance, group in sorted(grouped.items())]
    return {'macro_instance_trajectory_metrics': macro, 'action_counts': action_counts,
        'trajectory_metrics': trajectories, 'state_decisions': states,
        'per_instance_metrics': per_instance,
        'fixed_action_gain': {a: float(np.mean([row['fixed_action_gain'][a] for row in per_instance])) for a in ACTIONS},
        'oracle_gain': float(np.mean([row['oracle_gain'] for row in per_instance]))}


def comparison(summary, window, comparator, metric):
    grouped = defaultdict(list)
    sign = -1 if metric == 'brier' else 1
    for row in summary['trajectory_metrics']:
        grouped[row['instance']].append(sign*(row['variants']['E'][metric]-row['variants'][comparator][metric]))
    matrix = [grouped[i] for i in sorted(grouped)]
    means = [float(np.mean(v)) for v in matrix]
    return {'window': window, 'candidate': 'E', 'comparator': comparator, 'metric': metric,
        'positive_favors_E': True, 'mean_difference': float(np.mean(means)),
        'hierarchical_bootstrap_95ci': hierarchical_ci(matrix, repetitions=10000, seed=BOOTSTRAP_SEED),
        'instance_differences': dict(zip(sorted(grouped), means)),
        'leave_one_instance_out_mean_difference': {name: float(np.mean([v for other, v in zip(sorted(grouped), means) if other != name])) for name in sorted(grouped)},
        'instance_win_tie_loss': [sum(v > 1e-12 for v in means), sum(abs(v) <= 1e-12 for v in means), sum(v < -1e-12 for v in means)],
        **signed_rank(means)}


def dynamics_evaluation(bundle, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[*key(row), row['action']].append(row)
    by_trajectory = defaultdict(lambda: defaultdict(list))
    for (instance, trajectory, _, action), records in sorted(groups.items()):
        x = state_vector(records[0]['start_x'])
        y = np.mean([state_vector(r['end_x']) for r in records], axis=0)
        model = bundle['action_fp']['maps'][action]
        prediction = np.asarray(model['A']) @ x+np.asarray(model['c'])
        pooled = bundle['pooled_atomic_affine']
        pooled_prediction = np.asarray(pooled['A']) @ x+np.asarray(pooled['c'])
        for variant, value in [('conditioned', prediction), ('pooled', pooled_prediction), ('identity', x)]:
            by_trajectory[instance, trajectory][action, variant].append(weighted_norm(value-y))
    trajectory_errors = [{'instance': i, 'trajectory_id': t,
        'errors': {a: {v: float(np.mean(values[a, v])) for v in ('conditioned', 'pooled', 'identity')}
                   for a in ACTIONS}} for (i, t), values in sorted(by_trajectory.items())]
    by_instance = defaultdict(list)
    for row in trajectory_errors:
        by_instance[row['instance']].append(row)
    macro = {a: {v: float(np.mean([np.mean([r['errors'][a][v] for r in group]) for group in by_instance.values()]))
                 for v in ('conditioned', 'pooled', 'identity')} for a in ACTIONS}
    comparisons = []
    for action in ACTIONS:
        for baseline in ('pooled', 'identity'):
            matrix = [[r['errors'][action][baseline]-r['errors'][action]['conditioned'] for r in group]
                      for _, group in sorted(by_instance.items())]
            means = [float(np.mean(v)) for v in matrix]
            comparisons.append({'action': action, 'comparator': baseline, 'positive_favors_conditioned': True,
                'mean_error_reduction': float(np.mean(means)),
                'hierarchical_bootstrap_95ci': hierarchical_ci(matrix, repetitions=10000, seed=BOOTSTRAP_SEED),
                'instance_differences': dict(zip(sorted(by_instance), means)), **signed_rank(means)})
    holm(comparisons)
    return {'macro_weighted_state_error': macro, 'trajectory_errors': trajectory_errors,
            'comparisons': comparisons, 'family': '6 action-conditioned vs pooled/identity across3actions'}


def predict_and_analyze(protocol_path, bundle_path, collection_path, directory, output):
    protocol = assert_frozen_protocol(protocol_path, 'final_confirmation_analysis')
    if Path(output).exists():
        raise ValueError('Final confirmation is one shot; never overwrite')
    bundle = json.loads(Path(bundle_path).read_text())
    if bundle['status'] != 'FROZEN_FULL_DEVELOPMENT_ONLY_MODELS':
        raise ValueError('Only independently checked full-development frozen models accepted')
    if protocol['model_bundle_sha256'] != sha(bundle_path):
        raise ValueError('Analysis protocol model hash mismatch')
    settings = protocol['statistics']
    if settings != {'bootstrap_repetitions': 10000, 'bootstrap_seed': BOOTSTRAP_SEED,
                    'primary_family_size': 12, 'dynamics_family_size': 6}:
        raise ValueError('Final frozen statistics changed')
    atomic, fair, audit = load_confirmation(directory, collection_path, sha(bundle_path))
    atomic_probabilities = predict_atomic(bundle, atomic)
    # Probability scores are identical at the same saved state. The480-window
    # stress test does not fit or calibrate any separate target/window model.
    scores = {}
    start_states = {}
    for index, row in enumerate(atomic):
        identity = (*key(row), row['action'])
        value = {a: float(atomic_probabilities[a][index]) for a in ARMS}
        if identity in scores and scores[identity] != value:
            raise ValueError('Repeats produced different predecision scores')
        scores[identity] = value
        start_states[key(row)] = row['start_x']
    for row in fair:
        if not np.array_equal(state_vector(row['start_x']), state_vector(start_states[key(row)])):
            raise ValueError('Atomic/fair480 starting states differ')
    fair_probabilities = {a: np.asarray([scores[*key(r), r['action']][a] for r in fair]) for a in ARMS}
    atomic_summary, fair_summary = summary_metrics(atomic, atomic_probabilities), summary_metrics(fair, fair_probabilities)
    comparisons = [comparison(atomic_summary, 'atomic', c, m) for c in ('C', 'N', 'P')
                   for m in ('ap', 'brier', 'selected_gain')]
    comparisons += [comparison(fair_summary, 'fair480', c, 'selected_gain') for c in ('C', 'N', 'P')]
    holm(comparisons)
    dynamics = dynamics_evaluation(bundle, atomic)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'FINAL_FROZEN_OFFLINE_CONFIRMATION_COMPLETE',
        'online_M3': 'NOT_IMPLEMENTED_NOT_RUN', 'model_refits_during_confirmation': 0,
        'audit': audit, 'model_bundle_sha256': sha(bundle_path),
        'analysis_protocol_sha256': sha(protocol_path),
        'atomic_summary': atomic_summary, 'fair480_summary': fair_summary,
        'primary_comparisons': comparisons, 'dynamics': dynamics,
        'primary_family': '12 Holm: atomic EvsC/N/P times AP/Brier/gain9 + fair480 EvsC/N/P gain3',
        'notes': ['All20 instances and400 trajectories included; no outcome-based filtering.',
            'Fair480 probabilities are frozen atomic-window scores, an explicit window-transfer stress test.',
            'Predictions/selected branch returns are isolated counterfactual decisions, not online closed-loop Agent trajectories.',
            'Both family confidence intervals are marginal10000 paired instance-then-trajectory bootstrap intervals, not simultaneous intervals.'],
        'predictions': {window: [{**{k: r[k] for k in ('instance', 'trajectory_id', 'state_id', 'generation', 'action', 'repeat', 'gain', 'improved')},
            'probabilities': {a: float(probs[a][i]) for a in ARMS}} for i, r in enumerate(rows)]
            for window, rows, probs in [('atomic', atomic, atomic_probabilities), ('fair480', fair, fair_probabilities)]}}
    write_json_new(output, result)
    print(json.dumps({'status': result['status'], 'output': str(output), 'atomic_records': len(atomic), 'fair480_records': len(fair)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    draft = commands.add_parser('fit-protocol-template')
    draft.add_argument('--output', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--protocol', required=True)
    fit.add_argument('--bundle', required=True)
    fit.add_argument('--receipt', required=True)
    predict = commands.add_parser('predict-analyze')
    predict.add_argument('--protocol', required=True)
    predict.add_argument('--bundle', required=True)
    predict.add_argument('--collection-protocol', required=True)
    predict.add_argument('--directory', required=True)
    predict.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'fit-protocol-template':
        write_json_new(args.output, fit_protocol_template())
    elif args.command == 'fit':
        fit_models(args.protocol, args.bundle, args.receipt)
    else:
        predict_and_analyze(args.protocol, args.bundle, args.collection_protocol, args.directory, args.output)


if __name__ == '__main__':
    main()

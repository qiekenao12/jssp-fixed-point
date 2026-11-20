#!/usr/bin/env python3
"""One fixed, development-only representation diagnostic; never runs search.

The protocol must be frozen externally before --protocol is supplied.  Existing
M2 sources/results are read-only.  No validation/test cohort is accepted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from analyze_fp_action_diagnostics import (
    Logistic, average_precision, audit_records, base_features, common_context,
    rank_correlation,
)
from analyze_paper_evidence import hierarchical_ci, holm, signed_rank
from tools.action_fp import ACTIONS, WEIGHTS, ActionFPModel, state_vector

ARMS = ('A', 'B', 'C', 'D', 'E', 'N', 'P')
WIDTHS = dict(A=9, B=24, C=34, D=36, E=48, N=48, P=48)
SEED = 20261019
RIDGE = .1
L2 = 1.
ITERATIONS = 120
LAGS = (20, 40)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return str(row['instance']), str(row['trajectory_id']), str(row['state_id'])


def canonical_sort(rows):
    return sorted(rows, key=lambda r: (*key(r), ACTIONS.index(r['action']), int(r['repeat'])))


def attach_past_state_changes(payload):
    """Only past saved MAIN snapshots, never branch endpoints or future states."""
    snapshots = {int(s['generation']): s for s in payload['snapshots']}
    if len(snapshots) != len(payload['snapshots']):
        raise ValueError('Duplicate snapshot generation')
    result = []
    for source in payload['records']:
        row = dict(source)
        generation = int(row['generation'])
        current = state_vector(row['start_x'])
        if not np.array_equal(current, state_vector(snapshots[generation]['x'])):
            raise ValueError('Current state differs from main snapshot')
        deltas, masks = [], []
        for lag in LAGS:
            previous = snapshots.get(generation - lag)
            if previous is None:
                deltas.extend([0.] * 4)
                masks.append(0.)
            else:
                if int(previous['generation']) >= generation:
                    raise ValueError('History must strictly precede the decision')
                deltas.extend(current - state_vector(previous['x']))
                masks.append(1.)
        row['past_state_changes'] = list(map(float, deltas + masks))
        result.append(row)
    return result


def load_audited_development(directory, audit_path):
    """Use the immutable independent-auditor hashes, not today's core hashes.

    The M2 collector's old source lock correctly differs from today's semantic
    compatibility patch.  This loader never weakens that old collector loader;
    it separately checks the independent receipt's complete data identities.
    """
    directory, audit_path = Path(directory), Path(audit_path)
    audit = json.loads(audit_path.read_text())
    protocol_path = directory / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if audit['status'] != 'PASS_COMPLETE' or audit['split'] != 'development':
        raise ValueError('A completed development-only independent audit is required')
    if protocol['split'] != 'development':
        raise ValueError('Only development data can be fitted by this diagnostic')
    if sha(protocol_path) != audit['protocol_sha256']:
        raise ValueError('Audited collection protocol changed')
    if protocol['protocol_hash'] != audit['protocol_hash']:
        raise ValueError('Collection protocol identity mismatch')
    manifest_path = ROOT / 'projects/experiments/fp_v2_manifest.json'
    if sha(manifest_path) != audit['manifest_sha256']:
        raise ValueError('Audited cohort manifest changed')
    expected_instances = set(audit['instances'])
    if len(expected_instances) != 16 or protocol['runs'] != 20 or audit['runs'] != 20:
        raise ValueError('The fixed full development16 x20 grid is required')
    if set(protocol['instances']) != expected_instances:
        raise ValueError('Collection cohort differs from independent audit')
    expected_files = {f'{name}_{run:02d}.json' for name in expected_instances for run in range(20)}
    if set(audit['trajectory_sha256']) != expected_files:
        raise ValueError('Audit does not cover the full development grid')
    actual_files = {p.name for p in directory.glob('*.json') if p.name != 'protocol.json'}
    if actual_files != expected_files:
        raise ValueError('Raw development trajectory grid has changed')
    rows = []
    for filename in sorted(expected_files):
        path = directory / filename
        if sha(path) != audit['trajectory_sha256'][filename]:
            raise ValueError(f'Audited trajectory changed: {filename}')
        payload = json.loads(path.read_text())
        if payload['protocol_hash'] != audit['protocol_hash']:
            raise ValueError('Trajectory protocol identity mismatch')
        archive = payload['population_archive']
        if (archive not in audit['archive_sha256'] or
                sha(ROOT / archive) != audit['archive_sha256'][archive] or
                payload['population_archive_sha256'] != audit['archive_sha256'][archive]):
            raise ValueError(f'Audited population archive changed: {archive}')
        rows.extend(attach_past_state_changes(payload))
    rows = canonical_sort(rows)
    data_audit = audit_records(rows)
    if data_audit != dict(records=17280, instances=16, trajectories=320,
                          states=2880, complete_action_triples=True):
        raise ValueError('The complete fixed development dataset is required')
    return rows, {'audit_sha256': sha(audit_path), 'protocol_sha256': sha(protocol_path),
                  'manifest_sha256': sha(manifest_path), 'counts': data_audit,
                  'all_raw_and_population_hashes_match_audit': True}


class RandomStateNorms:
    """Fourteen fixed ordinary-x nonlinearities with no action-label fitting."""
    def __init__(self, training):
        unique = {key(r): state_vector(r['start_x']) for r in training}
        values = np.stack(list(unique.values()))
        self.center = values.mean(axis=0)
        self.scale = np.maximum(values.std(axis=0), 1e-6)
        rng = np.random.default_rng(SEED)
        self.matrix = rng.normal(size=(14, 4, 4)) / np.sqrt(4.)
        self.offset = rng.normal(size=(14, 4))

    def values(self, x):
        z = (state_vector(x) - self.center) / self.scale
        images = np.einsum('jkl,l->jk', self.matrix, z) + self.offset
        return np.sqrt(np.sum(WEIGHTS * images**2, axis=1))

    def summary(self):
        return {'seed': SEED, 'center': self.center.tolist(), 'scale': self.scale.tolist(),
                'matrix': self.matrix.tolist(), 'offset': self.offset.tolist(),
                'fit_inputs': 'unique outer-training-state x only; no action/outcome labels'}


class SignalCache:
    """Maps shared between symmetric inner folds; every excluded instance absent."""
    def __init__(self, all_rows):
        self.rows = all_rows
        self.models = {}
        self.pools = {}

    def model(self, excluded):
        excluded = tuple(sorted(excluded))
        if excluded not in self.models:
            train = [r for r in self.rows if r['instance'] not in excluded]
            if {r['instance'] for r in train} & set(excluded):
                raise ValueError('Excluded target instance leaked into FP map')
            model = ActionFPModel(ridge=RIDGE).fit(train)
            self.models[excluded], self.pools[excluded] = model, train
        return self.models[excluded], self.pools[excluded]

    def features(self, excluded, evaluation):
        model, pool = self.model(excluded)
        if {r['instance'] for r in evaluation} & set(model.training_instances):
            raise ValueError('FP target instance is present in model training')
        # Same fixed seed and training-pool reassignment as the old M2 control.
        rng = np.random.default_rng(20261018)
        state_features, controls = {}, {}
        for row in evaluation:
            state = key(row)
            if state not in state_features:
                fp, reliability, _ = model.features(row['start_x'])
                state_features[state] = np.r_[fp, reliability]
                source = pool[int(rng.integers(len(pool)))]
                f, q, _ = model.features(source['start_x'])
                controls[state] = np.r_[f, q]
        return (np.stack([state_features[key(r)] for r in evaluation]),
                np.stack([controls[key(r)] for r in evaluation]))

    def nested(self, outer_instance, training, evaluation):
        actual, control = np.empty((len(training), 14)), np.empty((len(training), 14))
        by_instance = defaultdict(list)
        for index, row in enumerate(training):
            by_instance[row['instance']].append(index)
        for instance, ids in sorted(by_instance.items()):
            held = [training[i] for i in ids]
            local, random = self.features({outer_instance, instance}, held)
            actual[ids], control[ids] = local, random
        test, random_test = self.features({outer_instance}, evaluation)
        return actual, control, test, random_test


def feature_matrix(rows, arm, actual, control, norms):
    if arm not in ARMS:
        raise ValueError('Unknown diagnostic representation')
    if len(rows) != len(actual) or len(rows) != len(control):
        raise ValueError('Feature rows are misaligned')
    result = []
    for index, row in enumerate(rows):
        _, b, original = base_features(row)
        features = np.r_[b, common_context(row)]
        if arm != 'A':
            features = np.r_[features, [b[i] * b[j] for i in range(5) for j in range(i, 5)]]
        if arm not in ('A', 'B'):
            history = np.asarray(row['past_state_changes'], float)
            if history.shape != (10,) or not np.all(np.isfinite(history)):
                raise ValueError('Invalid causal lag-state feature vector')
            features = np.r_[features, history]
        if arm == 'D':
            features = np.r_[features, original]
        elif arm == 'E':
            features = np.r_[features, actual[index]]
        elif arm == 'N':
            features = np.r_[features, norms.values(row['start_x'])]
        elif arm == 'P':
            features = np.r_[features, control[index]]
        if len(features) != WIDTHS[arm]:
            raise ValueError('Feature width violates the fixed protocol')
        onehot = np.asarray([float(row['action'] == a) for a in ACTIONS])
        result.append(np.r_[onehot, np.outer(onehot, features).ravel()])
    return np.asarray(result)


def trajectory_metrics(rows, probabilities):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['instance'], row['trajectory_id']].append(i)
    output, states_output = [], []
    for (instance, trajectory), ids in sorted(groups.items()):
        states = defaultdict(list)
        for i in ids:
            states[rows[i]['state_id']].append(i)
        labels = np.asarray([float(rows[i]['improved']) for i in ids])
        metrics = {arm: {'ap': average_precision(labels, probabilities[arm][ids]),
            'brier': float(np.mean((probabilities[arm][ids] - labels)**2)),
            'ap_by_action': {a: average_precision(
                [float(rows[i]['improved']) for i in ids if rows[i]['action'] == a],
                [float(probabilities[arm][i]) for i in ids if rows[i]['action'] == a]) for a in ACTIONS},
            'brier_by_action': {a: float(np.mean([
                (float(probabilities[arm][i]) - float(rows[i]['improved']))**2
                for i in ids if rows[i]['action'] == a])) for a in ACTIONS}}
            for arm in ARMS}
        gains = {a: [] for a in ACTIONS}
        oracle, selected, correct, ranks, choices, winner, score_differences = [], defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), [], defaultdict(list)
        tied_oracles, unique_oracles = 0, []
        for state, indices in sorted(states.items()):
            actual = np.asarray([np.mean([rows[i]['gain'] for i in indices if rows[i]['action'] == a]) for a in ACTIONS])
            maximum = float(actual.max())
            oracle.append(maximum)
            unique = int(np.sum(np.abs(actual - maximum) <= 1e-12)) == 1
            unique_oracles.append(unique)
            tied_oracles += int(not unique)
            winner.append(float(actual[1] > actual[2] + 1e-12))
            record = {'instance': instance, 'trajectory_id': trajectory, 'state_id': state,
                      'gain_by_action': dict(zip(ACTIONS, actual.tolist())), 'oracle_gain': maximum,
                      'arms': {}}
            for a, value in zip(ACTIONS, actual):
                gains[a].append(float(value))
            for arm in ARMS:
                scores = np.asarray([np.mean([probabilities[arm][i] for i in indices if rows[i]['action'] == a]) for a in ACTIONS])
                chosen = int(np.argmax(scores))
                selected[arm].append(float(actual[chosen]))
                correct[arm].append(float(abs(actual[chosen] - maximum) <= 1e-12))
                ranks[arm].append(rank_correlation(scores, actual))
                choices[arm].append(chosen)
                score_differences[arm].append(float(scores[1] - scores[2]))
                record['arms'][arm] = {'probability_by_action': dict(zip(ACTIONS, scores.tolist())),
                    'action': ACTIONS[chosen], 'gain': float(actual[chosen]),
                    'regret': maximum - float(actual[chosen]),
                    'oracle_tie_correct': bool(correct[arm][-1])}
            states_output.append(record)
        for arm in ARMS:
            unique_indices = np.flatnonzero(unique_oracles)
            metrics[arm].update(selected_gain=float(np.mean(selected[arm])),
                regret=float(np.mean(np.asarray(oracle) - selected[arm])),
                strict_correct_rate=float(np.mean(correct[arm])),
                action_rank=float(np.mean(ranks[arm])),
                choice_change_vs_C=float(np.mean(np.asarray(choices[arm]) != choices['C'])),
                ls_over_vnd_margin_ap=average_precision(winner, score_differences[arm]),
                unique_best_correct_rate=(float(np.mean(np.asarray(correct[arm])[unique_indices]))
                    if len(unique_indices) else None),
                chosen_action_counts={a: choices[arm].count(j) for j, a in enumerate(ACTIONS)})
        output.append({'instance': instance, 'trajectory_id': trajectory, 'states': len(states),
            'branches': len(ids), 'positive_events': int(labels.sum()),
            'zero_oracle_states': sum(v <= 1e-12 for v in oracle),
            'tied_oracle_states': tied_oracles,
            'unique_best_states': sum(unique_oracles),
            'ls_strictly_beats_vnd_states': int(sum(winner)), 'variants': metrics,
            'oracle_gain': float(np.mean(oracle)),
            'fixed_action_gain': {a: float(np.mean(value)) for a, value in gains.items()}})
    return output, states_output


def summarize(rows, probabilities):
    trajectories, states = trajectory_metrics(rows, probabilities)
    grouped = defaultdict(list)
    for row in trajectories:
        grouped[row['instance']].append(row)
    metrics = ('ap', 'brier', 'selected_gain', 'regret', 'strict_correct_rate',
               'action_rank', 'choice_change_vs_C', 'ls_over_vnd_margin_ap')
    macro = {arm: {m: float(np.mean([np.mean([r['variants'][arm][m] for r in group])
        for group in grouped.values()])) for m in metrics} for arm in ARMS}
    for arm in ARMS:
        macro[arm]['ap_by_action'] = {a: float(np.mean([np.mean([
            r['variants'][arm]['ap_by_action'][a] for r in group]) for group in grouped.values()])) for a in ACTIONS}
        macro[arm]['brier_by_action'] = {a: float(np.mean([np.mean([
            r['variants'][arm]['brier_by_action'][a] for r in group]) for group in grouped.values()])) for a in ACTIONS}
    comparisons = []
    for comparator in ('C', 'N', 'P'):
        for metric in ('ap', 'brier', 'selected_gain'):
            sign = -1 if metric == 'brier' else 1
            matrix = [[sign * (r['variants']['E'][metric] - r['variants'][comparator][metric])
                       for r in group] for _, group in sorted(grouped.items())]
            means = [float(np.mean(v)) for v in matrix]
            comparisons.append({'candidate': 'E', 'comparator': comparator, 'metric': metric,
                'positive_favors_E': True, 'mean_difference': float(np.mean(means)),
                'hierarchical_bootstrap_95ci': hierarchical_ci(matrix, repetitions=10000, seed=SEED),
                'instance_differences': dict(zip(sorted(grouped), means)),
                'leave_one_instance_out_mean_difference': {name: float(np.mean([
                    value for other, value in zip(sorted(grouped), means) if other != name]))
                    for name in sorted(grouped)},
                'instance_win_tie_loss': [sum(v > 1e-12 for v in means), sum(abs(v) <= 1e-12 for v in means), sum(v < -1e-12 for v in means)],
                **signed_rank(means)})
    holm(comparisons)
    per_instance = [{'instance': instance, 'trajectories': len(group),
        'variants': {arm: {m: float(np.mean([r['variants'][arm][m] for r in group]))
                    for m in metrics} for arm in ARMS}}
        for instance, group in sorted(grouped.items())]
    return {'macro_instance_trajectory_metrics': macro, 'comparisons': comparisons,
        'per_instance_metrics': per_instance,
        'trajectory_metrics': trajectories, 'state_decisions': states,
        'no_positive_event_trajectories': sum(r['positive_events'] == 0 for r in trajectories),
        'primary_family': '9: E vs C,N,P times AP,Brier,selected_gain',
        'evaluation_status': 'EXPLORATORY_DEVELOPMENT_ONLY; no independent confirmation',
        'correctness_convention': 'strict_correct_rate means chosen gain is within1e-12 of maximum; ties count as correct, not a unique winner',
        'diagnostic_margin_AP_label': 'LS repeat-mean gain strictly exceeds VND repeat-mean gain; score=pLS-pVND; no state filtering'}


def check_protocol(path, source_lock_path):
    """Require the root-authored frozen protocol and exact executable identity."""
    payload = json.loads(Path(path).read_text())
    if payload.get('status') != 'FROZEN_BEFORE_NEW_C_FITS':
        raise ValueError('Protocol must be frozen before any feature-arm fit')
    lock = json.loads(Path(source_lock_path).read_text())
    if lock.get('status') != 'FROZEN_BEFORE_FEATURE_FIT' or lock.get('protocol_sha256') != sha(path):
        raise ValueError('Source lock must precede fitting and match the frozen protocol')
    sources = lock.get('source_sha256', {})
    relative = 'scripts/compare_fp_decision_features.py'
    if sources.get(relative) != sha(ROOT / relative):
        raise ValueError('Diagnostic script differs from frozen protocol')
    for source, expected in sources.items():
        if sha(ROOT / source) != expected:
            raise ValueError(f'Frozen diagnostic dependency changed: {source}')
    for source, expected in payload['input_sha256'].items():
        if sha(ROOT / source) != expected:
            raise ValueError(f'Frozen protocol input changed: {source}')
    settings = payload['C_comparison']
    if (settings['ridge'] != RIDGE or settings['l2'] != L2 or
            settings['maximum_iterations'] != ITERATIONS or
            settings['N_definition']['seed'] != SEED):
        raise ValueError('Scientific settings differ from frozen protocol')
    return payload


def run(rows, data_audit):
    probabilities = {arm: np.zeros(len(rows)) for arm in ARMS}
    cache, optimizer_records, model_records, norm_records = SignalCache(rows), {}, {}, {}
    instances = sorted({r['instance'] for r in rows})
    if len(instances) != 16:
        raise ValueError('Only the full development16 cohort may be analyzed')
    for fold, instance in enumerate(instances):
        ids = [i for i, r in enumerate(rows) if r['instance'] == instance]
        training = [r for r in rows if r['instance'] != instance]
        evaluation = [rows[i] for i in ids]
        train_fp, train_control, test_fp, test_control = cache.nested(instance, training, evaluation)
        norms = RandomStateNorms(training)
        labels = np.asarray([float(r['improved']) for r in training])
        optimizer_records[instance] = {}
        for arm in ARMS:
            xt = feature_matrix(training, arm, train_fp, train_control, norms)
            xe = feature_matrix(evaluation, arm, test_fp, test_control, norms)
            classifier = Logistic(l2=L2, iterations=ITERATIONS).fit(xt, labels)
            probabilities[arm][ids] = classifier.predict(xe)
            optimizer_records[instance][arm] = {'iterations': classifier.iterations_run,
                'loss': classifier.loss, 'gradient_inf': classifier.gradient_inf,
                'parameters': len(classifier.weights), 'training_rows': len(training)}
        model, _ = cache.model({instance})
        model_records[instance] = model.summary
        norm_records[instance] = norms.summary()
        print(json.dumps({'completed_outer_fold': fold + 1, 'total_outer_folds': 16,
                          'heldout_instance': instance}), flush=True)
    return {'stage': 'development_only_final_decision_diagnostic', 'data_audit': data_audit,
        'config': {'ridge': RIDGE, 'l2': L2, 'iterations': ITERATIONS,
                   'gradient_inf_tolerance': 1e-6, 'random_norm_seed': SEED,
                   'negative_control_seed': 20261018, 'lags_generations': list(LAGS)},
        'feature_widths_before_action_interactions': WIDTHS,
        'summary': summarize(rows, probabilities), 'optimizer_records': optimizer_records,
        'outer_model_summaries': model_records, 'ordinary_random_norm_maps': norm_records,
        'fp_training_exclusions': [{'excluded_instances': list(excluded),
            'training_instances': list(model.training_instances),
            'all_targets_excluded': not bool(set(excluded) & set(model.training_instances))}
            for excluded, model in sorted(cache.models.items())],
        'predictions': [{**{k: row[k] for k in ('instance', 'trajectory_id', 'state_id', 'generation', 'action', 'repeat', 'gain', 'improved')},
            'probabilities': {arm: float(probabilities[arm][i]) for arm in ARMS}}
            for i, row in enumerate(rows)],
        'limitations': [
            'Development16 has informed earlier configuration choice; these OOF scores remain exploratory.',
            'Original FP D uses a five-generation mixed GA/LS state history not fully saved in C; D is descriptive, excluded from the matched-input attribution gate.',
            'Frozen action-FP maps/reliability are deterministic transforms of current x and training-fold estimates, not new information beyond their inputs.',
            'The probability objective, action windows, search primitives, and 120-iteration solver cap are unchanged.',
            'No validation refit/evaluation, locked-test access, alternative reward objective or online Agent is run.',
            'Repeated-mean oracle and winner labels are retrospective noisy references only, never current action inputs.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', default='results/fp_v2_development')
    parser.add_argument('--audit', default='projects/analysis/fp_action_development_independent_audit.json')
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--source-lock', default='projects/experiments/fp_agent_decision_c_source_lock.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Never refit or overwrite an existing diagnostic outcome')
    check_protocol(args.protocol, args.source_lock)
    rows, data_audit = load_audited_development(args.development, args.audit)
    result = run(rows, data_audit)
    result['protocol_sha256'] = sha(args.protocol)
    result['source_lock_sha256'] = sha(args.source_lock)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'status': 'DEVELOPMENT_ONLY_COMPLETE'}), flush=True)


if __name__ == '__main__':
    main()

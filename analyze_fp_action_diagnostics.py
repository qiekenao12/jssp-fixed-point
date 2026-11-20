#!/usr/bin/env python3
"""Grouped offline M2 experiment. Never launches search or an Agent.

Development configurations are selected using leave-one-instance-out outcomes.
Validation requires an explicit frozen development selection. Every branch from
one trajectory stays together. Counterfactual outcomes are labels only.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import math
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from tools.action_fp import ACTIONS, ActionFPModel, _Affine, collapsed_transitions, weighted_norm, pure_window, state_vector
from analyze_paper_evidence import signed_rank, hierarchical_ci, holm

VARIANTS = ('A', 'B', 'C', 'D', 'E', 'STATE_QUADRATIC', 'PERMUTED_FP')
CONFIGS = ({'ridge': 1e-3, 'l2': 1.}, {'ridge': 1e-3, 'l2': 10.},
           {'ridge': 1e-1, 'l2': 1.}, {'ridge': 1e-1, 'l2': 10.})
COMPARATORS = ('B', 'STATE_QUADRATIC', 'PERMUTED_FP')


def read_records(path):
    path = Path(path)
    if path.is_dir():
        protocol_path = path/'protocol.json'
        if not protocol_path.exists():
            raise ValueError('Real diagnostic directories require a frozen protocol.json')
        protocol = json.loads(protocol_path.read_text())
        payload = {k: v for k, v in protocol.items() if k != 'protocol_hash'}
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if actual != protocol.get('protocol_hash'):
            raise ValueError('Diagnostic protocol hash is invalid')
        root = Path(__file__).resolve().parent
        if 'run_fp_action_diagnostics.py' not in protocol.get('sources', {}):
            raise ValueError('Diagnostic runner source provenance is missing')
        for source, expected in protocol['sources'].items():
            if file_hash(root/source) != expected:
                raise ValueError(f'Diagnostic source changed: {source}')
        manifest = root/'projects/experiments/fp_v2_manifest.json'
        if file_hash(manifest) != protocol.get('manifest_sha256'):
            raise ValueError('Diagnostic manifest differs from frozen v2 manifest')
        settings = protocol['protocol']
        if (settings['generations'] != 180 or settings['snapshot_interval'] != 20
                or settings['local_iterations'] != 100 or settings['action_query_cap'] != 500):
            raise ValueError('Action strength or observation schedule differs from frozen M2')
        expected_generations = set(range(settings['snapshot_interval'], settings['generations']+1, settings['snapshot_interval']))
        expected_files = {f'{name}_{run:02d}.json' for name in protocol['instances'] for run in range(protocol['runs'])}
        observed = {p.name for p in path.glob('*.json') if p.name not in ('protocol.json', 'analysis.json') and not p.name.endswith('_analysis.json')}
        if observed != expected_files:
            raise ValueError(f'Incomplete diagnostic trajectory grid: missing={sorted(expected_files-observed)}, extra={sorted(observed-expected_files)}')
        rows = []
        for name in protocol['instances']:
            for run in range(protocol['runs']):
                saved = json.loads((path/f'{name}_{run:02d}.json').read_text())
                if saved.get('protocol_hash') != protocol['protocol_hash'] or saved.get('instance') != name or saved.get('run') != run:
                    raise ValueError('A trajectory has mismatched protocol/identity')
                archive = root/saved['population_archive']
                if file_hash(archive) != saved.get('population_archive_sha256'):
                    raise ValueError('Saved population archive hash mismatch')
                records = saved['records']
                expected = {(g, a, repeat) for g in expected_generations for a in ACTIONS for repeat in range(settings['repeats'])}
                observed_keys = [(r['generation'], r['action'], r['repeat']) for r in records]
                if len(observed_keys) != len(expected) or set(observed_keys) != expected:
                    raise ValueError('Incomplete state/action/repeat grid within trajectory')
                snapshots = {int(s['generation']): s for s in saved['snapshots']}
                if set(snapshots) != expected_generations:
                    raise ValueError('Snapshot grid differs from records')
                for row in records:
                    snap = snapshots[row['generation']]
                    if (row['instance'] != name or row['run'] != run or row['trajectory_id'] != f'{name}:{run}'
                            or row['state_fingerprint'] != snap['fingerprint']
                            or not np.array_equal(state_vector(row['start_x']), state_vector(snap['x']))):
                        raise ValueError('Branch identity/start state differs from saved snapshot')
                    if row['queries'] > settings['action_query_cap'] or sum(row['phase_queries'].values()) != row['queries']:
                        raise ValueError('Branch query accounting or cap mismatch')
                    if (row['elapsed_ga_generations'] != int(row['action'] == 'GA')
                            or row['bottom_operator'] != {'GA': 'evolve_generation_without_LS', 'LS': 'local_search_improved', 'VND': 'local_search_vnd'}[row['action']]):
                        raise ValueError('Branch action semantics differ from frozen protocol')
                    if not math.isclose(row['gain'], (row['start_makespan']-row['end_makespan'])/row['s0'], abs_tol=1e-10):
                        raise ValueError('Branch gain is inconsistent with endpoint makespans')
                    if not math.isclose(row['end_x'][0], row['end_makespan']/row['s0'], abs_tol=1e-10):
                        raise ValueError('Branch normalized endpoint is inconsistent')
                rows.extend(records)
    else:
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else payload['records']
    audit_records(rows)
    return sorted(rows, key=lambda r: (str(r['instance']), str(r['trajectory_id']), str(r['state_id']), ACTIONS.index(r['action']), int(r.get('repeat', 0))))


def audit_records(rows):
    if not rows:
        raise ValueError('No diagnostic rows')
    seen, states, trajectories = set(), defaultdict(list), {}
    for r in rows:
        action = pure_window(r)
        x = state_vector(r.get('start_x', r.get('x')))
        state_vector(r['end_x'])
        instance, trajectory, state = str(r['instance']), str(r['trajectory_id']), str(r['state_id'])
        if trajectory in trajectories and trajectories[trajectory] != instance:
            raise ValueError('A trajectory crosses instance identifiers')
        trajectories[trajectory] = instance
        key = (instance, trajectory, state, action, int(r.get('repeat', 0)))
        if key in seen:
            raise ValueError('Duplicate diagnostic branch')
        seen.add(key)
        if not np.isfinite(r['gain']) or r['gain'] < -1e-10:
            raise ValueError('Gain must be a finite nonnegative best-incumbent improvement')
        if bool(r['improved']) != bool(r['gain'] > 1e-12):
            raise ValueError('Improvement event disagrees with gain')
        states[instance, trajectory, state].append((action, int(r.get('repeat', 0)), x, r))
    for key, branches in states.items():
        if any(not np.array_equal(branches[0][2], b[2]) for b in branches):
            raise ValueError('Counterfactual actions must start at identical states')
        repeats = {a: {b[1] for b in branches if b[0] == a} for a in ACTIONS}
        if not repeats['GA'] or any(repeats[a] != repeats['GA'] for a in ACTIONS):
            raise ValueError('All three actions require identical repeat sets')
        fingerprints = {b[3].get('state_fingerprint') for b in branches if b[3].get('state_fingerprint') is not None}
        if len(fingerprints) > 1:
            raise ValueError('Counterfactual starting-state fingerprints differ')
    return {'records': len(rows), 'instances': len({r['instance'] for r in rows}),
            'trajectories': len(trajectories), 'states': len(states), 'complete_action_triples': True}


def sigmoid(z):
    z = np.clip(z, -40., 40.)
    return 1. / (1. + np.exp(-z))


class Logistic:
    """Same standardized L2 logistic regression and L-BFGS for every arm."""
    def __init__(self, l2=1., iterations=120):
        self.l2, self.iterations = l2, iterations

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.center = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-8)
        z = np.column_stack((np.ones(len(x)), (x-self.center)/self.scale))
        penalty = np.ones(z.shape[1]) * self.l2 / len(x)
        penalty[0] = 0.
        def objective(w):
            p = z @ w
            return (float(np.mean(np.logaddexp(0, p) - y*p) + .5*np.sum(penalty*w*w)),
                    z.T @ (sigmoid(p)-y)/len(x) + penalty*w)
        w = np.zeros(z.shape[1]); rate = np.clip(y.mean(), 1e-6, 1-1e-6)
        w[0] = math.log(rate/(1-rate))
        loss, gradient = objective(w)
        history = []
        for iteration in range(self.iterations):
            if np.linalg.norm(gradient, ord=np.inf) < 1e-6:
                break
            q, alphas = gradient.copy(), []
            for s, v, rho in reversed(history):
                alpha = rho * np.dot(s, q); alphas.append(alpha); q -= alpha*v
            scale = (np.dot(history[-1][0], history[-1][1]) / np.dot(history[-1][1], history[-1][1])) if history else 1.
            direction = scale*q
            for (s, v, rho), alpha in zip(history, reversed(alphas)):
                direction += s * (alpha-rho*np.dot(v, direction))
            direction = -direction
            if np.dot(gradient, direction) >= 0:
                direction = -gradient
            step = 1.
            for _ in range(25):
                candidate = w + step*direction
                newloss, newgradient = objective(candidate)
                if newloss <= loss + 1e-4*step*np.dot(gradient, direction):
                    break
                step *= .5
            s, v = candidate-w, newgradient-gradient
            curvature = np.dot(s, v)
            if curvature > 1e-12:
                history.append((s, v, 1./curvature)); history = history[-8:]
            w, loss, gradient = candidate, newloss, newgradient
        self.weights, self.loss = w, loss
        self.iterations_run = iteration+1
        self.gradient_inf = float(np.linalg.norm(gradient, ord=np.inf))
        return self

    def predict(self, x):
        x = np.asarray(x, float)
        return sigmoid(self.weights[0] + ((x-self.center)/self.scale)@self.weights[1:])


def base_features(row):
    x = state_vector(row.get('start_x', row.get('x')))
    stagnation = row.get('real_stagnant_generations', row.get('realstagnantgens', row.get('stagnant_generations', 0)))
    a = [math.log1p(max(0., float(stagnation))), float(x[3])]
    b = a + list(x[:3])  # diversity is already included once
    original = row.get('original_fp', {})
    residual = original.get('residual', 0.)
    if residual is None or not np.isfinite(residual):
        residual = 0.
    return np.asarray(a), np.asarray(b), np.asarray([float(residual), float(bool(original.get('model_ready', False)))])


def common_context(row):
    """Shared causal phase/payoff context; no diagnostic branch labels."""
    generation = float(row.get('generation', 0.))
    recent = np.asarray(row.get('recent_real_gains', []), dtype=float)
    if recent.ndim != 1 or not np.all(np.isfinite(recent)):
        raise ValueError('recent_real_gains must be a finite main-history vector')
    recent = recent[-6:]
    return np.asarray([generation/180., float(np.mean(recent)) if len(recent) else 0.,
        float(np.mean(recent > 0)) if len(recent) else 0., float(recent[-1]) if len(recent) else 0.])


def signal_rows(training, evaluation, ridge, reuse_identical_states=True):
    """Evaluation features use training instances only; training is inner OOF."""
    model = ActionFPModel(ridge=ridge).fit(training)
    signals, errors, permuted = [], [], []
    rng = np.random.default_rng(20261018)
    pool_indices = {}
    for row in evaluation:
        key = (str(row['instance']), str(row['trajectory_id']), str(row['state_id']))
        if key not in pool_indices:
            pool_indices[key] = int(rng.integers(len(training)))
    unified = _Affine(collapsed_transitions(training), ridge)
    state_cache = {}
    for row_index, r in enumerate(evaluation):
        key = (str(r['instance']), str(r['trajectory_id']), str(r['state_id']))
        cache_key = key if reuse_identical_states else (row_index,)
        if cache_key not in state_cache:
            source = training[pool_indices[key]]
            source_fp, source_reliability, _ = model.features(source.get('start_x', source.get('x')))
            x = state_vector(r.get('start_x', r.get('x')))
            fp, reliability, candidates = model.features(x)
            state_cache[cache_key] = (x, fp, reliability, candidates, np.r_[source_fp, source_reliability], unified.predict(x))
        x, fp, reliability, candidates, control, unified_prediction = state_cache[cache_key]
        permuted.append(control)
        signals.append((fp, reliability))
        current = candidates[ACTIONS.index(r['action'])]
        actual = state_vector(r['end_x'])
        errors.append({'instance': r['instance'], 'trajectory_id': r['trajectory_id'],
            'state_id': r['state_id'], 'action': r['action'], 'repeat': r.get('repeat', 0),
            'action_error': weighted_norm(np.asarray(current.predicted_state)-actual),
            'unified_one_action_error': weighted_norm(unified_prediction-actual),
            'identity_error': weighted_norm(x-actual), 'signal': current.as_dict()})
    return signals, errors, model.summary, np.asarray(permuted)


def crossfit_signals(rows, ridge):
    signals = [None]*len(rows)
    permuted = [None]*len(rows)
    indices = defaultdict(list)
    for i, r in enumerate(rows):
        indices[r['instance']].append(i)
    if len(indices) < 5:
        raise ValueError('At least five development instances required for nested OOF reliability')
    for instance, held_indices in sorted(indices.items()):
        train = [r for r in rows if r['instance'] != instance]
        held = [rows[i] for i in held_indices]
        local, _, _, local_permuted = signal_rows(train, held, ridge)
        for i, value, control in zip(held_indices, local, local_permuted):
            signals[i] = value
            permuted[i] = control
    return signals, np.asarray(permuted)


def feature_matrix(rows, signals, variant, shuffled=None):
    result = []
    for index, (row, (fp, reliability)) in enumerate(zip(rows, signals)):
        a, b, original = base_features(row)
        if variant == 'A':
            features = a
        elif variant == 'B':
            features = b
        elif variant == 'C':
            features = np.r_[b, original]
        elif variant == 'D':
            features = np.r_[b, fp]
        elif variant == 'E':
            features = np.r_[b, fp, reliability]
        elif variant == 'PERMUTED_FP':
            features = np.r_[b, shuffled[index]]
        elif variant == 'STATE_QUADRATIC':
            features = np.r_[b, [b[i]*b[j] for i in range(len(b)) for j in range(i, len(b))]]
        else:
            raise ValueError('Unknown feature arm')
        # Every arm shares identical causal phase and prior-main-payoff context.
        # The quadratic control expands only ordinary B state variables.
        features = np.r_[features, common_context(row)]
        # Identical architecture: action-specific linear probability scores.
        onehot = np.asarray([float(row['action'] == a) for a in ACTIONS])
        result.append(np.r_[onehot, np.outer(onehot, features).ravel()])
    return np.asarray(result)


def train_predict(training, evaluation, config, cached_signals=None):
    if {r['instance'] for r in training} & {r['instance'] for r in evaluation}:
        raise ValueError('Evaluation instances overlap training instances')
    if cached_signals is None:
        train_signals, shuffled_train = crossfit_signals(training, config['ridge'])
        test_signals, errors, summary, shuffled_test = signal_rows(training, evaluation, config['ridge'])
    else:
        train_signals, shuffled_train, test_signals, errors, summary, shuffled_test = cached_signals
    y = np.asarray([float(r['improved']) for r in training])
    output, optimizers = {}, {}
    for variant in VARIANTS:
        xt = feature_matrix(training, train_signals, variant, shuffled_train)
        xv = feature_matrix(evaluation, test_signals, variant, shuffled_test)
        classifier = Logistic(config['l2']).fit(xt, y)
        output[variant] = classifier.predict(xv)
        optimizers[variant] = {'iterations': classifier.iterations_run, 'loss': classifier.loss,
            'gradient_inf': classifier.gradient_inf, 'parameters': len(classifier.weights),
            'training_rows': len(training)}
    return output, errors, summary, optimizers


def average_precision(y, probability):
    """Stepwise AP with tied scores grouped; no positives -> 0 (reported)."""
    y, probability = np.asarray(y, float), np.asarray(probability, float)
    if not y.sum():
        return 0.
    order = np.argsort(-probability, kind='stable')
    y, probability = y[order], probability[order]
    cumulative = np.cumsum(y)
    endpoints = np.r_[np.flatnonzero(probability[1:] != probability[:-1]), len(y)-1]
    recall = cumulative[endpoints]/y.sum()
    precision = cumulative[endpoints]/(endpoints+1)
    return float(np.sum(np.diff(np.r_[0., recall])*precision))


def midranks(values):
    values = np.asarray(values)
    return np.asarray([1 + np.sum(values < v) + .5*(np.sum(values == v)-1) for v in values], float)


def rank_correlation(a, b):
    a, b = midranks(a), midranks(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.
    return float(np.corrcoef(a, b)[0, 1])


def trajectory_metrics(rows, probabilities):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['instance'], r['trajectory_id']].append(i)
    out = []
    for (instance, trajectory), indices in sorted(groups.items()):
        selected = [rows[i] for i in indices]
        y = np.asarray([float(r['improved']) for r in selected])
        states = defaultdict(list)
        for j, row in zip(indices, selected):
            states[str(row['state_id'])].append(j)
        metrics = {variant: {'ap': average_precision(y, np.asarray(probabilities[variant])[indices]),
            'brier': float(np.mean((np.asarray(probabilities[variant])[indices]-y)**2))}
            for variant in VARIANTS}
        gain_lists, rank_lists, chosen_actions = {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}
        oracle, action_gain = [], defaultdict(list)
        for state, ids in sorted(states.items()):
            gains = [float(np.mean([rows[i]['gain'] for i in ids if rows[i]['action'] == a])) for a in ACTIONS]
            oracle.append(max(gains))
            for a, value in zip(ACTIONS, gains):
                action_gain[a].append(value)
            for variant in VARIANTS:
                scores = [float(np.mean([probabilities[variant][i] for i in ids if rows[i]['action'] == a])) for a in ACTIONS]
                chosen = int(np.argmax(scores))  # Fixed GA,LS,VND tie break.
                gain_lists[variant].append(gains[chosen])
                rank_lists[variant].append(rank_correlation(scores, gains))
                chosen_actions[variant].append(chosen)
        for variant in VARIANTS:
            metrics[variant].update({'selected_gain': float(np.mean(gain_lists[variant])),
                'action_rank': float(np.mean(rank_lists[variant])),
                'oracle_gain': float(np.mean(oracle)),
                'decision_change_vs_B': float(np.mean(np.asarray(chosen_actions[variant]) != chosen_actions['B'])),
                'chosen_action_counts': {a: chosen_actions[variant].count(i) for i, a in enumerate(ACTIONS)}})
        out.append({'instance': instance, 'trajectory_id': trajectory, 'observations': len(indices),
            'states': len(states), 'positive_events': int(y.sum()), 'variants': metrics,
            'fixed_action_gain': {a: float(np.mean(v)) for a, v in action_gain.items()}})
    return out


def summarize(rows, probabilities, bootstrap=10000):
    trajectories = trajectory_metrics(rows, probabilities)
    by_instance = defaultdict(list)
    for row in trajectories:
        by_instance[row['instance']].append(row)
    macro = {variant: {metric: float(np.mean([np.mean([r['variants'][variant][metric] for r in group]) for group in by_instance.values()]))
              for metric in ('ap', 'brier', 'selected_gain', 'action_rank', 'decision_change_vs_B')}
             for variant in VARIANTS}
    comparisons = []
    # All 3 comparisons x 3 prespecified endpoints are one Holm family.
    for comparator in COMPARATORS:
        for metric in ('ap', 'brier', 'selected_gain'):
            sign = -1 if metric == 'brier' else 1
            matrix = [[sign*(r['variants']['E'][metric]-r['variants'][comparator][metric]) for r in group]
                      for _, group in sorted(by_instance.items())]
            means = [float(np.mean(v)) for v in matrix]
            comparisons.append({'candidate': 'E', 'comparator': comparator, 'metric': metric,
                'positive_favors_E': True, 'mean_difference': float(np.mean(means)),
                'hierarchical_bootstrap_95ci': hierarchical_ci(matrix, repetitions=bootstrap, seed=20261018),
                'instance_win_tie_loss': [sum(v>1e-12 for v in means), sum(abs(v)<=1e-12 for v in means), sum(v < -1e-12 for v in means)],
                'instance_differences': dict(zip(sorted(by_instance), means)), **signed_rank(means)})
    holm(comparisons)
    # In addition to the primary family, plain D vs B is separately reported,
    # never substituted for an unsuccessful predeclared gate.
    return {'macro_instance_trajectory_metrics': macro, 'comparisons': comparisons,
            'trajectory_metrics': trajectories,
            'no_positive_event_trajectories': sum(r['positive_events'] == 0 for r in trajectories),
            'instances': len(by_instance), 'trajectories': len(trajectories)}


def summarize_dynamics(errors, bootstrap=10000):
    tables = []
    comparisons = []
    for action in ACTIONS:
        grouped = defaultdict(lambda: defaultdict(list))
        for row in errors:
            if row['action'] == action:
                grouped[row['instance']][row['trajectory_id']].append(row)
        macro = {}
        for metric in ('action_error', 'unified_one_action_error', 'identity_error'):
            macro[metric] = float(np.mean([np.mean([np.mean([r[metric] for r in trajectory]) for trajectory in instance.values()]) for instance in grouped.values()]))
        tables.append({'action': action, **macro,
            'ready_fraction': float(np.mean([r['signal']['model_ready'] for r in errors if r['action'] == action])),
            'records': sum(len(t) for i in grouped.values() for t in i.values())})
        for comparator in ('unified_one_action_error', 'identity_error'):
            matrix = [[float(np.mean([r[comparator]-r['action_error'] for r in trajectory])) for trajectory in instance.values()] for _, instance in sorted(grouped.items())]
            means = [float(np.mean(v)) for v in matrix]
            comparisons.append({'action': action, 'comparator': comparator,
                'positive_favors_action_conditioning': True, 'mean_error_reduction': float(np.mean(means)),
                'hierarchical_bootstrap_95ci': hierarchical_ci(matrix, repetitions=bootstrap, seed=20261018),
                **signed_rank(means)})
    holm(comparisons)
    return {'macro_errors': tables, 'comparisons': comparisons,
        'unified_baseline': 'one affine map fit to pooled pure one-action transitions, action identity ignored; not the historical mixed five-generation operator',
        'historical_operator': 'original residual is evaluated as arm C; its mixed-action observation horizon is not silently treated as one atomic action'}


def prediction_rows(rows, predictions):
    return [{**{k: row[k] for k in ('instance', 'trajectory_id', 'state_id', 'action', 'repeat', 'gain', 'improved') if k in row},
        'start_x': row.get('start_x', row.get('x')),
        'probabilities': {variant: float(predictions[variant][i]) for variant in VARIANTS}}
        for i, row in enumerate(rows)]


def m2_gate(summary, *, independent_validation):
    if not independent_validation:
        return {'pass': False, 'status': 'DEVELOPMENT_ONLY', 'reason': 'Independent frozen validation has not been evaluated.'}
    comparisons = summary['comparisons']
    # Independent benefit over plain state AND a nonlinear state control AND
    # an equally dimensioned meaningless FP control is required.
    required = [r for r in comparisons if r['metric'] == 'selected_gain']
    gain_pass = len(required) == 3 and all(r['mean_difference'] > 0 and r['hierarchical_bootstrap_95ci'][0] > 0 and r['holm_p'] < .05 for r in required)
    info = [r for r in comparisons if r['comparator'] == 'B' and r['metric'] in ('ap', 'brier')]
    info_pass = len(info) == 2 and all(r['mean_difference'] > 0 and r['hierarchical_bootstrap_95ci'][0] > 0 and r['holm_p'] < .05 for r in info)
    return {'pass': bool(gain_pass and info_pass), 'status': 'PASS' if gain_pass and info_pass else 'STOP_M3',
        'independent_validation': True, 'positive_action_gain_vs_all_controls': gain_pass,
        'incremental_prediction_information_vs_B': info_pass,
        'reason': 'M3 is authorized only after positive paired instance-level evidence for both information and realized action gains; absence is not equivalence.'}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_identity(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def run_development(rows, bootstrap=10000):
    if len({r['instance'] for r in rows}) < 6:
        raise ValueError('At least six development instances required')
    configs = []
    signal_cache = {}  # same frozen ridge maps reused across classifier L2 values
    instances = sorted({r['instance'] for r in rows})
    for config_index, config in enumerate(CONFIGS):
        predictions = {v: np.zeros(len(rows)) for v in VARIANTS}
        errors, optimizer_records = [], {}
        for instance in instances:
            test_indices = [i for i,r in enumerate(rows) if r['instance'] == instance]
            training = [r for r in rows if r['instance'] != instance]
            test = [rows[i] for i in test_indices]
            cache_key = (config['ridge'], instance)
            if cache_key not in signal_cache:
                train_signals, shuffled_train = crossfit_signals(training, config['ridge'])
                test_signals, local_errors, model_summary, shuffled_test = signal_rows(training, test, config['ridge'])
                signal_cache[cache_key] = (train_signals, shuffled_train, test_signals, local_errors, model_summary, shuffled_test)
            predicted, local_errors, _, optimizers = train_predict(training, test, config, cached_signals=signal_cache[cache_key])
            for variant in VARIANTS:
                predictions[variant][test_indices] = predicted[variant]
            errors.extend(local_errors)
            optimizer_records[instance] = optimizers
        summary = summarize(rows, predictions, bootstrap)
        configs.append({'config_index': config_index, 'config': config, 'summary': summary,
            'dynamics_errors': errors, 'dynamics_summary': summarize_dynamics(errors, bootstrap),
            'predictions': prediction_rows(rows, predictions), 'optimizer_records': optimizer_records})
    # Proper scoring rule only; no heldout validation outcome has been used.
    selected = min(range(len(configs)), key=lambda i: (configs[i]['summary']['macro_instance_trajectory_metrics']['E']['brier'], i))
    return {'stage': 'development', 'audit': audit_records(rows), 'configuration_grid': list(CONFIGS),
        'development_identity': data_identity(rows), 'selected_config_index': selected, 'selected_config': dict(CONFIGS[selected]),
        'selection_rule': 'Smallest development outer-instance-OOF E Brier, fixed index tie break; all arms share this configuration.',
        'configurations': configs, 'm2_gate': m2_gate(configs[selected]['summary'], independent_validation=False),
        'information_caveat': 'Frozen pooled affine signals are deterministic transformations of x, not an additional information source. Superiority over linear B alone cannot establish FP-specific information.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', required=True)
    parser.add_argument('--validation')
    parser.add_argument('--selection', help='Frozen development analysis JSON required for validation')
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    development = read_records(args.development)
    if args.validation:
        if not args.selection:
            raise ValueError('Validation requires a preexisting frozen development selection')
        selection = json.loads(Path(args.selection).read_text())
        if selection.get('stage') != 'development':
            raise ValueError('Selection must be a development analysis')
        for path, expected in selection.get('source_sha256', {}).items():
            if file_hash(path) != expected:
                raise ValueError('Analysis source changed after development selection')
        if selection.get('development_identity') != data_identity(development):
            raise ValueError('Development data changed after selection')
        config = selection['selected_config']
        if config not in CONFIGS:
            raise ValueError('Configuration outside frozen grid')
        validation = read_records(args.validation)
        overlap = {r['instance'] for r in development} & {r['instance'] for r in validation}
        if overlap:
            raise ValueError(f'Development/validation instances overlap: {sorted(overlap)}')
        predictions, errors, models, optimizers = train_predict(development, validation, config)
        summary = summarize(validation, predictions, args.bootstrap)
        result = {'stage': 'validation', 'audit': audit_records(validation), 'selected_config': config,
            'selection_sha256': file_hash(args.selection), 'summary': summary, 'dynamics_errors': errors,
            'training_model_summary': models, 'dynamics_summary': summarize_dynamics(errors, args.bootstrap),
            'predictions': prediction_rows(validation, predictions), 'optimizer_records': optimizers,
            'm2_gate': m2_gate(summary, independent_validation=True)}
    else:
        result = run_development(development, args.bootstrap)
    result['source_sha256'] = {str(p): file_hash(p) for p in [Path(__file__), Path(__file__).parent/'src/tools/action_fp.py', Path(__file__).parent/'analyze_paper_evidence.py']}
    result['method'] = {'state': '[best/s0,mean/s0,std/s0,diversity]',
        'window': 'one pure atomic action application; costs differ and are reported, not equated',
        'W': [.35,.30,.15,.20], 'grouping': 'outer heldout instance; training features inner heldout instance; repeats collapsed only for dynamics fits',
        'classifier': 'identical standardized action-interacted regularized logistic; L-BFGS 120 iterations, tolerance1e-6',
        'common_context_all_arms': 'generation/180, last6 prior-main normalized gains mean, last6positivefraction, last actual prior-main gain; zero if absent; no diagnostic feedback',
        'feature_blocks': 'A stagnation+diversity; B A+four-dimensional state without repeated diversity; C B+originalresidual/readiness; D B+actionresidual/difference; E D+OOFerror/reliability/readiness. All add identical commoncontext; StateQuadratic expands only B ordinary variables.',
        'primary_multiplicity': 'all E-vs-3-controls times 3 endpoints in one Holm family',
        'AP': 'trajectory macro AP (tied-score step integration), no-positive trajectories=0; instance macro average',
        'bootstrap': 'paired instance then paired trajectory resampling, 10000 draws by default seed20261018',
        'negative_control': 'Current-state FP is replaced with FP evaluated at a randomly reassigned training-pool state. Each training target-instance control uses its inner-OOF model excluding that entire instance. Heldout control uses development model/pool only; no target future state used',
        'selection': 'all4configurations dev-only; no validation/test optimization',
        'no_global_convergence_claim': True,
        'reliability': 'LOO training-instance error relative to identity displacement, reduced outside the training95percentile standardized state radius; heuristic score, not calibrated probability'}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'stage': result['stage'], 'output': str(target), 'audit': result['audit'], 'm2_gate': result['m2_gate']}, ensure_ascii=False))

if __name__ == '__main__':
    main()

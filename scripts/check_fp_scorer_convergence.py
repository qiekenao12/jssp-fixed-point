#!/usr/bin/env python3
"""Independent numerical robustness check of the frozen development scorer.

No search operator, reward, representation, regularizer, or validation cohort is
changed.  The independent solver is damped Newton for the same convex objective.
Formal data fitting requires a root-authored, hash-locked supplement protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.compare_fp_decision_features import (
    ARMS, L2, WIDTHS, Logistic, RandomStateNorms, SignalCache,
    feature_matrix, load_audited_development, summarize,
)
from tools.action_fp import ACTIONS

GRADIENT_TOLERANCE = 1e-7
MAX_NEWTON_STEPS = 200
MAX_BACKTRACKS = 50
ARMIJO = 1e-4
CHECK_BLOCK_SIZE = 1024
FINITE_DIFFERENCE_STEP = 1e-4
FINITE_DIFFERENCE_TOLERANCE = 2e-7
SEED = 20261019


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_sigmoid(value):
    """Unclipped stable logistic function; frozen clipping differs below 5e-18."""
    value = np.asarray(value, float)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1. / (1. + np.exp(-value[positive]))
    e = np.exp(value[~positive])
    result[~positive] = e / (1. + e)
    return result


def independent_check(x, y, center, scale, weights, l2):
    """Rebuild standardized blocks; do not reuse Newton's matrix or gradient.

    Bernoulli loss is evaluated separately for observed positive/negative rows,
    and tanh computes probabilities independently of the solver's expit code.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, dimension = len(x), len(weights)
    gradient = np.zeros(dimension)
    loss = 0.
    for start in range(0, n, CHECK_BLOCK_SIZE):
        stop = min(n, start + CHECK_BLOCK_SIZE)
        block = np.column_stack((np.ones(stop-start), (x[start:stop]-center)/scale))
        margin = block @ weights
        target = y[start:stop]
        # This alternative expression has no subtraction y*large_margin.
        loss += float(np.sum(target*np.logaddexp(0., -margin)
                             + (1.-target)*np.logaddexp(0., margin)))
        probabilities = .5 * (1. + np.tanh(.5 * margin))
        gradient += block.T @ (probabilities-target)
    loss = loss/n + .5*l2/n*float(weights[1:] @ weights[1:])
    gradient /= n
    gradient[1:] += l2/n*weights[1:]
    return float(loss), gradient


def finite_difference_check(x, y, center, scale, weights, gradient, l2):
    rng = np.random.default_rng(SEED)
    records = []
    for _ in range(6):
        direction = rng.normal(size=len(weights))
        direction /= np.linalg.norm(direction)
        epsilon = FINITE_DIFFERENCE_STEP
        plus = independent_check(x, y, center, scale, weights+epsilon*direction, l2)[0]
        minus = independent_check(x, y, center, scale, weights-epsilon*direction, l2)[0]
        numerical = (plus-minus)/(2.*epsilon)
        analytical = float(direction @ gradient)
        records.append({'analytical': analytical, 'finite_difference': numerical,
                        'absolute_error': abs(analytical-numerical)})
    return {'directions': records, 'absolute_tolerance': FINITE_DIFFERENCE_TOLERANCE,
            'passed': all(r['absolute_error'] <= FINITE_DIFFERENCE_TOLERANCE for r in records)}


class ConvergedLogistic:
    """Damped Newton, fixed global convergence requirements across all arms."""
    def __init__(self, l2=1.):
        self.l2 = float(l2)

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 2 or y.shape != (len(x),) or not np.all(np.isin(y, [0., 1.])):
            raise ValueError('Finite binary-labelled matrix required')
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError('Nonfinite input')
        self.center = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-8)
        z = np.column_stack((np.ones(len(x)), (x-self.center)/self.scale))
        penalty = np.full(z.shape[1], self.l2/len(x))
        penalty[0] = 0.

        def objective(weights, hessian=False):
            margin = z @ weights
            p = stable_sigmoid(margin)
            loss = float(np.mean(np.logaddexp(0., margin)-y*margin)
                         + .5*np.sum(penalty*weights*weights))
            gradient = z.T @ (p-y)/len(x) + penalty*weights
            if not hessian:
                return loss, gradient
            # Exact analytic Hessian of the unchanged regularized loss.
            curvature = p*(1.-p)
            h = z.T @ (curvature[:, None]*z)/len(x)
            h.flat[::len(h)+1] += penalty
            return loss, gradient, h

        weights = np.zeros(z.shape[1])
        rate = np.clip(y.mean(), 1e-6, 1.-1e-6)
        weights[0] = np.log(rate/(1.-rate))
        self.path = []
        self.failure = None
        for iteration in range(MAX_NEWTON_STEPS+1):
            loss, gradient, hessian = objective(weights, hessian=True)
            self.path.append({'iteration': iteration, 'loss': loss,
                              'gradient_inf': float(np.max(np.abs(gradient)))})
            if np.max(np.abs(gradient)) <= GRADIENT_TOLERANCE:
                break
            if iteration == MAX_NEWTON_STEPS:
                self.failure = 'PREDECLARED_NEWTON_CAP'
                break
            try:
                direction = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                self.failure = 'SINGULAR_NEWTON_SYSTEM'
                break
            slope = float(gradient @ direction)
            if not np.isfinite(slope) or slope >= 0:
                self.failure = 'NON_DESCENT_DIRECTION'
                break
            accepted = False
            for backtrack in range(MAX_BACKTRACKS):
                step = .5**backtrack
                candidate = weights+step*direction
                candidate_loss, _ = objective(candidate)
                if np.isfinite(candidate_loss) and candidate_loss <= loss + ARMIJO*step*slope:
                    weights = candidate
                    self.path[-1].update(step=step, backtracks=backtrack)
                    accepted = True
                    break
            if not accepted:
                self.failure = 'PREDECLARED_ARMIJO_CAP'
                break
        self.weights = weights
        self.loss, self.gradient = independent_check(x, y, self.center, self.scale, weights, self.l2)
        self.gradient_inf = float(np.max(np.abs(self.gradient)))
        self.finite_difference = finite_difference_check(
            x, y, self.center, self.scale, weights, self.gradient, self.l2)
        self.converged = bool(self.failure is None and self.gradient_inf <= GRADIENT_TOLERANCE
                             and self.finite_difference['passed'] and np.isfinite(self.loss))
        if not self.converged and self.failure is None:
            self.failure = 'INDEPENDENT_CHECK_FAILED'
        self.iterations_run = len(self.path)-1
        return self

    def predict(self, x):
        x = np.asarray(x, float)
        return stable_sigmoid(self.weights[0] + ((x-self.center)/self.scale) @ self.weights[1:])

    def serialization(self):
        return {'center': self.center.tolist(), 'scale': self.scale.tolist(),
                'weights': self.weights.tolist(), 'l2': self.l2,
                'independent_loss': self.loss, 'independent_gradient_inf': self.gradient_inf,
                'newton_iterations': self.iterations_run, 'converged': self.converged,
                'failure': self.failure, 'finite_difference': self.finite_difference,
                'path': self.path}


def self_test():
    rng = np.random.default_rng(831)
    x = rng.normal(size=(220, 9))
    x[:, -1] = x[:, 0]+x[:, 1]  # collinearity challenges the numeric solver.
    x[:, -2] = 1.  # original standardization convention retained.
    target_weights = rng.normal(size=9)
    y = (rng.random(len(x)) < stable_sigmoid(x @ target_weights)).astype(float)
    model = ConvergedLogistic().fit(x, y)
    if not model.converged:
        raise AssertionError(model.failure)
    if not np.array_equal(model.scale, np.maximum(x.std(axis=0), 1e-8)):
        raise AssertionError('Frozen standardization changed')
    # The minimum is stationary and cannot be improved along fixed directions.
    loss0 = independent_check(x, y, model.center, model.scale, model.weights, 1.)[0]
    for _ in range(8):
        direction = rng.normal(size=len(model.weights)); direction /= np.linalg.norm(direction)
        for sign in (-1., 1.):
            perturbed = independent_check(x, y, model.center, model.scale,
                                         model.weights+sign*.01*direction, 1.)[0]
            if perturbed < loss0-1e-11:
                raise AssertionError('Stationary solution failed perturbation check')
    print(json.dumps({'status': 'PASS_SYNTHETIC_ONLY', 'converged': model.converged,
                      'iterations': model.iterations_run, 'gradient_inf': model.gradient_inf,
                      'finite_difference_max_error': max(r['absolute_error'] for r in model.finite_difference['directions'])}))


def assert_protocol(path):
    protocol = json.loads(Path(path).read_text())
    settings = protocol['stage1_numerics']
    expected = {'solver': 'independent_damped_Newton', 'gradient_inf_tolerance': GRADIENT_TOLERANCE,
                'max_newton_steps': MAX_NEWTON_STEPS, 'max_backtracks': MAX_BACKTRACKS,
                'armijo': ARMIJO, 'finite_difference_step': FINITE_DIFFERENCE_STEP,
                'finite_difference_absolute_tolerance': FINITE_DIFFERENCE_TOLERANCE,
                'l2': L2, 'arms': list(ARMS), 'development_only': True}
    for k, value in expected.items():
        if settings.get(k) != value:
            raise ValueError(f'Unfrozen or mismatching numerical requirement: {k}')
    if 'FROZEN' not in protocol.get('status', ''):
        raise ValueError('Supplement protocol must be frozen')
    sources = protocol['stage1_source_sha256']
    for source in ('scripts/check_fp_scorer_convergence.py',
                   'scripts/compare_fp_decision_features.py',
                   'analyze_fp_action_diagnostics.py', 'src/tools/action_fp.py',
                   'analyze_paper_evidence.py'):
        if sources.get(source) != sha(ROOT/source):
            raise ValueError(f'Frozen stage1 source mismatch: {source}')
    for source, expected_hash in protocol['stage1_input_sha256'].items():
        if sha(ROOT/source) != expected_hash:
            raise ValueError(f'Frozen stage1 input changed: {source}')
    return protocol


def run(rows, old, audit):
    old_predictions = old['predictions']
    if len(old_predictions) != len(rows):
        raise ValueError('Original comparison does not match full raw development data')
    identity = ('instance', 'trajectory_id', 'state_id', 'action', 'repeat', 'gain', 'improved')
    for a, b in zip(rows, old_predictions):
        if any(a[k] != b[k] for k in identity):
            raise ValueError('Frozen prediction row identities or targets changed')
    frozen_probabilities = {a: np.asarray([r['probabilities'][a] for r in old_predictions]) for a in ARMS}
    probabilities = {arm: np.zeros(len(rows)) for arm in ARMS}
    cache, fit_records = SignalCache(rows), {}
    instances = sorted({r['instance'] for r in rows})
    for fold, instance in enumerate(instances):
        ids = [i for i, r in enumerate(rows) if r['instance'] == instance]
        training = [r for r in rows if r['instance'] != instance]
        evaluation = [rows[i] for i in ids]
        train_fp, train_control, test_fp, test_control = cache.nested(instance, training, evaluation)
        norms = RandomStateNorms(training)
        labels = np.asarray([float(r['improved']) for r in training])
        fit_records[instance] = {}
        for arm in ARMS:
            started = time.monotonic()
            xt = feature_matrix(training, arm, train_fp, train_control, norms)
            xe = feature_matrix(evaluation, arm, test_fp, test_control, norms)
            # Reconstruct the original frozen scorer solely to verify the exact
            # original feature matrices, initialization and training protocol.
            original = Logistic(l2=L2, iterations=120).fit(xt, labels)
            reproduced = original.predict(xe)
            reproduction_error = float(np.max(np.abs(reproduced-frozen_probabilities[arm][ids])))
            if reproduction_error > 1e-10:
                raise ValueError(f'Original scorer does not reproduce {instance}/{arm}: {reproduction_error}')
            model = ConvergedLogistic(l2=L2).fit(xt, labels)
            probabilities[arm][ids] = model.predict(xe)
            previous_check = independent_check(xt, labels, original.center, original.scale, original.weights, L2)
            record = model.serialization()
            record.update(training_rows=len(training), parameters=len(model.weights),
                frozen_reproduction_max_abs_probability_error=reproduction_error,
                frozen_original_loss=previous_check[0],
                frozen_original_gradient_inf=float(np.max(np.abs(previous_check[1]))),
                original_independent_loss_difference=model.loss-previous_check[0],
                max_abs_probability_change=float(np.max(np.abs(probabilities[arm][ids]-frozen_probabilities[arm][ids]))),
                mean_abs_probability_change=float(np.mean(np.abs(probabilities[arm][ids]-frozen_probabilities[arm][ids]))),
                elapsed_seconds=time.monotonic()-started,
                original_center=original.center.tolist(), original_scale=original.scale.tolist(),
                original_weights=original.weights.tolist())
            fit_records[instance][arm] = record
            print(json.dumps({'fold': fold+1, 'heldout_instance': instance, 'arm': arm,
                'newton_steps': model.iterations_run, 'gradient_inf': model.gradient_inf,
                'converged': model.converged, 'elapsed_seconds': record['elapsed_seconds']}), flush=True)
    # Reuse exactly the already frozen metric aggregation, bootstrap and Holm
    # family.  It never runs a searcher or changes the action-selection target.
    current_summary = summarize(rows, probabilities)
    old_state = {(r['instance'], r['trajectory_id'], r['state_id']): r for r in old['summary']['state_decisions']}
    changes = {arm: Counter() for arm in ARMS}
    for row in current_summary['state_decisions']:
        previous = old_state[row['instance'], row['trajectory_id'], row['state_id']]
        for arm in ARMS:
            new, prior = row['arms'][arm], previous['arms'][arm]
            if new['action'] != prior['action']:
                changes[arm]['changed'] += 1
                delta = new['gain']-prior['gain']
                changes[arm]['beneficial' if delta > 1e-12 else 'harmful' if delta < -1e-12 else 'same_gain'] += 1
            changes[arm]['states'] += 1
    failures = [{'instance': i, 'arm': a, 'failure': fit_records[i][a]['failure']}
                for i in instances for a in ARMS if not fit_records[i][a]['converged']]
    deltas = {arm: {m: current_summary['macro_instance_trajectory_metrics'][arm][m]
                         - old['summary']['macro_instance_trajectory_metrics'][arm][m]
                   for m in ('ap', 'brier', 'selected_gain', 'regret', 'action_rank')}
              for arm in ARMS}
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'independent_numerical_robustness_development_only',
        'status': 'ALL_112_FITS_CONVERGED' if not failures else 'NUMERICAL_FAILURES_RETAINED',
        'data_audit': audit, 'fit_count': sum(map(len, fit_records.values())),
        'failed_fits': failures, 'fit_records': fit_records,
        'frozen_summary': old['summary'], 'converged_summary': current_summary,
        'macro_metric_change_vs_frozen': deltas,
        'choice_changes_vs_frozen': {a: dict(v) for a, v in changes.items()},
        'predictions': [{**{k: row[k] for k in ('instance', 'trajectory_id', 'state_id', 'generation', 'action', 'repeat', 'gain', 'improved')},
            'probabilities': {a: float(probabilities[a][i]) for a in ARMS},
            'frozen_probabilities': {a: float(frozen_probabilities[a][i]) for a in ARMS}}
            for i, row in enumerate(rows)],
        'limitations': ['Development-only numerical robustness; no independent confirmation.',
            'All original models, labels, features and outcomes are preserved; only convex numerical solver differs.',
            'No old validation refit, locked-test access, search rerun, reward change or model configuration search.',
            'Finite-difference and first-order stationarity checks are numerical checks, not independent scientific replication.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--protocol')
    parser.add_argument('--development', default='results/fp_v2_development')
    parser.add_argument('--audit', default='projects/analysis/fp_action_development_independent_audit.json')
    parser.add_argument('--frozen-scores', default='projects/analysis/fp_agent_decision_feature_comparison.json')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.protocol or not args.output:
        parser.error('--protocol and --output are required for formal fitting')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Original and supplementary outcomes must never be overwritten')
    assert_protocol(args.protocol)
    rows, audit = load_audited_development(args.development, args.audit)
    old = json.loads(Path(args.frozen_scores).read_text())
    started = time.monotonic()
    result = run(rows, old, audit)
    result['protocol_sha256'] = sha(args.protocol)
    result['input_sha256'] = {str(p): sha(p) for p in (args.audit, args.frozen_scores)}
    result['runtime_seconds'] = time.monotonic()-started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'status': result['status'], 'runtime_seconds': result['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()

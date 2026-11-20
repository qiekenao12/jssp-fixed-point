"""Strict accounting and experimental controls around the existing scheduler.

No search operator or numerical search parameter is defined here. The only
search intervention is the common objective-query cap, including cache hits.
"""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import time

import numpy as np

from tools.executor import Executor


ACTIVE_QUERY_METER = ContextVar('paper_q_meter', default=None)


class QueryBudgetExhausted(Exception):
    """The common objective-query budget has been spent."""


class VNDCallExhausted(Exception):
    """A VND invocation reached its reserved query boundary."""


@dataclass(frozen=True)
class QControls:
    policy: str = 'agent_fp'
    feedback: bool = True
    fixed_period: int = 10
    stagnation_threshold: int = 8

    def non_fp_signal(self, generation, stagnant_generations):
        eligible = (
            generation % self.fixed_period == 0 if self.policy == 'fixed'
            else stagnant_generations >= self.stagnation_threshold
        )
        return {
            'eligible': bool(eligible),
            'stagnant_generations': int(stagnant_generations),
            'fixed_clock': self.policy == 'fixed',
            'reason': 'fixed_generation_clock' if self.policy == 'fixed'
            else 'observer_stagnation',
        }


class QueryMeter:
    def __init__(self, budget, vnd_call_queries=500, vnd_total_queries=8000):
        if not isinstance(budget, int) or budget < 1:
            raise ValueError('objective query budget must be a positive integer')
        if not 2 <= vnd_call_queries <= vnd_total_queries <= budget:
            raise ValueError('require 2 <= VND call cap <= VND total cap <= query budget')
        self.budget = budget
        self.vnd_call_queries = vnd_call_queries
        self.vnd_total_queries = vnd_total_queries
        self.queries = 0
        self.decodes = 0
        self.vnd_queries = 0
        self.sa_queries = 0
        self.phase = 'initialization'
        self.vnd_depth = 0
        self.sa_depth = 0
        self.call_ceiling = None
        self.call_best = None
        self.call_cost = float('inf')
        self.best_cost = float('inf')
        self.best_individual = None
        self.phase_queries = Counter()
        self.phase_decodes = Counter()
        self.component_calls = Counter()
        self.component_seconds = Counter()
        self.trace = []
        self.known_costs = {}
        self.vnd_call_log = []
        self.started = time.perf_counter()
        self.initial_fingerprint = None

    @property
    def is_vnd(self):
        return self.vnd_depth > 0 or self.phase == 'vnd'

    @property
    def vnd_remaining(self):
        return self.vnd_total_queries - self.vnd_queries

    @contextmanager
    def activate(self):
        token = ACTIVE_QUERY_METER.set(self)
        try:
            yield self
        finally:
            ACTIVE_QUERY_METER.reset(token)

    def before_query(self):
        if self.queries >= self.budget:
            raise QueryBudgetExhausted
        if self.call_ceiling is not None and self.queries >= self.call_ceiling:
            raise VNDCallExhausted
        if self.is_vnd and self.vnd_remaining <= 0:
            raise VNDCallExhausted
        self.queries += 1
        self.phase_queries[self.phase] += 1
        if self.is_vnd:
            self.vnd_queries += 1
        if self.sa_depth:
            self.sa_queries += 1

    def after_query(self, individual, cost):
        self.known_costs[tuple(int(v) for v in individual)] = cost
        if self.call_ceiling is not None and cost < self.call_cost:
            self.call_cost, self.call_best = cost, individual.copy()
        if cost < self.best_cost:
            self.best_cost, self.best_individual = cost, individual.copy()
            self.trace.append({
                'query': self.queries, 'makespan': int(cost),
                'wall_seconds': time.perf_counter() - self.started,
                'phase': self.phase,
            })

    def record_decode(self):
        self.decodes += 1
        self.phase_decodes[self.phase] += 1

    @contextmanager
    def component(self, name):
        old_phase = self.phase
        if name == 'local_search' and old_phase == 'ga':
            self.phase = 'local_search'
        self.component_calls[name] += 1
        started = time.perf_counter()
        try:
            yield
        finally:
            self.component_seconds[name] += time.perf_counter() - started
            self.phase = old_phase


class QExecutor(Executor):
    """Original Executor with accounting, common caps, and no new operators."""
    def __init__(self, instance, meter, vnd_enabled=True):
        super().__init__(instance)
        self.query_meter = meter
        self.vnd_enabled = bool(vnd_enabled)

    @property
    def vnd_query_available(self):
        return self.vnd_enabled and self.query_meter.vnd_remaining > 0

    def evaluate(self, individual):
        meter = self.query_meter
        sa_verification = getattr(self, '_sa_verification_pending', False)
        self._sa_verification_pending = False
        if sa_verification:
            meter.sa_depth += 1
        try:
            meter.before_query()
            cost = super().evaluate(individual)
            meter.after_query(individual, cost)
            return cost
        finally:
            if sa_verification:
                meter.sa_depth -= 1

    def evaluate_population(self, individuals):
        meter = self.query_meter
        if meter.phase == 'initialization' and meter.initial_fingerprint is None:
            meter.initial_fingerprint = hashlib.sha256(
                np.asarray(individuals, dtype=np.int64).tobytes()
            ).hexdigest()
        return super().evaluate_population(individuals)

    def evolve_generation(self, *args, **kwargs):
        with self.query_meter.component('ga'):
            return super().evolve_generation(*args, **kwargs)

    def evaluate_local_search_result(self, individual):
        previous = self.query_meter.phase
        self.query_meter.phase = 'local_search'
        try:
            return self.evaluate(individual)
        finally:
            self.query_meter.phase = previous

    def local_search_improved(self, *args, **kwargs):
        with self.query_meter.component('local_search'):
            return super().local_search_improved(*args, **kwargs)

    def critical_path_local_search(self, *args, **kwargs):
        with self.query_meter.component('critical_path_local_search'):
            # A critical-path LS inside a VND/terminal stays in its parent
            # phase; generation-level critical LS is charged to ordinary LS.
            old_phase = self.query_meter.phase
            if old_phase == 'ga':
                self.query_meter.phase = 'local_search'
            try:
                return super().critical_path_local_search(*args, **kwargs)
            finally:
                self.query_meter.phase = old_phase

    def search_candidate(self, individual, method='local_search_vnd', **kwargs):
        meter = self.query_meter
        if not self.vnd_enabled:
            # VND is a capability boundary, including terminal VND starts.
            # Terminal critical-path LS and both SA methods remain available.
            return individual.copy(), meter.known_costs[tuple(int(v) for v in individual)]
        # Every scheduler VND invocation (including terminal VND and the
        # existing non-final improved-LS fallback) includes caller verification.
        remaining = min(meter.vnd_call_queries, meter.vnd_remaining)
        if remaining < 2:
            return individual.copy(), meter.known_costs[tuple(int(v) for v in individual)]
        before = meter.queries
        started = time.perf_counter()
        meter.component_calls['vnd'] += 1
        meter.vnd_depth += 1
        meter.call_best = individual.copy()
        meter.call_cost = meter.known_costs[tuple(int(v) for v in individual)]
        # Reserve one objective query for the original caller verification.
        meter.call_ceiling = min(meter.budget, before + remaining - 1)
        capped = False
        try:
            try:
                candidate = getattr(super(), method)(individual, **kwargs)
            except VNDCallExhausted:
                capped = True
                candidate = meter.call_best.copy()
            finally:
                meter.call_ceiling = None
            cost = self.evaluate(candidate)
            return candidate, cost
        finally:
            meter.vnd_depth -= 1
            elapsed = time.perf_counter() - started
            meter.component_seconds['vnd'] += elapsed
            meter.vnd_call_log.append({
                'phase': meter.phase, 'queries': meter.queries - before,
                'capped': capped, 'wall_seconds': elapsed,
            })

    def _sa(self, method, *args, **kwargs):
        meter = self.query_meter
        meter.sa_depth += 1
        try:
            with meter.component(method):
                candidate = getattr(super(), method)(*args, **kwargs)
            self._sa_verification_pending = True
            return candidate
        finally:
            meter.sa_depth -= 1

    def simulated_annealing_refine(self, *args, **kwargs):
        return self._sa('simulated_annealing_refine', *args, **kwargs)

    def reference_sa_search(self, *args, **kwargs):
        return self._sa('reference_sa_search', *args, **kwargs)


def record_decode():
    meter = ACTIVE_QUERY_METER.get()
    if meter is not None:
        meter.record_decode()

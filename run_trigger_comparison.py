#!/usr/bin/env python3
"""Trigger-only JSSP experiment with equal objective-query budgets and detached probes."""
import argparse
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from run_ablation_test import parse_jobshop1, TEST_INSTANCES
from tools.benchmark_data import BKS, bks_metadata, validate_makespan
from tools.controller import Controller, GAParams
from tools.executor import Executor, Population
from tools.fixed_point_detector import FixedPointDetector
from tools.jssp_core import JSSPUtils
from tools.meta_scheduler import MetaScheduler
from run_orchestration_baselines import make_instance

TRIGGERS = ("fixed", "stagnation", "state_drift", "fp_residual")


@dataclass(frozen=True)
class Protocol:
    seed: int = 20260916
    population: int = 80
    queries: int = 30000
    vnd_queries: int = 8000
    call_queries: int = 500
    vnd_iterations: int = 100
    ls_interval: int = 10
    ls_queries: int = 150
    ls_iterations: int = 20
    check_interval: int = 5
    warmup: int = 85
    cooldown: int = 10
    fixed_period: int = 20
    stagnant_generations: int = 15
    drift_threshold: float = 0.012
    fp_threshold: float = 0.012


def seed_for(protocol, name, run, generation, component):
    value = f"{protocol.seed}:{name}:{run}:{generation}:{component}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")


@contextmanager
def rng_stream(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class QueryLimit(Exception):
    pass


class MeteredExecutor(Executor):
    def __init__(self, instance, limit):
        super().__init__(instance)
        self.limit = limit
        self.queries = 0
        self.decodes = 0
        self.phase = "initialization"
        self.phase_queries = Counter()
        self.best_cost = float("inf")
        self.best_individual = None
        self.call_end = None
        self.call_best = None
        self.call_cost = float("inf")

    def evaluate(self, individual):
        if self.queries >= self.limit or (
            self.call_end is not None and self.queries >= self.call_end
        ):
            raise QueryLimit
        self.queries += 1
        self.phase_queries[self.phase] += 1
        key = tuple(int(value) for value in individual)
        self.decodes += int(key not in self._fitness_cache)
        cost = super().evaluate(individual)
        if cost < self.best_cost:
            self.best_cost, self.best_individual = cost, individual.copy()
        if self.call_end is not None and cost < self.call_cost:
            self.call_cost, self.call_best = cost, individual.copy()
        return cost

    def search(self, individual, cost, queries, iterations, component):
        started = time.perf_counter()
        before = self.queries
        self.phase = component
        self.call_end = min(self.limit, before + queries)
        self.call_cost, self.call_best = cost, individual.copy()
        capped = False
        try:
            if component in ("vnd", "probe"):
                self.local_search_vnd(individual.copy(), max_iter=iterations)
            else:
                self.local_search_improved(individual.copy(), max_iter=iterations)
        except QueryLimit:
            capped = True
        finally:
            self.call_end = None
        return self.call_best.copy(), self.call_cost, {
            "queries": self.queries - before, "capped": capped,
            "gain": float(cost - self.call_cost),
            "effective": bool(self.call_cost < cost),
            "elapsed": time.perf_counter() - started,
        }


def recommendations(features, protocol):
    return {
        "fixed": (features["generation"] - protocol.warmup) % protocol.fixed_period == 0,
        "stagnation": features["stagnant_generations"] >= protocol.stagnant_generations,
        "state_drift": features["state_drift"] is not None
        and features["state_drift"] <= protocol.drift_threshold,
        "fp_residual": features["fp_reliable"]
        and features["fp_residual"] <= protocol.fp_threshold,
    }


def run_trajectory(name, run, trigger, protocol, data, collect_probes=True):
    instance = make_instance(data)
    executor = MeteredExecutor(instance, protocol.queries)
    detector = FixedPointDetector(
        instance.num_jobs * instance.num_machines,
        max_generations=protocol.queries, model_mode="operator",
    )
    controller = Controller(instance.num_jobs, instance.num_machines)
    with rng_stream(seed_for(protocol, name, run, 0, "initialization")):
        scheduler = MetaScheduler(instance)
        individuals = scheduler._init_population(protocol.population)
        costs = executor.evaluate_population(individuals)
    initial_fingerprint = hashlib.sha256(
        np.asarray(individuals, dtype=np.int64).tobytes()
    ).hexdigest()
    population = Population(individuals, costs)
    params = GAParams(
        pc=0.85, pm=0.05, n=2, tournament_size=3,
        selection_method="tournament", selection_strategy="tournament",
        population_size=protocol.population,
    )
    params.crossover_rate, params.mutation_rate = 0.85, 0.05
    probe_executor = MeteredExecutor(instance, 10**12) if trigger == "probe" else None
    last_best, last_improvement, last_call = min(costs), 0, -protocol.cooldown
    previous_state = None
    generation, events, observations = 0, [], []
    started = time.perf_counter()
    while executor.queries < protocol.queries:
        generation += 1
        executor.phase = "ga"
        try:
            with rng_stream(seed_for(protocol, name, run, generation, "ga")):
                population, _ = executor.evolve_generation(
                    population, params, generation=generation,
                    max_generations=protocol.queries, apply_local_search=False,
                    diversity=JSSPUtils.calculate_diversity(population.individuals),
                )
        except QueryLimit:
            break
        if generation % protocol.ls_interval == 0:
            best_index = population.best_index
            with rng_stream(seed_for(protocol, name, run, generation, "ls")):
                candidate, cost, _ = executor.search(
                    population.individuals[best_index], population.best_fitness,
                    protocol.ls_queries, protocol.ls_iterations, "ls",
                )
            population.individuals[best_index] = candidate
            population.fitness_values[best_index] = cost
        if population.best_fitness < last_best:
            last_improvement, last_best = generation, population.best_fitness
        if generation % protocol.check_interval != 0:
            continue
        observation_started = time.perf_counter()
        status, _ = detector.detect(
            population.individuals, population.fitness_values, controller, generation,
        )
        state = detector.history["search_states"][-1]
        vector = np.asarray([state[key] for key in ("best", "mean", "dispersion", "diversity")])
        weights = np.sqrt([0.35, 0.30, 0.15, 0.20])
        drift = None if previous_state is None else float(np.linalg.norm(weights * (vector - previous_state)))
        change = None if previous_state is None else float(abs(vector[0] - previous_state[0]))
        previous_state = vector
        ready = status.operator_transitions >= detector.criteria.operator_min_transitions
        features = {
            "generation": generation, "queries_before": executor.queries,
            "progress": executor.queries / protocol.queries,
            "stagnant_generations": generation - last_improvement,
            "state_drift": drift, "fitness_change": change,
            "fp_residual": float(status.operator_residual) if ready else None,
            "fp_fit_error": float(status.operator_fit_error) if ready else None,
            "fp_transitions": status.operator_transitions,
            "fp_reliable": bool(ready and status.operator_fit_error <= detector.criteria.operator_fit_threshold),
            "observation_elapsed": time.perf_counter() - observation_started,
            "eligible": generation >= protocol.warmup,
        }
        signals = recommendations(features, protocol)
        features["recommendations"] = signals
        observations.append(features)
        if not features["eligible"]:
            continue
        if trigger == "probe":
            if not collect_probes:
                continue
            target_executor = probe_executor
            component, allowed = "probe", protocol.call_queries
        else:
            if not signals[trigger] or generation - last_call < protocol.cooldown:
                continue
            allowed = min(
                protocol.call_queries,
                protocol.vnd_queries - executor.phase_queries["vnd"],
                protocol.queries - executor.queries,
            )
            if allowed <= 0:
                continue
            target_executor, component = executor, "vnd"
        best_index = population.best_index
        before = population.best_fitness
        with rng_stream(seed_for(protocol, name, run, generation, "vnd")):
            candidate, cost, call = target_executor.search(
                population.individuals[best_index], before,
                allowed, protocol.vnd_iterations, component,
            )
        event = {
            **features, **call, "before": float(before), "after": float(cost),
            "gain_bks_pp": call["gain"] / BKS[name] * 100,
            "relative_gain": call["gain"] / max(float(before), 1),
        }
        if trigger != "probe":
            population.individuals[best_index] = candidate
            population.fitness_values[best_index] = cost
            last_call = generation
            if cost < last_best:
                last_improvement, last_best = generation, cost
        events.append(event)
    validate_makespan(name, executor.best_cost, BKS[name])
    assert executor.queries == protocol.queries
    assert executor.phase_queries["vnd"] <= protocol.vnd_queries
    return {
        "instance": name, "run": run, "trigger": trigger,
        "initial_fingerprint": initial_fingerprint,
        "makespan": int(executor.best_cost),
        "gap": (executor.best_cost - BKS[name]) / BKS[name] * 100,
        "elapsed": time.perf_counter() - started,
        "queries": executor.queries, "decodes": executor.decodes,
        "phase_queries": dict(executor.phase_queries),
        "generations": generation,
        "probe_queries": 0 if probe_executor is None else probe_executor.queries,
        "vnd_calls": len(events), "vnd_effective": sum(event["effective"] for event in events),
        "events": events, "observations": observations,
    }


def work_item(task):
    name, run, protocol, data = task
    return [run_trajectory(name, run, trigger, protocol, data)
            for trigger in (*TRIGGERS, "probe")]


def digest_sources():
    paths = [
        Path(__file__), ROOT / "src/tools/executor.py",
        ROOT / "src/tools/fixed_point_detector.py", ROOT / "src/tools/jssp_core.py",
        ROOT / "src/tools/meta_scheduler.py", ROOT / "jobshop1.txt",
    ]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default=",".join(TEST_INSTANCES))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--queries", type=int, default=30000)
    parser.add_argument("--vnd-queries", type=int, default=8000)
    parser.add_argument("--call-queries", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "results/trigger_controlled")
    args = parser.parse_args()
    protocol = Protocol(queries=args.queries, vnd_queries=args.vnd_queries, call_queries=args.call_queries)
    if args.runs < 1 or args.workers < 1 or args.queries < 10000:
        parser.error("runs/workers must be positive and queries >= 10000 (FP warm-up).")
    if not 0 < args.call_queries <= args.vnd_queries <= args.queries:
        parser.error("require 0 < call-queries <= vnd-queries <= queries")
    data = parse_jobshop1(str(ROOT / "jobshop1.txt"))
    instances = args.instances.split(",")
    if any(name not in data or name not in BKS for name in instances) or len(set(instances)) != len(instances):
        parser.error("instances must be distinct known benchmark names")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": asdict(protocol), "instances": instances, "runs": args.runs,
        "source_hashes": digest_sources(), "bks": {name: BKS[name] for name in instances},
        "bks_metadata": bks_metadata(),
        "budget_unit": "objective queries including cache hits; actual decodes also reported",
        "controlled_components": "Fixed GA parameters; scheduled LS; best-individual capped VND; terminal refinement disabled for all.",
        "fp_rule": "Reliable affine operator residual <= threshold; no legacy stage/stagnation gates.",
        "drift_rule": "Same normalized state and weighted L2 norm, one observation lag.",
        "probes": "All eligible states on independent GA+LS trajectories; never committed; RNG/cache/budget isolated.",
        "query_cap_behavior": "All search operators retain the best evaluated candidate on cap; unused VND allocation goes to GA+LS.",
        "runtime_note": "All arms compute FP diagnostics; timings include observation overhead, not native trigger-only runtime.",
        "threshold_note": "Frozen heuristic thresholds, no tuning on reported outcomes; no universal irreplaceability claim.",
        "python": sys.version, "numpy": np.__version__,
    }
    meta_path = args.output / "protocol.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Output directory belongs to a different protocol/source version")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    tasks = [(name, run, protocol, data[name]) for name in instances for run in range(args.runs)
             if not (args.output / f"{name}_{run:02d}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for group in pool.map(work_item, tasks):
            assert len({row["initial_fingerprint"] for row in group}) == 1
            path = args.output / f'{group[0]["instance"]}_{group[0]["run"]:02d}.json'
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(group, ensure_ascii=False, allow_nan=False))
            temporary.replace(path)
            print(f'{path.stem}: ' + " / ".join(
                f'{row["trigger"]} Gap={row["gap"]:.3f} calls={row["vnd_calls"]}'
                for row in group[:4]), flush=True)
    print(f"Completed {len(instances) * args.runs} paired groups: {args.output}", flush=True)


if __name__ == "__main__":
    main()

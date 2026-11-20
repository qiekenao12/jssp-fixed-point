#!/usr/bin/env python3
"""Frozen saved-state repeatability audit; no training or online controller.

The equal-work treatment repeats the same existing primitive under a common
480-request ceiling. It is an experimental intervention, never a new action
policy. Initialization fitness is known history, and all requests (including
cache hits) are counted. No dummy requests are made to fill a ceiling.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_fp_action_diagnostics import (ACTIONS, Protocol, branch, canonical,
    clone, fingerprint, parameters, sha, sources, stream_seed, vector)
from run_ablation_test import parse_jobshop1
from run_orchestration_baselines import make_instance
from run_trigger_comparison import rng_stream
from tools.executor import Population
from tools.jssp_core import JSSPUtils
from tools.paper_q import QueryBudgetExhausted, QueryMeter, QExecutor

SAMPLING = {'runs': [0, 1, 2, 3, 4], 'generations': [40, 100, 160]}
NEW_ATOMIC_REPEATS = [2, 3, 4, 5, 6]
WORK_REPEATS = [0, 1, 2, 3, 4]
WORK_QUERIES = 480
TOL = 1e-12


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Preserving existing evidence: {path}')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def plan(path):
    manifest_path = ROOT / 'projects/experiments/fp_v2_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    master_path = ROOT / 'projects/experiments/fp_submission_supplement_protocol.json'
    master = json.loads(master_path.read_text())
    saved, timings = [], []
    for name in manifest['splits']['development']:
        for run in SAMPLING['runs']:
            source = ROOT / f'results/fp_v2_development/{name}_{run:02d}.json'
            doc = json.loads(source.read_text())
            archive = ROOT / doc['population_archive']
            assert sha(archive) == doc['population_archive_sha256']
            selected = [s for s in doc['snapshots'] if s['generation'] in SAMPLING['generations']]
            assert len(selected) == len(SAMPLING['generations'])
            saved.append({'instance': name, 'run': run, 'trajectory_id': doc['trajectory_id'],
                'source_json': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
                'population_archive': doc['population_archive'],
                'population_archive_sha256': sha(archive),
                'states': [{'state_id': f'{name}:{run}:{s["generation"]}',
                            'generation': s['generation'], 'fingerprint': s['fingerprint']}
                           for s in selected]})
            # Historical duration is used only for capacity planning, not sample selection.
            timings.extend(r for r in doc['records'] if r['generation'] in SAMPLING['generations'])
    nstates = sum(len(r['states']) for r in saved)
    atomic_cpu = sum(r['wall_seconds'] for r in timings) * len(NEW_ATOMIC_REPEATS)/2
    work_cpu = sum(r['wall_seconds']*WORK_QUERIES/r['queries'] for r in timings) * len(WORK_REPEATS)/2
    payload = {'stage': 'submission_evidence_repeatability_v1',
        'master_protocol_path': str(master_path.relative_to(ROOT)),
        'master_protocol_sha256': sha(master_path),
        'sampling': SAMPLING, 'sampling_uses_outcomes': False,
        'split': 'development', 'manifest_sha256': sha(manifest_path),
        'data_path': manifest['data_path'], 'data_sha256': manifest['data_sha256'],
        'instances': manifest['splits']['development'], 'saved_trajectories': saved,
        'protocol': asdict(Protocol()), 'atomic_new_repeats': NEW_ATOMIC_REPEATS,
        'atomic_original_repeats': [0, 1], 'equal_work_repeats': WORK_REPEATS,
        'equal_work_query_ceiling': WORK_QUERIES,
        'equal_work_definition': {
            'unit': 'actual_objective_requests_including_cache_hits',
            'GA': 'repeat existing pure full generations; 80 requests/generation, exactly six generations',
            'LS': 'repeat existing improved-LS max_iter100 and caller verification from incumbent; final real call may be truncated',
            'VND': 'repeat existing VND max_iter100 from incumbent after natural return; each call reserves caller verification',
            'termination': '480 real requests, except VND can stop at479 when remaining1 cannot start valid call',
            'incumbent': 'best actually evaluated candidate always retained, including incomplete final call',
            'no_padding': True, 'not_equal_wall_time_or_structural_decodes': True,
            'not_controller_or_paper_innovation': True},
        'seeds': {'atomic': 'original stream_seed(Protocol seed20261018, instance,run,generation,branch_ACTION_repeat)',
                  'equal_work': 'original stream_seed with component submission_equal480_ACTION_repeat'},
        'estimates': {'instances': len(manifest['splits']['development']),
            'trajectories': len(saved), 'states': nstates,
            'new_atomic_branches': nstates*3*len(NEW_ATOMIC_REPEATS),
            'new_equal_work_branches': nstates*3*len(WORK_REPEATS),
            'new_atomic_request_upper_bound': nstates*3*len(NEW_ATOMIC_REPEATS)*500,
            'new_equal_work_request_upper_bound': nstates*3*len(WORK_REPEATS)*WORK_QUERIES,
            'historical_based_atomic_cpu_seconds': atomic_cpu,
            'historical_based_equal_work_cpu_seconds': work_cpu,
            'wall_estimate_8workers_minutes': [1, 3], 'conservative_wall_minutes': 10,
            'cpu_estimate_is_not_a_strict_runtime_bound': True},
        'analysis': {'repeat_summary': 'atomic original2/new5/pooled7 separately; work5 separately',
            'uncertainty': 'state t intervals descriptive; instance->trajectory paired bootstrap10000 for policy differences',
            'original_oracle': 'choose by original2, evaluate chosen action on independent new5; report GA/LS/VND fixed baselines',
            'winner_consistency': 'tie-aware repeat best-set overlap; leave-one-repeat-out action choice on independent omitted repeat',
            'selectors': 'reuse frozen original/converged scores supplied separately; no fitting on branch returns',
            'inference': 'development robustness only; no new validation or confirmation claim'},
        'locked_test_access': False,
        'sources': {**sources(), 'scripts/fp_empirical_repeatability.py': sha(__file__),
            'scripts/analyze_fp_empirical_repeatability.py': sha(ROOT/'scripts/analyze_fp_empirical_repeatability.py'),
            'scripts/audit_fp_action_data.py': sha(ROOT/'scripts/audit_fp_action_data.py'),
            'tests/test_fp_empirical_repeatability.py': sha(ROOT/'tests/test_fp_empirical_repeatability.py')}}
    payload['protocol_hash'] = canonical(payload)
    write_json(path, payload)
    return payload


def install_incumbent(pop, meter):
    if meter.best_cost < pop.best_fitness:
        index = pop.best_index
        pop.individuals[index] = meter.best_individual.copy()
        pop.fitness_values[index] = meter.best_cost


def work_window(instance, original, action, seed, generation, protocol, scale,
                ceiling=WORK_QUERIES):
    pop = clone(original)
    meter = QueryMeter(ceiling, ceiling, ceiling)
    executor = QExecutor(instance, meter)
    meter.best_cost = float(pop.best_fitness)
    meter.best_individual = pop.best_individual.copy()
    meter.known_costs.update({tuple(map(int, x)): float(y)
                             for x, y in zip(pop.individuals, pop.fitness_values)})
    meter.phase = action.lower()
    calls, reason = [], 'request_ceiling'
    start = time.perf_counter()
    with rng_stream(seed), meter.activate():
        while meter.queries < ceiling:
            remaining = ceiling-meter.queries
            if action == 'VND' and remaining < 2:
                reason = 'insufficient_minimum_call'
                break
            if action == 'GA' and remaining < len(pop.individuals):
                reason = 'insufficient_complete_generation'
                break
            before = meter.queries
            outcome = 'natural_return'
            try:
                if action == 'GA':
                    pop, _ = executor.evolve_generation(pop, parameters(protocol),
                        generation=generation+len(calls), max_generations=protocol.generations,
                        apply_local_search=False,
                        diversity=JSSPUtils.calculate_diversity(pop.individuals))
                elif action == 'LS':
                    candidate = executor.local_search_improved(pop.best_individual.copy(),
                        max_iter=protocol.local_iterations)
                    executor.evaluate(candidate)
                elif action == 'VND':
                    executor.search_candidate(pop.best_individual.copy(),
                        max_iter=protocol.local_iterations)
                    if meter.vnd_call_log[-1]['capped']:
                        outcome = 'request_boundary_truncation'
                else:
                    raise ValueError(action)
            except QueryBudgetExhausted:
                assert action == 'LS' and meter.queries == ceiling
                outcome = 'request_boundary_truncation'
            install_incumbent(pop, meter)
            used = meter.queries-before
            assert used > 0, 'A valid repeated primitive must perform real search requests'
            calls.append({'call': len(calls), 'queries': used, 'termination': outcome})
    install_incumbent(pop, meter)
    return {'end_x': vector(pop, scale), 'end_makespan': int(pop.best_fitness),
        'gain': float((original.best_fitness-pop.best_fitness)/scale),
        'improved': bool(pop.best_fitness < original.best_fitness),
        'queries': meter.queries, 'decodes': meter.decodes,
        'wall_seconds': time.perf_counter()-start,
        'best_chromosome': list(map(int, pop.best_individual)),
        'termination_reason': reason, 'unused_queries': ceiling-meter.queries,
        'primitive_calls': calls, 'vnd_call_log': meter.vnd_call_log,
        'incumbent_trace': meter.trace,
        'window': {'unit': 'objective_request_ceiling', 'span': ceiling,
                   'action_sequence': [action]*len(calls)},
        'elapsed_ga_generations': len(calls) if action == 'GA' else 0}


def run_task(task):
    output, saved, data, protocol_hash = task
    target = Path(output)/f'{saved["instance"]}_{saved["run"]:02d}.json'
    if target.exists():
        existing = json.loads(target.read_text())
        assert existing['protocol_hash'] == protocol_hash
        return {'file': target.name, 'existing': True}
    source = ROOT/saved['source_json']
    assert sha(source) == saved['source_sha256']
    old = json.loads(source.read_text())
    archive = ROOT/saved['population_archive']
    assert sha(archive) == saved['population_archive_sha256']
    instance, protocol = make_instance(data), Protocol()
    records = []
    with np.load(archive) as arrays:
        for state in saved['states']:
            i, = np.flatnonzero(arrays['generations'] == state['generation'])
            original = Population([x.copy() for x in arrays['populations'][i]],
                                  list(map(float, arrays['fitness'][i])))
            assert fingerprint(original) == state['fingerprint']
            for window, repeats in [('atomic', NEW_ATOMIC_REPEATS), ('equal480', WORK_REPEATS)]:
                for action in ACTIONS:
                    for repeat in repeats:
                        component = (f'branch_{action}_{repeat}' if window == 'atomic' else
                                     f'submission_equal480_{action}_{repeat}')
                        seed = stream_seed(protocol, saved['instance'], saved['run'],
                                           state['generation'], component)
                        function = branch if window == 'atomic' else work_window
                        outcome = function(instance, original, action, seed, state['generation'],
                                           protocol, old['s0'])
                        assert fingerprint(original) == state['fingerprint']
                        assert outcome['end_makespan'] <= original.best_fitness
                        records.append({'instance': saved['instance'], 'run': saved['run'],
                            'trajectory_id': saved['trajectory_id'], **state, 'window_mode': window,
                            'action': action, 'repeat': repeat, 'seed': seed,
                            's0': old['s0'], 'start_makespan': int(original.best_fitness),
                            'start_x': vector(original, old['s0']), **outcome})
    result = {'protocol_hash': protocol_hash, 'instance': saved['instance'],
        'run': saved['run'], 'trajectory_id': saved['trajectory_id'],
        'source_json': saved['source_json'], 'source_sha256': saved['source_sha256'],
        'population_archive': saved['population_archive'],
        'population_archive_sha256': saved['population_archive_sha256'],
        'records': records}
    write_json(target, result)
    return {'file': target.name, 'records': len(records),
            'queries': sum(r['queries'] for r in records)}


def run(lock, output, workers):
    protocol = json.loads(Path(lock).read_text())
    assert canonical({k:v for k,v in protocol.items() if k != 'protocol_hash'}) == protocol['protocol_hash']
    assert protocol['split'] == 'development' and not protocol['locked_test_access']
    assert sha(ROOT/protocol['master_protocol_path']) == protocol['master_protocol_sha256']
    for name, value in protocol['sources'].items():
        assert sha(ROOT/name) == value, f'Frozen source changed: {name}'
    assert sha(ROOT/protocol['data_path']) == protocol['data_sha256']
    data = parse_jobshop1(str(ROOT/protocol['data_path']))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    tasks = [(str(output), saved, data[saved['instance']], protocol['protocol_hash'])
             for saved in protocol['saved_trajectories']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_task, tasks, chunksize=1):
            print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'run'])
    parser.add_argument('--protocol', default='projects/experiments/fp_empirical_repeatability_protocol.json')
    parser.add_argument('--output', default='results/fp_empirical_repeatability_development')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.mode == 'plan':
        payload = plan(ROOT/args.protocol)
        print(json.dumps(payload['estimates'], indent=2))
    else:
        run(ROOT/args.protocol, ROOT/args.output, args.workers)


if __name__ == '__main__':
    main()

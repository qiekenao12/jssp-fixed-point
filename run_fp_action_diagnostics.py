#!/usr/bin/env python3
"""M2 only: detached, pure-action counterfactuals; no Agent is deployed."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_ablation_test import parse_jobshop1
from run_orchestration_baselines import make_instance
from run_trigger_comparison import rng_stream
from tools.controller import Controller, GAParams
from tools.executor import Population
from tools.fixed_point_detector import FixedPointDetector
from tools.jssp_core import JSSPUtils, JSSPEncoder
from tools.meta_scheduler import MetaScheduler
from tools.paper_q import QueryMeter, QExecutor

ACTIONS = ('GA', 'LS', 'VND')


@dataclass(frozen=True)
class Protocol:
    revision: int = 1
    seed: int = 20261018
    population: int = 80
    generations: int = 180
    check_interval: int = 5
    snapshot_interval: int = 20
    repeats: int = 2
    main_ls_interval: int = 10
    main_ls_iterations: int = 20
    local_iterations: int = 100
    action_query_cap: int = 500
    pc: float = .85
    pm: float = .05

    def __post_init__(self):
        for name in ('population', 'generations', 'check_interval', 'snapshot_interval',
                     'repeats', 'main_ls_interval', 'main_ls_iterations',
                     'local_iterations', 'action_query_cap'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.population < 4 or self.snapshot_interval % self.check_interval:
            raise ValueError('population>=4 and snapshots must coincide with FP observations')
        if self.generations % self.snapshot_interval:
            raise ValueError('generations must contain a whole number of snapshot windows')
        if self.action_query_cap < max(2*self.population, self.local_iterations+2):
            raise ValueError('cap must allow a complete GA/ordinary-LS application; only VND is truncated')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sources():
    names = ['run_fp_action_diagnostics.py', 'run_ablation_test.py',
             'run_orchestration_baselines.py', 'run_trigger_comparison.py']
    names += [f'src/tools/{name}.py' for name in (
        'controller', 'executor', 'fixed_point_detector', 'fp_signals',
        'jssp_core', 'meta_scheduler', 'observer', 'paper_q', 'search_orchestration')]
    return {name: sha(ROOT / name) for name in names}


def stream_seed(protocol, instance, run, generation, component):
    return int.from_bytes(hashlib.sha256(
        f'{protocol.seed}:{instance}:{run}:{generation}:{component}'.encode()
    ).digest()[:4], 'big')


def fingerprint(population):
    return canonical({'population': np.asarray(population.individuals, dtype=int).tolist(),
                      'fitness': list(map(float, population.fitness_values))})


def clone(population):
    return Population([x.copy() for x in population.individuals],
                      list(population.fitness_values))


def vector(population, scale):
    costs = np.asarray(population.fitness_values, float)
    return [float(costs.min()/scale), float(costs.mean()/scale),
            float(costs.std()/scale), float(JSSPUtils.calculate_diversity(population.individuals))]


def parameters(protocol):
    params = GAParams(pc=protocol.pc, pm=protocol.pm, n=2, tournament_size=3,
                      selection_method='tournament', selection_strategy='tournament',
                      population_size=protocol.population)
    params.crossover_rate, params.mutation_rate = protocol.pc, protocol.pm
    return params


def branch(instance, original, action, seed, generation, protocol, scale):
    """One pure application. Initialization costs are already-known history."""
    pop = clone(original)
    meter = QueryMeter(protocol.action_query_cap, protocol.action_query_cap,
                       protocol.action_query_cap)
    executor = QExecutor(instance, meter)
    meter.best_cost = float(pop.best_fitness)
    meter.best_individual = pop.best_individual.copy()
    meter.known_costs.update({tuple(map(int, x)): float(y)
                             for x, y in zip(pop.individuals, pop.fitness_values)})
    meter.phase = action.lower()
    started = time.perf_counter()
    with rng_stream(seed), meter.activate():
        if action == 'GA':
            pop, _ = executor.evolve_generation(
                pop, parameters(protocol), generation=generation,
                max_generations=protocol.generations, apply_local_search=False,
                diversity=JSSPUtils.calculate_diversity(pop.individuals))
        elif action == 'LS':
            candidate = executor.local_search_improved(
                pop.best_individual.copy(), max_iter=protocol.local_iterations)
            executor.evaluate(candidate)
        elif action == 'VND':
            executor.search_candidate(pop.best_individual.copy(),
                                      max_iter=protocol.local_iterations)
        else:
            raise ValueError(action)
        if action != 'GA' and meter.best_cost < pop.best_fitness:
            index = pop.best_index
            pop.individuals[index] = meter.best_individual.copy()
            pop.fitness_values[index] = meter.best_cost
    elapsed = time.perf_counter() - started
    # The incumbent keeps all evaluated candidates, independently of cache hits.
    if meter.best_cost < pop.best_fitness:
        index = pop.best_index
        pop.individuals[index] = meter.best_individual.copy()
        pop.fitness_values[index] = meter.best_cost
    end = vector(pop, scale)
    schedule = JSSPEncoder.decode_to_schedule(pop.best_individual, instance)
    assert max(item['end_time'] for item in schedule) == int(pop.best_fitness)
    assert pop.best_fitness <= original.best_fitness
    return {'end_x': end, 'end_makespan': int(pop.best_fitness),
            'gain': float((original.best_fitness-pop.best_fitness)/scale),
            'improved': bool(pop.best_fitness < original.best_fitness),
            'queries': meter.queries, 'decodes': meter.decodes,
            'wall_seconds': elapsed, 'phase_queries': dict(meter.phase_queries),
            'best_chromosome': list(map(int, pop.best_individual)),
            'incumbent_trace': meter.trace,
            'termination_reason': 'query_cap' if meter.queries == meter.budget else 'action_return',
            'window': {'unit': 'atomic_action_application', 'span': 1,
                       'action_sequence': [action]},
            'elapsed_ga_generations': int(action == 'GA'),
            'bottom_operator': {'GA': 'evolve_generation_without_LS',
                                'LS': 'local_search_improved', 'VND': 'local_search_vnd'}[action]}


def run_trajectory(name, run, data, protocol, collect=True, action_order=ACTIONS):
    instance = make_instance(data)
    meter = QueryMeter(10**12)
    executor = QExecutor(instance, meter)
    controller = Controller(instance.num_jobs, instance.num_machines)
    detector = FixedPointDetector(instance.num_jobs*instance.num_machines,
                                  max_generations=protocol.generations, model_mode='operator')
    with rng_stream(stream_seed(protocol, name, run, 0, 'initialization')), meter.activate():
        individuals = MetaScheduler(instance)._init_population(protocol.population)
        pop = Population(individuals, executor.evaluate_population(individuals))
    initial = fingerprint(pop)
    scale = float(pop.best_fitness)
    last_best, last_improvement, history_gains = pop.best_fitness, 0, []
    records, snapshots, main_history = [], [], []
    probe_seconds, main_started = 0., time.perf_counter()
    for generation in range(1, protocol.generations+1):
        before = pop.best_fitness
        meter.phase = 'ga'
        with rng_stream(stream_seed(protocol,name,run,generation,'main_ga')), meter.activate():
            pop, _ = executor.evolve_generation(
                pop, parameters(protocol), generation=generation,
                max_generations=protocol.generations, apply_local_search=False,
                diversity=JSSPUtils.calculate_diversity(pop.individuals))
        if generation % protocol.main_ls_interval == 0:
            meter.phase = 'local_search'
            with rng_stream(stream_seed(protocol,name,run,generation,'main_ls')), meter.activate():
                candidate = executor.local_search_improved(
                    pop.best_individual.copy(), max_iter=protocol.main_ls_iterations)
                value = executor.evaluate(candidate)
                if value < pop.best_fitness:
                    index = pop.best_index
                    pop.individuals[index], pop.fitness_values[index] = candidate.copy(), value
        history_gains.append(float((before-pop.best_fitness)/scale))
        main_history.append({'generation': generation, 'best_before': float(before),
                             'best_after': float(pop.best_fitness), 'main_queries': meter.queries})
        if pop.best_fitness < last_best:
            last_best, last_improvement = pop.best_fitness, generation
        if generation % protocol.check_interval:
            continue
        status, _ = detector.detect(pop.individuals, pop.fitness_values, controller, generation)
        if generation % protocol.snapshot_interval:
            continue
        state = vector(pop, scale)
        state_id = f'{name}:{run}:{generation}'
        fp = {'residual': float(status.operator_residual) if status.operator_transitions >= 16 else None,
              'fit_error_training': float(status.operator_fit_error),
              'model_ready': bool(status.operator_transitions >= 16),
              'observation_interval': int(status.operator_step),
              'training_transitions': int(status.operator_transitions),
              'scale': 'first_FP_observation_best (legacy scale; separate from action x s0)'}
        frozen = fingerprint(pop)
        snapshots.append({'generation': generation, 'x': state,
                          'population': np.asarray(pop.individuals,dtype=int).tolist(),
                          'fitness': list(map(float,pop.fitness_values)),
                          'fingerprint': frozen, 'original_fp': fp})
        if collect:
            detached_start = time.perf_counter()
            for action in action_order:
                for repeat in range(protocol.repeats):
                    outcome = branch(instance, pop, action,
                        stream_seed(protocol,name,run,generation,f'branch_{action}_{repeat}'),
                        generation, protocol, scale)
                    records.append({'instance': name, 'trajectory_id': f'{name}:{run}',
                        'run': run, 'generation': generation, 'state_id': state_id,
                        'action': action, 'repeat': repeat, 'start_x': state,
                        'start_makespan': int(pop.best_fitness), 's0': scale,
                        'state_fingerprint': frozen, 'original_fp': fp,
                        'real_stagnant_generations': generation-last_improvement,
                        'phase': generation/protocol.generations,
                        'recent_real_gains': history_gains[-6:], **outcome})
            probe_seconds += time.perf_counter()-detached_start
        assert fingerprint(pop) == frozen, 'diagnostic branch changed the main population'
    return {'instance': name, 'run': run, 'trajectory_id': f'{name}:{run}',
            'initial_fingerprint': initial, 's0': scale, 'records': records,
            'snapshots': snapshots, 'main_final_makespan': int(pop.best_fitness),
            'main_final_fingerprint': fingerprint(pop),
            'main_queries': meter.queries, 'main_decodes': meter.decodes,
            'main_incumbent_trace': meter.trace,
            'main_history': main_history,
            'diagnostic_queries': sum(r['queries'] for r in records),
            'diagnostic_decodes': sum(r['decodes'] for r in records),
            'diagnostic_seconds': probe_seconds,
            'main_wall_seconds_excluding_diagnostics': time.perf_counter()-main_started-probe_seconds}


def work(task):
    output, name, run, data, protocol, protocol_hash = task
    result = run_trajectory(name, run, data, protocol)
    snapshots = result.pop('snapshots')
    target = Path(output).resolve()/f'{name}_{run:02d}'
    np.savez_compressed(target.with_suffix('.npz'),
        populations=np.asarray([s['population'] for s in snapshots],dtype=np.int32),
        fitness=np.asarray([s['fitness'] for s in snapshots]),
        generations=np.asarray([s['generation'] for s in snapshots]))
    result.update(protocol_hash=protocol_hash, snapshots=[{k:v for k,v in s.items()
                  if k not in ('population','fitness')} for s in snapshots],
                  population_archive=str(target.with_suffix('.npz').relative_to(ROOT)),
                  population_archive_sha256=sha(target.with_suffix('.npz')))
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,allow_nan=False,separators=(',',':')))
    temporary.replace(target.with_suffix('.json'))
    return {'instance':name,'run':run,'records':len(result['records'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'projects/experiments/fp_v2_manifest.json')
    parser.add_argument('--split',choices=('development','validation'),required=True)
    parser.add_argument('--runs',type=int,default=20)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--instances')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    manifest=json.loads(args.manifest.read_text()); protocol=Protocol()
    names=manifest['splits'][args.split]
    if args.instances:
        selected=args.instances.split(',');assert set(selected)<=set(names);names=selected
    payload={'phase':'M2','split':args.split,'instances':names,'runs':args.runs,
             'protocol':asdict(protocol),'sources':sources(),'manifest_sha256':sha(args.manifest),
             'numpy_version':np.__version__,'python_version':sys.version,
             'window_definition':'one pure atomic action; not equal wall time or equal GA generations; action costs recorded',
             'diagnostics_isolated':True,'locked_test_execution_allowed':False}
    protocol_hash=canonical(payload);payload['protocol_hash']=protocol_hash
    args.output.mkdir(parents=True,exist_ok=True)
    lock=args.output/'protocol.json'
    if lock.exists():assert json.loads(lock.read_text())==payload,'source/protocol changed'
    else:lock.write_text(json.dumps(payload,indent=2))
    data=parse_jobshop1(str(ROOT/manifest['data_path']))
    tasks=[]
    for name in names:
        for run in range(args.runs):
            path=args.output/f'{name}_{run:02d}.json'
            if path.exists():
                saved=json.loads(path.read_text());assert saved['protocol_hash']==protocol_hash
                assert sha(ROOT/saved['population_archive'])==saved['population_archive_sha256']
                continue
            tasks.append((str(args.output),name,run,data[name],protocol,protocol_hash))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(work,tasks,chunksize=1):print(json.dumps(result),flush=True)
    assert all((args.output/f'{name}_{run:02d}.json').exists()
               for name in names for run in range(args.runs))
    print(json.dumps({'complete_trajectories':len(names)*args.runs,
                      'expected_records':len(names)*args.runs*9*protocol.repeats*3}),flush=True)


if __name__=='__main__':
    main()

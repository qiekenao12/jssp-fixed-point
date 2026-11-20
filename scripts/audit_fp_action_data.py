#!/usr/bin/env python3
"""Independent M2 archive, endpoint-DAG and causal-history verification.

Does not import the action model, diagnostic runner, optimizer or scheduler.
An independent numeric parser loads frozen matrices. Diagnostic incumbents are
recomputed by job/machine precedence DAG longest paths.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ACTIONS = ('GA', 'LS', 'VND')
EXPECTED_SETTINGS = dict(revision=1, seed=20261018, population=80,
    generations=180, check_interval=5, snapshot_interval=20, repeats=2,
    main_ls_interval=10, main_ls_iterations=20, local_iterations=100,
    action_query_cap=500, pc=.85, pm=.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_numeric_blocks(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    result, index = {}, 0
    while index < len(lines):
        if not lines[index].startswith('instance '):
            index += 1
            continue
        name = lines[index].split()[1]
        require(name not in result, 'Duplicate numeric instance block')
        index += 1
        while index < len(lines):
            parts = lines[index].split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                break
            index += 1
        jobs, machines = map(int, lines[index].split())
        matrix = [list(map(int, line.split())) for line in lines[index+1:index+1+jobs]]
        require(len(matrix) == jobs and all(len(row) == 2*machines for row in matrix),
                'Numeric matrix dimension mismatch')
        require(all(sorted(row[::2]) == list(range(machines)) and min(row[1::2]) >= 0 for row in matrix),
                'Numeric matrix route/duration invalid')
        result[name] = dict(num_jobs=jobs, num_machines=machines,
            machine_sequences=[row[::2] for row in matrix],
            processing_times=[row[1::2] for row in matrix],
            matrix_sha256=canonical(matrix))
        index += jobs+1
    return result


def dag_makespan(chromosome, data):
    """Independent topological reconstruction from machine/job precedences."""
    jobs, machines = data['num_jobs'], data['num_machines']
    require(all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                for v in chromosome), 'Chromosome genes must be integer job identifiers')
    genes = [int(v) for v in chromosome]
    require(len(genes) == jobs*machines and
            Counter(genes) == {j: machines for j in range(jobs)},
            'Invalid job-operation multiplicities')
    durations = [int(v) for row in data['processing_times'] for v in row]
    parents = [set() for _ in durations]
    occurrence, machine_last = [0]*jobs, [None]*machines
    for job in genes:
        operation = occurrence[job]
        node = job*machines + operation
        machine = int(data['machine_sequences'][job][operation])
        if operation:
            parents[node].add(node-1)
        if machine_last[machine] is not None:
            parents[node].add(machine_last[machine])
        machine_last[machine] = node
        occurrence[job] += 1
    children = [[] for _ in durations]
    indegree = [len(p) for p in parents]
    for child, predecessors in enumerate(parents):
        for parent in predecessors:
            children[parent].append(child)
    ready = deque(i for i, n in enumerate(indegree) if n == 0)
    completed, finish = 0, [0]*len(durations)
    while ready:
        node = ready.popleft()
        finish[node] = durations[node] + max((finish[p] for p in parents[node]), default=0)
        completed += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    require(completed == len(durations), 'Job/machine precedence cycle')
    return max(finish)


def diversity(population, jobs):
    n, length = population.shape
    if n < 2:
        return 0.0
    sum_squares = np.zeros(length, dtype=np.int64)
    for job in range(jobs):
        counts = np.sum(population == job, axis=0)
        sum_squares += counts*counts
    return float(np.sum(n*n-sum_squares)/(n*(n-1)*length))


def audit(directory, *, allow_partial_cohort=False):
    directory = Path(directory).resolve()
    lock_path = directory/'protocol.json'
    protocol = json.loads(lock_path.read_text())
    payload = {k: v for k, v in protocol.items() if k != 'protocol_hash'}
    require(canonical(payload) == protocol['protocol_hash'], 'Protocol hash mismatch')
    require(protocol['phase'] == 'M2' and protocol['split'] in ('development', 'validation'),
            'M2 cannot use locked-test solver data')
    manifest_path = ROOT/'projects/experiments/fp_v2_manifest.json'
    require(sha(manifest_path) == protocol['manifest_sha256'], 'Manifest identity mismatch')
    manifest = json.loads(manifest_path.read_text())
    names = protocol['instances']
    expected_names = manifest['splits'][protocol['split']]
    require(len(set(names)) == len(names) and set(names) <= set(expected_names),
            'Instance membership / duplicate violation')
    if not allow_partial_cohort:
        require(names == expected_names and protocol['runs'] == 20,
                'Formal receipt requires the complete frozen 20-seed cohort')
    require(protocol['protocol'] == EXPECTED_SETTINGS, 'Frozen action/schedule settings mismatch')
    for name, expected in protocol['sources'].items():
        require(sha(ROOT/name) == expected, f'Diagnostic source changed: {name}')
    input_path = ROOT/manifest['data_path']
    require(sha(input_path) == manifest['data_sha256'], 'Benchmark matrix file identity mismatch')
    matrices = parse_numeric_blocks(input_path)
    expected_files = {f'{name}_{run:02d}.json' for name in names for run in range(protocol['runs'])}
    actual_files = {p.name for p in directory.glob('*.json') if p.name != 'protocol.json'}
    require(actual_files == expected_files, 'Missing or extra trajectory JSON file')
    generations = list(range(20, 181, 20))
    json_hashes, archive_hashes = {}, {}
    counts = Counter()
    for name in names:
        data = matrices[name]
        require(data['matrix_sha256'] == manifest['instances'][name]['matrix_sha256'],
                'Independent matrix parser differs from frozen instance')
        jobs, machines = data['num_jobs'], data['num_machines']
        for run in range(protocol['runs']):
            path = directory/f'{name}_{run:02d}.json'
            saved = json.loads(path.read_text())
            json_hashes[path.name] = sha(path)
            require((saved['instance'], saved['run'], saved['trajectory_id'], saved['protocol_hash']) ==
                    (name, run, f'{name}:{run}', protocol['protocol_hash']), 'Trajectory identity mismatch')
            scale = float(saved['s0'])
            require(math.isfinite(scale) and scale > 0, 'Invalid initial normalization')
            archive_path = ROOT/saved['population_archive']
            archive_hashes[str(archive_path.relative_to(ROOT))] = sha(archive_path)
            require(sha(archive_path) == saved['population_archive_sha256'], 'NPZ SHA mismatch')
            with np.load(archive_path, allow_pickle=False) as arrays:
                populations, fitness = arrays['populations'], arrays['fitness']
                require(arrays['generations'].tolist() == generations, 'NPZ generation grid mismatch')
            require(populations.shape == (9, 80, jobs*machines), 'Population archive shape mismatch')
            require(fitness.shape == (9, 80) and np.all(np.isfinite(fitness)), 'Fitness archive shape/value mismatch')
            snapshots = {s['generation']: s for s in saved['snapshots']}
            require(len(saved['snapshots']) == 9 and set(snapshots) == set(generations), 'Snapshot grid mismatch')
            main_history = saved.get('main_history')
            require(main_history is not None and len(main_history) == 180,
                    'Per-generation main history is required for causal payoff audit')
            gain_history, last_improvement, previous, previous_queries = [], 0, scale, 0
            causal = {}
            for g, history in enumerate(main_history, start=1):
                require(history['generation'] == g and history['best_before'] == previous,
                        'Main history temporal continuity mismatch')
                after = history['best_after']
                require(after <= previous and history['main_queries'] >= previous_queries,
                        'Main history incumbent/query monotonicity violation')
                gain_history.append((previous-after)/scale)
                if after < previous:
                    last_improvement = g
                causal[g] = dict(best=after, recent=gain_history[-6:],
                                 stagnant=g-last_improvement)
                previous, previous_queries = after, history['main_queries']
            require(previous == saved['main_final_makespan'] and previous_queries == saved['main_queries'],
                    'Main history final result mismatch')
            for index, generation in enumerate(generations):
                population, values, snap = populations[index], fitness[index], snapshots[generation]
                require(np.all(np.sort(population, axis=1) == np.repeat(np.arange(jobs), machines)),
                        'Snapshot contains invalid chromosomes')
                fp = canonical({'population': population.astype(int).tolist(),
                                'fitness': values.astype(float).tolist()})
                require(fp == snap['fingerprint'], 'Saved state fingerprint differs from NPZ')
                state = [float(values.min()/scale), float(values.mean()/scale),
                         float(values.std()/scale), diversity(population, jobs)]
                require(np.allclose(state, snap['x'], atol=1e-12, rtol=0), 'Snapshot state differs from NPZ')
                require(values.min() == causal[generation]['best'], 'Saved snapshot differs from actual main history')
                require(dag_makespan(population[int(np.argmin(values))], data) == int(values.min()),
                        'Snapshot best fitness differs from independent DAG makespan')
                counts['snapshot_DAG_checks'] += 1
            expected_rows = {(g, a, repeat) for g in generations for a in ACTIONS for repeat in range(2)}
            keys = [(r['generation'], r['action'], r['repeat']) for r in saved['records']]
            require(len(keys) == 54 and set(keys) == expected_rows, 'Action/repeat grid mismatch')
            for row in saved['records']:
                g, action = row['generation'], row['action']
                snap = snapshots[g]
                require((row['instance'], row['run'], row['trajectory_id'], row['state_id']) ==
                        (name, run, f'{name}:{run}', f'{name}:{run}:{g}'), 'Branch identity mismatch')
                require(row['state_fingerprint'] == snap['fingerprint'] and row['start_x'] == snap['x'],
                        'Counterfactual branch does not start at exact saved state')
                require(row['start_makespan'] == causal[g]['best'] and row['s0'] == scale,
                        'Branch start/normalization mismatch')
                require(row['phase'] == g/180 and
                        row['real_stagnant_generations'] == causal[g]['stagnant'] and
                        np.allclose(row['recent_real_gains'], causal[g]['recent'], atol=1e-12, rtol=0),
                        'Predecision context does not equal preceding real main history')
                require(row['original_fp'] == snap['original_fp'], 'Original FP status differs within saved state')
                require(row['window'] == {'unit':'atomic_action_application','span':1,'action_sequence':[action]},
                        'Mixed or unequal action window')
                require(row['elapsed_ga_generations'] == int(action == 'GA') and
                        row['bottom_operator'] == {'GA':'evolve_generation_without_LS','LS':'local_search_improved','VND':'local_search_vnd'}[action],
                        'Action semantics mismatch')
                require(0 < row['queries'] <= 500 and sum(row['phase_queries'].values()) == row['queries'],
                        'Action query cap/accounting mismatch')
                require({p for p,v in row['phase_queries'].items() if v} == {action.lower()},
                        'Action query phases are mixed')
                cost = dag_makespan(row['best_chromosome'], data)
                require(cost == row['end_makespan'] and cost <= row['start_makespan'],
                        'Branch endpoint differs from independent DAG cost')
                gain = (row['start_makespan']-cost)/scale
                require(math.isclose(gain, row['gain'], abs_tol=1e-12) and
                        bool(row['improved']) == (gain > 0) and
                        math.isclose(row['end_x'][0], cost/scale, abs_tol=1e-12),
                        'Counterfactual improvement label/endpoint mismatch')
                require(math.isfinite(row['wall_seconds']) and row['wall_seconds'] >= 0,
                        'Invalid execution wall time')
                counts['branch_DAG_checks'] += 1
                counts['branch_queries'] += row['queries']
            counts['trajectories'] += 1
    return {'status':'PASS_COMPLETE' if not allow_partial_cohort else 'PASS_SMOKE_ONLY',
        'split':protocol['split'], 'instances':names, 'runs':protocol['runs'],
        'counts':dict(counts), 'protocol_sha256':sha(lock_path),
        'protocol_hash':protocol['protocol_hash'], 'manifest_sha256':sha(manifest_path),
        'auditor_sha256':sha(__file__), 'trajectory_sha256':json_hashes,
        'archive_sha256':archive_hashes,
        'independent_checks':['NPZ identities and states','all endpoint chromosomes by precedence DAG',
            'complete prespecified grid','causal generation/phase/recent payoff/stagnation history',
            'pure action windows and real query accounting'],
        'limitation':'Full endpoint population is not saved; end mean/std/diversity rely on frozen runner. Endpoint best chromosome and makespan are independently verified. No score-based row removal.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial-cohort', action='store_true',
                        help='Development smoke audit only; never a formal receipt')
    args = parser.parse_args()
    result = audit(args.directory, allow_partial_cohort=args.allow_partial_cohort)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('trajectory_sha256','archive_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

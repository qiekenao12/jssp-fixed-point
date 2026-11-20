#!/usr/bin/env python3
"""Independent schedule/data verifier for ext_standard_jssp raw archives.

Standalone: imports ONLY the standard library. Does not import the solver,
collector, feature builder, or any scorer. Every saved current/incumbent machine
order and every branch endpoint is re-decoded by synchronous monotone time
relaxation, independent of the solver's heap/topological longest-path decoder.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / 'experiments/ext_standard_jssp'
ACTIONS = ('CB_BEST_DESCENT', 'CB_FIRST_DESCENT', 'CB_TABU')


def read(p):
    with (gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)) as f:
        return json.load(f)


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(c, m):
    if not c:
        raise ValueError(m)


def relaxed_makespan(data, orders):
    """Synchronous monotone relaxation; independent of solver decode."""
    j, m = data['jobs'], data['machines']
    n = j * m
    require(len(orders) == m, 'machine count')
    require(all(len(o) == j for o in orders), 'machine row length')
    nodes = [x for o in orders for x in o]
    require(sorted(nodes) == list(range(n)), 'machine order permutation')
    for machine, o in enumerate(orders):
        require(all(data['routes'][u // m][u % m] == machine for u in o), 'machine assignment')
    p = [v for r in data['durations'] for v in r]
    arcs = [(job * m + k, job * m + k + 1) for job in range(j) for k in range(m - 1)]
    arcs += [(u, v) for o in orders for u, v in zip(o, o[1:])]
    finish = p[:]
    for _ in range(n):
        newer = finish[:]
        for u, v in arcs:
            newer[v] = max(newer[v], finish[u] + p[v])
        if newer == finish:
            return max(finish)
        finish = newer
    raise ValueError('cyclic machine order')


def check_file(arg):
    rel, expected_lock = arg
    path = ROOT / rel
    d = read(path)
    di = d['instance_data']
    run = d['run']
    require(digest({k: v for k, v in d.items() if k != 'content_hash'}) == d['content_hash'],
            'trajectory payload hash')
    require(d['lock_hash'] == expected_lock, 'lock mismatch')
    require(d['solver'] == 'ext_tabu', 'solver tag')
    snapshots = {s['state_id']: s for s in d['snapshots']}
    require(set(snapshots) == {f"{d['instance']}:{run}:{p}" for p in range(2)}, 'snapshot grid')
    cache = {}
    checks = 0

    def verify_state(st):
        nonlocal checks
        for field, ms in [('machine_orders', 'current_makespan'), ('best_orders', 'best_makespan')]:
            q = tuple(map(tuple, st[field]))
            if q not in cache:
                cache[q] = relaxed_makespan(di, q)
                checks += 1
            require(cache[q] == st[ms], 'independent schedule makespan mismatch')
        require(0 < st['best_makespan'] <= st['current_makespan']
                and st['best_makespan'] <= st['initial_makespan'], 'incumbent invariant')

    for s in snapshots.values():
        require(digest(s['state']) == s['state_hash'], 'snapshot state hash')
        verify_state(s['state'])
    rows = d['records']
    require(len(rows) == 2 * 3 * 5, 'branch number')
    keys = [(r['state_id'], r['action'], r['repeat']) for r in rows]
    require(len(keys) == len(set(keys)) and
            set(keys) == {(f"{d['instance']}:{run}:{p}", a, rep)
                          for p in range(2) for a in ACTIONS for rep in range(5)}, 'branch grid')
    seeds = []
    for r in rows:
        s = snapshots[r['state_id']]
        require((r['instance'], r['run'], r['trajectory_id']) ==
                (d['instance'], run, d['trajectory_id']), 'branch identity')
        require(math.isfinite(r['gain']) and r['gain'] >= 0, 'invalid reward')
        st = s['state']; o = r['outcome']
        verify_state(o['state'])
        require(r['state_hash'] == s['state_hash'] and r['observed'] == s['features'],
                'source snapshot mismatch')
        require(r['start_makespan'] == st['best_makespan'] and r['s0'] == st['initial_makespan'],
                'start mismatch')
        require(o['state']['best_makespan'] == r['end_makespan'], 'endpoint mismatch')
        require(o['decode_evaluations'] == o['candidate_evaluations'] + 2
                and o['accepted_steps'] <= 8, 'cost mismatch')
        require(o['accepted_steps'] == o['state']['accepted_total'] - st['accepted_total'],
                'accepted count')
        for k in ('history', 'gain_history'):
            require(o['state'][k][:-1] == st[k], 'branch history contains other branches')
        require(o['state']['gain_history'][-1] == r['start_makespan'] - r['end_makespan'],
                'history gain')
        require(math.isfinite(o['elapsed_seconds']) and o['elapsed_seconds'] >= 0, 'elapsed')
        expected = (r['start_makespan'] - r['end_makespan']) / r['s0']
        require(abs(expected - r['gain']) < 1e-12 and r['improved'] == (expected > 0), 'gain mismatch')
        require(abs(r['end_x'][0] - r['end_makespan'] / r['s0']) < 1e-12, 'endpoint best coordinate')
        seeds.append(r['seed'])
    return {'file': rel, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'branches': len(rows), 'states': len(snapshots),
            'distinct_schedule_decodes_checked': checks, 'seeds': seeds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['development', 'test'], required=True)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    started = time.perf_counter()
    lock = read(EXT / 'outputs/manifest_lock.json')
    instances = {i['name']: i for i in read(EXT / 'instances.json') if i['split'] == args.split}
    runs = lock['protocol']['runs'][args.split]
    directory = EXT / 'raw' / args.split
    files = sorted(directory.glob('*.json.gz'))
    expected = {f'{name}_seed{run:02}.json.gz' for name in instances for run in range(runs)}
    require({p.name for p in files} == expected, 'incomplete or unexpected trajectory grid')
    # instance_data must equal the locked manifest entry
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(check_file, [(str(p.relative_to(ROOT)), lock['lock_hash']) for p in files],
                            chunksize=1):
            results.append(res)
    all_seeds = [s for r in results for s in r['seeds']]
    require(len(all_seeds) == len(set(all_seeds)), 'seed collision across branches')
    for r in results:
        d = read(ROOT / r['file'])
        require(d['instance_data'] == instances[d['instance']], 'instance manifest mismatch')
    report = {'status': 'PASS_COMPLETE_GRID_HASH_REWARD_AND_INDEPENDENT_SCHEDULE_CHECKS',
              'cohort': 'ext_standard_jssp', 'split': args.split,
              'instances': len(instances), 'trajectories': len(files),
              'states': sum(r['states'] for r in results),
              'branches': sum(r['branches'] for r in results),
              'distinct_schedule_decodes_checked': sum(r['distinct_schedule_decodes_checked'] for r in results),
              'unique_branch_seeds': len(set(all_seeds)),
              'file_hashes': {r['file']: r['sha256'] for r in results},
              'wall_seconds': time.perf_counter() - started,
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'independence': 'stdlib only; synchronous time relaxation, no solver/scorer import',
              'scope': 'All archived start schedules and branch endpoint best/current schedules.'}
    out = EXT / 'outputs' / f'verification_schedules_{args.split}.json'
    if out.exists():
        raise ValueError('Never overwrite audit')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'file_hashes'}, indent=2))


if __name__ == '__main__':
    main()

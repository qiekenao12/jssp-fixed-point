#!/usr/bin/env python3
"""Prospective branch collection for ext_standard_jssp (isolated archives).

Exact native protocol of scripts/run_fp_decision_collection.py (tabu branch):
neutral CB_TABU warm-up of 8 then 16 accepted moves, 2 saved states per
trajectory, 3 native actions x 5 independent repeats x <=8 accepted moves.
New seed namespace; write-once gzip archives with content hashes; resume-safe.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import (ROOT, EXT, RAW, OUT, RUNS, NEUTRAL_CHUNKS, ACTION_STEPS,
                        REPEATS, canonical, load, seed, sha256_file, write_new)
from tools.independent_tabu_jssp import Instance, initial_state, run_action, features, ACTIONS


def verify_lock():
    lock = load(OUT / 'manifest_lock.json')
    assert canonical({k: v for k, v in lock.items() if k != 'lock_hash'}) == lock['lock_hash']
    for rel, h in {**lock['code_hashes_new'], **lock['code_hashes_reused_frozen'],
                   **lock['benchmark_hashes']}.items():
        assert sha256_file(ROOT / rel) == h, f'Changed locked file {rel}'
    return lock


def task(args):
    instdata, run, lock_hash = args
    name, split = instdata['name'], instdata['split']
    path = RAW / split / f'{name}_seed{run:02}.json.gz'
    if path.exists():
        x = load(path)
        assert x['lock_hash'] == lock_hash and x['content_hash'] == canonical(
            {k: v for k, v in x.items() if k != 'content_hash'})
        return {'file': str(path.relative_to(ROOT)), 'resumed': True, 'branches': len(x['records'])}
    started = time.perf_counter()
    inst = Instance(tuple(map(tuple, instdata['routes'])),
                    tuple(map(tuple, instdata['durations'])), name)
    assert inst.fingerprint == instdata['fingerprint']
    state = initial_state(inst, seed(name, run, 'init'))
    snapshots, records, neutral = [], [], []
    for point, steps in enumerate(NEUTRAL_CHUNKS):
        actual = run_action(inst, state, 'CB_TABU', seed(name, run, point, 'neutral'), steps=steps)
        state = actual.state
        neutral.append(actual.to_dict())
        feature_start = time.perf_counter()
        observed = features(inst, state)
        feature_seconds = time.perf_counter() - feature_start
        before = canonical(state.to_dict())
        state_id = f'{name}:{run}:{point}'
        snapshots.append({'state_id': state_id, 'point': point, 'state': state.to_dict(),
                          'features': observed, 'feature_seconds': feature_seconds,
                          'state_hash': before})
        for action in ACTIONS:
            for repeat in range(REPEATS):
                branch_seed = seed(name, run, point, action, repeat)
                outcome = run_action(inst, state, action, branch_seed, steps=ACTION_STEPS)
                ef_start = time.perf_counter()
                end_features = features(inst, outcome.state)
                ef_seconds = time.perf_counter() - ef_start
                records.append({'instance': name, 'trajectory_id': f'{name}:{run}', 'run': run,
                                'state_id': state_id, 'point': point, 'action': action,
                                'repeat': repeat, 'seed': branch_seed, 'state_hash': before,
                                'start_x': observed['dynamics'], 'observed': observed,
                                'end_x': end_features['dynamics'],
                                'endpoint_feature_seconds': ef_seconds,
                                'gain': outcome.normalized_improvement,
                                'improved': outcome.improvement > 0,
                                'start_makespan': state.best_makespan, 's0': state.initial_makespan,
                                'end_makespan': outcome.state.best_makespan,
                                'outcome': outcome.to_dict()})
                assert canonical(state.to_dict()) == before, 'Diagnostic branch modified source'
    result = {'instance': name, 'run': run, 'trajectory_id': f'{name}:{run}',
              'snapshots': snapshots, 'neutral': neutral, 'records': records,
              'main_final_makespan': state.best_makespan,
              'solver': 'ext_tabu', 'split': split, 'instance_data': instdata,
              'lock_hash': lock_hash,
              'elapsed_collection_seconds': time.perf_counter() - started}
    result['content_hash'] = canonical(result)
    write_new(path, result)
    return {'file': str(path.relative_to(ROOT)), 'resumed': False,
            'branches': len(records), 'seconds': result['elapsed_collection_seconds']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['development', 'test'], required=True)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    lock = verify_lock()
    lock_hash = lock['lock_hash']
    instances = load(EXT / 'instances.json')
    assert canonical(instances) == lock['instances_hash']
    if args.split == 'test':
        # model bundle must already be frozen from development-only fitting
        assert (EXT / 'models/ext_models.json').exists(), 'Fit dev models before test collection'
        bundle = load(EXT / 'models/ext_models.json')
        assert bundle['lock_hash'] == lock_hash
        assert not (set(bundle['training_instances']) &
                    {i['name'] for i in instances if i['split'] == 'test'})
    tasks = [(i, r, lock_hash) for i in instances if i['split'] == args.split
             for r in range(RUNS[args.split])]
    started = time.perf_counter()
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for receipt in pool.map(task, tasks, chunksize=1):
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    report = {'status': 'COMPLETE', 'cohort': 'ext_standard_jssp', 'split': args.split,
              'trajectories': len(receipts), 'branches': sum(r['branches'] for r in receipts),
              'wall_seconds': time.perf_counter() - started,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'lock_hash': lock_hash, 'receipts': receipts}
    write_new(RAW / f'{args.split}_collection_receipt.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'receipts'}, indent=2))


if __name__ == '__main__':
    main()

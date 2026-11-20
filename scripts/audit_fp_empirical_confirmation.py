#!/usr/bin/env python3
"""Independent saved-data audit of the single frozen FP empirical confirmation.

Imports no searcher, selector, model, FP implementation or training routine.
Checks complete grids, source/protocol identities, full saved populations by a
separate DAG evaluator, causal main histories, seeds and primitive accounting.
GA endpoint population is not persisted, so only its best-coordinate is
independently reconstructed; local-action endpoint states can be derived from
the frozen single-incumbent replacement rule. This coverage limit is explicit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from audit_fp_action_data import canonical, dag_makespan, diversity, parse_numeric_blocks, require, sha

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ('GA', 'LS', 'VND')
GENERATIONS = list(range(20, 181, 20))
WORK_GENERATIONS = (40, 100, 160)
REPEATS = range(5)
SETTINGS = dict(revision=1, seed=20261018, population=80, generations=180,
    check_interval=5, snapshot_interval=20, repeats=5, main_ls_interval=10,
    main_ls_iterations=20, local_iterations=100, action_query_cap=500, pc=.85, pm=.05)


def array_equal(actual, expected, message):
    require(np.asarray(actual).shape == np.asarray(expected).shape and
            np.allclose(actual, expected, atol=1e-12, rtol=0), message)


def expected_seed(name, run, generation, action, repeat, mode):
    component = (f'branch_{action}_{repeat}' if mode == 'atomic' else
                 f'equal480:{action}:{repeat}')
    value = f'20261018:{name}:{run}:{generation}:{component}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'big')


def state(population, fitness, scale, jobs):
    return [float(fitness.min()/scale), float(fitness.mean()/scale),
            float(fitness.std()/scale), diversity(population, jobs)]


def causal_history(saved, scale):
    history = saved['main_history']
    require(len(history) == 180, 'Main history must retain all180 generations')
    previous, queries, last = scale, 0, 0
    gains, context = [], {}
    for generation, row in enumerate(history, start=1):
        require(row['generation'] == generation and row['best_before'] == previous,
                'Main history chronological continuity mismatch')
        after = row['best_after']
        require(math.isfinite(after) and 0 < after <= previous and row['main_queries'] >= queries,
                'Main incumbent/query history is not monotone')
        gains.append((previous-after)/scale)
        if after < previous:
            last = generation
        context[generation] = dict(best=after, recent=gains[-6:], stagnant=generation-last)
        previous, queries = after, row['main_queries']
    require(previous == saved['main_final_makespan'] and queries == saved['main_queries'],
            'Main final result does not match its actual history')
    return context


def branch_audit(row, *, mode, name, run, snap, population, fitness,
                 scale, data, context, seed_identities, cached_cost, counts):
    generation, action, repeat = row['generation'], row['action'], row['repeat']
    require(row['window_mode'] == mode, 'Branch window_mode metadata mismatch')
    require((row['instance'], row['run'], row['trajectory_id'], row['state_id']) ==
        (name, run, f'{name}:{run}', f'{name}:{run}:{generation}'), 'Branch identity mismatch')
    require(row.get('state_fingerprint', row.get('fingerprint')) == snap['fingerprint'],
            'Branch does not identify exact saved population')
    array_equal(row['start_x'], snap['x'], 'Branch input state mismatch')
    require(row['start_makespan'] == context['best'] and row['s0'] == scale,
            'Branch start incumbent/normalization mismatch')
    require(row['phase'] == generation/180 and row['real_stagnant_generations'] == context['stagnant'],
            'Branch stage/stagnation uses incorrect main history')
    array_equal(row['recent_real_gains'], context['recent'], 'Causal main payoff history mismatch')
    require(row['original_fp'] == snap['original_fp'], 'Predecision original FP differs within state')
    seed = expected_seed(name, run, generation, action, repeat, mode)
    require(row['seed'] == seed, 'Action repetition seed does not follow frozen derivation')
    require(seed not in seed_identities, f'Duplicate32-bit seed across branches: {seed}')
    seed_identities[seed] = (name, run, generation, action, repeat, mode)
    queries, decodes = row['queries'], row['decodes']
    require(isinstance(queries, int) and isinstance(decodes, int) and 0 <= decodes <= queries,
            'Invalid objective request/decode accounting')
    require(math.isfinite(row['wall_seconds']) and row['wall_seconds'] >= 0, 'Invalid action elapsed time')
    require(all(math.isfinite(v) for v in row['end_x']) and len(row['end_x']) == 4,
            'Nonfinite or malformed endpoint state')
    require(row['end_x'][2] >= 0 and 0 <= row['end_x'][3] <= 1+1e-12,
            'Invalid endpoint spread/diversity')
    cost = cached_cost(row['best_chromosome'])
    require(cost == row['end_makespan'] and cost <= row['start_makespan'],
            'Independent branch DAG endpoint mismatch')
    trace_previous,trace_query = row['start_makespan'],0
    for point in row['incumbent_trace']:
        require(isinstance(point['query'],int) and trace_query < point['query'] <= queries and point['makespan'] < trace_previous, 'Incumbent trace request/cost monotonicity mismatch')
        trace_previous,trace_query = point['makespan'],point['query']
    require(trace_previous == cost, 'Final best differs from actual evaluated incumbent trace')
    gain = (row['start_makespan']-cost)/scale
    require(math.isclose(gain, row['gain'], abs_tol=1e-12, rel_tol=0) and
            bool(row['improved']) == (gain > 0), 'Actual gain/improvement event mismatch')
    require(math.isclose(row['end_x'][0], cost/scale, abs_tol=1e-12, rel_tol=0),
            'Endpoint best coordinate mismatch')
    if mode == 'atomic':
        require(row['window'] == {'unit':'atomic_action_application','span':1,'action_sequence':[action]},
                'Atomic window contains mixed actions or wrong span')
        require(row['elapsed_ga_generations'] == int(action == 'GA'), 'Atomic GA generation count mismatch')
        require(row['termination_reason'] == ('query_cap' if queries == 500 else 'action_return'), 'Atomic cap/reason metadata mismatch')
        require(row['bottom_operator'] == {'GA':'evolve_generation_without_LS',
            'LS':'local_search_improved','VND':'local_search_vnd'}[action], 'Incorrect atomic primitive')
        require(0 < queries <= 500 and sum(row['phase_queries'].values()) == queries,
                'Atomic request cap/phase sum mismatch')
        require({k for k,v in row['phase_queries'].items() if v} == {action.lower()},
                'Atomic action mixed objective-request phases')
        require(queries == 80 if action == 'GA' else queries == 102 if action == 'LS' else True,
                'Atomic GA/LS primitive intensity mismatch')
    else:
        require(queries in ([479,480] if action == 'VND' else [480]), 'Equal480 actual work mismatch')
        calls = row['primitive_calls']
        require(calls and all(c['call'] == i and c['queries'] > 0 for i,c in enumerate(calls)),
                'Malformed/empty primitive call or dummy padding')
        require(sum(c['queries'] for c in calls) == queries, 'Primitive requests do not sum to real requests')
        require(row['unused_queries'] == 480-queries, 'Undisclosed unused request allowance')
        require(row['window'] == {'unit':'objective_request_ceiling','span':480,
                'action_sequence':[action]*len(calls)}, 'Equal480 contains mixed action applications')
        if action == 'GA':
            require(len(calls) == 6 and all(c['queries']==80 and c['termination']=='natural_return' for c in calls)
                    and row['elapsed_ga_generations'] == 6, 'Equal480 GA must complete six pure generations')
        else:
            require(row['elapsed_ga_generations'] == 0, 'Local action spuriously reports elapsed GA generations')
        if queries == 479:
            require(action == 'VND' and row['termination_reason'] == 'insufficient_minimum_call',
                    '479 requests need disclosed VND minimum-call termination')
        else:
            require(row['termination_reason'] == 'request_ceiling', '480-request termination reason mismatch')
        if action == 'LS':
            require(all(c['queries']==102 and c['termination']=='natural_return' for c in calls[:-1]) and
                    calls[-1]['queries']==72 and calls[-1]['termination']=='request_boundary_truncation',
                    'Equal480 LS must be four full102 calls plus a real truncated72 call')
        if action == 'VND':
            require(len(row['vnd_call_log']) == len(calls) and
                    all(c['queries']==v['queries'] and
                        (c['termination']=='request_boundary_truncation')==bool(v['capped'])
                        for c,v in zip(calls,row['vnd_call_log'])), 'VND natural/capped call metadata mismatch')
    # LS/VND only replace the incumbent member. This reconstructs all4 endpoint
    # coordinates under the fixed primitive semantics; GA population is absent.
    if action in ('LS','VND'):
        end_population, end_fitness = population.copy(), fitness.copy()
        if cost < fitness.min():
            index = int(np.argmin(fitness))
            end_population[index] = np.asarray(row['best_chromosome'])
            end_fitness[index] = cost
        array_equal(row['end_x'], state(end_population,end_fitness,scale,data['num_jobs']),
                    'Local-action endpoint differs from single-incumbent replacement')
        counts['local_endpoint_all4_coordinates_reconstructed'] += 1
    else:
        counts['GA_endpoint_best_coordinate_only'] += 1
    counts[f'{mode}_branch_DAG_checks'] += 1
    counts[f'{mode}_queries'] += queries
    counts[f'{mode}_decodes'] += decodes


def audit(directory, *, manifest_path=None, model_bundle=None, collection_lock=None):
    directory = Path(directory).resolve()
    lock_path = Path(collection_lock).resolve() if collection_lock else directory/'protocol.json'
    lock = json.loads(lock_path.read_text())
    require(canonical({k:v for k,v in lock.items() if k!='protocol_hash'}) == lock['protocol_hash'],
            'Collection protocol canonical hash mismatch')
    manifest_path = Path(manifest_path).resolve() if manifest_path else ROOT/'projects/experiments/fp_v2_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(sha(manifest_path) == lock['manifest_sha256'], 'Original exposure/split manifest changed')
    names = manifest['splits']['locked_test']
    require(lock['instances'] == names and len(names) == 20 and len(set(names)) == 20,
            'Final confirmation must retain all20 previously locked instances')
    require(lock['runs'] == 20 and lock['protocol'] == SETTINGS, 'Confirmation trajectory/action configuration mismatch')
    require(lock['split'] == 'locked_test' and lock['stage'] == 'FINAL_CONFIRMATION' and lock['status'] == 'FROZEN', 'Confirmation data role/freeze must be explicit')
    require(lock['equal_work_generations'] == [40,100,160] and lock['equal_work_repeats'] == 5 and lock['equal_work_query_ceiling'] == 480, 'Secondary window protocol mismatch')
    require(sha(ROOT/lock['master_protocol_path']) == lock['master_protocol_sha256'], 'Master final protocol identity mismatch')
    require(lock['data_path'] == manifest['data_path'] and lock['data_sha256'] == manifest['data_sha256'], 'Collector data identity differs from exposure manifest')
    for source,digest in lock['sources'].items():
        require(sha(ROOT/source) == digest, f'Frozen confirmation source mismatch: {source}')
    input_path = ROOT/manifest['data_path']
    require(sha(input_path) == manifest['data_sha256'], 'Frozen benchmark matrices changed')
    matrices = parse_numeric_blocks(input_path)
    require(model_bundle is not None, 'Explicit frozen model bundle path is required')
    model_bundle = Path(model_bundle).resolve()
    require(sha(model_bundle) == lock['model_bundle_sha256'], 'Frozen selector model bundle changed')
    expected_files = {f'{name}_{run:02d}.json' for name in names for run in range(20)}
    if collection_lock:
        require(json.loads((directory/'protocol.json').read_text()) == lock, 'Directory and external collection locks differ')
    files = {p.name for p in directory.glob('*.json') if p.name not in {'protocol.json',lock_path.name}}
    require(files == expected_files, 'Final cohort missing/extra trajectory files')
    counts, json_hashes, npz_hashes, seeds = Counter(), {}, {}, {}
    request_histogram = {f'{mode}/{action}':Counter() for mode in ('atomic','equal480') for action in ACTIONS}
    for name in names:
        data = matrices[name]
        require(data['matrix_sha256'] == manifest['instances'][name]['matrix_sha256'], 'Instance matrix identity mismatch')
        jobs,machines = data['num_jobs'],data['num_machines']
        for run in range(20):
            path=directory/f'{name}_{run:02d}.json';saved=json.loads(path.read_text());json_hashes[path.name]=sha(path)
            require((saved['instance'],saved['run'],saved['trajectory_id'],saved['protocol_hash']) ==
                (name,run,f'{name}:{run}',lock['protocol_hash']), 'Saved trajectory identity/protocol mismatch')
            require(saved['payload_sha256'] == canonical({k:v for k,v in saved.items() if k!='payload_sha256'}), 'Saved trajectory content hash mismatch')
            require(saved['top_protocol'] == lock and saved['sources'] == lock['sources'], 'Saved full protocol/sources differ from external frozen lock')
            require(saved['model_bundle_sha256'] == lock['model_bundle_sha256'], 'Trajectory model identity mismatch')
            scale=float(saved['s0']);require(math.isfinite(scale) and scale>0,'Invalid initial normalization')
            archive=ROOT/saved['population_archive'];require(archive.resolve().parent == directory and archive.stem == path.stem, 'Trajectory archive not paired with exact saved run')
            require(sha(archive)==saved['population_archive_sha256'],'NPZ hash mismatch')
            npz_hashes[str(archive.relative_to(ROOT))]=sha(archive)
            with np.load(archive,allow_pickle=False) as arrays:
                populations,fitness=arrays['populations'],arrays['fitness'];require(arrays['generations'].tolist()==GENERATIONS,'Snapshot generation grid mismatch')
            require(populations.shape==(9,80,jobs*machines) and fitness.shape==(9,80) and np.all(np.isfinite(fitness)), 'Malformed full-population archive')
            require(np.issubdtype(populations.dtype,np.integer),'Population genes must have integer dtype')
            snapshots={s['generation']:s for s in saved['snapshots']};require(len(saved['snapshots'])==9 and set(snapshots)==set(GENERATIONS),'JSON snapshot grid mismatch')
            context=causal_history(saved,scale);archive_context={};dag_cache={}
            def cached_cost(chromosome):
                require(all(isinstance(v,(int,np.integer)) and not isinstance(v,bool) for v in chromosome) and Counter(map(int,chromosome)) == {j:machines for j in range(jobs)}, 'Invalid branch/snapshot chromosome before DAG cache lookup')
                key=np.asarray(chromosome,dtype=np.int32).tobytes()
                if key not in dag_cache:
                    dag_cache[key]=dag_makespan(chromosome,data);counts['unique_DAG_evaluations']+=1
                return dag_cache[key]
            for index,generation in enumerate(GENERATIONS):
                pop,values,snap=populations[index],fitness[index],snapshots[generation]
                require(np.all(np.sort(pop,axis=1)==np.repeat(np.arange(jobs),machines)),'Snapshot invalid chromosomes')
                require(canonical({'population':pop.astype(int).tolist(),'fitness':values.astype(float).tolist()})==snap['fingerprint'],'Snapshot fingerprint mismatch')
                array_equal(snap['x'],state(pop,values,scale,jobs),'Snapshot full4D state mismatch')
                require(values.min()==context[generation]['best'],'Snapshot does not match main causal history')
                for chromosome,value in zip(pop,values):
                    require(cached_cost(chromosome)==value,'Snapshot member fitness fails independent DAG check');counts['snapshot_member_fitness_checks']+=1
                archive_context[generation]=(snap,pop,values)
                counts['snapshots']+=1
            for mode,field,generations in [('atomic','records',GENERATIONS),('equal480','equal_work_records',WORK_GENERATIONS)]:
                rows=saved[field];expected={(g,a,r) for g in generations for a in ACTIONS for r in REPEATS};keys=[(r['generation'],r['action'],r['repeat']) for r in rows]
                require(len(keys)==len(expected) and set(keys)==expected,'Incomplete/duplicate action-repeat grid')
                for row in rows:
                    snap,pop,values=archive_context[row['generation']]
                    branch_audit(row,mode=mode,name=name,run=run,snap=snap,population=pop,fitness=values,scale=scale,data=data,context=context[row['generation']],seed_identities=seeds,cached_cost=cached_cost,counts=counts)
                    request_histogram[f'{mode}/{row["action"]}'][row['queries']]+=1
            require(saved['diagnostic_queries'] == sum(r['queries'] for r in saved['records']) and saved['diagnostic_decodes'] == sum(r['decodes'] for r in saved['records']), 'Atomic diagnostic totals mismatch')
            require(saved['equal_work_queries'] == sum(r['queries'] for r in saved['equal_work_records']) and saved['equal_work_decodes'] == sum(r['decodes'] for r in saved['equal_work_records']), 'Secondary diagnostic totals mismatch')
            require(saved['total_diagnostic_queries'] == saved['diagnostic_queries'] + saved['equal_work_queries'], 'Combined diagnostic totals mismatch')
            counts['trajectories']+=1
        print(json.dumps({'audited_instance':name,'trajectories':counts['trajectories'],'branch_checks':counts['atomic_branch_DAG_checks']+counts['equal480_branch_DAG_checks']}),flush=True)
    require(counts['trajectories']==400 and counts['snapshots']==3600 and
            counts['snapshot_member_fitness_checks']==288000 and counts['atomic_branch_DAG_checks']==54000 and
            counts['equal480_branch_DAG_checks']==18000 and len(seeds)==72000,'Final complete cohort counts mismatch')
    return {'status':'PASS_COMPLETE','stage':'single_frozen_empirical_confirmation','scope':'Independent saved-data audit; no model/selector/search imports or execution.',
        'protocol_sha256':sha(lock_path),'protocol_hash':lock['protocol_hash'],'manifest_sha256':sha(manifest_path),'model_bundle_sha256':sha(model_bundle),
        'instances':names,'runs':20,'counts':dict(counts),'globally_unique_branch_seeds':len(seeds),
        'request_histogram':{k:dict(v) for k,v in request_histogram.items()},'trajectory_json_sha256':json_hashes,'population_archive_sha256':npz_hashes,
        'endpoint_coverage':{'saved_start_population':'All80 members at all3600 snapshots DAG-checked; all4 state coordinates independently reconstructed.',
            'local_actions':'All4 endpoint coordinates reconstructed under frozen single-incumbent replacement semantics from the saved population and independently checked best schedule.',
            'GA':'Only bestendpoint/end_x[0] independently reconstructed. Endpoint population is not persisted, so GA mean/std/diversity rely on the frozen tested collector.',
            'causality':'Predecision stagnation,stage,recent real main gains checked against all180 main-history steps. No future diagnostic gains accepted as context.'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--model-bundle',required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--collection-lock')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    require(not output.exists(),'Never overwrite an independent final-confirmation audit receipt')
    result=audit(args.directory,manifest_path=args.manifest,model_bundle=args.model_bundle,collection_lock=args.collection_lock)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'counts':result['counts']}),flush=True)


if __name__=='__main__':
    main()

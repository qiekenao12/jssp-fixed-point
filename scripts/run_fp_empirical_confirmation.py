#!/usr/bin/env python3
"""One explicitly frozen, detached final confirmation; no online Agent.

Without --collect this command validates metadata and prints the cost bound.
It never repairs or overwrites an incomplete trajectory. Endpoint populations
are not saved: full start populations and independently checkable endpoint best
chromosomes are saved, exactly as in the frozen M2 primitive interfaces.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_fp_action_diagnostics import (ACTIONS, Protocol, canonical, fingerprint,
    run_trajectory, sha, sources, stream_seed)
from run_ablation_test import parse_jobshop1
from run_orchestration_baselines import make_instance
from tools.executor import Population
from scripts.fp_empirical_repeatability import work_window


def require(condition, message):
    if not condition:
        raise ValueError(message)


def required_sources():
    return {**sources(),
        'scripts/fp_empirical_repeatability.py': sha(ROOT/'scripts/fp_empirical_repeatability.py'),
        'scripts/run_fp_empirical_confirmation.py': sha(__file__)}


def validate_protocol(path):
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    require(doc.get('status') == 'FROZEN', 'Final confirmation requires status=FROZEN')
    require(doc.get('stage') == 'FINAL_CONFIRMATION', 'Wrong scientific stage')
    require(doc.get('split') == 'locked_test', 'Final confirmation requires the fixed locked_test cohort')
    require(doc.get('protocol_hash') == canonical({k:v for k,v in doc.items() if k != 'protocol_hash'}),
            'Frozen protocol canonical hash mismatch')
    for prefix in ('manifest', 'master_protocol', 'model_bundle'):
        require(sha(ROOT/doc[prefix+'_path']) == doc[prefix+'_sha256'], f'{prefix} hash mismatch')
    manifest = json.loads((ROOT/doc['manifest_path']).read_text())
    require(doc['instances'] == manifest['splits']['locked_test'], 'Do not change, filter or reorder locked instances')
    require(len(doc['instances']) == 20 and len(set(doc['instances'])) == 20,
            'Exactly twenty locked instances required')
    require(doc['runs'] == 20 and not isinstance(doc['runs'],bool), 'Exactly20 predeclared run indices required')
    settings = Protocol(**doc['protocol'])
    require(asdict(settings) == asdict(Protocol(repeats=5)),
            'Final protocol must retain all M2 search settings, with five repetitions')
    require(doc['equal_work_generations'] == [40,100,160] and doc['equal_work_repeats'] == 5
            and doc['equal_work_query_ceiling'] == 480, 'Frozen secondary window differs')
    require(doc['data_path'] == manifest['data_path']
            and doc['data_sha256'] == manifest['data_sha256']
            and sha(ROOT/doc['data_path']) == doc['data_sha256'], 'Frozen benchmark bytes mismatch')
    current = required_sources()
    require(set(current) <= set(doc['sources']), 'Final source lock omits a collector/primitive dependency')
    for name,value in doc['sources'].items():
        require(sha(ROOT/name) == value, f'Frozen source changed: {name}')
    require(all(doc['sources'][name] == value for name,value in current.items()),
            'Collector source lock differs')
    return doc


def expected_keys(name, run, settings, generations, repeats):
    return {(f'{name}:{run}:{g}',a,r) for g in generations for a in ACTIONS for r in range(repeats)}


def validate_saved(path, top):
    """Resume verifies complete identity; invalid/partial results abort intact."""
    path = Path(path)
    doc = json.loads(path.read_text())
    require(doc.get('payload_sha256') == canonical({k:v for k,v in doc.items() if k != 'payload_sha256'}),
            'Saved trajectory payload hash mismatch')
    require(doc['top_protocol'] == top and doc['protocol_hash'] == top['protocol_hash'],
            f'Frozen identity mismatch: {path}')
    require(doc['sources'] == top['sources'] and doc['model_bundle_sha256'] == top['model_bundle_sha256'],
            'Result source/model identity mismatch')
    name, run = doc['instance'],doc['run']
    require(name in top['instances'] and run in range(top['runs']), 'Unexpected instance/run')
    require(doc['trajectory_id'] == f'{name}:{run}' and path.name == f'{name}_{run:02d}.json',
            'Trajectory filename/identity mismatch')
    settings = Protocol(**top['protocol'])
    generations = list(range(settings.snapshot_interval,settings.generations+1,settings.snapshot_interval))
    for field,gs,reps in [('records',generations,settings.repeats),
                         ('equal_work_records',top['equal_work_generations'],top['equal_work_repeats'])]:
        rows = doc[field]
        keys = [(r['state_id'],r['action'],r['repeat']) for r in rows]
        require(len(keys) == len(set(keys)) and set(keys) == expected_keys(name,run,settings,gs,reps),
                f'Incomplete or duplicated {field}: {path}')
        for row in rows:
            require(row['instance']==name and row['run']==run and row['trajectory_id']==doc['trajectory_id']
                    and row['state_id']==f'{name}:{run}:{row["generation"]}', 'Branch identity mismatch')
            component = (f'branch_{row["action"]}_{row["repeat"]}' if field=='records' else
                         f'equal480:{row["action"]}:{row["repeat"]}')
            require(row['seed']==stream_seed(settings,name,run,row['generation'],component), 'Branch seed mismatch')
    archive = ROOT/doc['population_archive']
    require(archive.resolve().parent == path.resolve().parent and archive.stem == path.stem,
            'Archive must be paired with trajectory')
    require(sha(archive)==doc['population_archive_sha256'], 'Saved population archive hash mismatch')
    with np.load(archive,allow_pickle=False) as arrays:
        require(arrays['generations'].tolist()==generations, 'Snapshot generation grid differs')
        require(len(doc['snapshots'])==len(generations), 'Saved snapshot metadata incomplete')
        require(arrays['populations'].shape[0]==arrays['fitness'].shape[0]==len(generations)
                and arrays['populations'].shape[1]==arrays['fitness'].shape[1]==settings.population,
                'Archive dimensions differ')
        for index,snapshot in enumerate(doc['snapshots']):
            require(snapshot['generation']==generations[index], 'Snapshot metadata order differs')
            pop=Population([x.copy() for x in arrays['populations'][index]],
                           list(map(float,arrays['fitness'][index])))
            require(fingerprint(pop)==snapshot['fingerprint'], 'Population fingerprint mismatch')
    require(len(doc['main_history'])==settings.generations, 'Causal main trajectory history incomplete')
    require(doc['diagnostic_queries']==sum(r['queries'] for r in doc['records']) and
            doc['equal_work_queries']==sum(r['queries'] for r in doc['equal_work_records']),
            'Diagnostic query totals differ')
    return doc


def collect_trajectory(task):
    output, name, run, data, top = task
    output=Path(output)
    target=output/f'{name}_{run:02d}'
    json_path=target.with_suffix('.json')
    archive=target.with_suffix('.npz')
    if json_path.exists():
        doc=validate_saved(json_path,top)
        return {'instance':name,'run':run,'resumed_complete':True,
                'atomic_records':len(doc['records']),'equal_work_records':len(doc['equal_work_records'])}
    require(not archive.exists() and not target.with_suffix('.tmp').exists()
            and not target.with_suffix('.npz.tmp').exists(),
            f'Incomplete trajectory preserved; explicit failure review required: {target}')
    settings=Protocol(**top['protocol'])
    result=run_trajectory(name,run,data,settings,collect=True)
    snapshots=result.pop('snapshots')
    by_generation={s['generation']:s for s in snapshots}
    instance=make_instance(data)
    equal_records=[]
    equal_started=time.perf_counter()
    for generation in top['equal_work_generations']:
        saved=by_generation[generation]
        original=Population([np.asarray(x,dtype=int).copy() for x in saved['population']],
                            list(map(float,saved['fitness'])))
        frozen=fingerprint(original)
        require(frozen==saved['fingerprint'], 'Saved population not recovered exactly')
        history=next(r for r in result['records'] if r['generation']==generation)
        context={k:history[k] for k in ('instance','trajectory_id','run','generation','state_id',
            'start_x','start_makespan','s0','state_fingerprint','original_fp',
            'real_stagnant_generations','phase','recent_real_gains')}
        for action in ACTIONS:
            for repeat in range(top['equal_work_repeats']):
                seed=stream_seed(settings,name,run,generation,f'equal480:{action}:{repeat}')
                outcome=work_window(instance,original,action,seed,generation,settings,result['s0'],
                                    ceiling=top['equal_work_query_ceiling'])
                require(fingerprint(original)==frozen, 'Detached secondary branch modified its saved state')
                equal_records.append({**context,'action':action,'repeat':repeat,'seed':seed,
                                      'window_mode':'equal480',**outcome})
    for row in result['records']:
        row['seed']=stream_seed(settings,name,run,row['generation'],
                               f'branch_{row["action"]}_{row["repeat"]}')
        row['window_mode']='atomic'
    equal_seconds=time.perf_counter()-equal_started
    result.update(equal_work_records=equal_records,
        equal_work_queries=sum(r['queries'] for r in equal_records),
        equal_work_decodes=sum(r['decodes'] for r in equal_records),
        equal_work_wall_seconds=equal_seconds,
        total_diagnostic_queries=result['diagnostic_queries']+sum(r['queries'] for r in equal_records),
        protocol_hash=top['protocol_hash'], top_protocol=top,
        sources=top['sources'], model_bundle_sha256=top['model_bundle_sha256'],
        endpoint_verification_scope='Full archived start populations; endpoint best chromosome available. Pure LS/VND endpoints can be reconstructed from the frozen incumbent-only replacement rule; GA endpoint mean/std/diversity require the frozen primitive implementation because endpoint populations are not archived.')
    temporary_archive=target.with_suffix('.npz.tmp')
    with temporary_archive.open('wb') as handle:
        np.savez_compressed(handle,
            populations=np.asarray([s['population'] for s in snapshots],dtype=np.int32),
            fitness=np.asarray([s['fitness'] for s in snapshots]),
            generations=np.asarray([s['generation'] for s in snapshots]))
    temporary_archive.replace(archive)
    result.update(population_archive=str(archive.resolve().relative_to(ROOT)),
                  population_archive_sha256=sha(archive),
                  snapshots=[{k:v for k,v in s.items() if k not in ('population','fitness')}
                             for s in snapshots])
    result['payload_sha256']=canonical(result)
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,allow_nan=False,separators=(',',':')))
    temporary.replace(json_path)
    validate_saved(json_path,top)
    return {'instance':name,'run':run,'atomic_records':len(result['records']),
            'equal_work_records':len(equal_records),
            'diagnostic_queries':result['total_diagnostic_queries']}


def run(protocol_path,output,workers,collect):
    top=validate_protocol(protocol_path)
    settings=Protocol(**top['protocol'])
    count=len(top['instances'])*top['runs']
    atomic=count*(settings.generations//settings.snapshot_interval)*len(ACTIONS)*settings.repeats
    secondary=count*len(top['equal_work_generations'])*len(ACTIONS)*top['equal_work_repeats']
    estimate={'trajectories':count,'atomic_records':atomic,'equal_work_records':secondary,
        'diagnostic_request_upper':atomic*settings.action_query_cap+secondary*top['equal_work_query_ceiling'],
        'neutral_main_estimate_requests':count*(settings.population+settings.generations*settings.population+
            (settings.generations//settings.main_ls_interval)*(settings.main_ls_iterations+2)),
        'collect_requested':collect}
    print(json.dumps(estimate),flush=True)
    if not collect:
        return
    require(workers>=1,'workers must be positive')
    output=Path(output).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'protocol.json'
    if lock.exists():
        require(json.loads(lock.read_text())==top,'Output directory already has another frozen protocol')
    else:
        require(not list(output.iterdir()),'Refuse to initialize a nonempty output directory')
        lock.write_text(json.dumps(top,indent=2))
    expected_files={f'{name}_{run:02d}.json' for name in top['instances'] for run in range(top['runs'])}
    require({p.name for p in output.glob('*.json')} <= expected_files|{'protocol.json'},
            'Unexpected result files in final confirmation directory')
    # Parsing matrices occurs only after explicit --collect with validated freeze.
    data=parse_jobshop1(str(ROOT/top['data_path']))
    tasks=[(str(output),name,run,data[name],top)
           for name in top['instances'] for run in range(top['runs'])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(collect_trajectory,tasks,chunksize=1):
            print(json.dumps(result),flush=True)
    for name in top['instances']:
        for run_index in range(top['runs']):
            validate_saved(output/f'{name}_{run_index:02d}.json',top)
    print(json.dumps({'status':'COMPLETE_FROZEN_CONFIRMATION','trajectories':count,
                     'atomic_records':atomic,'equal_work_records':secondary}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/fp_empirical_confirmation')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--collect',action='store_true',help='Explicitly execute the one frozen confirmation')
    args=parser.parse_args()
    run(args.protocol,args.output,args.workers,args.collect)


if __name__=='__main__':
    main()

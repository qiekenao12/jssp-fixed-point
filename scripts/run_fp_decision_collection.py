#!/usr/bin/env python3
"""Versioned isolated action interventions; old search primitives stay untouched."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.independent_tabu_jssp import Instance, initial_state, run_action, features, ACTIONS
from scripts.prepare_fp_decision_study import canonical

EXPERIMENT=ROOT/'projects/experiments/fp_decision_study'
RESULTS=ROOT/'results/fp_decision_study'

def seed(*parts):
    return int.from_bytes(hashlib.sha256((':'.join(map(str,(2026102501,*parts)))).encode()).digest()[:8],'big')

def write_new(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise ValueError(f'Cannot overwrite {path}')
    data=json.dumps(obj,separators=(',',':'),allow_nan=False).encode()
    temp=path.with_suffix(path.suffix+'.tmp')
    if temp.exists(): raise ValueError(f'Incomplete result retained: {temp}')
    if path.suffix=='.gz':
        with gzip.open(temp,'wb') as f:f.write(data)
    else:temp.write_bytes(data)
    temp.replace(path)

def load(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f:return json.load(f)

def task(args):
    instdata,run,solver,lock_hash=args
    name,split=instdata['name'],instdata['split']
    path=RESULTS/solver/split/f'{name}_seed{run:02}.json.gz'
    if path.exists():
        x=load(path)
        assert x['lock_hash']==lock_hash and x['content_hash']==canonical({k:v for k,v in x.items() if k!='content_hash'})
        return {'file':str(path.relative_to(ROOT)),'resumed':True,'branches':len(x['records'])+len(x.get('equal_work_records',[]))}
    started=time.perf_counter()
    if solver=='tabu':
        inst=Instance(tuple(map(tuple,instdata['routes'])),tuple(map(tuple,instdata['durations'])),name)
        state=initial_state(inst,seed(name,run,'init'))
        snapshots=[];records=[];neutral=[]
        for point,steps in enumerate((8,16)):
            actual=run_action(inst,state,'CB_TABU',seed(name,run,point,'neutral'),steps=steps)
            state=actual.state
            neutral.append(actual.to_dict())
            feature_start=time.perf_counter();observed=features(inst,state)
            feature_seconds=time.perf_counter()-feature_start
            before=canonical(state.to_dict())
            state_id=f'{name}:{run}:{point}'
            snapshots.append({'state_id':state_id,'point':point,'state':state.to_dict(),'features':observed,
                              'feature_seconds':feature_seconds,'state_hash':before})
            for action in ACTIONS:
                for repeat in range(5):
                    branch_seed=seed(name,run,point,action,repeat)
                    outcome=run_action(inst,state,action,branch_seed,steps=8)
                    ef_start=time.perf_counter();end_features=features(inst,outcome.state)
                    ef_seconds=time.perf_counter()-ef_start
                    records.append({'instance':name,'trajectory_id':f'{name}:{run}','run':run,
                        'state_id':state_id,'point':point,'action':action,'repeat':repeat,'seed':branch_seed,
                        'state_hash':before,'start_x':observed['dynamics'],'observed':observed,
                        'end_x':end_features['dynamics'],'endpoint_feature_seconds':ef_seconds,
                        'gain':outcome.normalized_improvement,'improved':outcome.improvement>0,
                        'start_makespan':state.best_makespan,'s0':state.initial_makespan,
                        'end_makespan':outcome.state.best_makespan,'outcome':outcome.to_dict()})
                    assert canonical(state.to_dict())==before,'Diagnostic branch modified source'
        result={'instance':name,'run':run,'trajectory_id':f'{name}:{run}','snapshots':snapshots,
                'neutral':neutral,'records':records,'main_final_makespan':state.best_makespan}
    else:
        from run_fp_action_diagnostics import Protocol, run_trajectory, fingerprint, stream_seed
        from run_orchestration_baselines import make_instance
        from tools.executor import Population
        from scripts.fp_empirical_repeatability import work_window
        settings=Protocol(repeats=5)
        data={'num_jobs':instdata['jobs'],'num_machines':instdata['machines'],
              'processing_times':instdata['durations'],'machine_sequences':instdata['routes'],
              'upper_bound':sum(map(sum,instdata['durations']))}
        result=run_trajectory(name,run,data,settings,collect=True)
        inst=make_instance(data);equal=[]
        for row in result['records']:
            row['window_mode']='atomic'
            row['seed']=stream_seed(settings,name,run,row['generation'],f"branch_{row['action']}_{row['repeat']}")
        for saved in result['snapshots']:
            g=saved['generation']
            if g not in (40,100,160):continue
            original=Population([np.asarray(x,dtype=int) for x in saved['population']],list(map(float,saved['fitness'])))
            assert fingerprint(original)==saved['fingerprint']
            first=next(r for r in result['records'] if r['generation']==g)
            context={k:first[k] for k in ('instance','trajectory_id','run','generation','state_id',
                  'start_x','start_makespan','s0','state_fingerprint','original_fp','real_stagnant_generations','phase','recent_real_gains')}
            for action in ('GA','LS','VND'):
                for repeat in range(5):
                    bs=stream_seed(settings,name,run,g,f'equal480:{action}:{repeat}')
                    outcome=work_window(inst,original,action,bs,g,settings,result['s0'],ceiling=480)
                    equal.append({**context,'action':action,'repeat':repeat,'seed':bs,'window_mode':'equal480',**outcome})
                    assert fingerprint(original)==saved['fingerprint']
        result.update(equal_work_records=equal,settings=asdict(settings))
    result.update(solver=solver,split=split,instance_data=instdata,lock_hash=lock_hash,
                  elapsed_collection_seconds=time.perf_counter()-started)
    result['content_hash']=canonical(result)
    write_new(path,result)
    return {'file':str(path.relative_to(ROOT)),'resumed':False,
            'branches':len(result['records'])+len(result.get('equal_work_records',[])),
            'seconds':result['elapsed_collection_seconds']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--solver',choices=['tabu','original'],required=True)
    ap.add_argument('--split',choices=['development','validation','final'],required=True)
    ap.add_argument('--workers',type=int,default=12);args=ap.parse_args()
    design=load(EXPERIMENT/'design.json');instances=load(EXPERIMENT/'instances.json')
    assert canonical(instances)==design['instances_hash']
    assert canonical({k:v for k,v in design.items() if k!='design_hash'})==design['design_hash']
    if args.split=='final':
        lock=load(EXPERIMENT/'final_lock.json')
        for rel,h in lock['hashes'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,f'Changed frozen file {rel}'
        lock_hash=canonical(lock)
    else:lock_hash=design['design_hash']
    tasks=[(i,s,args.solver,lock_hash) for i in instances if i['split']==args.split for s in range(design['runs'][args.split])]
    started=time.perf_counter();receipts=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for receipt in pool.map(task,tasks,chunksize=1):
            receipts.append(receipt);print(json.dumps(receipt),flush=True)
    report={'status':'COMPLETE','solver':args.solver,'split':args.split,'trajectories':len(receipts),
            'branches':sum(r['branches'] for r in receipts),'wall_seconds':time.perf_counter()-started,
            'created_utc':datetime.now(timezone.utc).isoformat(),'lock_hash':lock_hash,'receipts':receipts}
    write_new(RESULTS/args.solver/f'{args.split}_collection_receipt.json',report)

if __name__=='__main__':main()

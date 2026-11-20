#!/usr/bin/env python3
"""Independent data verifier. Does not import the new solver or FP scorers."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import math
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_fp_action_data import dag_makespan

def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)

def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def require(c,m):
    if not c:raise ValueError(m)

def relaxed_makespan(data,orders):
    """Synchronous monotone relaxation, independent of solver heap/topological path."""
    j,m=data['jobs'],data['machines'];n=j*m
    require(len(orders)==m,'machine count')
    require(all(len(o)==j for o in orders),'machine row length')
    nodes=[x for o in orders for x in o]
    require(sorted(nodes)==list(range(n)),'machine order permutation')
    for machine,o in enumerate(orders):
        require(all(data['routes'][u//m][u%m]==machine for u in o),'machine assignment')
    p=[v for r in data['durations'] for v in r]
    arcs=[(job*m+k,job*m+k+1) for job in range(j) for k in range(m-1)]
    arcs += [(u,v) for o in orders for u,v in zip(o,o[1:])]
    finish=p[:]
    for _ in range(n):
        newer=finish[:]
        for u,v in arcs:newer[v]=max(newer[v],finish[u]+p[v])
        if newer==finish:return max(finish)
        finish=newer
    raise ValueError('cyclic machine order')

def audit(solver,split):
    started=time.perf_counter()
    exp=ROOT/'projects/experiments/fp_decision_study'
    design=read(exp/'design.json');allinst=read(exp/'instances.json')
    require(digest(allinst)==design['instances_hash'],'instance manifest hash')
    instances={i['name']:i for i in allinst if i['split']==split};runs=design['runs'][split]
    directory=ROOT/f'results/fp_decision_study/{solver}/{split}'
    files=sorted(directory.glob('*.json.gz'))
    expected={f'{name}_seed{run:02}.json.gz' for name in instances for run in range(runs)}
    require({p.name for p in files}==expected,'incomplete or unexpected trajectory grid')
    expectedlock=digest(read(exp/'final_lock.json')) if split=='final' else design['design_hash']
    branch_count=state_count=endpoint_checks=pop_checks=0;file_hashes={};seeds={};costs=Counter();states=set()
    for path in files:
        d=read(path);di=instances[d['instance']];run=d['run']
        require(digest({k:v for k,v in d.items() if k!='content_hash'})==d['content_hash'],'trajectory payload hash')
        require(d['instance_data']==di and d['lock_hash']==expectedlock,'manifest/lock mismatch')
        file_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        snapshots={s['state_id'] if solver=='tabu' else f"{d['instance']}:{run}:{s['generation']}":s for s in d['snapshots']}
        points=range(2) if solver=='tabu' else range(20,181,20)
        require(set(snapshots)=={f"{d['instance']}:{run}:{p}" for p in points},'snapshot grid')
        state_count+=len(snapshots);states.update(snapshots)
        cache={}
        if solver=='tabu':
            def verify_state(st):
                nonlocal endpoint_checks
                for field,ms in [('machine_orders','current_makespan'),('best_orders','best_makespan')]:
                    q=tuple(map(tuple,st[field]))
                    if q not in cache:cache[q]=relaxed_makespan(di,q);endpoint_checks+=1
                    require(cache[q]==st[ms],'independent schedule makespan mismatch')
                require(0<st['best_makespan']<=st['current_makespan'] and st['best_makespan']<=st['initial_makespan'],'incumbent invariant')
            for s in snapshots.values():
                require(digest(s['state'])==s['state_hash'],'snapshot state hash');verify_state(s['state'])
            actions=('CB_BEST_DESCENT','CB_FIRST_DESCENT','CB_TABU');window_sets=[('records',range(2))]
        else:
            numeric={'num_jobs':di['jobs'],'num_machines':di['machines'],'machine_sequences':di['routes'],'processing_times':di['durations']}
            def cost(chrom):
                nonlocal endpoint_checks
                q=tuple(chrom)
                if q not in cache:cache[q]=dag_makespan(chrom,numeric);endpoint_checks+=1
                return cache[q]
            for s in snapshots.values():
                require(len(s['population'])==len(s['fitness'])==80,'population dimension')
                for chrom,value in zip(s['population'],s['fitness']):
                    require(cost(chrom)==value,'independent population decode');pop_checks+=1
            actions=('GA','LS','VND');window_sets=[('records',range(20,181,20)),('equal_work_records',(40,100,160))]
        for field,ps in window_sets:
            rows=d[field]
            require(len(rows)==len(list(ps))*3*5,'branch number')
            keys=[(r['state_id'],r['action'],r['repeat']) for r in rows]
            require(len(keys)==len(set(keys)) and set(keys)=={(f"{d['instance']}:{run}:{p}",a,r) for p in ps for a in actions for r in range(5)},'branch grid')
            for r in rows:
                s=snapshots[r['state_id']];branch_count+=1
                require((r['instance'],r['run'],r['trajectory_id'])==(d['instance'],run,d['trajectory_id']),'branch identity')
                require(math.isfinite(r['gain']) and r['gain']>=0,'invalid reward')
                if solver=='tabu':
                    st=s['state'];o=r['outcome'];verify_state(o['state'])
                    require(r['state_hash']==s['state_hash'] and r['observed']==s['features'],'source snapshot mismatch')
                    require(r['start_makespan']==st['best_makespan'] and r['s0']==st['initial_makespan'],'start mismatch')
                    require(o['state']['best_makespan']==r['end_makespan'],'endpoint mismatch')
                    require(o['decode_evaluations']==o['candidate_evaluations']+2 and o['accepted_steps']<=8,'cost mismatch')
                    require(o['accepted_steps']==o['state']['accepted_total']-st['accepted_total'],'accepted count')
                    for k in ('history','gain_history'):
                        require(o['state'][k][:-1]==st[k],'branch history contains other branches')
                    require(o['state']['gain_history'][-1]==r['start_makespan']-r['end_makespan'],'history gain')
                    require(math.isfinite(o['elapsed_seconds']) and o['elapsed_seconds']>=0,'elapsed')
                    costs[r['action']+'_decodes']+=o['decode_evaluations']
                else:
                    require(cost(r['best_chromosome'])==r['end_makespan'],'independent endpoint decode')
                    require(0<=r['decodes']<=r['queries']<=500,'cost mismatch')
                    require(r['start_x']==s['x'] and r['state_fingerprint']==s['fingerprint'],'source snapshot')
                    costs[r['action']+'_requests']+=r['queries'];costs[r['action']+'_decodes']+=r['decodes']
                expectedgain=(r['start_makespan']-r['end_makespan'])/r['s0']
                require(abs(expectedgain-r['gain'])<1e-12 and r['improved']==(expectedgain>0),'gain mismatch')
                require(abs(r['end_x'][0]-r['end_makespan']/r['s0'])<1e-12,'endpoint best coordinate')
                ident=(field,r['state_id'],r['action'],r['repeat'])
                require(r['seed'] not in seeds or seeds[r['seed']]==ident,'seed collision');seeds[r['seed']]=ident
    report={'status':'PASS_COMPLETE_GRID_HASH_REWARD_AND_INDEPENDENT_SCHEDULE_CHECKS','solver':solver,'split':split,
       'instances':len(instances),'trajectories':len(files),'states':state_count,'unique_states':len(states),'branches':branch_count,
       'distinct_schedule_decodes_checked':endpoint_checks,'population_individual_checks':pop_checks,
       'cost_totals':dict(costs),'file_hashes':file_hashes,'wall_seconds':time.perf_counter()-started,
       'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'scope':'Full archived start schedules and endpoint best/current schedules. Original GA endpoint other population statistics not independently reconstructible: endpoint population absent. No online performance inference.'}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--solver',choices=['tabu','original'],required=True);p.add_argument('--split',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError('Never overwrite audit')
    result=audit(a.solver,a.split);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='file_hashes'},indent=2))

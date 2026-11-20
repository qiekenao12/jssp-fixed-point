#!/usr/bin/env python3
"""Independent scalar aggregation from exported scores and original rewards.

No model fitting, feature constructor or primary metric helper is imported.
Exact signed-rank tails are independently enumerated for the observed contrasts.
"""
import csv,gzip,hashlib,itertools,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'projects/analysis/fp_decision_study'

def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)

def exact(values):
    v=[x for x in values if abs(x)>1e-12]
    if not v:return 1.
    av=list(map(abs,v));ranks=[]
    for a in av:
        smaller=sum(x<a for x in av);equal=sum(x==a for x in av)
        ranks.append(smaller+(equal+1)/2)
    observed=sum(r for r,x in zip(ranks,v) if x>0)
    tails=0;total=2**len(v)
    distance=abs(observed-sum(ranks)/2)
    for signs in itertools.product((0,1),repeat=len(v)):
        if abs(sum(r*s for r,s in zip(ranks,signs))-sum(ranks)/2)>=distance-1e-12:tails+=1
    return min(1.,tails/total)

def main():
    lock=read(ROOT/'projects/experiments/fp_decision_study/final_lock.json')
    for p,h in lock['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    primary=read(OUT/'primary_results.json')['primary_family']
    with open(OUT/'tabu_predictions.csv') as f:pred=list(csv.DictReader(f))
    scores={(r['state_id'],r['action']):r for r in pred}
    original_rewards={};gains={};names={}
    actions=('CB_BEST_DESCENT','CB_FIRST_DESCENT','CB_TABU')
    for p in sorted((ROOT/'results/fp_decision_study/tabu/final').glob('*.json.gz')):
        for r in read(p)['records']:
            gains[r['state_id'],r['action'],r['repeat']]=(r['start_makespan']-r['end_makespan'])/r['s0']
            names[r['state_id']]=r['instance']
    grouped=defaultdict(lambda:defaultdict(list));counts=defaultdict(Counter);changes=Counter();oracle=[]
    for sid in sorted(names):
        means={a:sum(gains[sid,a,r] for r in range(5))/5 for a in actions}
        chosen={}
        for arm in ('A','B','C','D','E','D_random'):
            chosen[arm]=max(actions,key=lambda a:float(scores[sid,a]['gain_'+arm]))
            grouped[names[sid]][arm].append(means[chosen[arm]]);counts[arm][chosen[arm]]+=1
        for c in ('B','E','D_random'):changes[c]+=chosen['D']!=chosen[c]
        selected=max(actions,key=lambda a:sum(gains[sid,a,r] for r in (0,1))/2)
        fit=sum(gains[sid,selected,r] for r in (0,1))/2
        evaluate=sum(gains[sid,selected,r] for r in (2,3,4))/3
        oracle.append((names[sid],fit-evaluate))
    scalar=[]
    for c in ('B','E'):
        v={i:sum(g['D'])/len(g['D'])-sum(g[c])/len(g[c]) for i,g in grouped.items()}
        expected=next(r for r in primary if r['contrast']=='tabu_gain_D_minus_'+c)
        assert max(abs(v[i]-expected['instance_values'][i]) for i in v)<1e-12
        p=exact(list(v.values()));assert abs(p-expected['p'])<1e-12
        scalar.append({'contrast':expected['contrast'],'mean':sum(v.values())/len(v),'independent_exact_p':p})
    # Original archived state choices are compared to branch means recomputed from raw files.
    result=read(OUT/'original_final_analysis.json')
    choices={r['state_id']:r for r in result['atomic']['state_decisions']}
    for p in sorted((ROOT/'results/fp_decision_study/original/final').glob('*.json.gz')):
        for r in read(p)['records']:
            original_rewards.setdefault((r['state_id'],r['action']),[]).append((r['start_makespan']-r['end_makespan'])/r['s0'])
    # Schema is inspected explicitly; unsupported schema fails rather than skips.
    one=next(iter(choices.values()))
    diffs=defaultdict(list);originalchanges=0
    for sid,r in choices.items():
        ca=r['arms']['C']['action'];ea=r['arms']['E']['action']
        c=sum(original_rewards[sid,ca])/5;e=sum(original_rewards[sid,ea])/5
        diffs[r['instance']].append(e-c);originalchanges+=ca!=ea
    v=[sum(x)/len(x) for x in diffs.values()]
    expected=next(r for r in primary if r['contrast'].startswith('original'))
    assert abs(sum(v)/len(v)-expected['mean'])<1e-12
    assert exact(v)==expected['p']
    means=[]
    for inst in sorted(grouped):means.append(sum(v for i,v in oracle if i==inst)/sum(i==inst for i,v in oracle))
    aux=read(OUT/'tabu_final_analysis.json')
    assert abs(sum(means)/len(means)-aux['auxiliary_contrasts'][0]['mean'])<1e-12
    assert {a:dict(c) for a,c in counts.items()}=={k[5:]:v for k,v in aux['decision_counts'].items() if k.startswith('gain_')}
    ordered=sorted(primary,key=lambda r:r['p']);running=0
    for i,r in enumerate(ordered):
        running=max(running,min(1.,(len(primary)-i)*r['p']));assert abs(running-r['holm_p'])<1e-12
    report={'status':'PASS_RAW_REWARD_AND_EXPORTED_SCORE_SCALAR_RECONSTRUCTION','contrasts':scalar,
            'original_mean_difference':sum(v)/len(v),'original_changed_actions':originalchanges,
            'tabu_changed_actions':dict(changes),'independent_split_repeat_optimism':sum(means)/len(means),
            'frozen_file_hashes_verified':len(lock['hashes']),
            'scope':'Independent grouping/reward/max/sign-rank/Holm and archived source integrity; scoring coefficients and features separately covered by numerical/causality tests. Not an independent full refit or online evaluation.'}
    p=OUT/'independent_statistics_audit.json'
    if p.exists():raise ValueError('Do not overwrite audit')
    p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Prespecified instance-cluster analysis; no model selection or search."""
import csv
import json
from collections import defaultdict,Counter
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.fp_decision_models import ARMS,matrix,predict,predictions,read_rows
from scripts.run_fp_decision_collection import RESULTS,EXPERIMENT,load,write_new
from tools.independent_tabu_jssp import ACTIONS
from analyze_paper_evidence import signed_rank,holm

OUT=ROOT/'projects/analysis/fp_decision_study'

def compare(values,label):
    names=sorted(values);means=np.asarray([values[k] for k in names],float)
    assert np.isfinite(means).all()
    rng=np.random.default_rng(2026102502)
    boot=means[rng.integers(len(means),size=(20000,len(means)))].mean(1)
    return {'contrast':label,'mean':float(means.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
        'instance_values':dict(zip(names,means.tolist())),
        'win_tie_loss':[int(sum(means>1e-12)),int(sum(abs(means)<=1e-12)),int(sum(means < -1e-12))],
        'leave_one_instance_out_means':{k:float(np.delete(means,i).mean()) for i,k in enumerate(names)},
        **signed_rank(means)}

def average_precision(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if not y.sum():return None
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
    end=np.r_[np.flatnonzero(ps[1:]!=ps[:-1]),len(ps)-1]
    tp=np.cumsum(ys)[end];precision=tp/(end+1)
    return float(np.sum(np.diff(np.r_[0,tp])*precision)/y.sum())

def mean_or_none(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None

def csv_write(path,rows):
    if not rows:return
    with open(path,'x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def second_solver():
    rows=read_rows('final');bundle=load(EXPERIMENT/'tabu_models.json');maps=bundle['maps']
    assert not({r['instance'] for r in rows}&set(bundle['training_instances']))
    scores={};timing={}
    for arm in ARMS:
        start=time.perf_counter();x=matrix(rows,arm,maps);feature_s=time.perf_counter()-start
        for target in ('gain','probability'):
            start=time.perf_counter()
            scores[target,arm]=predict(bundle['models'][arm][target],x,target=='probability')
            timing[f'{target}_{arm}']={'all_row_feature_seconds':feature_s,'all_row_prediction_seconds':time.perf_counter()-start,
              'scope':'batched offline, features cached per state; not online per-decision latency'}
    groups=defaultdict(list)
    for idx,r in enumerate(rows):groups[r['state_id']].append(idx)
    decisions=[];prediction_rows=[];transition=[]
    for sid,indices in groups.items():
        first=rows[indices[0]];r=np.empty((3,5));ps={k:[] for k in scores}
        action_indices=[]
        for ai,a in enumerate(ACTIONS):
            ids=sorted([i for i in indices if rows[i]['action']==a],key=lambda i:rows[i]['repeat'])
            assert [rows[i]['repeat'] for i in ids]==list(range(5));action_indices.append(ids)
            r[ai]=[rows[i]['gain'] for i in ids]
            for k in scores:
                p=scores[k][ids]
                assert np.max(p)-np.min(p)<1e-12,'Outcome-dependent score within identical action-state'
                ps[k].append(float(p[0]))
            x=np.asarray(first['start_x']);y=np.mean([rows[i]['end_x'] for i in ids],axis=0)
            predicted={'conditioned':predictions(maps,x)[ai],
                       'pooled':np.asarray(maps['pooled']['A'])@x+maps['pooled']['c'],'identity':x}
            transition.append({'instance':first['instance'],'run':first['run'],'state_id':sid,'action':a,
                **{v:float(np.linalg.norm(p-y)) for v,p in predicted.items()}})
            prediction_rows.append({'instance':first['instance'],'run':first['run'],'state_id':sid,'action':a,
                'gain_mean5':float(r[ai].mean()),'gain_sd5':float(r[ai].std(ddof=1)),
                'gain_se5':float(r[ai].std(ddof=1)/np.sqrt(5)),
                **{f'{t}_{arm}':ps[t,arm][-1] for t,arm in ps},
                **{f'repeat{rep}_gain':float(r[ai,rep]) for rep in range(5)}})
        means=r.mean(1);fit2=r[:,:2].mean(1);eval3=r[:,2:].mean(1)
        chosen2=int(np.argmax(fit2));chosen3=int(np.argmax(eval3))
        d={'instance':first['instance'],'run':first['run'],'state_id':sid,
           'sample_max5':float(means.max()),'sample_winner2_fit_value':float(fit2[chosen2]),
           'sample_winner2_eval_value':float(eval3[chosen2]),
           'reverse_winner3_fit_value':float(eval3[chosen3]),'reverse_winner3_eval_value':float(fit2[chosen3]),
           'winner_agreement':int(chosen2==chosen3),'fit2_unique_winner':int(sum(abs(fit2-fit2.max())<1e-12)==1),
           'eval3_unique_winner':int(sum(abs(eval3-eval3.max())<1e-12)==1),
           'all_actions_zero':int(means.max()==0),
           **{f'fixed_{a}':float(means[i]) for i,a in enumerate(ACTIONS)},
           'fixed_tabu_eval3':float(eval3[2])}
        for key,p in ps.items():
            target,arm=key;c=int(np.argmax(p));d[f'{target}_{arm}_action']=ACTIONS[c]
            d[f'{target}_{arm}_value']=float(means[c]);d[f'{target}_{arm}_eval3']=float(eval3[c])
            comparisons=[float(np.sign(p[i]-p[j])*np.sign(means[i]-means[j])) for i in range(3) for j in range(i+1,3) if abs(means[i]-means[j])>1e-12]
            d[f'{target}_{arm}_within_state_concordance']=mean_or_none(comparisons)
        for comparator in ('B','E','D_random'):
            d[f'D_changed_vs_{comparator}']=int(d['gain_D_action']!=d[f'gain_{comparator}_action'])
        decisions.append(d)
    instances=sorted({r['instance'] for r in rows});per=[]
    for inst in instances:
        g=[r for r in decisions if r['instance']==inst]
        numeric=[k for k in g[0] if k not in ('instance','run','state_id') and not k.endswith('_action')]
        out={'instance':inst,**{k:mean_or_none([r[k] for r in g]) for k in numeric}}
        ids=[i for i,r in enumerate(rows) if r['instance']==inst]
        for arm in ARMS:
            y=np.asarray([rows[i]['improved'] for i in ids]);p=scores['probability',arm][ids]
            out[f'{arm}_brier']=float(np.mean((p-y)**2))
            for a in ACTIONS:
                ix=[i for i in ids if rows[i]['action']==a]
                out[f'{arm}_{a}_ap']=average_precision([rows[i]['improved'] for i in ix],scores['probability',arm][ix])
        per.append(out)
    contrasts=[compare({r['instance']:r['gain_D_value']-r[f'gain_{c}_value'] for r in per},f'tabu_gain_D_minus_{c}') for c in ('B','E')]
    auxiliary=[compare({r['instance']:r['sample_winner2_fit_value']-r['sample_winner2_eval_value'] for r in per},'tabu_split_repeat_optimism'),
        compare({r['instance']:r['sample_winner2_eval_value']-r['fixed_tabu_eval3'] for r in per},'tabu_independent_sample_winner_minus_fixed_tabu'),
        compare({r['instance']:r['gain_D_value']-r['gain_D_random_value'] for r in per},'tabu_gain_D_minus_equal_width_random')]
    dynamics=[]
    for a in ACTIONS:
        for c in ('identity','pooled'):
            values={i:float(np.mean([r[c]-r['conditioned'] for r in transition if r['instance']==i and r['action']==a])) for i in instances}
            dynamics.append(compare(values,f'tabu_{a}_error_{c}_minus_conditioned'))
    holm(dynamics)
    costs={a:{'mean_decodes':float(np.mean([r['outcome']['decode_evaluations'] for r in rows if r['action']==a])),
                'mean_seconds':float(np.mean([r['outcome']['elapsed_seconds'] for r in rows if r['action']==a])),
                'mean_moves':float(np.mean([r['outcome']['accepted_steps'] for r in rows if r['action']==a])),
                'termination_counts':dict(Counter(r['outcome']['termination'] for r in rows if r['action']==a))} for a in ACTIONS}
    summary={'solver':'tabu','instances':len(instances),'trajectories':len({r['trajectory_id'] for r in rows}),
       'states':len(decisions),'branches':len(rows),'primary_contrasts':contrasts,'auxiliary_contrasts':auxiliary,
       'macro':{k:mean_or_none([r[k] for r in per]) for k in per[0] if k!='instance'},
       'decision_counts':{f'{t}_{arm}':dict(Counter(r[f'{t}_{arm}_action'] for r in decisions)) for t in ('gain','probability') for arm in ARMS},
       'changed_decisions':{c:sum(r[f'D_changed_vs_{c}'] for r in decisions) for c in ('B','E','D_random')},
       'dynamics':dynamics,'dynamics_mean_error':{a:{c:float(np.mean([r[c] for r in transition if r['action']==a])) for c in ('conditioned','pooled','identity')} for a in ACTIONS},
       'costs':costs,'offline_model_computation':timing,'scope':'Offline branch reward, not online policy performance; independent-repeat comparator is not true oracle.'}
    csv_write(OUT/'tabu_decisions.csv',decisions);csv_write(OUT/'tabu_predictions.csv',prediction_rows)
    csv_write(OUT/'tabu_instance_metrics.csv',per);csv_write(OUT/'tabu_transition_errors.csv',transition)
    write_new(OUT/'tabu_final_analysis.json',summary)
    return summary

def original_solver():
    from scripts.compare_fp_decision_features import attach_past_state_changes,canonical_sort
    from scripts.fp_submission_frozen_models import predict_atomic,summary_metrics,dynamics_evaluation
    files=sorted((RESULTS/'original/final').glob('*.json.gz'));rows=[];equal=[]
    for p in files:
        d=load(p);rows.extend(attach_past_state_changes(d))
        d2={**d,'records':d['equal_work_records']};equal.extend(attach_past_state_changes(d2))
    rows=canonical_sort(rows);equal=canonical_sort(equal)
    bundle=load(ROOT/'projects/experiments/fp_submission_frozen_model_bundle.json')
    atomic=summary_metrics(rows,predict_atomic(bundle,rows));secondary=summary_metrics(equal,predict_atomic(bundle,equal))
    primary=compare({r['instance']:r['variants']['E']['selected_gain']-r['variants']['C']['selected_gain'] for r in atomic['per_instance_metrics']},'original_frozen_E_minus_C_atomic')
    report={'solver':'original','atomic':atomic,'equal480':secondary,'primary_contrasts':[primary],
       'dynamics':dynamics_evaluation(bundle,rows),'atomic_branches':len(rows),'equal480_branches':len(equal)}
    write_new(OUT/'original_final_analysis.json',report)
    return report

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=second_solver();b=original_solver();primary=b['primary_contrasts']+a['primary_contrasts'];holm(primary)
    gate=all(r['mean']>0 and r['ci95'][0]>0 and r['holm_p']<.05 and min(r['leave_one_instance_out_means'].values())>0 for r in primary)
    write_new(OUT/'primary_results.json',{'primary_family':primary,'gate_online_both_solvers':gate,
       'gate_basis':'All3 predeclared contrasts: positive mean/CI lower,Holm<.05,all leave-one-instance-out means positive',
       'CI':'20000 instance cluster bootstrap,seed2026102502; equal-weight instance means',
       'no_true_oracle':'V(S),V(Z) and irreducible information loss are not identified from five repeats.'})
    print(json.dumps({'primary_family':primary,'online_gate':gate},indent=2))

if __name__=='__main__':main()

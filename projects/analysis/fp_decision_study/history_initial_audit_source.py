#!/usr/bin/env python3
"""Read-only historical evidence audit for the decision-sufficiency manuscript.

No search, model fit, model selection, or historical file modification occurs.
Fresh instance-bootstrap intervals are explicitly retrospective sensitivity
analyses, not replacements for the archived frozen hierarchical intervals.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ('GA', 'LS', 'VND')
TOL = 1e-12
READ_PATHS = [
 'projects/docs/fp_agent_submission_manuscript.md',
 'projects/docs/fp_agent_empirical_draft.md',
 'projects/docs/fp_empirical_revised_draft.md',
 'projects/docs/fp_empirical_submission_ready.md',
 'projects/docs/fp_agent_evidence_report.md',
 'projects/docs/fp_agent_semantic_audit.md',
 'projects/docs/fp_agent_method_v2.md',
 'projects/docs/fp_agent_decision_diagnosis.md',
 'projects/docs/fp_agent_final_go_nogo.md',
 'projects/docs/fp_empirical_scorer_convergence.md',
 'projects/docs/fp_empirical_repeatability_report.md',
 'projects/docs/fp_agent_exposure_audit.md',
 'projects/docs/safe_agent_results.md',
 'projects/docs/safe_agent_method.md',
 'projects/docs/safe_agent_ablation.md',
 'projects/docs/safe_agent_independent_audit.md',
 'projects/docs/brfp_fixed_point_definition.md',
 'projects/docs/brfp_return_diagnostic.md',
 'projects/docs/brfp_decision_logic_audit.md',
 'projects/docs/brfp_submission_go_nogo.md',
 'projects/docs/brfp_agent_method.md',
 'projects/docs/brfp_ablation_results.md',
 'projects/docs/brfp_novelty_audit.md',
 'projects/docs/fp_proof_reuse_security_audit.md',
 'projects/docs/fp_proof_reuse_corrected_exhaustive_audit.md',
 'projects/docs/fp_proof_reuse_final_research_verdict.md',
 'scripts/run_fp_action_diagnostics.py',
 'scripts/fp_empirical_repeatability.py',
 'scripts/analyze_fp_empirical_repeatability.py',
 'scripts/fp_submission_frozen_models_loader_compat.py',
 'scripts/compare_fp_decision_features.py',
 'scripts/diagnose_fp_action_oracle.py',
 'scripts/analyze_safe_agent.py',
 'scripts/audit_brfp_decision_logic.py',
 'src/tools/action_fp.py',
 'src/tools/safe_agent_policy.py',
 'src/tools/brfp_closure.py',
 'src/tools/fp_proof_reuse.py',
 'projects/experiments/fp_v2_manifest.json',
 'projects/experiments/fp_v2_exposure.json',
 'projects/experiments/fp_submission_final_confirmation_protocol.json',
 'projects/experiments/fp_empirical_repeatability_protocol.json',
 'projects/experiments/fp_submission_frozen_model_bundle.json',
 'projects/analysis/fp_empirical_confirmation_independent_audit.json',
 'projects/analysis/fp_submission_final_independent_audit.json',
 'projects/analysis/fp_empirical_repeatability_independent_audit.json',
 'projects/analysis/safe_agent_independent_audit.json',
 'projects/analysis/safe_agent_development_analysis.json',
 'projects/analysis/brfp_decision_logic_audit.json',
 'projects/analysis/fp_proof_reuse_remediation_final_receipt.json',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_read(path):
    return json.loads((ROOT/path).read_text())


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list,tuple)) else v
                         for k,v in row.items()} for row in rows)


def average_precision(y, p):
    """Independent tied-score AP, zero for no positives, without sklearn."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-p, kind='stable'); y=y[order]; p=p[order]
    ends=np.r_[np.flatnonzero(p[1:] != p[:-1]),len(y)-1]
    hits=np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0,hits])/y.sum()*hits/(ends+1)))


def summarize_difference(rows, field, seed=20261025, repetitions=20000):
    """Instance-only paired bootstrap sensitivity, conditional on trained models."""
    grouped=defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['instance']][row['trajectory_id']].append(float(row[field]))
    means={i:float(np.mean([np.mean(v) for v in t.values()])) for i,t in sorted(grouped.items())}
    x=np.asarray(list(means.values()))
    rng=np.random.default_rng(seed)
    boot=x[rng.integers(len(x),size=(repetitions,len(x)))].mean(axis=1)
    return {'effect':float(x.mean()),'instance_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
            'instance_values':means,'instance_win_tie_loss':[int(sum(x>TOL)),int(sum(abs(x)<=TOL)),int(sum(x < -TOL))],
            'n_instances':len(x),'bootstrap_repetitions':repetitions,'bootstrap_seed':seed,
            'inference_role':'retrospective instance-only sensitivity; no new confirmatory test or model selection'}


def load_raw():
    specs=[('development','fp_v2_development'),('exposed_validation','fp_v2_validation'),
           ('exposed_confirmation','fp_empirical_confirmation'),('development_new_repeats','fp_empirical_repeatability_development')]
    rows={}; provenance=[]; files=[]; snapshot_hashes={}; counts=[]
    for cohort,folder in specs:
        instances=set(); trajectories=set(); states=set(); windows=Counter(); total=0
        for path in sorted((ROOT/'results'/folder).glob('*.json')):
            d=json.loads(path.read_text())
            if 'trajectory_id' not in d or 'records' not in d:
                continue
            files.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'cohort':cohort})
            instances.add(d['instance']); trajectories.add(d['trajectory_id'])
            archive=ROOT/d['population_archive']
            if archive not in snapshot_hashes:
                snapshot_hashes[archive]=sha(archive)
            assert snapshot_hashes[archive] == d['population_archive_sha256'], archive
            records=[('atomic',r) for r in d['records']]
            if 'equal_work_records' in d:
                records += [('fair480',r) for r in d['equal_work_records']]
            for window,r in records:
                window=r.get('window_mode',window)
                if window == 'equal480': window='fair480'
                assert window in ('atomic','fair480')
                key=(cohort,window,r['state_id'],r['action'],r['repeat'])
                assert key not in rows,key
                gain=(r['start_makespan']-r['end_makespan'])/r['s0']
                assert abs(gain-r['gain'])<TOL
                assert bool(r['improved']) == (r['end_makespan']<r['start_makespan'])
                assert r['queries'] >= r['decodes'] >= 0
                rows[key]={k:r[k] for k in ('instance','trajectory_id','state_id','generation','action','repeat','gain','improved','queries','decodes','wall_seconds','start_x','end_x')}
                rows[key].update(cohort=cohort,window=window)
                states.add(r['state_id']); windows[window]+=1;total+=1
        counts.append({'cohort':cohort,'raw_trajectory_json':len(trajectories),'instances':len(instances),
                       'trajectories':len(trajectories),'states':len(states),'branches':total,
                       'atomic_branches':windows['atomic'],'fair480_branches':windows['fair480'],
                       'instance_names':sorted(instances),'directory':f'results/{folder}',
                       'role':'all now historically exposed; offline intervention, not FP online trajectories'})
    assert sum(r['branches'] for r in counts)==109440
    assert len({r['instance'] for r in rows.values()})==48
    assert len({r['trajectory_id'] for r in rows.values()})==960
    assert len({r['state_id'] for r in rows.values()})==8640
    return rows,counts,files


def evaluate_predictions(predictions, raw, cohort, window, label):
    grouped=defaultdict(list)
    for p in predictions:
        key=(cohort,window,p['state_id'],p['action'],p['repeat'])
        assert key in raw,key
        r=raw[key]
        assert abs(r['gain']-p['gain'])<TOL and r['improved']==p['improved']
        grouped[p['state_id']].append(p)
    states=[]; trajectories=defaultdict(list); arms=list(predictions[0]['probabilities'])
    for sid,pr in sorted(grouped.items()):
        action_groups={a:[r for r in pr if r['action']==a] for a in ACTIONS}
        gains=np.asarray([np.mean([r['gain'] for r in action_groups[a]]) for a in ACTIONS])
        for a in ACTIONS:
            assert len({json.dumps(r['probabilities'],sort_keys=True) for r in action_groups[a]})==1
        s={'cohort':cohort,'window':window,'evaluation':label,'instance':pr[0]['instance'],
           'trajectory_id':pr[0]['trajectory_id'],'state_id':sid,'generation':pr[0]['generation'],
           'sample_oracle_gain':float(gains.max())}
        for a,g in zip(ACTIONS,gains):s[a+'_gain']=float(g)
        for arm in arms:
            scores=[action_groups[a][0]['probabilities'][arm] for a in ACTIONS]
            index=int(np.argmax(scores)); action=ACTIONS[index]
            s[arm+'_action']=action;s[arm+'_gain']=float(gains[index])
            s[arm+'_sample_regret']=float(gains.max()-gains[index])
            s[arm+'_LS_VND_margin']=float(scores[1]-scores[2])
        ref='C' if 'C' in arms and label != 'original_m2' else 'B'
        s['FP_minus_NoFP']=s['E_gain']-s[ref+'_gain']
        s['FP_changed']=s['E_action']!=s[ref+'_action']
        s['oracle_minus_VND']=float(gains.max()-gains[2])
        s['LS_minus_VND']=float(gains[1]-gains[2])
        s['reference_arm']=ref
        states.append(s);trajectories[s['instance'],s['trajectory_id']].extend(pr)
    trs=[]
    for (instance,tid),pr in sorted(trajectories.items()):
        ss=[s for s in states if s['trajectory_id']==tid]
        y=[r['improved'] for r in pr]
        for arm in arms:
            p=[r['probabilities'][arm] for r in pr]
            trs.append({'evaluation':label,'cohort':cohort,'window':window,'instance':instance,
                        'trajectory_id':tid,'arm':arm,'ap':average_precision(y,p),
                        'brier':float(np.mean((np.asarray(y)-p)**2)),
                        'selected_gain':float(np.mean([s[arm+'_gain'] for s in ss])),
                        'sample_regret':float(np.mean([s[arm+'_sample_regret'] for s in ss]))})
    macro=[]
    for arm in arms:
        aa=[r for r in trs if r['arm']==arm]
        by=defaultdict(list)
        for r in aa:by[r['instance']].append(r)
        macro.append({'evaluation':label,'cohort':cohort,'window':window,'arm':arm,
           **{k:float(np.mean([np.mean([r[k] for r in rr]) for rr in by.values()]))
              for k in ('ap','brier','selected_gain','sample_regret')},
           **{a+'_choices':sum(s[arm+'_action']==a for s in states) for a in ACTIONS},'states':len(states)})
    return states,trs,macro


def repeat_analysis(raw, converged_states):
    choices={s['state_id']:s for s in converged_states}
    ids=sorted({r['state_id'] for k,r in raw.items() if k[0]=='development_new_repeats'})
    out=[]
    for sid in ids:
        old=np.asarray([[raw['development','atomic',sid,a,i]['gain'] for a in ACTIONS] for i in (0,1)])
        winner=int(np.argmax(old.mean(axis=0)))
        tied=np.flatnonzero(old.mean(axis=0)>=old.mean(axis=0).max()-TOL)
        vnd_tie=2 if 2 in tied else winner
        for window,reps in [('atomic',range(2,7)),('fair480',range(5))]:
            new=np.asarray([[raw['development_new_repeats',window,sid,a,i]['gain'] for a in ACTIONS] for i in reps])
            pooled=np.vstack([old,new]) if window=='atomic' else new
            mu=new.mean(axis=0); s=choices[sid]
            row={'instance':s['instance'],'trajectory_id':s['trajectory_id'],'state_id':sid,'window':window,
                 'old2_oracle_minus_VND':float(old.mean(axis=0).max()-old[:,2].mean()),
                 'old2_choice_independent5_minus_VND':float(mu[winner]-mu[2]),
                 'old2_choice_VNDtie_independent5_minus_VND':float(mu[vnd_tie]-mu[2]),
                 'selection_optimism':float(old.mean(axis=0).max()-old[:,2].mean()-(mu[winner]-mu[2])),
                 'LS_minus_VND':float(mu[1]-mu[2]),
                 'pooled_LS_minus_VND':float(pooled[:,1].mean()-pooled[:,2].mean()),
                 'new5_sample_oracle_minus_VND':float(mu.max()-mu[2]),
                 'old2_action':ACTIONS[winner],'new5_action':ACTIONS[int(np.argmax(mu))]}
            for a,i in zip(ACTIONS,range(3)):
                row[a+'_gain']=float(mu[i]);row[a+'_within_state_sd']=float(new[:,i].std(ddof=1))
            for arm in ('C','E','N','P'):
                row[arm+'_action']=s[arm+'_action'];row[arm+'_gain']=float(mu[ACTIONS.index(s[arm+'_action'])])
            row['FP_minus_NoFP']=row['E_gain']-row['C_gain'];row['FP_changed']=row['E_action']!=row['C_action']
            out.append(row)
    return out


def dynamics(raw):
    bundle=json_read('projects/experiments/fp_submission_frozen_model_bundle.json')
    groups=defaultdict(list)
    for k,r in raw.items():
        if k[:2]==('exposed_confirmation','atomic'):
            groups[r['instance'],r['trajectory_id'],r['state_id'],r['action']].append(r)
    out=[];w=np.asarray([.35,.30,.15,.20])
    for (i,t,s,a),rr in sorted(groups.items()):
        x=np.asarray(rr[0]['start_x']);y=np.mean([r['end_x'] for r in rr],axis=0)
        d={'instance':i,'trajectory_id':t,'state_id':s,'action':a}
        for arm,model in [('conditioned',bundle['action_fp']['maps'][a]),('pooled',bundle['pooled_atomic_affine']),('identity',None)]:
            pred=x if model is None else np.asarray(model['A'])@x+model['c']
            d[arm+'_error']=float(np.sqrt(np.sum(w*(pred-y)**2)))
        d['pooled_minus_conditioned']=d['pooled_error']-d['conditioned_error']
        d['identity_minus_conditioned']=d['identity_error']-d['conditioned_error'];out.append(d)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='projects/analysis/fp_decision_study');args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'history_audit.json').exists():raise FileExistsError('Use a fresh output directory; historical audits are immutable')
    inventory=[]
    for rel in READ_PATHS:
        p=ROOT/rel
        inventory.append({'path':rel,'exists':p.exists(),'sha256':sha(p) if p.exists() else None,
                          'bytes':p.stat().st_size if p.exists() else None,
                          'role':'historical context/source; not pooled as algorithm-success evidence'})
    raw,counts,files=load_raw()
    print('raw grids loaded',sum(c['branches'] for c in counts),flush=True)
    final=json_read('projects/analysis/fp_submission_final_confirmation_results.json')
    conv=json_read('projects/analysis/fp_submission_convergence_results.json')
    original=json_read('projects/analysis/fp_agent_decision_feature_comparison.json')
    evals=[('exposed_confirmation','atomic','frozen_confirmation',final['predictions']['atomic']),
           ('exposed_confirmation','fair480','frozen_confirmation',final['predictions']['fair480']),
           ('development','atomic','converged_development',conv['predictions']),
           ('development','atomic','original_structured_development',original['predictions'])]
    states=[];trs=[];macro=[];effects=[];converged_states=None
    for cohort,window,label,pred in evals:
        ss,tt,mm=evaluate_predictions(pred,raw,cohort,window,label)
        states+=ss;trs+=tt;macro+=mm
        if label=='converged_development':converged_states=ss
        eff=summarize_difference(ss,'FP_minus_NoFP')
        eff.update(cohort=cohort,window=window,evaluation=label,contrast='E_minus_C',
                   changed_choices=sum(s['FP_changed'] for s in ss),states=len(ss),
                   changed_gain_positive=sum(s['FP_changed'] and s['FP_minus_NoFP']>TOL for s in ss),
                   changed_gain_negative=sum(s['FP_changed'] and s['FP_minus_NoFP'] < -TOL for s in ss),
                   changed_gain_tie=sum(s['FP_changed'] and abs(s['FP_minus_NoFP'])<=TOL for s in ss))
        if label=='frozen_confirmation':
            archived=next(c for c in final['primary_comparisons'] if c['window']==window and c['comparator']=='C' and c['metric']=='selected_gain')
            assert abs(eff['effect']-archived['mean_difference'])<1e-14
            eff['archived_hierarchical_95ci']=archived['hierarchical_bootstrap_95ci'];eff['archived_holm_p']=archived['holm_p']
            summary=final['atomic_summary' if window=='atomic' else 'fair480_summary']
            for m in mm:
                old=summary['macro_instance_trajectory_metrics'][m['arm']]
                for metric in ('ap','brier','selected_gain'):assert abs(m[metric]-old[metric])<1e-12,(m,old)
        effects.append(eff)
        print('evaluated',label,window,len(ss),flush=True)
    repeat=repeat_analysis(raw,converged_states)
    repeat_effects=[]
    for window in ('atomic','fair480'):
        rr=[r for r in repeat if r['window']==window]
        for field in ('FP_minus_NoFP','LS_minus_VND','pooled_LS_minus_VND','old2_oracle_minus_VND','old2_choice_independent5_minus_VND',
                      'old2_choice_VNDtie_independent5_minus_VND','selection_optimism','new5_sample_oracle_minus_VND'):
            repeat_effects.append({'window':window,'contrast':field,**summarize_difference(rr,field)})
    # Match final atomic checkpoints to the fair480 state grid before window comparison.
    window_effects=[]
    for window in ('atomic','fair480'):
        ss=[s for s in states if s['evaluation']=='frozen_confirmation' and s['window']==window and s['generation'] in (40,100,160)]
        window_effects.append({'cohort':'exposed_confirmation_matched_1200','window':window,
                              'contrast':'LS_minus_VND',**summarize_difference(ss,'LS_minus_VND')})
    dyn=dynamics(raw);dyn_effects=[]
    for a in ACTIONS:
        rr=[r for r in dyn if r['action']==a]
        for ref in ('pooled','identity'):
            eff=summarize_difference(rr,ref+'_minus_conditioned')
            archived=next(c for c in final['dynamics']['comparisons'] if c['action']==a and c['comparator']==ref)
            assert abs(eff['effect']-archived['mean_error_reduction'])<1e-12
            dyn_effects.append({'action':a,'comparator':ref,**eff,'archived_hierarchical_95ci':archived['hierarchical_bootstrap_95ci'],
                               'archived_holm_p':archived['holm_p']})
    sources=['projects/analysis/fp_submission_final_confirmation_results.json','projects/analysis/fp_submission_convergence_results.json',
             'projects/analysis/fp_agent_decision_feature_comparison.json']
    for p in sources:inventory.append({'path':p,'exists':True,'sha256':sha(ROOT/p),'bytes':(ROOT/p).stat().st_size,'role':'frozen probabilities replayed; all matched raw rewards'})
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'historical read-only evidence; no refit or new search',
             'counts':counts,'unique_instances':48,'unique_neutral_trajectories':960,'unique_saved_states':8640,'diagnostic_branches':109440,
             'missing_paths':[r['path'] for r in inventory if not r['exists']],
             'prediction_replay_status':'PASS_ALL_FROZEN_FINAL_AP_BRIER_GAIN_AND_DYNAMICS_MEANS',
             'selected_gain_effects':effects,'repeat_effects':repeat_effects,'matched_window_effects':window_effects,'dynamics_effects':dyn_effects,
             'historical_final_20_now_exposed':True,'scorer_retrained':False,'online_fp_agent_tested':False,
             'new_CI_semantics':'20000 paired instance-only bootstrap seed20261025; retrospective sensitivity; historical hierarchical intervals separately retained',
             'limits':['No new endpoint decoding; previous independent endpoint audits remain separate sources.',
                       'Historical C/E share the scorer family but differ in width106/148; N/P match E148; endpoint auxiliary supervision not matched by C.',
                       'Observed sample-oracle regret is not true conditional-mean regret or identified representation insufficiency.',
                       'All historical48 now exposed; old20 cannot become a new final confirmation.',
                       'Supplementary repeats reuse80 of960 trajectories; SAFE/BRFP/CP are separate exploratory lines, never added to109440.',
                       'No online FP policy performance, independent solver replication, or universal FP failure follows from this history.'],
             'script_sha256':sha(__file__)}
    for stem,rows in [('sources',inventory),('raw_files',files),('cohorts',counts),('state_decisions',states),('trajectory_metrics',trs),('model_metrics',macro),
                      ('selected_gain_effects',effects),('repeat_states',repeat),('repeat_effects',repeat_effects),('matched_window_effects',window_effects),
                      ('dynamics_states',dyn),('dynamics_effects',dyn_effects)]:
        csv_write(out/f'history_{stem}.csv',rows)
    (out/'history_audit.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({k:receipt[k] for k in ('diagnostic_branches','unique_instances','unique_neutral_trajectories','unique_saved_states','missing_paths')},ensure_ascii=False))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Frozen empirical affine representations and shared lightweight scorers.

No test outcomes enter predictions. Transition features for fitting are
leave-one-instance-out; all representations use identical fitted maps.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.run_fp_decision_collection import load,write_new,EXPERIMENT,RESULTS
from scripts.check_fp_scorer_convergence import ConvergedLogistic,stable_sigmoid
from tools.independent_tabu_jssp import ACTIONS

ARMS=('A','B','C','D','E','D_random')

def read_rows(split,solver='tabu'):
    files=sorted((RESULTS/solver/split).glob('*.json.gz'))
    if not files:raise ValueError('Empty dataset')
    return [r for p in files for r in load(p)['records']]

def affine(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['state_id'],r['action'])].append(r)
    x=np.asarray([g[0]['start_x'] for g in grouped.values()]);y=np.asarray([np.mean([r['end_x'] for r in g],axis=0) for g in grouped.values()])
    center=x.mean(0);scale=np.maximum(x.std(0),1e-6)
    z=np.column_stack(((x-center)/scale,np.ones(len(x))))
    penalty=np.eye(5)*.1;penalty[-1,-1]=0
    beta=np.linalg.solve(z.T@z+penalty,z.T@(y-x))
    return {'A':(np.eye(4)+beta[:4].T/scale).tolist(),
            'c':(beta[-1]-(center/scale)@beta[:4]).tolist(),'training_states':len(x)}

def fit_maps(rows):
    return {**{a:affine([r for r in rows if r['action']==a]) for a in ACTIONS},'pooled':affine(rows)}

def predictions(maps,x):
    return np.array([np.asarray(maps[a]['A'])@x+maps[a]['c'] for a in ACTIONS])

def vector(row,arm,maps):
    x=np.asarray(row['start_x'],float);obs=row['observed']
    # Ordinary4 includes actual stagnation and progress; B contains all four x coordinates.
    a=np.r_[obs['ordinary'],obs['history']]
    if arm=='A':return a
    b=np.r_[a,obs['structure'],[x[i]*x[j] for i in range(4) for j in range(i,4)]]
    if arm=='B':return b
    y=predictions(maps,x);delta=y-x
    diffs=np.asarray([y[i]-y[j] for i in range(3) for j in range(i+1,3)])
    if arm=='C':extra=y.ravel()
    elif arm=='D':extra=np.r_[np.linalg.norm(delta,axis=1),np.linalg.norm(diffs,axis=1)]
    elif arm=='E':extra=np.r_[delta.ravel(),(delta**2).ravel(),diffs.ravel(),(diffs**2).ravel()]
    elif arm=='D_random':
        rng=np.random.default_rng(2026102501)
        extra=np.linalg.norm(np.einsum('ijk,k->ij',rng.normal(size=(6,4,4)),x)+rng.normal(size=(6,4)),axis=1)
    else:raise ValueError(arm)
    return np.r_[b,extra]

def matrix(rows,arm,maps,by_instance=None):
    states={}
    for r in rows:
        if r['state_id'] not in states:
            states[r['state_id']]=vector(r,arm,by_instance[r['instance']] if by_instance else maps)
    result=[]
    for r in rows:
        v=states[r['state_id']];one=np.asarray([float(r['action']==a) for a in ACTIONS])
        result.append(np.r_[one,np.outer(one,v).ravel()])
    return np.array(result)

def fit_ridge(x,y):
    center=x.mean(0);scale=np.maximum(x.std(0),1e-8)
    z=np.column_stack((np.ones(len(x)),(x-center)/scale))
    p=np.eye(z.shape[1]);p[0,0]=0
    w=np.linalg.solve(z.T@z+p,z.T@y)
    gradient=z.T@(z@w-y)/len(x)+p@w/len(x)
    residual=float(np.max(np.abs(gradient)))
    if not residual<=1e-7:raise ArithmeticError('Ridge stationarity failed')
    # Separate QR least-squares reconstruction verifies the normal-equation solve.
    augmented=np.r_[z,np.sqrt(p)]
    independent=np.linalg.lstsq(augmented,np.r_[y,np.zeros(len(w))],rcond=None)[0]
    prediction_check=float(np.max(np.abs(z@w-z@independent)))
    if prediction_check>1e-7:raise ArithmeticError('Independent QR fit disagreement')
    return dict(center=center.tolist(),scale=scale.tolist(),weights=w.tolist(),
                gradient_inf=residual,independent_prediction_max_error=prediction_check,converged=True)

def predict(model,x,logistic=False):
    m=np.asarray(model['weights']);v=m[0]+((x-model['center'])/model['scale'])@m[1:]
    return stable_sigmoid(v) if logistic else v

def fit():
    target=EXPERIMENT/'tabu_models.json'
    if target.exists():raise ValueError('Models already frozen')
    rows=read_rows('development');instances=sorted({r['instance'] for r in rows})
    design=load(EXPERIMENT/'design.json')
    assert instances==sorted(design['split_names']['development'])
    maps=fit_maps(rows);cross={}
    for i in instances:cross[i]=fit_maps([r for r in rows if r['instance']!=i])
    g=np.asarray([r['gain'] for r in rows]);binary=(g>0).astype(float)
    models={};receipts={}
    started=time.perf_counter()
    for arm in ARMS:
        x=matrix(rows,arm,maps,cross)
        gain=fit_ridge(x,g)
        event=ConvergedLogistic(l2=1.).fit(x,binary)
        if not event.converged:raise ArithmeticError(f'Logistic failure {arm}: {event.failure}')
        models[arm]={'gain':gain,'probability':event.serialization()}
        receipts[arm]={'features_including_action_interactions':x.shape[1],
            'gain_gradient_inf':gain['gradient_inf'],'gain_independent_prediction_error':gain['independent_prediction_max_error'],
            'event_gradient_inf':event.gradient_inf,'event_iterations':event.iterations_run,
            'event_finite_difference':event.finite_difference}
        print(json.dumps({'arm':arm,**receipts[arm]}),flush=True)
    bundle={'schema':'fp_decision_study_tabu_frozen_v1','maps':maps,'models':models,
       'training_instances':instances,'training_rows':len(rows),'fit_seconds':time.perf_counter()-started,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'crossfit_exclusions':{i:[j for j in instances if j!=i] for i in instances},'numerical_receipts':receipts}
    write_new(target,bundle)
    # Validation is an engineering-only check. It cannot select any fitted variant.
    v=read_rows('validation');assert not(set(instances)&{r['instance'] for r in v})
    receipt={'status':'PASS_ALL_FITS_AND_FINITE_VALIDATION_PREDICTIONS','validation_rows':len(v),
             'training_instances':instances,'validation_instances':sorted({r['instance'] for r in v}),
             'variants_retained':list(ARMS),'selection_performed':False,'numerics':receipts}
    for arm in ARMS:
        vx=matrix(v,arm,maps)
        for target_name in ('gain','probability'):
            p=predict(models[arm][target_name],vx,target_name=='probability')
            assert np.isfinite(p).all()
    write_new(EXPERIMENT/'tabu_fit_receipt.json',receipt)

if __name__=='__main__':fit()

#!/usr/bin/env python3
"""Post-results algebra audit, no reward data or new search experiments.

For each fixed affine map, C's endpoints and E's coordinates/squares are
linear combinations of the constant, x_i and all x_i*x_j already in B.
The independent sampled numeric check only illustrates this exact identity.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.fp_decision_models import vector

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='projects/analysis/fp_decision_study/fixed_map_span_replay.json');args=parser.parse_args()
    maps=json.loads((ROOT/'projects/experiments/fp_decision_study/tabu_models.json').read_text())['maps']
    rng=np.random.default_rng(2026102604);rows=[]
    for i in range(128):
        x=rng.uniform(.1,1.5,4)
        rows.append({'start_x':x.tolist(),'observed':{'ordinary':np.r_[x[:2],rng.normal(size=2)].tolist(),
             'structure':np.r_[x[2:],rng.normal(size=3)].tolist(),'history':rng.normal(size=4).tolist()}})
    b=np.column_stack([np.ones(len(rows)),np.array([vector(r,'B',maps) for r in rows])]);checks={}
    for arm in ('C','E'):
        z=np.array([vector(r,arm,maps) for r in rows]);coef=np.linalg.lstsq(b,z,rcond=None)[0]
        error=float(np.max(np.abs(b@coef-z)));assert error<1e-10
        checks[arm]={'B_intercept_design_rank':int(np.linalg.matrix_rank(b)),
            'augmented_design_rank':int(np.linalg.matrix_rank(np.column_stack([b,z]))),
            'max_reconstruction_error':error,'added_raw_coordinates':z.shape[1]-23}
    path=Path(args.output)
    if path.exists():raise ValueError('Do not overwrite audit')
    path.write_text(json.dumps({'status':'PASS_FIXED_MAP_POLYNOMIAL_SPAN_AUDIT','checks':checks,
       'scope':'No search or reward data. Fixed-map test-time unregularized span only; OOF training maps vary by instance and induced penalties differ.'},indent=2)+'\n')

if __name__=='__main__':main()

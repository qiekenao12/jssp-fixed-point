#!/usr/bin/env python3
"""Read-only analysis of independently repeated, frozen saved-state actions."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_fp_action_data import canonical, dag_makespan, parse_numeric_blocks, sha

ACTIONS = ('GA', 'LS', 'VND')
TOL = 1e-12
T_CRITICAL_975 = {4: 2.7764451051977987, 6: 2.446911848791681}


def interval(values):
    values = np.asarray(values, float)
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    # Only the frozen n=5 and n=7 designs occur. Standard Student-t quantiles
    # avoid an optional SciPy dependency; this is not a selected CI method.
    radius = float(T_CRITICAL_975[len(values)-1]*se)
    return float(values.mean()), float(values.std(ddof=1)), [float(values.mean()-radius), float(values.mean()+radius)]


def best(gains):
    return [a for a, g in zip(ACTIONS, gains) if g >= max(gains)-TOL]


def choose(gains, preserve_vnd=False):
    winners = best(gains)
    return 'VND' if preserve_vnd and 'VND' in winners else winners[0]


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list,dict)) else v for k,v in r.items()} for r in rows)


def bootstrap(rows, field, seed=20261019, repetitions=10000):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['instance']][row['trajectory_id']].append(row[field])
    values = [np.asarray([np.mean(v) for _,v in sorted(tr.items())]) for _,tr in sorted(groups.items())]
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(values), size=(repetitions, len(values)))
    sampled = np.empty(indices.shape)
    for column in range(len(values)):
        for i,value in enumerate(values):
            mask = indices[:,column] == i
            if mask.any():
                sampled[mask,column] = value[rng.integers(len(value), size=(int(mask.sum()),len(value)))].mean(axis=1)
    instance_means = np.asarray([v.mean() for v in values])
    return {'mean': float(instance_means.mean()),
        'ci95': np.quantile(sampled.mean(axis=1), [.025,.975]).tolist(),
        'instance_win_tie_loss': [int(np.sum(instance_means>TOL)), int(np.sum(abs(instance_means)<=TOL)), int(np.sum(instance_means < -TOL))],
        'instances': len(values), 'bootstrap_repetitions': repetitions,
        'interpretation': 'exploratory development instance->paired trajectory bootstrap; repeats/states never treated as independent instances'}


def policy_choices(predictions, prefix, variants):
    grouped = defaultdict(dict)
    for row in predictions:
        if row['repeat'] == 0:
            grouped[row['state_id']][row['action']] = row['probabilities']
    return {f'{prefix}_{variant}': {state: ACTIONS[int(np.argmax([r[a][variant] for a in ACTIONS]))]
            for state,r in grouped.items()} for variant in variants}


def aggregate(rows, keys, numeric):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, values in sorted(grouped.items()):
        row = dict(zip(keys,key)); row['states'] = len(values)
        row.update({field: float(np.mean([r[field] for r in values])) for field in numeric})
        out.append(row)
    return out


def analyze(protocol_path, directory, output, selector_path=None):
    protocol_path, directory, output = Path(protocol_path), Path(directory), Path(output)
    protocol = json.loads(protocol_path.read_text())
    assert protocol['protocol_hash'] == canonical({k:v for k,v in protocol.items() if k!='protocol_hash'})
    assert protocol['split'] == 'development' and not protocol['locked_test_access']
    matrices = parse_numeric_blocks(ROOT/protocol['data_path'])
    assert sha(ROOT/protocol['data_path']) == protocol['data_sha256']
    inputs = {str(protocol_path.relative_to(ROOT)): sha(protocol_path)}
    m2_path = ROOT/'projects/analysis/fp_v2_development_analysis.json'
    m2 = json.loads(m2_path.read_text()); m2 = m2['configurations'][m2['selected_config_index']]
    choices = policy_choices(m2['predictions'], 'm2_original', ['B','E'])
    decision_path = ROOT/'projects/analysis/fp_agent_decision_feature_comparison.json'
    decision = json.loads(decision_path.read_text())
    choices.update(policy_choices(decision['predictions'], 'decision_original', ['C','E']))
    inputs[str(m2_path.relative_to(ROOT))] = sha(m2_path)
    inputs[str(decision_path.relative_to(ROOT))] = sha(decision_path)
    if selector_path:
        selectors = json.loads(Path(selector_path).read_text())
        assert not set(selectors['selectors']) & set(choices)
        choices.update(selectors['selectors'])
        inputs[str(Path(selector_path).relative_to(ROOT))] = sha(selector_path)
    branch_rows, states = [], []
    for saved in protocol['saved_trajectories']:
        path = directory/f'{saved["instance"]}_{saved["run"]:02d}.json'
        doc = json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT))] = sha(path)
        assert doc['protocol_hash'] == protocol['protocol_hash']
        assert sha(ROOT/saved['source_json']) == saved['source_sha256']
        assert sha(ROOT/saved['population_archive']) == saved['population_archive_sha256']
        old = json.loads((ROOT/saved['source_json']).read_text())
        groups = defaultdict(dict)
        for row in doc['records']:
            assert row['instance'] == saved['instance'] and row['run'] == saved['run']
            assert row['state_id'] in {s['state_id'] for s in saved['states']}
            assert row['end_makespan'] == dag_makespan(row['best_chromosome'], matrices[row['instance']])
            assert abs(row['gain'] - (row['start_makespan']-row['end_makespan'])/row['s0']) < TOL
            assert row['queries'] >= row['decodes'] >= 0
            if row['window_mode'] == 'equal480':
                assert row['queries'] == sum(c['queries'] for c in row['primitive_calls'])
                assert row['queries'] in ([479,480] if row['action']=='VND' else [480])
                assert all(c['queries']>0 for c in row['primitive_calls'])
                if row['action']=='GA':
                    assert len(row['primitive_calls']) == 6
                    assert all(c['queries']==80 for c in row['primitive_calls'])
            key = row['window_mode'], row['action'],row['repeat']
            assert key not in groups[row['state_id']]
            groups[row['state_id']][key] = row
            branch_rows.append({k:row[k] for k in ('instance','trajectory_id','run','generation','state_id',
                'window_mode','action','repeat','seed','gain','improved','queries','decodes','wall_seconds','termination_reason')})
        for state in saved['states']:
            records = groups[state['state_id']]
            expected = {('atomic',a,r) for a in ACTIONS for r in protocol['atomic_new_repeats']} | {('equal480',a,r) for a in ACTIONS for r in protocol['equal_work_repeats']}
            assert set(records) == expected
            old_records = {(r['action'],r['repeat']):r for r in old['records'] if r['state_id'] == state['state_id']}
            old_gains = np.asarray([[old_records[a,r]['gain'] for a in ACTIONS] for r in [0,1]])
            for window in ('atomic','equal480'):
                rows = [r for (w,a,repeat),r in records.items() if w==window]
                new = np.asarray([[records[window,a,r]['gain'] for a in ACTIONS]
                    for r in (protocol['atomic_new_repeats'] if window=='atomic' else protocol['equal_work_repeats'])])
                gains = np.vstack([old_gains,new]) if window=='atomic' else new
                row = {'instance': saved['instance'], 'run': saved['run'], 'trajectory_id': saved['trajectory_id'],
                    **state, 'window_mode': window, 'repetitions': len(gains), 's0': old['s0']}
                for i,a in enumerate(ACTIONS):
                    avg, sd, ci = interval(gains[:,i])
                    row.update({f'{a}_gain': avg, f'{a}_repeat_sd': sd,
                        f'{a}_mean_t_ci95': ci, f'{a}_new5_gain': float(new[:,i].mean()),
                        f'{a}_improvement_frequency': float(np.mean(gains[:,i] > TOL)),
                        f'{a}_queries_mean': float(np.mean([r['queries'] for r in rows if r['action']==a]))})
                avg = gains.mean(axis=0); new_avg = new.mean(axis=0)
                old_avg = old_gains.mean(axis=0)
                old_action = choose(old_avg)
                old_vndtie = choose(old_avg, True)
                row.update({'sample_oracle_action': choose(avg), 'sample_oracle_best_set': '|'.join(best(avg)),
                    'sample_oracle_gain': float(max(avg)), 'fixed_VND_gain': float(avg[2]),
                    'sample_oracle_minus_VND': float(max(avg)-avg[2]),
                    'VND_strictly_nonoptimal': int('VND' not in best(avg)),
                    'original2_oracle_action': old_action,
                    'original2_oracle_minus_VND': float(max(old_avg)-old_avg[2]),
                    'original2_choice_new5_gain': float(new_avg[ACTIONS.index(old_action)]),
                    'original2_choice_new5_minus_VND': float(new_avg[ACTIONS.index(old_action)]-new_avg[2]),
                    'original2_choice_preserveVNDties_new5_minus_VND': float(new_avg[ACTIONS.index(old_vndtie)]-new_avg[2]),
                    'original2_oracle_selection_optimism': float(max(old_avg)-old_avg[2] - (new_avg[ACTIONS.index(old_action)]-new_avg[2])),
                    'original2_choice_new_best': int(old_action in best(new_avg)),
                    'LS_minus_VND': float(avg[1]-avg[2])})
                sets = [best(g) for g in gains]
                row['winner_pair_bestset_overlap'] = float(np.mean([bool(set(sets[i]) & set(sets[j]))
                    for i in range(len(sets)) for j in range(i)]))
                row['winner_pair_same_unique'] = float(np.mean([sets[i]==sets[j] and len(sets[i])==1
                    for i in range(len(sets)) for j in range(i)]))
                for preserve,label in [(False,'LOO_oracle'),(True,'LOO_oracle_preserveVNDties')]:
                    heldout = [gains[i,ACTIONS.index(choose(np.delete(gains,i,axis=0).mean(axis=0),preserve))] for i in range(len(gains))]
                    row[label+'_gain'] = float(np.mean(heldout))
                    row[label+'_minus_VND'] = float(np.mean(heldout)-avg[2])
                for label,mapping in choices.items():
                    action = mapping[state['state_id']]
                    assert action in ACTIONS
                    row[label+'_action'] = action
                    row[label+'_gain'] = float(new_avg[ACTIONS.index(action)])
                    row[label+'_minus_VND'] = float(new_avg[ACTIONS.index(action)]-new_avg[2])
                for prefix in ('m2_original','decision_original','m2_converged','decision_converged'):
                    baseline = 'B' if prefix.startswith('m2') else 'C'
                    if f'{prefix}_E_gain' in row and f'{prefix}_{baseline}_gain' in row:
                        row[prefix+'_FP_minus_NoFP'] = row[f'{prefix}_E_gain'] - row[f'{prefix}_{baseline}_gain']
                states.append(row)
    assert len(branch_rows) == protocol['estimates']['new_atomic_branches']+protocol['estimates']['new_equal_work_branches']
    numeric = [k for k,v in states[0].items() if isinstance(v,(int,float)) and k not in ('run','generation','s0','repetitions')]
    trajectory = aggregate(states, ('window_mode','instance','trajectory_id'), numeric)
    instances = aggregate(trajectory, ('window_mode','instance'), numeric)
    summaries = {}
    for window in ('atomic','equal480'):
        rows = [r for r in states if r['window_mode']==window]
        costs = [r for r in branch_rows if r['window_mode']==window]
        effects = [k for k in numeric if k.endswith('_minus_VND') or k.endswith('FP_minus_NoFP') or k in ('original2_oracle_selection_optimism','VND_strictly_nonoptimal')]
        summaries[window] = {'states': len(rows), 'instances': len({r['instance'] for r in rows}),
            'trajectories': len({r['trajectory_id'] for r in rows}),
            'means': {k:float(np.mean([r[k] for r in rows])) for k in numeric},
            'intervals': {k:bootstrap(rows,k) for k in effects},
            'selector_action_counts': {label:dict(Counter(r[label+'_action'] for r in rows)) for label in choices},
            'costs': {a:{'branches':sum(r['action']==a for r in costs),
                **{c:{'mean':float(np.mean([r[c] for r in costs if r['action']==a])),
                    'min':min(r[c] for r in costs if r['action']==a), 'max':max(r[c] for r in costs if r['action']==a)} for c in ('queries','decodes','wall_seconds')},
                'termination_counts':dict(Counter(r['termination_reason'] for r in costs if r['action']==a))} for a in ACTIONS}}
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix,rows in [('branches',branch_rows),('states',states),('trajectories',trajectory),('instances',instances)]:
        csv_write(output.with_name(output.stem+'_'+suffix+'.csv'), rows)
    result = {'stage':'frozen_development_repeatability_audit', 'protocol_hash':protocol['protocol_hash'],
        'inputs':inputs, 'counts':{'new_branches':len(branch_rows),'sampled_states':len(states)//2},
        'endpoint_independent_DAG_audit':'PASS_ALL_NEW_BRANCHES', 'summaries':summaries,
        'limitations':['Development robustness, not new confirmation data.',
            'State t intervals with 5 or7 repeats are descriptive and neither distribution-free nor simultaneous.',
            'Sample oracle is optimistic and is not an estimate of the true conditional-mean oracle.',
            'Selectors are frozen atomic-window learners; equal480 selection is window-transport audit without refitting.',
            '480 requests equalizes evaluated requests, not fresh decodes, primitive call count or wall time.',
            'Original2 oracle evaluation on equal480 additionally changes execution window; interpret transfer rather than same-window winner bias.']}
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default='projects/experiments/fp_empirical_repeatability_protocol.json')
    parser.add_argument('--directory',default='results/fp_empirical_repeatability_development')
    parser.add_argument('--output',default='projects/analysis/fp_empirical_repeatability.json')
    parser.add_argument('--selectors')
    args=parser.parse_args()
    result=analyze(ROOT/args.protocol,ROOT/args.directory,ROOT/args.output,ROOT/args.selectors if args.selectors else None)
    print(json.dumps({'counts':result['counts'],'independent_audit':result['endpoint_independent_DAG_audit']}))


if __name__=='__main__':
    main()

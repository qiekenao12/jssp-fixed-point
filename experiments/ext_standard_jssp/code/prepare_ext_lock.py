#!/usr/bin/env python3
"""Freeze manifest_lock.json BEFORE any test collection or test-outcome access.

Contents: code SHA256 hashes (new + reused frozen), exact benchmark file hashes,
fixed stratified dev/test split, all random seeds, hyperparameter grids, protocol
parameters, environment versions, and a full SHA256 snapshot of every frozen
asset (this checkout has an empty .git/, so hash snapshots replace git diff).
"""
from __future__ import annotations
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import (ROOT, EXT, BENCH, OUT, CODE, BLOCKS, RUNS, NEUTRAL_CHUNKS,
                        ACTION_STEPS, REPEATS, SPLIT_SEED, COLLECTION_NAMESPACE,
                        BOOTSTRAP_SEED, BOOTSTRAP_DRAWS, GBDT_RANDOM_STATE,
                        GBDT_GRID, GBDT_FIXED, L2_GRID, RFF_SEED, RFF_COMPONENTS,
                        D_RANDOM_SEED, ARMS, SCORERS, TARGETS, FIXED_REFERENCE,
                        canonical, load, parse_jsplib, proposed_split, sha256_file,
                        write_new)

REUSED_FROZEN_CODE = [
    'src/tools/independent_tabu_jssp.py',
    'scripts/run_fp_decision_collection.py',
    'scripts/prepare_fp_decision_study.py',
    'scripts/fp_decision_models.py',
    'scripts/revision_response_analysis.py',
    'scripts/check_fp_scorer_convergence.py',
    'scripts/compare_fp_decision_features.py',
    'scripts/audit_fp_decision_collection.py',
    'scripts/verify_fp_decision_analysis_independent.py',
    'scripts/verify_fp_decision_statistics.py',
    'analyze_paper_evidence.py',
    'src/tools/action_fp.py',
]
SNAPSHOT_DIRS = ['results', 'projects', 'scripts', 'src', 'tests']
SNAPSHOT_ROOT_GLOBS = ['*.py', '*.txt', '*.md', '*.toml', '*.yml', '*.lock',
                       '*.docx', '*.pdf', '*.zip', '*.tar.gz', '.gitignore', '.env.example']


def frozen_snapshot():
    files = {}
    for d in SNAPSHOT_DIRS:
        for p in sorted((ROOT / d).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                files[str(p.relative_to(ROOT))] = sha256_file(p)
    for pattern in SNAPSHOT_ROOT_GLOBS:
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file():
                files[str(p.relative_to(ROOT))] = sha256_file(p)
    return files


def main():
    import numpy, scipy, sklearn, pandas, matplotlib
    prov = load(BENCH / 'provenance.json')
    dev_names, test_names = proposed_split()
    assert len(dev_names) == 20 and len(test_names) == 30
    assert not (set(dev_names) & set(test_names))

    instances = []
    for name in sorted(prov['instances']):
        meta = prov['instances'][name]
        inst = parse_jsplib(ROOT / meta['snapshot_file'])
        assert inst.fingerprint == meta['fingerprint']
        block = next(b for b, names in BLOCKS.items() if name in names)
        instances.append({
            'name': name, 'split': 'development' if name in dev_names else 'test',
            'jobs': inst.jobs, 'machines': inst.machines,
            'size': f'{inst.jobs}x{inst.machines}', 'block': block,
            'routes': [list(r) for r in inst.routes],
            'durations': [list(p) for p in inst.durations],
            'fingerprint': inst.fingerprint,
            'source': meta['source'],
            'in_historical_genetic_48': meta['in_historical_genetic_48'],
            'genetic48_name': meta['genetic48_name'],
            'in_frozen_benchmark_46': meta['in_frozen_benchmark_46'],
            'new_to_critical_block_stack': True,
        })
    assert len({i['fingerprint'] for i in instances}) == 50
    dev_fp = {i['fingerprint'] for i in instances if i['split'] == 'development'}
    test_fp = {i['fingerprint'] for i in instances if i['split'] == 'test'}
    assert not (dev_fp & test_fp), 'dev/test duplicate content'

    write_new(EXT / 'instances.json', instances)

    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p)
                   for p in sorted(CODE.glob('*.py'))}
    reused_hashes = {rel: sha256_file(ROOT / rel) for rel in REUSED_FROZEN_CODE}
    benchmark_hashes = {rel: sha256_file(ROOT / rel) for rel in [
        'jobshop1.txt', 'projects/experiments/benchmark_jobshop.txt',
        'projects/experiments/fp_v2_jobshop.txt',
        'experiments/ext_standard_jssp/benchmarks/provenance.json']}
    for p in sorted((BENCH / 'instances').glob('*.txt')):
        benchmark_hashes[str(p.relative_to(ROOT))] = sha256_file(p)
    for p in sorted(BENCH.glob('*_mirror.txt')):
        benchmark_hashes[str(p.relative_to(ROOT))] = sha256_file(p)

    print('Hashing frozen-asset snapshot (results/ projects/ scripts/ src/ tests/ + root files)...',
          flush=True)
    snapshot = frozen_snapshot()
    write_new(OUT / 'frozen_snapshot_before.json',
              {'created_utc': datetime.now(timezone.utc).isoformat(),
               'file_count': len(snapshot), 'hashes': snapshot,
               'exclusions': ['__pycache__', 'venv*', '.git (empty directory; not a git repository)'],
               'note': 'SHA256 before/after snapshot replaces git status/diff (empty .git).'})

    design = {
        'status': 'LOCKED_BEFORE_ANY_TEST_COLLECTION_OR_TEST_OUTCOME_ACCESS',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'cohort': 'ext_standard_jssp',
        'title': 'Prospective standard-benchmark extension of the independent critical-block action-selection study',
        'solver': 'src/tools/independent_tabu_jssp.py (frozen, unmodified)',
        'protocol': {
            'neutral_action': 'CB_TABU', 'neutral_chunks': list(NEUTRAL_CHUNKS),
            'saved_opportunities': 2, 'actions': list(('CB_BEST_DESCENT', 'CB_FIRST_DESCENT', 'CB_TABU')),
            'action_steps': ACTION_STEPS, 'tabu_tenure': 5, 'repeats': REPEATS,
            'runs': RUNS,
            'state': ['best/s0', 'current/s0', 'critical_path_operation_fraction',
                      'operations_in_critical_blocks_fraction'],
            'targets': {'raw': 'R_a=(C_best(S)-C_best(S_a))/s0',
                        'cng': 'R_tilde_a_scaled=R_a/decode_evaluations_a*1000'},
            'fixed_reference': FIXED_REFERENCE,
            'maps': 'displacement ridge 0.1 raw normal equations, unpenalized intercept, standardized starts, W=I; LOIO cross-fitted maps for dev scorer training features; full-dev maps for test',
            'representations': 'frozen A,B,C,D,E,D_random (fp_decision_models.vector); D_random seed frozen',
            'scorers': list(SCORERS),
            'statistics': 'instance-cluster means (state selected value = mean of 5 repeats; instance = mean of its states; macro = equal instance weight); exact two-sided Wilcoxon signed-rank; Holm within declared families; 20000-draw instance-cluster percentile bootstrap',
            'split_repeat': 'repeats {0,1} -> {2,3,4} and reverse, under BOTH raw and cng targets, vs best fixed static action per regime',
        },
        'seeds': {'split': SPLIT_SEED, 'collection_namespace': COLLECTION_NAMESPACE,
                  'bootstrap': BOOTSTRAP_SEED, 'bootstrap_draws': BOOTSTRAP_DRAWS,
                  'gbdt_random_state': GBDT_RANDOM_STATE, 'rff': RFF_SEED,
                  'rff_components': RFF_COMPONENTS, 'd_random': D_RANDOM_SEED},
        'grids': {'ridge_l2': L2_GRID, 'gbdt': GBDT_GRID, 'gbdt_fixed': GBDT_FIXED,
                  'rff_l2': L2_GRID,
                  'gbdt_selection': 'per arm x target, min mean leave-one-development-instance-out MSE; ties -> first grid entry; fold-train features use LOIO maps, fold-test features use full-dev maps; trees fit on raw action-interacted design (scale invariant); OMP_NUM_THREADS=1 per fit'},
        'split_names': {'development': dev_names, 'test': test_names},
        'instances_hash': canonical(instances),
        'overlap_disclosure': {
            'historical_genetic_48': prov['genetic48_overlap_names'],
            'note': 'Overlap is with the historical GENETIC study only; all 50 instances are new to the critical-block stack, whose frozen study used 32 synthetic instances. Per-instance fingerprints recorded.',
        },
        'environment': {'python': platform.python_version(), 'numpy': numpy.__version__,
                        'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__,
                        'pandas': pandas.__version__, 'matplotlib': matplotlib.__version__,
                        'venv': 'experiments/ext_standard_jssp/venv_ext',
                        'gbdt_impl': 'sklearn.ensemble.HistGradientBoostingRegressor',
                        'determinism': 'PYTHONDONTWRITEBYTECODE=1, OMP_NUM_THREADS=1, fixed random_state, early_stopping=False'},
        'code_hashes_new': code_hashes,
        'code_hashes_reused_frozen': reused_hashes,
        'benchmark_hashes': benchmark_hashes,
        'frozen_snapshot_before_file': 'experiments/ext_standard_jssp/outputs/frozen_snapshot_before.json',
        'frozen_snapshot_file_count': len(snapshot),
        'failure_rules': 'No outcome-based dropping/rerunning/replacing of instances or seeds; complete grids only; abort on any verification gate failure; frozen assets must hash identical before/after.',
    }
    design['lock_hash'] = canonical({k: v for k, v in design.items() if k != 'lock_hash'})
    write_new(OUT / 'manifest_lock.json', design)
    print(json.dumps({'status': 'LOCKED', 'lock_hash': design['lock_hash'],
                      'dev': len(dev_names), 'test': len(test_names),
                      'frozen_files_hashed': len(snapshot),
                      'code_files_hashed': len(code_hashes) + len(reused_hashes),
                      'benchmark_files_hashed': len(benchmark_hashes)}, indent=2))


if __name__ == '__main__':
    main()

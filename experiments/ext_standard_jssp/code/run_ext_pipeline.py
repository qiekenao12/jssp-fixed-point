#!/usr/bin/env python3
"""Orchestrator for ext_standard_jssp with mandatory verification gates.

Order: benchmarks -> lock -> dev collection -> dev schedule audit -> dev-only
fitting -> test collection -> test schedule audit -> frozen evaluation ->
independent scalar replay -> figures -> frozen-asset after-snapshot comparison.
Any gate failure aborts the pipeline. Writes outputs/verification_report.json.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import ROOT, EXT, OUT, RAW, MODELS, load, write_new

PY = EXT / 'venv_ext/bin/python'
ENV = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def run(script, *args):
    cmd = [str(PY), str(EXT / 'code' / script), *map(str, args)]
    print('>>>', ' '.join(cmd), flush=True)
    t = time.perf_counter()
    subprocess.run(cmd, check=True, env=ENV, cwd=str(ROOT))
    print(f'<<< {script} done in {time.perf_counter() - t:.1f}s', flush=True)


def main():
    gates = []
    started = time.perf_counter()

    def gate(name, ok, detail):
        gates.append({'gate': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})
        if not ok:
            finish(gates, started, failed=True)
            raise SystemExit(f'GATE FAILED: {name}')

    if not (EXT / 'benchmarks/provenance.json').exists():
        run('prepare_benchmarks.py')
    prov = load(EXT / 'benchmarks/provenance.json')
    gate('benchmark_acquisition_and_mirror_fidelity',
         prov['cohort_size'] == 50 and prov['unique_fingerprints'] == 50
         and len(prov['mirror_fidelity_checks_passed']) == 5,
         {'instances': 50, 'mirror_checks': prov['mirror_fidelity_checks_passed'],
          'genetic48_overlap': prov['genetic48_overlap_names']})

    if not (OUT / 'manifest_lock.json').exists():
        run('prepare_ext_lock.py')
    lock = load(OUT / 'manifest_lock.json')
    gate('manifest_lock_before_test', lock['status'].startswith('LOCKED_BEFORE'),
         {'lock_hash': lock['lock_hash'], 'dev': len(lock['split_names']['development']),
          'test': len(lock['split_names']['test'])})

    if not (RAW / 'development_collection_receipt.json').exists():
        run('run_ext_collection.py', '--split', 'development', '--workers', 16)
    rec = load(RAW / 'development_collection_receipt.json')
    gate('dev_collection_complete', rec['status'] == 'COMPLETE' and rec['branches'] == 6000,
         {'trajectories': rec['trajectories'], 'branches': rec['branches'],
          'wall_seconds': rec['wall_seconds']})

    if not (OUT / 'verification_schedules_development.json').exists():
        run('verify_ext_schedules.py', '--split', 'development', '--workers', 16)
    sd = load(OUT / 'verification_schedules_development.json')
    gate('dev_schedule_validity_time_relaxation', sd['status'].startswith('PASS'),
         {k: sd[k] for k in ('states', 'branches', 'distinct_schedule_decodes_checked',
                             'unique_branch_seeds')})

    if not (MODELS / 'ext_models.json').exists():
        run('fit_ext_scorers.py')
    bundle = load(MODELS / 'ext_models.json')
    num_ok = True
    num_detail = {}
    for arm, rec_arm in bundle['numerical_receipts'].items():
        for t in ('raw', 'cng'):
            r = rec_arm[t]
            ok = (r['frozen_gradient_inf'] <= 1e-7
                  and r['frozen_independent_prediction_error'] <= 1e-7
                  and r['loocv_gradient_inf'] <= 1e-7)
            num_ok &= ok
            num_detail[f'{arm}_{t}'] = r
        num_ok &= rec_arm['event_gradient_inf'] <= 1e-7
    for t in ('raw', 'cng'):
        for arm in bundle['models']:
            num_ok &= bundle['models'][arm][t]['rff_ridge']['reproducibility'] == 'EXACT_DOUBLE_FIT_MATCH'
            num_ok &= bundle['models'][arm][t]['gbdt']['reproducibility'] == 'EXACT_DOUBLE_FIT_MATCH'
    gate('numerical_solver_gates_and_reproducibility', bool(num_ok),
         {'gradient_tolerance': 1e-7, 'receipts': num_detail,
          'rff_gbdt_double_fit': 'EXACT_DOUBLE_FIT_MATCH (all arms/targets)',
          'dev_only_gate': bundle['dev_only_gate']})

    if not (RAW / 'test_collection_receipt.json').exists():
        run('run_ext_collection.py', '--split', 'test', '--workers', 16)
    rt = load(RAW / 'test_collection_receipt.json')
    gate('test_collection_complete', rt['status'] == 'COMPLETE' and rt['branches'] == 18000,
         {'trajectories': rt['trajectories'], 'branches': rt['branches'],
          'wall_seconds': rt['wall_seconds']})

    if not (OUT / 'verification_schedules_test.json').exists():
        run('verify_ext_schedules.py', '--split', 'test', '--workers', 16)
    st = load(OUT / 'verification_schedules_test.json')
    gate('test_schedule_validity_time_relaxation', st['status'].startswith('PASS'),
         {k: st[k] for k in ('states', 'branches', 'distinct_schedule_decodes_checked',
                             'unique_branch_seeds')})

    if not (OUT / 'ext_analysis.json').exists():
        run('analyze_ext.py')
    gate('frozen_test_evaluation', (OUT / 'ext_analysis.json').exists(),
         {'tables': [p.name for p in sorted(OUT.glob('table_ext_*.csv'))]})

    if not (OUT / 'verification_replay.json').exists():
        run('verify_ext_replay.py')
    rp = load(OUT / 'verification_replay.json')
    gate('independent_scalar_replay', rp['status'].startswith('PASS'),
         {'paired_contrasts_reconstructed': rp['paired_contrasts_reconstructed'],
          'max_abs_errors': rp['max_abs_errors']})

    run('plot_ext_summary.py')
    gate('figures_300dpi', (OUT / 'fig_ext_summary.png').exists() and (OUT / 'fig_ext_summary.pdf').exists(),
         {'files': ['fig_ext_summary.png', 'fig_ext_summary.pdf']})

    # frozen-asset after-snapshot (git unavailable: empty .git directory)
    from prepare_ext_lock import frozen_snapshot
    before = load(OUT / 'frozen_snapshot_before.json')['hashes']
    print('Hashing frozen-asset after-snapshot...', flush=True)
    after = frozen_snapshot()
    write_new(OUT / 'frozen_snapshot_after.json',
              {'created_utc': datetime.now(timezone.utc).isoformat(),
               'file_count': len(after), 'hashes': after})
    changed = [k for k in before if before[k] != after.get(k)]
    removed = [k for k in before if k not in after]
    added = [k for k in after if k not in before]
    gate('zero_modification_of_frozen_assets', not changed and not removed and not added,
         {'files_hashed_before': len(before), 'files_hashed_after': len(after),
          'changed': changed, 'removed': removed, 'added': added,
          'method': 'SHA256 before/after snapshot (empty .git; git diff unavailable)'})

    finish(gates, started, failed=False)


def finish(gates, started, failed):
    report = {'status': 'FAIL' if failed else 'PASS_ALL_GATES',
              'cohort': 'ext_standard_jssp',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'wall_seconds': time.perf_counter() - started,
              'gates': gates}
    path = OUT / 'verification_report.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'],
                      'gates': [(g['gate'], g['status']) for g in gates]}, indent=2))


if __name__ == '__main__':
    main()

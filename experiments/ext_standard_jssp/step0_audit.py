#!/usr/bin/env python3
"""Step 0 audit for ext_standard_jssp: benchmark inventory, overlap hashes, timing.

Read-only with respect to every frozen asset. Writes only inside
experiments/ext_standard_jssp/step0/. No model fitting, no test outcomes.
"""
from __future__ import annotations
import hashlib, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.independent_tabu_jssp import Instance, initial_state, run_action, features, ACTIONS

OUT = ROOT / 'experiments/ext_standard_jssp/step0'
OUT.mkdir(parents=True, exist_ok=True)


def parse_orlib_blocks(path):
    """Generic OR-Library / JSPLIB-snapshot block parser (independent, auditable).

    Handles both `instance NAME` + description + `j m` + j rows (jobshop1.txt)
    and `instance NAME` + 'JSPLIB frozen snapshot' + `j m` + j rows.
    """
    text = Path(path).read_text()
    blocks = re.split(r'^\s*instance\s+(\w+)\s*$', text, flags=re.M)
    out = {}
    for i in range(1, len(blocks), 2):
        name, body = blocks[i], blocks[i + 1]
        rows = []
        for line in body.splitlines():
            s = line.strip()
            if not s:
                continue
            toks = s.split()
            try:
                vals = [int(t) for t in toks]
            except ValueError:
                continue  # description / separator lines are never all-integer
            rows.append(vals)
        # first numeric row is dimensions; keep exactly j following rows
        j, m = rows[0][0], rows[0][1]
        data = rows[1:1 + j]
        assert len(data) == j and all(len(r) == 2 * m for r in data), (name, path)
        routes = tuple(tuple(r[::2]) for r in data)
        durations = tuple(tuple(r[1::2]) for r in data)
        try:
            out[name] = Instance(routes, durations, name)
        except ValueError as exc:
            SKIPPED.append({'name': name, 'source': str(path), 'reason': str(exc),
                            'jobs': j, 'machines': m})
    return out


SKIPPED = []


def inventory():
    orlib = parse_orlib_blocks(ROOT / 'jobshop1.txt')
    frozen46 = parse_orlib_blocks(ROOT / 'projects/experiments/benchmark_jobshop.txt')
    genetic48 = parse_orlib_blocks(ROOT / 'projects/experiments/fp_v2_jobshop.txt')
    inv = {}
    for label, d in [('orlib_jobshop1', orlib), ('frozen_benchmark_46', frozen46),
                     ('historical_genetic_48', genetic48)]:
        inv[label] = {n: {'jobs': i.jobs, 'machines': i.machines, 'fingerprint': i.fingerprint}
                      for n, i in sorted(d.items())}
    # candidate cohort: Lawrence la21-la40 (all local) + locally present Taillard
    cand = {n: i for n, i in orlib.items() if re.fullmatch(r'la(2[1-9]|3[0-9]|40)', n)}
    ta_local = {n: i for n, i in frozen46.items() if n.startswith('ta')}
    # also yn (20x20) and swv medium blocks as potential supplements
    extra = {n: i for n, i in orlib.items() if n.startswith(('yn', 'swv', 'abz', 'ft20'))}
    fp_gen = {i.fingerprint for i in genetic48.values()}
    fp_f46 = {i.fingerprint for i in frozen46.values()}
    fp_orl = {i.fingerprint for i in orlib.values()}
    overlap = {}
    for label, d in [('lawrence_la21_la40', cand), ('taillard_local', ta_local),
                     ('other_local_medium', extra)]:
        overlap[label] = {}
        for n, i in sorted(d.items()):
            overlap[label][n] = {
                'size': f'{i.jobs}x{i.machines}', 'fingerprint': i.fingerprint,
                'in_historical_genetic_48': i.fingerprint in fp_gen,
                'in_frozen_benchmark_46': i.fingerprint in fp_f46,
                'in_orlib_jobshop1': i.fingerprint in fp_orl,
                'genetic48_names_same_fingerprint': sorted(
                    k for k, v in genetic48.items() if v.fingerprint == i.fingerprint),
            }
    # cross-check: parser agreement between orlib and frozen snapshots for shared names
    mismatch = []
    for n, i in frozen46.items():
        if n in orlib and orlib[n].fingerprint != i.fingerprint:
            mismatch.append(n)
    report = {'inventory_counts': {k: len(v) for k, v in inv.items()},
              'parser_cross_check_frozen46_vs_orlib_mismatches': mismatch,
              'skipped_nonpositive_duration_instances': SKIPPED,
              'candidate_overlap': overlap, 'inventory': inv}
    (OUT / 'benchmark_inventory.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'inventory'}, indent=2)[:4000])
    return orlib, frozen46, genetic48


def micro(orlib, name):
    inst = orlib[name]
    seed = int.from_bytes(hashlib.sha256(f'ext_micro:{name}'.encode()).digest()[:8], 'big')
    t0 = time.perf_counter()
    state = initial_state(inst, seed)
    t_init = time.perf_counter() - t0
    timings = {'instance': name, 'size': f'{inst.jobs}x{inst.machines}',
               'initial_makespan': state.initial_makespan, 'init_seconds': t_init,
               'points': []}
    branch_seconds = []
    for point, steps in enumerate((8, 16)):
        t = time.perf_counter()
        neutral = run_action(inst, state, 'CB_TABU', seed + 1000 + point, steps=steps)
        state = neutral.state
        neutral_s = time.perf_counter() - t
        t = time.perf_counter(); features(inst, state); feat_s = time.perf_counter() - t
        per_action = {}
        for ai, action in enumerate(ACTIONS):
            secs = []
            for rep in range(5):
                t = time.perf_counter()
                out = run_action(inst, state, action, seed + 2000 + point * 100 + ai * 10 + rep, steps=8)
                secs.append(time.perf_counter() - t)
                t = time.perf_counter(); features(inst, out.state); feat_s += time.perf_counter() - t
                branch_seconds.append(secs[-1])
            per_action[action] = {'mean_branch_seconds': sum(secs) / 5, 'max': max(secs),
                                  'mean_decodes': out.decode_evaluations}
        timings['points'].append({'point': point, 'neutral_seconds': neutral_s,
                                  'feature_seconds_total': feat_s, 'actions': per_action})
    timings['total_seconds'] = time.perf_counter() - t0
    timings['mean_branch_seconds'] = sum(branch_seconds) / len(branch_seconds)
    return timings


def main():
    orlib, frozen46, genetic48 = inventory()
    micros = [micro(orlib, n) for n in ('yn1', 'la31', 'la36')]
    (OUT / 'microbenchmark.json').write_text(json.dumps(micros, indent=2))
    for m in micros:
        print(json.dumps({'instance': m['instance'], 'size': m['size'],
                          'total_trajectory_seconds': round(m['total_seconds'], 3),
                          'mean_branch_seconds': round(m['mean_branch_seconds'], 4)}))


if __name__ == '__main__':
    main()

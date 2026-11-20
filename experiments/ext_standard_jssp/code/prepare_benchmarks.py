#!/usr/bin/env python3
"""Acquire and verify the 50-instance standard benchmark cohort (read-only elsewhere).

- Lawrence la21-la40: parsed from the local OR-Library file jobshop1.txt.
- Taillard ta01-ta30: 5 members exist in the frozen 46-instance archive; the other
  25 are downloaded from the public JSPLIB mirror. Every locally known Taillard
  fingerprint must match the mirror copy exactly (mirror fidelity gate).
- Writes per-instance JSPLIB snapshots + provenance.json under benchmarks/.
"""
from __future__ import annotations
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import (ROOT, BENCH, BLOCKS, LOCAL_TA, MIRROR_URL, canonical,
                        parse_jsplib, parse_orlib_blocks, sha256_file, write_new)


def main():
    (BENCH / 'instances').mkdir(parents=True, exist_ok=True)
    orlib = parse_orlib_blocks(ROOT / 'jobshop1.txt')
    frozen46 = parse_orlib_blocks(ROOT / 'projects/experiments/benchmark_jobshop.txt')
    genetic48 = parse_orlib_blocks(ROOT / 'projects/experiments/fp_v2_jobshop.txt')
    fp_gen = {i.fingerprint: n for n, i in genetic48.items()}
    fp_f46 = {i.fingerprint: n for n, i in frozen46.items()}

    provenance = {'mirror_url_template': MIRROR_URL, 'instances': {}}
    all_names = [n for names in BLOCKS.values() for n in names]
    assert len(all_names) == 50 and len(set(all_names)) == 50

    for name in all_names:
        if name.startswith('la'):
            inst = orlib[name]
            source = {'kind': 'local_orlib', 'file': 'jobshop1.txt',
                      'file_sha256': sha256_file(ROOT / 'jobshop1.txt')}
            snap = BENCH / 'instances' / f'{name}.txt'
            if not snap.exists():
                lines = [f'{inst.jobs} {inst.machines}']
                for r, p in zip(inst.routes, inst.durations):
                    lines.append(' '.join(f'{m} {t}' for m, t in zip(r, p)))
                snap.write_text('\n'.join(lines) + '\n')
            assert parse_jsplib(snap).fingerprint == inst.fingerprint
        else:
            mirror_path = BENCH / f'{name}_mirror.txt'
            if not mirror_path.exists():
                url = MIRROR_URL.format(name=name)
                import subprocess
                tmp = mirror_path.with_suffix('.part')
                subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '90',
                                '-o', str(tmp), url], check=True)
                tmp.replace(mirror_path)
            inst = parse_jsplib(mirror_path)
            source = {'kind': 'jsplib_mirror', 'url': MIRROR_URL.format(name=name),
                      'downloaded_file': str(mirror_path.relative_to(ROOT)),
                      'downloaded_sha256': sha256_file(mirror_path)}
            if name in LOCAL_TA:
                local = frozen46[name]
                assert inst.fingerprint == local.fingerprint, f'MIRROR FIDELITY FAILURE {name}'
                source['local_frozen_copy'] = 'projects/experiments/benchmark_jobshop.txt'
                source['local_frozen_fingerprint_match'] = True
            snap = BENCH / 'instances' / f'{name}.txt'
            if not snap.exists():
                snap.write_bytes(mirror_path.read_bytes())
            assert parse_jsplib(snap).fingerprint == inst.fingerprint

        assert inst.jobs > 0 and all(t > 0 for row in inst.durations for t in row)
        provenance['instances'][name] = {
            'jobs': inst.jobs, 'machines': inst.machines,
            'size': f'{inst.jobs}x{inst.machines}',
            'fingerprint': inst.fingerprint,
            'snapshot_file': str(snap.relative_to(ROOT)),
            'snapshot_sha256': sha256_file(snap),
            'source': source,
            'in_historical_genetic_48': inst.fingerprint in fp_gen,
            'genetic48_name': fp_gen.get(inst.fingerprint),
            'in_frozen_benchmark_46': inst.fingerprint in fp_f46,
            'frozen46_name': fp_f46.get(inst.fingerprint),
            'new_to_critical_block_stack': True,
        }

    fps = [v['fingerprint'] for v in provenance['instances'].values()]
    assert len(set(fps)) == 50, 'duplicate instance content in cohort'
    provenance['cohort_size'] = 50
    provenance['unique_fingerprints'] = 50
    provenance['mirror_fidelity_checks_passed'] = sorted(LOCAL_TA)
    provenance['genetic48_overlap_names'] = sorted(
        n for n, v in provenance['instances'].items() if v['in_historical_genetic_48'])
    write_new(BENCH / 'provenance.json', provenance)
    print(json.dumps({'status': 'BENCHMARKS_READY', 'instances': 50,
                      'mirror_fidelity_passed': sorted(LOCAL_TA),
                      'genetic48_overlap': provenance['genetic48_overlap_names']}, indent=2))


if __name__ == '__main__':
    main()

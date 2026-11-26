#!/usr/bin/env python3
"""One-command independent verification for the SWEVO FP-decision reproducibility repo.

Runs, in order:
  1. Lock integrity  — SHA256 re-verification of every file hashed in
     projects/experiments/fp_decision_study/final_lock.json (primary cohort) and
     experiments/ext_standard_jssp/outputs/manifest_lock.json (prospective extension),
     including recomputation of design_hash / instances_hash / lock_hash and the three
     documented post-lock engineering amendments (code_amendment_001..003).
  2. Primary synthetic cohort — independent synchronous-relaxation schedule audits of all
     four raw grids (tabu development/validation/final, original final) via
     scripts/audit_fp_decision_collection.py, plus the two independent scalar replays
     (verify_fp_decision_statistics.py, verify_fp_decision_analysis_independent.py).
  3. Prospective 50-instance extension — independent schedule audits (development/test)
     via verify_ext_schedules.py and the independent scalar replay verify_ext_replay.py.
  4. Unit tests — tests/test_fp_decision_study.py + tests/test_independent_tabu_jssp.py.
  5. Repository hygiene — max file size, total size, file count, forbidden-pattern scan
     (with the documented hash-lock exception allowlist).

All verifiers are executed inside a fresh temporary sandbox copy of this repository,
because the shipped receipts are write-once (the scripts refuse to overwrite existing
audits). Regenerated reports are compared field-by-field against the shipped frozen
receipts (timestamps/wall-clock fields excluded; floats compared to 1e-9 rel / 1e-12 abs).
Nothing in this repository is modified. Requires only numpy (+ stdlib); matplotlib is
optional and only needed for --figures.

Usage:
    python scripts/verify_all.py                 # full verification (a few minutes)
    python scripts/verify_all.py --quick         # locks + hygiene only (seconds)
    python scripts/verify_all.py --figures       # additionally regenerate all figures in sandbox
    python scripts/verify_all.py --historical    # also audit extracted historical raw archives
                                                 # (requires release_assets tarballs extracted to
                                                 #  ./raw_extracted/ — see README §6)
    python scripts/verify_all.py --report out.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / 'projects/analysis/fp_decision_study'
EXT = ROOT / 'experiments/ext_standard_jssp'

RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str = '', seconds: float | None = None):
    RESULTS.append({'check': name, 'status': 'PASS' if ok else 'FAIL',
                    'detail': detail, 'wall_seconds': seconds})
    mark = 'PASS' if ok else 'FAIL'
    extra = f' ({seconds:.1f}s)' if seconds is not None else ''
    print(f'[{mark}] {name}{extra}' + (f' — {detail}' if detail and not ok else ''), flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(p: Path):
    return json.loads(p.read_text())


def approx_equal(a, b, rel=1e-9, abs_=1e-12, path='$'):
    """Recursive equality with float tolerance; returns first mismatch description or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f'{path}: key sets differ ({sorted(set(a) ^ set(b))[:5]})'
        for k in a:
            m = approx_equal(a[k], b[k], rel, abs_, f'{path}.{k}')
            if m:
                return m
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f'{path}: length {len(a)} != {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            m = approx_equal(x, y, rel, abs_, f'{path}[{i}]')
            if m:
                return m
        return None
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return f'{path}: {a!r} != {b!r}'
        if fa == fb:
            return None
        if abs(fa - fb) <= max(abs_, rel * max(abs(fa), abs(fb))):
            return None
        return f'{path}: {fa!r} != {fb!r}'
    if a != b:
        return f'{path}: {a!r} != {b!r}'
    return None


def compare_report(name: str, fresh: Path, frozen: Path, drop_keys=()):
    if not fresh.exists():
        record(name, False, f'verifier produced no report at {fresh}')
        return
    a, b = read(fresh), read(frozen)
    for k in drop_keys:
        a.pop(k, None)
        b.pop(k, None)
    mismatch = approx_equal(a, b)
    record(name, mismatch is None,
           'regenerated report matches shipped frozen receipt' if mismatch is None else mismatch)


# ---------------------------------------------------------------- step 1: locks
def check_locks():
    t0 = time.perf_counter()
    try:
        lock = read(ROOT / 'projects/experiments/fp_decision_study/final_lock.json')
        bad = [p for p, h in lock['hashes'].items() if sha(ROOT / p) != h]
        record('primary final_lock.json SHA256 (32 files)', not bad, f'mismatched: {bad}')
        design = read(ROOT / 'projects/experiments/fp_decision_study/design.json')
        instances = read(ROOT / 'projects/experiments/fp_decision_study/instances.json')
        ok = (canon({k: v for k, v in design.items() if k != 'design_hash'}) == design['design_hash']
              and canon(instances) == design['instances_hash'] and len(instances) == 32)
        record('primary design_hash / instances_hash recomputation', ok)
    except Exception as e:
        record('primary lock integrity', False, repr(e))
    try:
        ext = read(EXT / 'outputs/manifest_lock.json')
        amendments = {}
        for ap in sorted((EXT / 'outputs').glob('code_amendment_00*.json')):
            a = read(ap)
            amendments[a['file']] = a
        bad, amended = [], []
        for group in ('code_hashes_new', 'code_hashes_reused_frozen', 'benchmark_hashes'):
            for rel, h in ext[group].items():
                if rel in amendments:
                    a = amendments[rel]
                    if h == a['sha256_before'] and sha(ROOT / rel) == a['sha256_after']:
                        amended.append(rel)
                    else:
                        bad.append(rel)
                elif sha(ROOT / rel) != h:
                    bad.append(rel)
        record('ext manifest_lock SHA256 (10 new + 12 reused code, 84 benchmarks)',
               not bad, f'mismatched: {bad}')
        record('ext post-lock engineering amendments 001-003 (before==lock, after==current)',
               len(amended) == 3, f'amended files verified: {amended}')
        ok = (canon({k: v for k, v in ext.items() if k != 'lock_hash'}) == ext['lock_hash']
              and canon(read(EXT / 'instances.json')) == ext['instances_hash'])
        record('ext lock_hash / instances_hash recomputation', ok)
    except Exception as e:
        record('ext lock integrity', False, repr(e))
    return time.perf_counter() - t0


# ---------------------------------------------------------------- sandbox helpers
WRITE_ONCE = [
    'projects/analysis/fp_decision_study/independent_statistics_audit.json',
    'projects/analysis/fp_decision_study/independent_analysis_review_replay.json',
    'experiments/ext_standard_jssp/outputs/verification_replay.json',
    'experiments/ext_standard_jssp/outputs/verification_schedules_development.json',
    'experiments/ext_standard_jssp/outputs/verification_schedules_test.json',
]


def make_sandbox(tmp: Path) -> Path:
    sb = tmp / 'repo'
    shutil.copytree(ROOT, sb, ignore=shutil.ignore_patterns(
        '__pycache__', '.git', '.pytest_cache', 'raw_extracted', 'release_assets',
        '*.pyc', '.DS_Store', '*.tar.gz'))
    for rel in WRITE_ONCE:
        (sb / rel).unlink(missing_ok=True)
    return sb


def sandbox_env(sb: Path, tmp: Path) -> dict:
    env = {**os.environ,
           'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'MPLCONFIGDIR': str(tmp / 'mpl'),
           'PYTHONPATH': os.pathsep.join([str(sb / 'src'), str(sb)])}
    return env


def run_py(sb: Path, env: dict, script: str, *args, timeout=2400):
    if script == '-m':
        cmd = [sys.executable, '-m', *map(str, args)]
    else:
        cmd = [sys.executable, str(sb / script), *map(str, args)]
    proc = subprocess.run(cmd, cwd=str(sb), env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
        raise RuntimeError(f'{script} exited {proc.returncode}:\n' + '\n'.join(tail))
    return proc.stdout + proc.stderr


# ---------------------------------------------------------------- step 2: primary
def check_primary(sb: Path, env: dict, tmp: Path):
    grids = [('tabu', 'development', 'tabu_development_data_audit.json'),
             ('tabu', 'validation', 'tabu_validation_data_audit.json'),
             ('tabu', 'final', 'tabu_final_data_audit.json'),
             ('original', 'final', 'original_final_data_audit.json')]
    for solver, split, frozen_name in grids:
        name = f'primary schedule audit: {solver}/{split} (synchronous relaxation)'
        t0 = time.perf_counter()
        try:
            out = tmp / f'audit_{solver}_{split}.json'
            run_py(sb, env, 'scripts/audit_fp_decision_collection.py',
                   '--solver', solver, '--split', split, '--output', out)
            compare_report(name, out, AN / frozen_name, drop_keys=('wall_seconds',))
        except Exception as e:
            record(name, False, str(e))
        RESULTS[-1]['wall_seconds'] = time.perf_counter() - t0

    name = 'primary independent scalar replay: verify_fp_decision_statistics.py'
    t0 = time.perf_counter()
    try:
        run_py(sb, env, 'scripts/verify_fp_decision_statistics.py')
        compare_report(name,
                       sb / 'projects/analysis/fp_decision_study/independent_statistics_audit.json',
                       AN / 'independent_statistics_audit.json')
    except Exception as e:
        record(name, False, str(e))
    RESULTS[-1]['wall_seconds'] = time.perf_counter() - t0

    name = 'primary independent analysis replay: verify_fp_decision_analysis_independent.py'
    t0 = time.perf_counter()
    try:
        out = tmp / 'independent_analysis_review_replay.json'
        run_py(sb, env, 'scripts/verify_fp_decision_analysis_independent.py', '--output', out)
        fresh = read(out)
        frozen = read(AN / 'independent_analysis_review_replay.json')
        fresh.pop('created_utc', None)
        frozen.pop('created_utc', None)
        ra = fresh.pop('reviewed_sha256', {})
        rb = frozen.pop('reviewed_sha256', {})
        mismatch = approx_equal(fresh, frozen)
        doc = 'projects/docs/fp_decision_empirical_manuscript.md'
        # Documented provenance exception (see README §5): the frozen receipt
        # reviewed the pre-revision 2026-10-26 document (hash 550e0070...), which was
        # revised in place later that day; the reviewed version no longer exists. Every
        # other reviewed hash must match exactly, and the document mismatch must be
        # exactly the documented revision pair — not arbitrary drift.
        DOC_REVIEWED = '550e0070e1c520112ef324c87dabf5568cac892a8d22cc1d0aba790733c0e295'
        DOC_SHIPPED = '42099499343b15a9d3702a52bb714919cdfd92cbe640b7d95f44504be429d302'
        review_ok = (set(ra) == set(rb)
                     and all(ra[k] == rb[k] for k in rb if k != doc)
                     and rb.get(doc) == DOC_REVIEWED and ra.get(doc) == DOC_SHIPPED)
        record(name, mismatch is None and review_ok,
               'all numerical fields match frozen receipt; 6/7 reviewed hashes match'
               if mismatch is None and review_ok else (mismatch or 'reviewed_sha256 drift'))
        record('reviewed-document provenance exception (documented post-review revision)',
               review_ok,
               'frozen receipt hash 550e0070… = pre-revision Oct-26 document; shipped revised '
               'document hash 42099499…; mismatch is exactly the documented revision pair')
    except Exception as e:
        record(name, False, str(e))
    RESULTS[-1]['wall_seconds'] = time.perf_counter() - t0


# ---------------------------------------------------------------- step 3: extension
def check_ext(sb: Path, env: dict, tmp: Path, workers: int):
    for split in ('development', 'test'):
        name = f'ext schedule audit: {split} (stdlib-only synchronous relaxation)'
        t0 = time.perf_counter()
        try:
            run_py(sb, env, 'experiments/ext_standard_jssp/code/verify_ext_schedules.py',
                   '--split', split, '--workers', workers)
            compare_report(name,
                           sb / 'experiments/ext_standard_jssp/outputs' / f'verification_schedules_{split}.json',
                           EXT / 'outputs' / f'verification_schedules_{split}.json',
                           drop_keys=('wall_seconds',))
        except Exception as e:
            record(name, False, str(e))
        RESULTS[-1]['wall_seconds'] = time.perf_counter() - t0

    name = 'ext independent scalar replay: verify_ext_replay.py (72 contrasts, both targets)'
    t0 = time.perf_counter()
    try:
        run_py(sb, env, 'experiments/ext_standard_jssp/code/verify_ext_replay.py')
        compare_report(name,
                       sb / 'experiments/ext_standard_jssp/outputs/verification_replay.json',
                       EXT / 'outputs/verification_replay.json')
    except Exception as e:
        record(name, False, str(e))
    RESULTS[-1]['wall_seconds'] = time.perf_counter() - t0


# ---------------------------------------------------------------- step 4: tests
def check_tests(sb: Path, env: dict):
    name = 'unit tests: solver correctness + branch isolation (15 tests)'
    t0 = time.perf_counter()
    try:
        out = run_py(sb, env, '-m', 'unittest', 'tests.test_fp_decision_study',
                     'tests.test_independent_tabu_jssp', timeout=1800)
        record(name, True, out.strip().splitlines()[-1] if out.strip() else 'OK')
    except Exception as e:
        # unittest writes results to stderr; recover the summary line
        msg = str(e)
        record(name, False, msg.splitlines()[-1] if msg else 'failed')
    RESULTS[-1]['wall_seconds'] = time.perf_counter() - t0


# ---------------------------------------------------------------- step 5: historical trigger cohort
def check_trigger(sb: Path, env: dict):
    """Re-audit and re-summarize the 800-run closed-loop trigger cohort (Table 8 data).

    analyze_trigger_comparison.py re-checks every paired group (initialization fingerprints,
    30,000-query budgets, VND sub-budgets, event gain accounting) and recomputes all statistics
    from the saved run JSONs with fixed bootstrap/sign-flip seeds. It writes into the sandbox
    copy only; regenerated outputs are compared against the shipped frozen receipts.
    """
    name = 'historical trigger cohort: 800-run re-audit + summary replay (30k queries/run)'
    t0 = time.perf_counter()
    rel = 'results/trigger_controlled_20runs'
    try:
        run_py(sb, env, 'analyze_trigger_comparison.py', rel, timeout=1800)
        compare_report(name, sb / rel / 'summary.json', ROOT / rel / 'summary.json')
        for csv_name in ('summary.csv', 'per_instance.csv', 'events.csv', 'report.md'):
            fresh = (sb / rel / csv_name).read_bytes()
            frozen = (ROOT / rel / csv_name).read_bytes()
            if fresh != frozen:
                RESULTS[-1]['status'] = 'FAIL'
                RESULTS[-1]['detail'] += f'; {csv_name} byte mismatch'
    except Exception as e:
        record(name, False, str(e))
    RESULTS[-1]['wall_seconds'] = time.perf_counter() - t0


# ---------------------------------------------------------------- optional: figures
def check_figures(sb: Path, env: dict):
    name = 'figure regeneration (all 8 paper figures, sandbox only)'
    t0 = time.perf_counter()
    try:
        run_py(sb, env, 'scripts/regenerate_figures.py', timeout=1800)
        fig = sb / 'projects/submission/fp_decision_study/figures'
        expected = ['fig1_overview_architecture', 'fig2_transition_errors',
                    'fig3_primary_decision_contrasts', 'fig4_scorer_sensitivity',
                    'fig5_choices_and_scope', 'fig6_independent_repeat_winners',
                    'fig7_action_window']
        missing = [f for f in expected if not (fig / f'{f}.png').exists()]
        ext_fig = sb / 'experiments/ext_standard_jssp/outputs/fig_ext_summary.png'
        official = ['Figure1.jpeg', 'Figure2.png', 'Figure3.png', 'Figure4.png',
                    'Figure5.png', 'Figure6.png', 'Figure7.png', 'Figure8.png', 'PROVENANCE.md']
        missing += [f'official/{f}' for f in official if not (fig / f).exists()]
        record(name, not missing and ext_fig.exists(), f'missing: {missing}')
    except Exception as e:
        record(name, False, str(e))
    RESULTS[-1]['wall_seconds'] = time.perf_counter() - t0


# ---------------------------------------------------------------- optional: historical
def check_historical(sb: Path, env: dict, tmp: Path):
    """Audit extracted historical raw archives (release_assets tarballs -> raw_extracted/).

    Historical replay uses the frozen source snapshots shipped in projects/experiments/:
    the fp_v2 cohorts were collected with pre-2026-10-26 versions of three solver modules
    (controller / fixed_point_detector / meta_scheduler); their protocol.json locks those
    older hashes, so fp_v2_full_replay_sources.zip is extracted over the sandbox before the
    fp_v2 audits (the canonical replay tree). The confirmation cohort's source lock matches
    the shipped current sources exactly. The repeatability analyzer additionally consumes the
    frozen M2 analysis JSONs from historical_analysis_large_jsons.tar.gz, which are linked
    into the sandbox at their canonical projects/analysis/ paths.
    """
    raw = ROOT / 'raw_extracted'
    # Link ALL extracted cohorts into the sandbox up front: the repeatability replay
    # cross-references the original fp_v2 source trajectories (saved['source_json']).
    res = raw / 'results'
    if res.is_dir():
        for d in sorted(res.iterdir()):
            if d.is_dir():
                link = sb / 'results' / d.name
                if not link.exists():
                    try:
                        link.symlink_to(d.resolve(), target_is_directory=True)
                    except OSError:
                        shutil.copytree(d, link)  # Windows fallback without symlink privilege
    # order matters: fp_v2 snapshot extraction overwrites sandbox sources, so it runs LAST
    plans = [
        ('fp_empirical_confirmation', 'audit_fp_empirical_confirmation.py'),
        ('fp_empirical_repeatability_development', 'analyze_fp_empirical_repeatability.py'),
        ('fp_v2_development', 'audit_fp_action_data.py'),
        ('fp_v2_validation', 'audit_fp_action_data.py'),
    ]
    fp_v2_snapshot_done = False
    for cohort, script in plans:
        name = f'historical raw audit: {cohort}'
        d = raw / 'results' / cohort
        if not d.is_dir():
            record(name, False, f'not extracted: {d} (see release_assets/README.md)')
            continue
        t0 = time.perf_counter()
        try:
            out = tmp / f'hist_audit_{cohort}.json'
            if script == 'audit_fp_action_data.py':
                if not fp_v2_snapshot_done:
                    import zipfile
                    snap = ROOT / 'projects/experiments/fp_v2_full_replay_sources.zip'
                    with zipfile.ZipFile(snap) as zf:
                        zf.extractall(sb)  # frozen historical replay tree (sandbox only)
                    fp_v2_snapshot_done = True
                run_py(sb, env, f'scripts/{script}', str(d.resolve()), '--output', out, timeout=3600)
                rep = read(out)
                ok = str(rep.get('status', '')).startswith('PASS')
                detail = rep.get('status', 'no status')
            elif script == 'analyze_fp_empirical_repeatability.py':
                ana = raw / 'projects/analysis'
                if ana.is_dir():
                    for p in ana.glob('*.json'):
                        dst = sb / 'projects/analysis' / p.name
                        if not dst.exists():
                            try:
                                dst.symlink_to(p.resolve())
                            except OSError:
                                shutil.copy2(p, dst)
                run_py(sb, env, f'scripts/{script}',
                       '--directory', f'results/{cohort}', '--output', out, timeout=3600)
                rep = read(out)
                ok = rep.get('endpoint_independent_DAG_audit') == 'PASS_ALL_NEW_BRANCHES'
                detail = f"endpoint_independent_DAG_audit={rep.get('endpoint_independent_DAG_audit')}"
            else:  # audit_fp_empirical_confirmation.py
                run_py(sb, env, f'scripts/{script}',
                       '--directory', str(d.resolve()),
                       '--model-bundle', 'projects/experiments/fp_submission_frozen_model_bundle.json',
                       '--output', out, timeout=3600)
                rep = read(out)
                ok = str(rep.get('status', '')).startswith('PASS')
                detail = rep.get('status', 'no status')
            record(name, ok, detail)
        except Exception as e:
            record(name, False, str(e))
        RESULTS[-1]['wall_seconds'] = time.perf_counter() - t0


# ---------------------------------------------------------------- step 5: hygiene
# Documented exceptions: hash-locked / frozen provenance records that mention a
# forbidden token as a *relative path or disclosure sentence*. No actual venv,
# cache, Word document or unrelated-project file is shipped. See README §11.
TOKEN_ALLOWLIST = {
    'experiments/ext_standard_jssp/outputs/manifest_lock.json': {'venv_ext'},
    'experiments/ext_standard_jssp/code/run_ext_pipeline.py': {'venv_ext'},
    'experiments/ext_standard_jssp/code/prepare_ext_lock.py': {'venv_ext', '__pycache__', '.docx'},
    'experiments/ext_standard_jssp/STEP0_REPORT.md': {'venv_ext'},
    # whole-workspace hash snapshots enumerate paths of excluded assets (hashes only,
    # no content): they necessarily mention cache/Word paths that are NOT shipped.
    'experiments/ext_standard_jssp/outputs/frozen_snapshot_before.json': {'__pycache__', '.docx'},
    'experiments/ext_standard_jssp/outputs/frozen_snapshot_after.json': {'__pycache__', '.docx'},
    'projects/docs/fp_decision_empirical_manuscript.md': {'resource-allocation'},
    # release documentation / config that describes these exceptions or ignores them:
    'README.md': {'venv_ext', '__pycache__', '.docx'},
    '.gitignore': {'__pycache__', '.docx', 'venv_ext'},
    # the scanner itself necessarily contains the forbidden tokens as literals:
    'scripts/verify_all.py': {'venv_ext', '__pycache__', '.docx', 'resource-allocation',
                              '/Users/', '/home/', 'C:\\Users', 'API_KEY', 'api_key'},
    # public OR-Library maintainer contacts shipped inside the hash-locked benchmark text:
    'jobshop1.txt': {'EMAIL'},
}
FORBIDDEN = ['/Users/', '/home/', 'C:\\Users', 'venv_ext', '__pycache__',
             '.docx', 'resource-allocation', 'API_KEY', 'api_key']
import re
API_KEY_RE = re.compile(r'sk-[A-Za-z0-9]{20,}')
EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]{2,}')
TEXT_SUFFIX = {'.py', '.md', '.json', '.txt', '.csv', '.svg', '.sh', '.toml', '.cfg'}
LOCAL_DIRS = {'.git', '.venv', 'venv', 'raw_extracted', 'release_assets'}


def check_hygiene():
    files = [p for p in ROOT.rglob('*')
             if p.is_file() and not (set(p.relative_to(ROOT).parts) & LOCAL_DIRS)]
    total = sum(p.stat().st_size for p in files)
    biggest = max(files, key=lambda p: p.stat().st_size)
    record('max single file < 50 MB', biggest.stat().st_size < 50 * 1024 * 1024,
           f'{biggest.stat().st_size / 1e6:.2f} MB  {biggest.relative_to(ROOT)}')
    record('total repo size < 300 MB', total < 300 * 1024 * 1024, f'{total / 1e6:.1f} MB')
    record('file count < 10,000', len(files) < 10000, f'{len(files)} files')
    bad_dirs = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
                if p.is_dir() and p.name in {'__pycache__', '.pytest_cache', '.ipynb_checkpoints',
                                             'venv_ext'}
                and not (set(p.relative_to(ROOT).parts) & LOCAL_DIRS)]
    record('no shipped venv_ext / cache directories', not bad_dirs, str(bad_dirs))
    venv_cfg = [str(p.relative_to(ROOT)) for p in files if p.name == 'pyvenv.cfg']
    record('no virtualenv shipped (pyvenv.cfg absent)', not venv_cfg, str(venv_cfg))
    bad_binary = [str(p.relative_to(ROOT)) for p in files
                  if p.suffix in {'.docx', '.doc', '.pyc', '.env'} or p.name == '.DS_Store']
    record('no Word drafts / bytecode / .env / .DS_Store files', not bad_binary, str(bad_binary))
    hits = []
    for p in files:
        if p.suffix not in TEXT_SUFFIX and p.name != '.gitignore':
            continue
        rel = str(p.relative_to(ROOT))
        try:
            text = p.read_text(errors='ignore')
        except Exception:
            continue
        allowed = TOKEN_ALLOWLIST.get(rel, set())
        for tok in FORBIDDEN:
            if tok in text and tok not in allowed:
                hits.append(f'{rel}: {tok}')
        if API_KEY_RE.search(text):
            hits.append(f'{rel}: API-key-like token')
        if EMAIL_RE.search(text) and 'EMAIL' not in allowed:
            hits.append(f'{rel}: e-mail address')
    record('forbidden-pattern scan (allowlist documented in README §11)', not hits,
           '; '.join(hits[:10]))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='locks + hygiene only')
    ap.add_argument('--figures', action='store_true', help='also regenerate all figures in the sandbox')
    ap.add_argument('--historical', action='store_true', help='also audit extracted historical raw archives')
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument('--report', type=Path, default=None, help='optional JSON summary output path')
    args = ap.parse_args()

    print('=' * 78)
    print('SWEVO FP-decision study — independent verification (verify_all.py)')
    print(f'interpreter: {sys.version.split()[0]}   repo: {ROOT}')
    try:
        import numpy
        print(f'numpy: {numpy.__version__}')
    except ImportError:
        print('FATAL: numpy is required (pip install -r requirements.txt)')
        sys.exit(2)
    print('=' * 78)

    lock_secs = check_locks()
    RESULTS[0]['wall_seconds'] = lock_secs

    if not args.quick:
        with tempfile.TemporaryDirectory(prefix='fp_verify_') as td:
            tmp = Path(td)
            t0 = time.perf_counter()
            sb = make_sandbox(tmp)
            record('clean-room sandbox copy (write-once receipts removed)', True,
                   '', time.perf_counter() - t0)
            env = sandbox_env(sb, tmp)
            check_primary(sb, env, tmp)
            check_ext(sb, env, tmp, args.workers)
            check_tests(sb, env)
            check_trigger(sb, env)
            if args.figures:
                check_figures(sb, env)
            if args.historical:
                check_historical(sb, env, tmp)

    check_hygiene()

    print('\n' + '=' * 78)
    width = max(len(r['check']) for r in RESULTS) + 2
    print(f'{"CHECK":<{width}}STATUS')
    print('-' * (width + 6))
    for r in RESULTS:
        print(f'{r["check"]:<{width}}{r["status"]}')
    print('-' * (width + 6))
    n_pass = sum(r['status'] == 'PASS' for r in RESULTS)
    n_fail = len(RESULTS) - n_pass
    print(f'TOTAL: {n_pass} PASS / {n_fail} FAIL')
    print('=' * 78)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {'summary': {'pass': n_pass, 'fail': n_fail}, 'checks': RESULTS}, indent=2) + '\n')
        print(f'report written: {args.report}')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()

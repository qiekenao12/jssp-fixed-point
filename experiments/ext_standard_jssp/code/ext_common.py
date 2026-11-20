#!/usr/bin/env python3
"""Shared constants/helpers for the ext_standard_jssp prospective extension cohort.

Isolated extension: imports frozen protocol code read-only; never modifies it.
All new artifacts live under experiments/ext_standard_jssp/.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / 'experiments/ext_standard_jssp'
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from tools.independent_tabu_jssp import Instance, ACTIONS  # noqa: E402

BENCH = EXT / 'benchmarks'
RAW = EXT / 'raw'
MODELS = EXT / 'models'
OUT = EXT / 'outputs'
CODE = EXT / 'code'

ARMS = ('A', 'B', 'C', 'D', 'E', 'D_random')
SCORERS = ('frozen_ridge', 'loocv_ridge', 'rff_ridge', 'gbdt')
TARGETS = ('raw', 'cng')  # raw: R_a ; cng: R_a / decodes_a * 1e3
FIXED_REFERENCE = {'raw': 'CB_TABU', 'cng': 'CB_FIRST_DESCENT'}

SPLIT_SEED = 2026112000
COLLECTION_NAMESPACE = 2026112001
BOOTSTRAP_SEED = 2026112002
GBDT_RANDOM_STATE = 2026112003
RFF_SEED = 20261112          # frozen protocol value (revision_response_analysis)
RFF_COMPONENTS = 256
D_RANDOM_SEED = 2026102501   # frozen inside fp_decision_models.vector

L2_GRID = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
GBDT_GRID = [{'max_depth': d, 'learning_rate': lr, 'max_iter': it, 'l2_regularization': l2}
             for d in (2, 3) for lr in (0.03, 0.1) for it in (50, 100, 200) for l2 in (1.0, 10.0)]
GBDT_FIXED = {'min_samples_leaf': 20, 'early_stopping': False, 'loss': 'squared_error',
              'random_state': GBDT_RANDOM_STATE, 'max_bins': 255}

RUNS = {'development': 10, 'test': 20}
NEUTRAL_CHUNKS = (8, 16)
ACTION_STEPS = 8
REPEATS = 5
BOOTSTRAP_DRAWS = 20000

BLOCKS = {
    'ta_15x15': [f'ta{i:02d}' for i in range(1, 11)],
    'ta_20x15': [f'ta{i:02d}' for i in range(11, 21)],
    'ta_20x20': [f'ta{i:02d}' for i in range(21, 31)],
    'la_15x10': [f'la{i:02d}' for i in range(21, 26)],
    'la_20x10': [f'la{i:02d}' for i in range(26, 31)],
    'la_30x10': [f'la{i:02d}' for i in range(31, 36)],
    'la_15x15': [f'la{i:02d}' for i in range(36, 41)],
}
MIRROR_URL = 'https://raw.githubusercontent.com/tamy0612/JSPLIB/master/instances/{name}'
LOCAL_TA = {'ta01', 'ta02', 'ta11', 'ta12', 'ta21'}


def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256((':'.join(map(str, (COLLECTION_NAMESPACE, *parts)))).encode()).digest()[:8], 'big')


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f'Cannot overwrite {path}')
    data = json.dumps(obj, separators=(',', ':'), allow_nan=False).encode()
    temp = path.with_suffix(path.suffix + '.tmp')
    if temp.exists():
        raise ValueError(f'Incomplete result retained: {temp}')
    if path.suffix == '.gz':
        with gzip.open(temp, 'wb') as f:
            f.write(data)
    else:
        temp.write_bytes(data)
    temp.replace(path)


def load(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)


def parse_orlib_blocks(path):
    """Generic OR-Library / JSPLIB-snapshot block parser (independent, auditable)."""
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
            try:
                vals = [int(t) for t in s.split()]
            except ValueError:
                continue
            rows.append(vals)
        j, m = rows[0][0], rows[0][1]
        data = rows[1:1 + j]
        assert len(data) == j and all(len(r) == 2 * m for r in data), (name, path)
        routes = tuple(tuple(r[::2]) for r in data)
        durations = tuple(tuple(r[1::2]) for r in data)
        try:
            out[name] = Instance(routes, durations, name)
        except ValueError:
            continue  # e.g. orb07 contains zero durations; rejected by the solver
    return out


def parse_jsplib(path):
    """Plain JSPLIB file: first line `j m`, then j rows of (machine, time) pairs."""
    rows = [list(map(int, line.split())) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    j, m = rows[0]
    assert len(rows) == j + 1 and all(len(r) == 2 * m for r in rows[1:]), path
    return Instance(tuple(tuple(r[::2]) for r in rows[1:]),
                    tuple(tuple(r[1::2]) for r in rows[1:]), Path(path).name)


def proposed_split():
    import random
    rng = random.Random(SPLIT_SEED)
    dev, test = [], []
    for names in BLOCKS.values():
        k = 4 if len(names) == 10 else 2
        d = sorted(rng.sample(sorted(names), k))
        dev += d
        test += sorted(set(names) - set(d))
    return sorted(dev), sorted(test)


def cng_value(record):
    return record['gain'] / max(record['outcome']['decode_evaluations'], 1) * 1e3

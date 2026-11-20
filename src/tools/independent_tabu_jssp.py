"""Independent machine-order JSSP critical-block search for empirical replication.

This module imports no existing JSSP decoder/search operator. It is a deliberately
small, auditable implementation, not a claim of a new or state-of-the-art solver.
Node j*m+k denotes operation k of job j. All random choices use local RNGs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import heapq
import json
import random
import time
from typing import Optional


ACTIONS = ("CB_BEST_DESCENT", "CB_FIRST_DESCENT", "CB_TABU")
DEFAULT_STEPS = 8
TABU_TENURE = 5


@dataclass(frozen=True)
class Instance:
    routes: tuple[tuple[int, ...], ...]
    durations: tuple[tuple[int, ...], ...]
    name: str = "unnamed"

    def __post_init__(self):
        if not self.routes or not self.routes[0]:
            raise ValueError("empty instance")
        m = len(self.routes[0])
        if len(self.routes) != len(self.durations):
            raise ValueError("route/time mismatch")
        for r, p in zip(self.routes, self.durations):
            if len(r) != m or len(p) != m or sorted(r) != list(range(m)):
                raise ValueError("each route must be a permutation of machines")
            if any(type(v) is not int or v <= 0 for v in p):
                raise ValueError("durations must be positive integers")
            if any(type(v) is not int for v in r):
                raise ValueError("machine IDs must be integers")

    @property
    def jobs(self):
        return len(self.routes)

    @property
    def machines(self):
        return len(self.routes[0])

    @property
    def size(self):
        return self.jobs * self.machines

    @property
    def fingerprint(self):
        body = json.dumps([self.routes, self.durations], separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(map(tuple, d["routes"])), tuple(map(tuple, d["durations"])), d.get("name", "unnamed"))


Orders = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Decoded:
    makespan: int
    starts: tuple[int, ...]
    finishes: tuple[int, ...]
    critical_path: tuple[int, ...]
    critical_blocks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SearchState:
    instance_fingerprint: str
    machine_orders: Orders
    best_orders: Orders
    current_makespan: int
    best_makespan: int
    initial_makespan: int
    accepted_total: int = 0
    stagnant_moves: int = 0
    # (node1, node2, expiry), unordered swap pair prevents immediate reversal.
    tabu_until: tuple[tuple[int, int, int], ...] = ()
    # All recorded endpoint observations (best, current, accepted total).
    history: tuple[tuple[int, int, int], ...] = ()
    # Actual previous macro-action incumbent gains, not counterfactual outcomes.
    gain_history: tuple[int, ...] = ()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        x = dict(d)
        for key in ("machine_orders", "best_orders", "tabu_until", "history"):
            x[key] = tuple(map(tuple, x[key]))
        x["gain_history"] = tuple(x.get("gain_history", ()))
        return cls(**x)


@dataclass(frozen=True)
class ActionResult:
    state: SearchState
    action: str
    seed: int
    requested_steps: int
    accepted_steps: int
    decode_evaluations: int
    candidate_evaluations: int
    infeasible_candidates: int
    elapsed_seconds: float
    termination: str
    improvement: int
    normalized_improvement: float
    # (accepted move index, elapsed seconds, incumbent makespan)
    anytime: tuple[tuple[int, float, int], ...]

    def to_dict(self):
        return asdict(self)


def validate_orders(inst: Instance, orders: Orders):
    if len(orders) != inst.machines:
        raise ValueError("wrong machine count")
    seen = set()
    for machine, row in enumerate(orders):
        if len(row) != inst.jobs:
            raise ValueError("wrong operations per machine")
        for node in row:
            if type(node) is not int or not 0 <= node < inst.size or node in seen:
                raise ValueError("illegal or duplicate operation")
            if inst.routes[node // inst.machines][node % inst.machines] != machine:
                raise ValueError("operation assigned to wrong machine")
            seen.add(node)


def _decode(inst: Instance, orders: Orders) -> Optional[Decoded]:
    """Unchecked inner kernel; public decode validates exact machine permutations."""
    n, m = inst.size, inst.machines
    outgoing = [[] for _ in range(n)]
    indegree = [0] * n
    for j in range(inst.jobs):
        for k in range(m - 1):
            u, v = j * m + k, j * m + k + 1
            outgoing[u].append(v)
            indegree[v] += 1
    for row in orders:
        for u, v in zip(row, row[1:]):
            outgoing[u].append(v)
            indegree[v] += 1
    ready = [i for i, d in enumerate(indegree) if d == 0]
    heapq.heapify(ready)
    start = [0] * n
    finish = [0] * n
    parent = [-1] * n
    count = 0
    while ready:
        u = heapq.heappop(ready)
        finish[u] = start[u] + inst.durations[u // m][u % m]
        count += 1
        for v in outgoing[u]:
            if finish[u] > start[v] or (finish[u] == start[v] and (parent[v] < 0 or u < parent[v])):
                start[v], parent[v] = finish[u], u
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(ready, v)
    if count != n:
        return None
    terminal = min(range(n), key=lambda i: (-finish[i], i))
    path = []
    while terminal >= 0:
        path.append(terminal)
        terminal = parent[terminal]
    path.reverse()
    blocks = []
    current = []
    previous_machine = None
    for node in path:
        machine = inst.routes[node // m][node % m]
        if machine != previous_machine and current:
            if len(current) > 1:
                blocks.append(tuple(current))
            current = []
        current.append(node)
        previous_machine = machine
    if len(current) > 1:
        blocks.append(tuple(current))
    return Decoded(max(finish), tuple(start), tuple(finish), tuple(path), tuple(blocks))


def decode(inst: Instance, orders: Orders) -> Optional[Decoded]:
    validate_orders(inst, orders)
    return _decode(inst, orders)


def initial_state(inst: Instance, seed: int) -> SearchState:
    """Random job-frontier linear extension, hence a feasible machine-order DAG."""
    rng = random.Random(seed)
    next_op = [0] * inst.jobs
    active = list(range(inst.jobs))
    orders = [[] for _ in range(inst.machines)]
    while active:
        j = rng.choice(active)
        k = next_op[j]
        orders[inst.routes[j][k]].append(j * inst.machines + k)
        next_op[j] += 1
        if next_op[j] == inst.machines:
            active.remove(j)
    frozen = tuple(map(tuple, orders))
    decoded = decode(inst, frozen)
    assert decoded is not None
    c = decoded.makespan
    return SearchState(inst.fingerprint, frozen, frozen, c, c, c, history=((c, c, 0),))


def critical_swaps(inst: Instance, orders: Orders, decoded: Decoded):
    """Finite specified neighborhood, no assertion of exhaustive N5/N6 coverage."""
    result = []
    for block in decoded.critical_blocks:
        machine = inst.routes[block[0] // inst.machines][block[0] % inst.machines]
        positions = {node: i for i, node in enumerate(orders[machine])}
        for u, v in zip(block, block[1:]):
            index = positions[u]
            if positions[v] != index + 1:
                raise AssertionError("critical same-machine edge is not adjacent")
            result.append((machine, index, min(u, v), max(u, v)))
    return result


def _swapped(orders: Orders, machine: int, index: int):
    row = list(orders[machine])
    row[index], row[index + 1] = row[index + 1], row[index]
    return orders[:machine] + (tuple(row),) + orders[machine + 1:]


def validate_state(inst: Instance, state: SearchState):
    if state.instance_fingerprint != inst.fingerprint:
        raise ValueError("state belongs to another instance")
    current, best = decode(inst, state.machine_orders), decode(inst, state.best_orders)
    if current is None or best is None:
        raise ValueError("infeasible state")
    if current.makespan != state.current_makespan or best.makespan != state.best_makespan:
        raise ValueError("state makespan mismatch")
    if not (0 < state.best_makespan <= state.current_makespan and state.best_makespan <= state.initial_makespan):
        raise ValueError("invalid incumbent/normalization")
    return current


def run_action(inst: Instance, state: SearchState, action: str, seed: int, steps: int = DEFAULT_STEPS) -> ActionResult:
    if action not in ACTIONS or type(steps) is not int or steps < 0:
        raise ValueError("invalid action or steps")
    clock = time.perf_counter()
    decoded = validate_state(inst, state)
    rng = random.Random(seed)
    orders, best_orders, best = state.machine_orders, state.best_orders, state.best_makespan
    # No shared mutable storage with the saved state or any diagnostic branch.
    tabu = {(u, v): expiry for u, v, expiry in state.tabu_until}
    accepted = candidates_checked = infeasible = 0
    stagnant = state.stagnant_moves
    reason = "step_limit"
    anytime = [(0, time.perf_counter() - clock, best)]
    while accepted < steps:
        moves = critical_swaps(inst, orders, decoded)
        if not moves:
            reason = "no_critical_block_moves"
            break
        rng.shuffle(moves)
        chosen = None
        chosen_value = None
        total_index = state.accepted_total + accepted
        for machine, index, u, v in moves:
            trial_orders = _swapped(orders, machine, index)
            trial = _decode(inst, trial_orders)
            candidates_checked += 1
            if trial is None:
                infeasible += 1
                continue
            if action != "CB_TABU" and trial.makespan >= decoded.makespan:
                continue
            if action == "CB_TABU" and tabu.get((u, v), -1) > total_index and trial.makespan >= best:
                continue
            if chosen_value is None or trial.makespan < chosen_value:
                chosen = (trial_orders, trial, (u, v))
                chosen_value = trial.makespan
            if action == "CB_FIRST_DESCENT":
                break
        if chosen is None:
            reason = "no_admissible_move" if action == "CB_TABU" else "no_strict_improving_move"
            break
        orders, decoded, pair = chosen
        accepted += 1
        if action == "CB_TABU":
            tabu[pair] = state.accepted_total + accepted + TABU_TENURE
        if decoded.makespan < best:
            best, best_orders, stagnant = decoded.makespan, orders, 0
        else:
            stagnant += 1
        anytime.append((accepted, time.perf_counter() - clock, best))
    total = state.accepted_total + accepted
    gain = state.best_makespan - best
    endpoint = replace(state, machine_orders=orders, best_orders=best_orders,
                       current_makespan=decoded.makespan, best_makespan=best,
                       accepted_total=total, stagnant_moves=stagnant,
                       tabu_until=tuple(sorted((u, v, t) for (u, v), t in tabu.items() if t > total)),
                       history=state.history + ((best, decoded.makespan, total),),
                       gain_history=state.gain_history + (gain,))
    return ActionResult(endpoint, action, seed, steps, accepted, 2 + candidates_checked,
                        candidates_checked, infeasible, time.perf_counter() - clock,
                        reason, gain, gain / state.initial_makespan, tuple(anytime))


def features(inst: Instance, state: SearchState):
    """Only the observed saved state and past real macro-action outcomes."""
    decoded = validate_state(inst, state)
    scale = state.initial_makespan
    path_fraction = len(decoded.critical_path) / inst.size
    block_nodes = sum(map(len, decoded.critical_blocks))
    dynamics = [state.best_makespan / scale, state.current_makespan / scale,
                path_fraction, block_nodes / inst.size]
    last = state.gain_history[-3:]
    previous = state.history[-2] if len(state.history) > 1 else state.history[-1]
    ordinary = dynamics[:2] + [state.stagnant_moves / inst.size, state.accepted_total / inst.size]
    history = [sum(last) / (max(len(last), 1) * scale),
               (last[-1] if last else 0) / scale,
               (previous[0] - state.best_makespan) / scale,
               (previous[1] - state.current_makespan) / scale]
    structure = [path_fraction, block_nodes / inst.size,
                 len(decoded.critical_blocks) / inst.machines,
                 max(map(len, decoded.critical_blocks), default=0) / inst.jobs,
                 len(critical_swaps(inst, state.machine_orders, decoded)) / inst.size]
    return {"dynamics": dynamics, "ordinary": ordinary, "history": history, "structure": structure,
            "dynamics_names": ["best/s0", "current/s0", "critical_path_fraction", "critical_block_node_fraction"],
            "ordinary_names": ["best/s0", "current/s0", "stagnant_moves/n", "accepted_moves/n"],
            "history_names": ["last3_mean_gain/s0", "last_gain/s0", "last_best_delta/s0", "last_current_delta/s0"],
            "structure_names": ["critical_path_fraction", "critical_block_node_fraction", "blocks/m", "max_block/jobs", "critical_swaps/n"]}


def read_jsplib(path):
    """Plain JSPLIB parser, independent from every existing project parser."""
    from pathlib import Path
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    j, m = map(int, rows[0])
    if len(rows) != j + 1 or any(len(row) != 2 * m for row in rows[1:]):
        raise ValueError("invalid JSPLIB dimensions")
    values = [list(map(int, row)) for row in rows[1:]]
    return Instance(tuple(tuple(row[::2]) for row in values), tuple(tuple(row[1::2]) for row in values), Path(path).name)

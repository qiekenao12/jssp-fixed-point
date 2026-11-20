"""Independent synchronous-relaxation oracle; never calls existing GA/LS code."""
import itertools
import json
import random
import unittest
from dataclasses import replace
from unittest.mock import patch

from tools import independent_tabu_jssp as solver


def independent_reference(inst, orders):
    n, m = inst.size, inst.machines
    arcs = [(j * m + k, j * m + k + 1) for j in range(inst.jobs) for k in range(m - 1)]
    arcs += [(u, v) for row in orders for u, v in zip(row, row[1:])]
    durations = sum((list(p) for p in inst.durations), [])
    start = [0] * n
    for iteration in range(n):
        nxt = start.copy()
        for u, v in arcs:
            nxt[v] = max(nxt[v], start[u] + durations[u])
        if nxt == start:
            return max(s + p for s, p in zip(start, durations)), tuple(start)
        start = nxt
    return None


class IndependentTabuTests(unittest.TestCase):
    def setUp(self):
        self.inst = solver.Instance(((0, 1, 2), (1, 2, 0), (2, 0, 1)),
                                    ((3, 2, 4), (2, 4, 3), (4, 3, 2)), "toy3x3")

    def test_exhaustive_all_216_machine_orderings(self):
        machines = [[j * 3 + k for j in range(3) for k in range(3) if self.inst.routes[j][k] == a] for a in range(3)]
        counts = {True: 0, False: 0}
        for orders in itertools.product(*(list(itertools.permutations(row)) for row in machines)):
            observed = solver.decode(self.inst, orders)
            expected = independent_reference(self.inst, orders)
            counts[expected is not None] += 1
            self.assertEqual(observed is None, expected is None)
            if expected is not None:
                self.assertEqual((observed.makespan, observed.starts), expected)
                duration = sum(self.inst.durations[v // 3][v % 3] for v in observed.critical_path)
                self.assertEqual(duration, observed.makespan)
                for machine, index, _, _ in solver.critical_swaps(self.inst, orders, observed):
                    trial_orders = solver._swapped(orders, machine, index)
                    self.assertEqual(solver.decode(self.inst, trial_orders) is None,
                                     independent_reference(self.inst, trial_orders) is None)
        self.assertEqual(sum(counts.values()), 216)
        self.assertGreater(counts[True], 0)
        self.assertGreater(counts[False], 0)

    def test_random_dags_and_endpoint_schedule_correctness(self):
        rng = random.Random(119)
        for size in ((3, 3), (4, 3), (6, 6), (10, 5)):
            j, m = size
            for repeat in range(5):
                inst = solver.Instance(tuple(tuple(rng.sample(range(m), m)) for _ in range(j)),
                                       tuple(tuple(rng.randint(1, 30) for _ in range(m)) for _ in range(j)))
                state = solver.initial_state(inst, repeat)
                for action in solver.ACTIONS:
                    result = solver.run_action(inst, state, action, repeat + 3, steps=8)
                    for orders, reported in ((result.state.machine_orders, result.state.current_makespan),
                                             (result.state.best_orders, result.state.best_makespan)):
                        reference = independent_reference(inst, orders)
                        self.assertIsNotNone(reference)
                        self.assertEqual(reference[0], reported)
                    self.assertGreaterEqual(result.improvement, 0)
                    self.assertLessEqual(result.accepted_steps, 8)
                    self.assertEqual(result.improvement, state.best_makespan - result.state.best_makespan)
                    self.assertEqual(result.normalized_improvement, result.improvement / state.initial_makespan)

    def test_serialization_and_branch_isolation(self):
        state = solver.initial_state(self.inst, 3)
        before = json.dumps(state.to_dict(), sort_keys=True)
        roundtrip = solver.SearchState.from_dict(json.loads(before))
        self.assertEqual(state, roundtrip)
        self.assertEqual(self.inst, solver.Instance.from_dict(json.loads(json.dumps(self.inst.to_dict()))))
        results = [solver.run_action(self.inst, state, a, 43) for a in solver.ACTIONS]
        self.assertEqual(before, json.dumps(state.to_dict(), sort_keys=True))
        for result in results:
            repeat = solver.run_action(self.inst, roundtrip, result.action, 43)
            self.assertEqual(result.state, repeat.state)
            self.assertEqual(result.decode_evaluations, repeat.decode_evaluations)

    def test_rng_is_local_and_replicates_are_seeded(self):
        random.seed(234)
        global_before = random.getstate()
        state = solver.initial_state(self.inst, 4)
        for seed in range(10):
            solver.run_action(self.inst, state, "CB_TABU", seed)
        self.assertEqual(global_before, random.getstate())

    def test_real_cost_matches_all_decoder_calls(self):
        state = solver.initial_state(self.inst, 2)
        for action in solver.ACTIONS:
            original = solver._decode
            with patch.object(solver, "_decode", wraps=original) as called:
                result = solver.run_action(self.inst, state, action, 7)
            self.assertEqual(called.call_count, result.decode_evaluations)
            self.assertEqual(result.decode_evaluations, result.candidate_evaluations + 2)
            self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_tabu_tenure_is_five_future_choices_and_aspiration(self):
        selected = None
        aspirating = None
        for seed in range(20):
            state = solver.initial_state(self.inst, seed)
            decoded = solver.decode(self.inst, state.machine_orders)
            for move in solver.critical_swaps(self.inst, state.machine_orders, decoded):
                machine, index, u, v = move
                trial = solver.decode(self.inst, solver._swapped(state.machine_orders, machine, index))
                if trial is None:
                    continue
                if trial.makespan >= state.best_makespan and selected is None:
                    selected = state, move
                if trial.makespan < state.best_makespan and aspirating is None:
                    aspirating = state, move
        self.assertIsNotNone(selected)
        self.assertIsNotNone(aspirating)
        state, move = selected
        for elapsed_accepted in range(7):
            # Swap forbidden after a hypothetical accepted move at t=0.
            # Checks before choices 1..5 see counts 0..4; choice 6 sees 5.
            snapshot = replace(state, accepted_total=elapsed_accepted,
                               tabu_until=((move[2], move[3], 5),))
            with patch.object(solver, "critical_swaps", return_value=[move]):
                result = solver.run_action(self.inst, snapshot, "CB_TABU", 3, steps=1)
            self.assertEqual(result.accepted_steps, int(elapsed_accepted >= 5))
        state, move = aspirating
        snapshot = replace(state, tabu_until=((move[2], move[3], 5),))
        with patch.object(solver, "critical_swaps", return_value=[move]):
            result = solver.run_action(self.inst, snapshot, "CB_TABU", 3, steps=1)
        self.assertEqual(result.accepted_steps, 1)
        self.assertGreater(result.improvement, 0)

    def test_zero_steps_and_natural_termination_are_explicit(self):
        state = solver.initial_state(self.inst, 1)
        result = solver.run_action(self.inst, state, "CB_TABU", 0, steps=0)
        self.assertEqual(result.accepted_steps, 0)
        self.assertEqual(result.candidate_evaluations, 0)
        self.assertEqual(result.improvement, 0)
        self.assertEqual(result.termination, "step_limit")
        small = solver.Instance(((0,),), ((4,),))
        done = solver.run_action(small, solver.initial_state(small, 0), "CB_TABU", 0)
        self.assertEqual(done.termination, "no_critical_block_moves")
        self.assertEqual(done.accepted_steps, 0)

    def test_feature_history_has_no_diagnostic_outcomes(self):
        state = solver.initial_state(self.inst, 1)
        before = solver.features(self.inst, state)
        for action in solver.ACTIONS:
            solver.run_action(self.inst, state, action, 5)
        self.assertEqual(before, solver.features(self.inst, state))
        self.assertEqual(len(before["dynamics"]), 4)
        self.assertEqual(before["history"], [0.0, 0.0, 0.0, 0.0])

    def test_invalid_orders_instances_and_cross_instance_snapshots(self):
        with self.assertRaises(ValueError):
            solver.Instance(((0, 0),), ((1, 2),))
        with self.assertRaises(ValueError):
            solver.Instance(((0,),), ((0,),))
        state = solver.initial_state(self.inst, 5)
        bad = (state.machine_orders[0][:-1] + (state.machine_orders[0][0],),) + state.machine_orders[1:]
        with self.assertRaises(ValueError):
            solver.decode(self.inst, bad)
        other = solver.Instance(self.inst.routes, ((4, 2, 4), (2, 4, 3), (4, 3, 2)))
        with self.assertRaises(ValueError):
            solver.run_action(other, state, "CB_TABU", 0)


if __name__ == "__main__":
    unittest.main()

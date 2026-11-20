"""Small independent checks of collection and numerical/input semantics.

Uses only manufactured toy states and temporary output, never confirmation data.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import fp_decision_models as models
from scripts import run_fp_decision_collection as collection
from tools.independent_tabu_jssp import ACTIONS, Instance, SearchState, features, initial_state, run_action


def synthetic_row():
    inst = Instance(((0, 1, 2), (1, 2, 0), (2, 0, 1)),
                    ((3, 2, 4), (2, 4, 3), (4, 3, 2)), "synthetic")
    state = initial_state(inst, 92)
    observed = features(inst, state)
    row = dict(instance="synthetic", state_id="synthetic:0:0", action=ACTIONS[0],
               observed=observed, start_x=observed["dynamics"], end_x=[0.3] * 4,
               gain=0.2, improved=True, repeat=0)
    return row


def maps_with_offset(offset=0.0):
    return {a: {"A": np.eye(4).tolist(), "c": [offset + i * .1] * 4} for i, a in enumerate(ACTIONS)}


class DecisionStudyTests(unittest.TestCase):
    def test_ridge_exact_scalar_objective_is_sum_squared_error_plus_l2(self):
        # min_b,w sum((b+w*x-y)^2)+w^2 gives b=1/2,w=1/3.
        x = np.array([[-1.], [1.]])
        y = np.array([0., 1.])
        model = models.fit_ridge(x, y)
        np.testing.assert_allclose(model["weights"], [.5, 1 / 3], atol=1e-13)
        np.testing.assert_allclose(models.predict(model, x), [1 / 6, 5 / 6], atol=1e-13)
        # Replication increases the data term, as required by raw-sum penalty.
        duplicated = models.fit_ridge(np.repeat(x, 2, axis=0), np.repeat(y, 2))
        np.testing.assert_allclose(duplicated["weights"], [.5, .4], atol=1e-13)

    def test_feature_vectors_do_not_read_future_outcomes(self):
        row = synthetic_row()
        changed = copy.deepcopy(row)
        changed.update(end_x=[999.] * 4, gain=999., improved=False,
                       oracle_action=ACTIONS[-1], future_returns=[1, 1, 1])
        for arm in models.ARMS:
            np.testing.assert_array_equal(models.vector(row, arm, maps_with_offset()),
                                          models.vector(changed, arm, maps_with_offset()))

    def test_all_start_coordinates_available_to_strong_nofp_and_widths_disclosed(self):
        row = synthetic_row()
        b = models.vector(row, "B", maps_with_offset())
        # best/current are ordinary entries 0,1; path/block fraction are structure 0,1.
        np.testing.assert_allclose(b[[0, 1, 8, 9]], row["start_x"])
        expected = {"A": 8, "B": 23, "C": 35, "D": 29, "E": 71, "D_random": 29}
        for arm, width in expected.items():
            self.assertEqual(len(models.vector(row, arm, maps_with_offset())), width)
        self.assertEqual(len(models.vector(row, "D", maps_with_offset())),
                         len(models.vector(row, "D_random", maps_with_offset())))

    def test_action_interactions_are_block_diagonal_and_crossfit_map_selected(self):
        row = synthetic_row()
        rows = [{**row, "action": a} for a in ACTIONS]
        selected = maps_with_offset(.5)
        x = models.matrix(rows, "D", maps_with_offset(100.), {row["instance"]: selected})
        expected = models.vector(row, "D", selected)
        for i in range(3):
            np.testing.assert_array_equal(x[i, :3], np.eye(3)[i])
            blocks = x[i, 3:].reshape(3, -1)
            np.testing.assert_allclose(blocks[i], expected)
            for j in set(range(3)) - {i}:
                np.testing.assert_array_equal(blocks[j], 0)

    def test_affine_raw_conversion_matches_independent_augmented_fit(self):
        rng = np.random.default_rng(12)
        xs = rng.normal(size=(12, 4))
        ys = xs + xs @ rng.normal(size=(4, 4)) * .1 + .3
        rows = [dict(state_id=str(k), action=ACTIONS[0], start_x=x.tolist(),
                     end_x=(ys[k] + (-1 if repeat == 0 else 1) * .01).tolist())
                for k, x in enumerate(xs) for repeat in range(2)]
        fitted = models.affine(rows)
        z = np.column_stack(((xs - xs.mean(0)) / xs.std(0), np.ones(len(xs))))
        penalty = np.diag([np.sqrt(.1)] * 4 + [0.])
        coef = np.linalg.lstsq(np.vstack([z, penalty]),
                              np.vstack([ys - xs, np.zeros((5, 4))]), rcond=None)[0]
        expected = xs + z @ coef
        observed = xs @ np.asarray(fitted["A"]).T + fitted["c"]
        np.testing.assert_allclose(observed, expected, atol=1e-12)

    def test_collection_preserves_neutral_trajectory_and_full_repeat_grid(self):
        instance = dict(name="synthetic_readonly_test", split="development", jobs=3, machines=3,
                        routes=[[0, 1, 2], [1, 2, 0], [2, 0, 1]],
                        durations=[[3, 2, 4], [2, 4, 3], [4, 3, 2]])
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(collection, "ROOT", Path(temp)), patch.object(collection, "RESULTS", Path(temp) / "results"):
                receipt = collection.task((instance, 0, "tabu", "synthetic_test_lock"))
                saved = collection.load(Path(temp) / receipt["file"])
                resumed = collection.task((instance, 0, "tabu", "synthetic_test_lock"))
        self.assertEqual(receipt["branches"], 30)
        self.assertTrue(resumed["resumed"])
        self.assertEqual(saved["content_hash"], collection.canonical({k: v for k, v in saved.items() if k != "content_hash"}))
        inst = Instance.from_dict(instance)
        neutral = initial_state(inst, collection.seed(instance["name"], 0, "init"))
        for point, steps in enumerate((8, 16)):
            neutral = run_action(inst, neutral, "CB_TABU", collection.seed(instance["name"], 0, point, "neutral"), steps).state
            self.assertEqual(neutral, SearchState.from_dict(saved["snapshots"][point]["state"]))
            rows = [r for r in saved["records"] if r["point"] == point]
            self.assertEqual({(r["action"], r["repeat"]) for r in rows},
                             {(a, repeat) for a in ACTIONS for repeat in range(5)})
            for row in rows:
                self.assertEqual(row["state_hash"], collection.canonical(neutral.to_dict()))
                self.assertEqual(row["observed"], features(inst, neutral))
                self.assertEqual(row["gain"], (row["start_makespan"] - row["end_makespan"]) / row["s0"])
        self.assertEqual(len({r["seed"] for r in saved["records"]}), 30)


if __name__ == "__main__":
    unittest.main()

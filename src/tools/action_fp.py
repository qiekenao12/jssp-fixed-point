"""Action-conditioned *empirical* affine dynamics (explicit M2 opt-in only).

No optimizer imports this module.  F_a is fitted to pure one-action counterfactual
transitions, not GA+LS/VND mixtures.  A small residual is not an optimality claim.
Held-out errors describe prediction quality, never global convergence guarantees.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Mapping
import numpy as np

ACTIONS = ("GA", "LS", "VND")
WEIGHTS = np.array([.35, .30, .15, .20], dtype=float)
MIN_TRANSITIONS = 24
MIN_INSTANCES = 3


def state_vector(value):
    x = np.asarray(value, dtype=float)
    if x.shape != (4,) or not np.all(np.isfinite(x)):
        raise ValueError("A search state must contain four finite values")
    return x


def weighted_norm(value):
    return float(np.sqrt(np.sum(WEIGHTS * np.asarray(value, dtype=float) ** 2)))


def pure_window(row):
    action = str(row["action"]).upper()
    if action not in ACTIONS:
        raise ValueError(f"Unknown action {action}")
    window = row["window"]
    if (window.get("unit") != "atomic_action_application" or window.get("span") != 1
            or window.get("action_sequence") != [action]):
        raise ValueError("Only a pure single atomic action window is admissible")
    return action


def collapsed_transitions(records):
    """Average repeats at one state/action; they are not independent samples."""
    groups = defaultdict(list)
    for row in records:
        action = pure_window(row)
        start = state_vector(row.get("start_x", row.get("x")))
        end = state_vector(row["end_x"])
        key = (str(row["instance"]), str(row["trajectory_id"]), str(row["state_id"]), action)
        groups[key].append((start, end))
    out = []
    for (instance, trajectory, state_id, action), pairs in sorted(groups.items()):
        if any(not np.array_equal(pairs[0][0], pair[0]) for pair in pairs):
            raise ValueError("Repeated branch transitions have different starting states")
        out.append({"instance": instance, "trajectory_id": trajectory,
                    "state_id": state_id, "action": action,
                    "start_x": pairs[0][0],
                    "end_x": np.mean([p[1] for p in pairs], axis=0)})
    return out


@dataclass(frozen=True)
class ActionFPSignal:
    action: str
    predicted_state: tuple[float, ...]
    residual: float
    difference_from_ga: float
    prediction_error: float | None
    reliability: float
    observation_interval: int
    model_ready: bool
    training_transitions: int
    training_instances: int

    def as_dict(self):
        return dict(self.__dict__)


class _Affine:
    def __init__(self, rows, ridge):
        x = np.stack([r["start_x"] for r in rows])
        y = np.stack([r["end_x"] for r in rows])
        self.center = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-6)
        self.support_radius = float(np.quantile(np.linalg.norm((x-self.center)/self.scale, axis=1), .95))
        z = np.column_stack(((x - self.center) / self.scale, np.ones(len(x))))
        penalty = np.eye(5) * float(ridge)
        penalty[-1, -1] = 0
        # Ridge on displacement shrinks towards the identity map.
        self.beta = np.linalg.solve(z.T @ z + penalty, z.T @ (y - x))
        self.A = np.eye(4) + self.beta[:4].T / self.scale
        self.c = self.beta[-1] - (self.center / self.scale) @ self.beta[:4]

    def predict(self, x):
        x = state_vector(x)
        return self.A @ x + self.c


class ActionFPModel:
    """Pooled maps with leave-one-training-instance-out reliability.

    `prediction_error` is the mean *held-out training-instance* weighted
    transition error.  It is not the unavailable error of the current forecast.
    No target-instance outcome is required by predict().
    """
    def __init__(self, ridge=1e-3, min_transitions=MIN_TRANSITIONS,
                 min_instances=MIN_INSTANCES):
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.ridge = float(ridge)
        self.min_transitions = int(min_transitions)
        self.min_instances = int(min_instances)
        self.models = {}
        self.summary = {}
        self.training_instances = ()

    def fit(self, records: Iterable[Mapping]):
        rows = collapsed_transitions(records)
        self.models, self.summary = {}, {}
        self.training_instances = tuple(sorted({r["instance"] for r in rows}))
        for action in ACTIONS:
            selected = [r for r in rows if r["action"] == action]
            instances = sorted({r["instance"] for r in selected})
            count, groups = len(selected), len(instances)
            enough = count >= self.min_transitions and groups >= self.min_instances
            errors, improvements, per_instance = [], [], []
            if enough:
                for instance in instances:
                    train = [r for r in selected if r["instance"] != instance]
                    held = [r for r in selected if r["instance"] == instance]
                    # A fold with inadequate support does not become reliable.
                    if len(train) < self.min_transitions or len({r['instance'] for r in train}) < self.min_instances:
                        continue
                    model = _Affine(train, self.ridge)
                    local = [weighted_norm(model.predict(r["start_x"]) - r["end_x"]) for r in held]
                    errors.extend(local)
                    per_instance.append(float(np.mean(local)))
                    improvements.append(float(np.mean([weighted_norm(r["end_x"] - r["start_x"]) for r in held])))
                self.models[action] = _Affine(selected, self.ridge)
            error = float(np.mean(per_instance)) if len(per_instance) == groups and groups else None
            movement = float(np.mean(improvements)) if improvements else 0.
            ready = bool(enough and error is not None)
            reliability = (1. / (1. + error / max(movement, 1e-8))) if ready else 0.
            self.summary[action] = {"training_transitions": count, "training_instances": groups,
                "prediction_error": error, "reliability": float(reliability),
                "model_ready": ready, "oof_instances": len(per_instance),
                "baseline_identity_oof_error": movement,
                "A": self.models[action].A.tolist() if action in self.models else None,
                "c": self.models[action].c.tolist() if action in self.models else None}
        return self

    def predict(self, x, action, window=None):
        action = str(action).upper()
        if action not in ACTIONS:
            raise ValueError("Unknown action")
        if window is not None:
            pure_window({"action": action, "window": window})
        x = state_vector(x)
        summary = self.summary.get(action, {})
        ready = bool(summary.get("model_ready", False))
        # Missing/unsupported models explicitly return identity + ready=False.
        predicted = self.models[action].predict(x) if ready else x.copy()
        ga_ready = bool(self.summary.get("GA", {}).get("model_ready", False))
        ga = self.models["GA"].predict(x) if ga_ready else x.copy()
        # Reliability is a heuristic support score, not a calibrated probability.
        extrapolation = max(0., float(np.linalg.norm((x-self.models[action].center)/self.models[action].scale)) - self.models[action].support_radius) if ready else 0.
        reliability = float(summary.get('reliability', 0.)) / (1.+extrapolation)
        return ActionFPSignal(action, tuple(float(v) for v in predicted),
            weighted_norm(predicted - x), weighted_norm(predicted - ga) if ga_ready and ready else 0.,
            summary.get("prediction_error"), reliability, 1,
            ready, int(summary.get("training_transitions", 0)), int(summary.get("training_instances", 0)))

    def features(self, x):
        signals = [self.predict(x, a) for a in ACTIONS]
        # Every action's candidate signals are available before choosing one.
        fp = [s.residual for s in signals] + [s.difference_from_ga for s in signals[1:]]
        reliability = [s.prediction_error or 0. for s in signals] + [s.reliability for s in signals] + [float(s.model_ready) for s in signals]
        return np.asarray(fp), np.asarray(reliability), signals

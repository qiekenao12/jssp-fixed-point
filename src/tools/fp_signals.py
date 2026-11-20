"""Explicit FP observation contracts, without assumptions of predictive value.

All generation quantities refer to actual generation timestamps. Observation
counts never share a field with generation counts. A training fit error is
deliberately not exposed as a held-out prediction error or as reliability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class FPSignal:
    """Read-only signal available before a search decision.

    ``prediction_error`` is a grouped, out-of-sample estimate when supplied by
    an independently validated model; ``None`` means unavailable. ``model_ready``
    only indicates that the model can be evaluated, not that it is reliable.
    """

    residual: Optional[float]
    prediction_error: Optional[float]
    reliability: float
    observation_interval: int
    model_ready: bool
    stagnant_generations: int
    stagnant_observations: int
    observation_count: int
    internal_model_stagnant_generations: int
    generation: int
    source_model: str
    training_fit_error: Optional[float] = None

    def __post_init__(self):
        for name in ('observation_interval', 'stagnant_generations',
                     'stagnant_observations', 'observation_count',
                     'internal_model_stagnant_generations', 'generation'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if type(self.model_ready) is not bool:
            raise TypeError('model_ready must be a bool')
        if not math.isfinite(self.reliability) or not 0 <= self.reliability <= 1:
            raise ValueError('reliability must be finite and in [0, 1]')
        for name in ('residual', 'prediction_error', 'training_fit_error'):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be None or finite and nonnegative')
        if not self.model_ready and self.reliability != 0:
            raise ValueError('an unready model cannot have positive reliability')
        if self.prediction_error is None and self.reliability != 0:
            raise ValueError('positive reliability requires held-out prediction evidence')
        if not self.source_model:
            raise ValueError('source_model must identify the estimated mapping')

    def to_dict(self):
        return asdict(self)


class FPObservationTracker:
    """Count sparse FP observations separately from Observer generations."""

    def __init__(self):
        self.observation_count = 0
        self.stagnant_observations = 0
        self.last_generation: Optional[int] = None
        self.last_best: Optional[float] = None

    def observe(self, *, generation: int, best_fitness: float,
                stagnant_generations: int, residual: Optional[float],
                model_ready: bool, source_model: str,
                internal_model_stagnant_generations: int,
                training_fit_error: Optional[float] = None,
                prediction_error: Optional[float] = None,
                reliability: float = 0.0) -> FPSignal:
        if self.last_generation is not None and generation <= self.last_generation:
            raise ValueError('FP observations require strictly increasing generations')
        if not math.isfinite(best_fitness):
            raise ValueError('best_fitness must be finite')
        interval = 0 if self.last_generation is None else generation - self.last_generation
        stagnant_observations = (
            self.stagnant_observations + 1
            if self.last_best is not None and best_fitness >= self.last_best - 1e-6
            else 0
        )
        signal = FPSignal(
            residual=residual, prediction_error=prediction_error,
            reliability=reliability, observation_interval=interval,
            model_ready=model_ready, stagnant_generations=stagnant_generations,
            stagnant_observations=stagnant_observations,
            observation_count=self.observation_count + 1,
            internal_model_stagnant_generations=internal_model_stagnant_generations,
            generation=generation, source_model=source_model,
            training_fit_error=training_fit_error,
        )
        self.last_generation = generation
        self.last_best = best_fitness
        self.stagnant_observations = stagnant_observations
        self.observation_count += 1
        return signal

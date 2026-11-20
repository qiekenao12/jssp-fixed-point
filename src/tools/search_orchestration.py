"""Shared contracts for budget-aware component-level search orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class SearchState:
    """Read-only state observed before the Agent schedules components."""

    generation: int
    max_generations: int
    diversity: float
    improvement_rate: float
    stagnant_generations: int
    best_fitness: float
    convergence_level: str = "unknown"
    component_roi: Dict[str, float] = field(default_factory=dict)
    budget_remaining: Optional[float] = None

    @property
    def progress(self) -> float:
        return (self.generation + 1) / max(self.max_generations, 1)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SearchAction:
    """An Agent decision: component, target population, and work budget."""

    component: str
    enabled: bool
    target: str
    target_size: float
    budget: float
    budget_unit: str = "work_calls"
    mode: str = "default"
    reason: str = "baseline"
    expected_roi: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class BudgetLedger:
    """Tracks requested, accepted, and consumed component work.

    The ledger is deliberately independent of the random generators.  In
    ``shadow`` mode it is an observability layer only; in ``enforce`` mode the
    accepted plan budget is the admission limit.  ``work_calls`` means one
    invocation of a component on a candidate (or one convergence check for
    FP), rather than an internal neighborhood evaluation count.
    """

    def __init__(self, total_budget: Optional[float] = None,
                 mode: str = "shadow"):
        if mode not in ("shadow", "enforce"):
            raise ValueError("budget mode must be 'shadow' or 'enforce'")
        self.mode = mode
        self.total_budget = (
            None if total_budget is None else max(0.0, float(total_budget))
        )
        # ``planned`` is retained as a compatibility alias for the historical
        # output, but now means accepted plan budget.  ``requested`` exposes
        # rejected requests separately.
        self.planned: Dict[str, float] = {}
        self.requested: Dict[str, float] = {}
        self.rejected: Dict[str, float] = {}
        self.consumed: Dict[str, float] = {}
        self.gain: Dict[str, float] = {}
        self.actions = 0
        self.rejected_actions = 0
        self.action_log = []

    @property
    def consumed_total(self) -> float:
        return sum(self.consumed.values())

    @property
    def allocated_total(self) -> float:
        """Budget reserved by accepted plans."""
        return sum(self.planned.values())

    @property
    def remaining(self) -> Optional[float]:
        if self.total_budget is None:
            return None
        return max(0.0, self.total_budget - self.allocated_total)

    def authorize(self, action: SearchAction) -> bool:
        self.actions += 1
        requested = max(0.0, float(action.budget))
        self.requested[action.component] = (
            self.requested.get(action.component, 0.0) + requested
        )
        accepted = bool(action.enabled)
        rejection_reason = None
        if not action.enabled:
            rejection_reason = "disabled"
        elif (self.mode == "enforce" and self.remaining is not None
              and requested > self.remaining):
            self.rejected_actions += 1
            self.rejected[action.component] = (
                self.rejected.get(action.component, 0.0) + requested
            )
            rejection_reason = "budget_exhausted"
            accepted = False
        if accepted:
            self.planned[action.component] = (
                self.planned.get(action.component, 0.0) + requested
            )
        self.action_log.append({
            "action": action.to_dict(),
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "execution_match": None,
        })
        # A run can contain hundreds of generations. Keep diagnostics useful
        # without allowing the orchestration trace to dominate result files.
        if len(self.action_log) > 500:
            del self.action_log[:-500]
        return accepted

    def record(self, component: str, work: float, gain: float = 0.0):
        work = max(0.0, float(work))
        gain = max(0.0, float(gain))
        self.consumed[component] = (
            self.consumed.get(component, 0.0) + work
        )
        self.gain[component] = (
            self.gain.get(component, 0.0) + gain
        )
        # Match positive execution with the most recent accepted plan for the
        # component. Zero-work observations are intentionally left pending:
        # they may correspond to a generation without a VND action.
        if work > 0.0:
            for item in reversed(self.action_log):
                action = item.get("action", {})
                if (action.get("component") == component
                        and item.get("accepted")
                        and item.get("execution_match") is None):
                    item["execution_match"] = {
                        "work": work, "gain": gain,
                    }
                    break

    def roi(self, component: str) -> float:
        return self.gain.get(component, 0.0) / max(
            self.consumed.get(component, 0.0), 1e-12
        )

    def summary(self) -> Dict[str, object]:
        components = sorted(
            set(self.requested) | set(self.planned) | set(self.consumed)
        )
        matched = sum(
            1 for item in self.action_log if item.get("execution_match")
        )
        return {
            "mode": self.mode,
            "total_budget": self.total_budget,
            "remaining": self.remaining,
            "allocated": self.allocated_total,
            "requested": dict(self.requested),
            "planned": dict(self.planned),
            "rejected": dict(self.rejected),
            "consumed": dict(self.consumed),
            "gain": dict(self.gain),
            "roi": {name: self.roi(name) for name in components},
            "actions": self.actions,
            "rejected_actions": self.rejected_actions,
            "execution_matches": matched,
            "action_log": list(self.action_log),
        }

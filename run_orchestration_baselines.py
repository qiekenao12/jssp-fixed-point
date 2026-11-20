#!/usr/bin/env python3
"""Controlled scheduling-policy experiment for the Agent-FP design.

This script compares component scheduling policies while keeping the same
instance parser, seeds, population, and optimizer implementation.  It is an
experiment entry point, not part of the default performance runs.

Policies:
  fixed:      fixed component permissions (no Agent decisions)
  stagnation: trigger VND from phenotype stagnation only (no FP signal)
  agent:      the current Agent + FP orchestration
  budget:     the same Agent policy with an enforced work-call budget

Examples:
  JSSP_NUM_RUNS=5 ORCH_INSTANCES=ft06,la21 python3 run_orchestration_baselines.py
  JSSP_NUM_RUNS=5 ORCH_BUDGET=80 python3 run_orchestration_baselines.py
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from run_ablation_test import parse_jobshop1, seed_run, TEST_INSTANCES
from tools.benchmark_data import BKS, validate_makespan
from tools.controller import Controller
from tools.jssp_core import JSSPInstance
from tools.meta_scheduler import MetaScheduler


POLICIES = {
    "fixed": {
        "adaptive_params": False, "fixed_point": False,
        "agent_commander": False, "budget_mode": "shadow",
    },
    "stagnation": {
        "adaptive_params": False, "fixed_point": False,
        "agent_commander": False, "budget_mode": "shadow",
    },
    "agent": {
        "adaptive_params": True, "fixed_point": True,
        "agent_commander": True, "budget_mode": "shadow",
    },
    "budget": {
        "adaptive_params": True, "fixed_point": True,
        "agent_commander": True, "budget_mode": "enforce",
    },
}


def make_instance(data):
    return JSSPInstance(
        num_jobs=data["num_jobs"], num_machines=data["num_machines"],
        processing_times=data["processing_times"],
        machine_sequences=data["machine_sequences"],
        upper_bound=data["upper_bound"],
    )


def run_policy(instance, name, policy, instance_name, run_index, budget,
               enforce_budget=False):
    seed_run(instance_name, run_index)
    scheduler = MetaScheduler(instance)
    if policy["adaptive_params"]:
        scheduler.controller = Controller(instance.num_jobs, instance.num_machines)
        scheduler.controller.set_max_generations(scheduler.max_generations)
        scheduler.controller.set_early_stop_stagnant(scheduler.early_stop_stagnant)
    started = time.perf_counter()
    result = scheduler.optimize(
        use_adaptive_params=policy["adaptive_params"],
        use_fixed_point=policy["fixed_point"],
        use_local_search=True, use_vnd=True,
        use_terminal_refinement=True,
        use_agent_commander=policy["agent_commander"],
        budget_mode=("enforce" if enforce_budget else policy["budget_mode"]),
        search_budget=(budget if enforce_budget or name == "budget" else None),
        vnd_schedule=("fixed" if name == "fixed" else
                      "stagnation" if name == "stagnation" else "none"),
    )
    elapsed = time.perf_counter() - started
    return {
        "makespan": int(result.best_makespan),
        "gap": (result.best_makespan - BKS[instance_name]) / BKS[instance_name] * 100,
        "time": elapsed,
        "orchestration": result.orchestration,
    }


def main():
    runs = int(os.getenv("JSSP_NUM_RUNS", "5"))
    budget = float(os.getenv("ORCH_BUDGET", "100"))
    requested = os.getenv("ORCH_INSTANCES", ",".join(TEST_INSTANCES)).split(",")
    requested = [name.strip() for name in requested if name.strip()]
    data = parse_jobshop1(os.path.join(os.path.dirname(__file__), "jobshop1.txt"))
    selected = [name for name in requested if name in data and name in BKS]
    results = defaultdict(list)
    for name in selected:
        instance = make_instance(data[name])
        for policy_name, policy in POLICIES.items():
            print(f"Running {policy_name} on {name} x{runs}...", flush=True)
            for run in range(runs):
                row = run_policy(instance, policy_name, policy, name, run, budget)
                validate_makespan(name, row["makespan"], BKS[name])
                results[policy_name].append({"instance": name, "run": run, **row})

    summary = {}
    for policy_name, rows in results.items():
        gaps = [row["gap"] for row in rows]
        summary[policy_name] = {
            "runs": len(rows), "avg_gap": float(np.mean(gaps)),
            "std_gap": float(np.std(gaps)), "median_gap": float(np.median(gaps)),
            "bks_hit_rate": float(np.mean([row["makespan"] == BKS[row["instance"]] for row in rows])),
            "avg_time": float(np.mean([row["time"] for row in rows])),
            "avg_consumed": _average_component_value(rows, "consumed"),
            "avg_gain": _average_component_value(rows, "gain"),
            "avg_roi": _average_component_value(rows, "roi"),
        }
    output = {
        "experiment": "orchestration_baselines", "num_runs": runs,
        "budget_unit": "work_calls", "enforced_budget": budget,
        "instances": selected, "policies": POLICIES,
        "summary": summary, "rows": dict(results),
        "note": "The budget policy is an additional controlled experiment; default performance runs remain shadow mode.",
    }
    path = os.path.join(os.path.dirname(__file__), "results", "orchestration_baselines.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {path}")


def _average_component_value(rows, field):
    totals = defaultdict(float)
    count = max(1, len(rows))
    for row in rows:
        for component, value in (row.get("orchestration", {}).get(field, {}) or {}).items():
            totals[component] += float(value)
    return {component: value / count for component, value in totals.items()}


if __name__ == "__main__":
    main()

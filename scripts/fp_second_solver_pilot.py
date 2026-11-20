#!/usr/bin/env python3
"""Small exposed-instance timing; no final data and no model training."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from tools.independent_tabu_jssp import ACTIONS, features, initial_state, read_jsplib, run_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="projects/analysis/fp_decision_study/solver_pilot.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refuse to overwrite pilot evidence")
    started = time.perf_counter()
    rows = []
    for name in ("la01", "la11", "abz5"):
        inst = read_jsplib(Path("projects/baselines/jsplib/instances") / name)
        for seed in range(3):
            state = initial_state(inst, seed)
            state = run_action(inst, state, "CB_TABU", seed + 100, steps=8).state
            feature_start = time.perf_counter()
            observed = features(inst, state)
            feature_elapsed = time.perf_counter() - feature_start
            for action in ACTIONS:
                for repeat in range(5):
                    result = run_action(inst, state, action, 1000 + seed * 10 + repeat)
                    rows.append({"instance": name, "jobs": inst.jobs, "machines": inst.machines,
                                 "state_seed": seed, "repeat": repeat, "action": action,
                                 "current": state.current_makespan, "best": state.best_makespan,
                                 "dynamics": observed["dynamics"], "feature_seconds": feature_elapsed,
                                 **{k: v for k, v in result.to_dict().items() if k not in ("state", "anytime")}})
    grouped = {}
    for name in ("la01", "la11", "abz5"):
        for action in ACTIONS:
            g = [r for r in rows if r["instance"] == name and r["action"] == action]
            grouped[name + ":" + action] = {"branches": len(g), "mean_seconds": statistics.mean(r["elapsed_seconds"] for r in g),
                "mean_decodes": statistics.mean(r["decode_evaluations"] for r in g),
                "mean_moves": statistics.mean(r["accepted_steps"] for r in g),
                "early_endings": sum(r["accepted_steps"] < r["requested_steps"] for r in g)}
    means = [g["mean_seconds"] for g in grouped.values()]
    report = {"schema": "independent_tabu_pilot_v1", "scope": "exposed development instances only, timing not a policy-selection result",
              "source_sha256": hashlib.sha256(Path("src/tools/independent_tabu_jssp.py").read_bytes()).hexdigest(),
              "branch_count": len(rows), "total_seconds": time.perf_counter() - started, "groups": grouped,
              "design_projection": {"formula": "20 instances * 20 seeds * 2 states * 3 actions * 5 repeats",
                  "branches": 12000, "projected_seconds_using_all_pilot_branch_mean": 12000 * statistics.mean(r["elapsed_seconds"] for r in rows),
                  "projected_seconds_using_slowest_instance_action_mean": 12000 * max(means),
                  "limitations": "not a guarantee: final instance sizes and termination differ; excludes state generation, features, file I/O, fitting and independent schedule validation"},
              "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

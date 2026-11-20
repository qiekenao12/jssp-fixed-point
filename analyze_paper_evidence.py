#!/usr/bin/env python3
"""Audit frozen Protocol Q runs and compute instance-level paired evidence.

This analysis never invokes or changes an optimizer. Positive differences mean
the comparator has a higher value than Agent-FP; for Gap/AUC this favors FP.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import math

import numpy as np

CORE = ("agent_nofp", "fixed", "stagnation")
ABLATIONS = ("no_feedback", "no_ls", "no_vnd", "no_terminal")
METRICS = ("final_gap", "anytime_gap_auc", "actual_queries", "wall_seconds")


def signed_rank(values):
    """Exact two-sided Wilcoxon random-sign distribution, midranks for ties.

    Zeros are discarded (Wilcox convention). Integer doubled ranks permit an
    exact dynamic program even when absolute differences are tied. This tests
    instance means, never individual states or pooled seeds.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.abs(data) > 1e-12]
    if len(data) == 0:
        return {"p": 1.0, "rank_biserial": 0.0, "nonzero_instances": 0}
    absolute = np.abs(data)
    order = np.argsort(absolute, kind="stable")
    ranks = np.empty(len(data), dtype=int)
    i = 0
    while i < len(data):
        j = i + 1
        while j < len(data) and absolute[order[j]] == absolute[order[i]]:
            j += 1
        ranks[order[i:j]] = i + 1 + j  # twice mean of ranks i+1,...,j
        i = j
    total = int(ranks.sum())
    positive = int(ranks[data > 0].sum())
    ways = [0] * (total + 1)
    ways[0] = 1
    current = 0
    for rank in ranks:
        rank = int(rank)
        for subtotal in range(current, -1, -1):
            ways[subtotal + rank] += ways[subtotal]
        current += rank
    extreme = abs(2 * positive - total)
    count = sum(ways[s] for s in range(total + 1) if abs(2 * s - total) >= extreme)
    return {"p": min(1.0, count / (2 ** len(data))),
            "rank_biserial": (2 * positive - total) / total,
            "nonzero_instances": len(data)}


def hierarchical_ci(paired_differences, *, repetitions=10000, seed=20261017):
    """Resample instances, then paired seeds within each sampled instance."""
    if not paired_differences:
        return None
    arrays = [np.asarray(row, dtype=float) for row in paired_differences]
    if any(len(a) == 0 for a in arrays):
        raise ValueError("Empty paired instance")
    rng = np.random.default_rng(seed)
    selected = rng.integers(len(arrays), size=(repetitions, len(arrays)))
    sums = np.zeros(repetitions)
    for position in range(len(arrays)):
        for index, values in enumerate(arrays):
            mask = selected[:, position] == index
            count = int(mask.sum())
            if count:
                draws = rng.integers(len(values), size=(count, len(values)))
                sums[mask] += values[draws].mean(axis=1)
    return np.quantile(sums / len(arrays), [0.025, 0.975]).tolist()


def holm(records, field="p", output="holm_p"):
    ordered = sorted(records, key=lambda row: row[field])
    running = 0.0
    for i, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * row[field]))
        row[output] = running


def trace_points(row):
    points = [(int(p["query"]), float(p["makespan"])) for p in row["trace"]]
    if not points or points[0][0] != 1:
        raise ValueError("Incumbent trace must start at the first actual query")
    for i, (q, value) in enumerate(points):
        if not math.isfinite(value) or q > row["actual_queries"]:
            raise ValueError("Invalid trace value or query")
        if i and (q <= points[i-1][0] or value > points[i-1][1]):
            raise ValueError("Trace is not an increasing-query incumbent trace")
    return points


def incumbent_at(points, query):
    index = int(np.searchsorted([p[0] for p in points], query, side="right")) - 1
    if index < 0:
        raise ValueError("No actual incumbent at this position")
    return points[index][1]


def gap_auc(row, reference, horizon=None):
    """Mean Gap after queries 1,...,horizon; natural stops carry incumbent."""
    horizon = int(row["budget"] if horizon is None else horizon)
    points = trace_points(row)
    total = 0.0
    for i, (q, value) in enumerate(points):
        if q > horizon:
            break
        next_q = points[i+1][0] if i+1 < len(points) else horizon + 1
        total += (min(next_q, horizon + 1) - q) * (value / reference - 1) * 100
    return total / horizon


def audit_row(row, reference):
    for field in ("final_gap", "best_gap", "wall_seconds", "anytime_gap_auc"):
        if not math.isfinite(float(row[field])):
            raise ValueError(f"Nonfinite {field}")
    q = int(row["actual_queries"])
    if not (0 < q <= int(row["budget"])):
        raise ValueError("Actual queries are outside the strict budget")
    if sum(row["phase_queries"].values()) != q:
        raise ValueError("Phase queries do not reconcile")
    points = trace_points(row)
    makespan = points[-1][1]
    if makespan != row["best_makespan"] or row["final_makespan"] != makespan:
        raise ValueError("Final result loses an evaluated incumbent")
    expected = (makespan / reference - 1) * 100
    for field in ("best_gap", "final_gap"):
        if abs(expected - row[field]) > 1e-8:
            raise ValueError(f"{field} disagrees with trace/reference")
    if not row.get("termination_reason"):
        raise ValueError("Missing termination reason")
    if row["wall_seconds"] < 0 or row["actual_decodes"] < 0:
        raise ValueError("Invalid computational cost")
    # Direct decoding for critical paths and final output can exceed objective
    # cache misses: actual_decodes is deliberately not bounded by query count.
    if "anytime_gap_auc" in row and abs(gap_auc(row, reference) - row["anytime_gap_auc"]) > 1e-6:
        raise ValueError("AUC convention differs from query-after-evaluation convention")


def audit_controls(row, metadata):
    controls = metadata.get("protocol", {})
    if row["vnd_queries"] > controls.get("vnd_total_queries", 8000):
        raise ValueError("VND cumulative query cap violated")
    if any(call["queries"] > controls.get("vnd_call_queries", 500)
           for call in row.get("vnd_call_log", [])):
        raise ValueError("VND per-call query cap violated")
    policy = row["policy"]
    if policy == "no_vnd" and row["vnd_queries"] != 0:
        raise ValueError("No-VND still executes VND")
    if policy == "no_ls" and row["ls_queries"] != 0:
        raise ValueError("No-LS still executes ordinary LS")
    if policy == "no_terminal" and (row["terminal_queries"] != 0 or row["sa_queries"] != 0):
        raise ValueError("No-terminal still executes terminal/SA")
    if policy in ("agent_fp", "agent_nofp"):
        if row.get("vnd_decision_generations") != row.get("fp_check_generations"):
            raise ValueError("Agent-FP and Agent-NoFP do not share decision opportunities")


def paired_analysis(rows, budgets, instances, *, repetitions=10000):
    lookup = {(r["instance"], int(r["seed"]), r["policy"], int(r["budget"])): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate instance/seed/policy/budget")
    comparisons, common_rows, per_instance = [], [], []
    available_policies = {row["policy"] for row in rows}
    for budget in budgets:
        for comparator in (*CORE, *ABLATIONS):
            if comparator not in available_policies:
                continue
            matrices = {m: [] for m in (*METRICS, "common_query_auc", "common_endpoint_gap")}
            included = []
            for name in instances:
                seeds_a = {s for n, s, p, b in lookup if n == name and p == "agent_fp" and b == budget}
                seeds_b = {s for n, s, p, b in lookup if n == name and p == comparator and b == budget}
                if seeds_a != seeds_b:
                    raise ValueError(f"Unpaired seeds: {name}/{budget}/{comparator}")
                if not seeds_a:
                    continue
                differences = defaultdict(list)
                for seed in sorted(seeds_a):
                    full = lookup[name, seed, "agent_fp", budget]
                    other = lookup[name, seed, comparator, budget]
                    if full["initial_fingerprint"] != other["initial_fingerprint"]:
                        raise ValueError("Paired initialization mismatch")
                    reference = full["bks"]
                    if reference != other["bks"]:
                        raise ValueError("Paired reference mismatch")
                    for metric in METRICS:
                        left = other[metric] if metric != "anytime_gap_auc" else gap_auc(other, reference)
                        right = full[metric] if metric != "anytime_gap_auc" else gap_auc(full, reference)
                        differences[metric].append(left - right)
                    horizon = min(full["actual_queries"], other["actual_queries"])
                    differences["common_query_auc"].append(gap_auc(other, reference, horizon) - gap_auc(full, reference, horizon))
                    endpoint = (incumbent_at(trace_points(other), horizon) - incumbent_at(trace_points(full), horizon)) / reference * 100
                    differences["common_endpoint_gap"].append(endpoint)
                    positions = sorted({1, horizon, *[q for q, _ in trace_points(full) if q <= horizon],
                                        *[q for q, _ in trace_points(other) if q <= horizon]})
                    for q in positions:
                        common_rows.append({"instance": name, "seed": seed, "budget": budget,
                                            "comparator": comparator, "query": q, "common_horizon": horizon,
                                            "agent_fp_gap": (incumbent_at(trace_points(full), q) / reference - 1) * 100,
                                            "comparator_gap": (incumbent_at(trace_points(other), q) / reference - 1) * 100})
                included.append(name)
                for metric, values in differences.items():
                    matrices[metric].append(values)
                    per_instance.append({"instance": name, "budget": budget, "comparator": comparator,
                                         "metric": metric, "paired_seeds": len(values),
                                         "comparator_minus_agent_fp": float(np.mean(values))})
            if not included:
                continue
            for metric, matrix in matrices.items():
                means = np.asarray([np.mean(x) for x in matrix])
                stats = signed_rank(means)
                comparisons.append({"budget": budget, "comparator": comparator, "metric": metric,
                                    "family": "core" if comparator in CORE else "ablation",
                                    "instances": len(included), "instance_names": included,
                                    "paired_seeds_per_instance": [len(a) for a in matrix],
                                    "mean_difference": float(means.mean()),
                                    "hierarchical_bootstrap_95ci": hierarchical_ci(matrix, repetitions=repetitions),
                                    "instance_win_tie_loss": [int(sum(means > 1e-12)), int(sum(abs(means) <= 1e-12)), int(sum(means < -1e-12))],
                                    "median_instance_difference": float(np.median(means)), **stats})
    # Separate prespecified families by metric; all three budgets corrected
    # jointly, never one favorable budget selected after seeing the results.
    for family in ("core", "ablation"):
        for metric in (*METRICS, "common_query_auc", "common_endpoint_gap"):
            holm([r for r in comparisons if r["family"] == family and r["metric"] == metric])
    return comparisons, per_instance, common_rows


def title_decision(comparisons, complete_final_test):
    values = [c for c in comparisons if c["comparator"] == "agent_nofp" and c["metric"] == "final_gap"]
    if not complete_final_test or not values:
        return {"case": "UNRESOLVED", "title": "State-Aware Hybrid Search Orchestration under Strict Computational Budgets",
                "reason": "Independent complete final-test evidence is required; absence of significance is not equivalence."}
    if all(c["mean_difference"] > 0 and c["hierarchical_bootstrap_95ci"][0] > 0 and c["holm_p"] < .05 for c in values):
        return {"case": "1", "title": "FP may remain a main contribution", "reason": "Positive independent effect with CI and multiplicity support at all frozen budgets."}
    if all(c["mean_difference"] < 0 and c["hierarchical_bootstrap_95ci"][1] < 0 for c in values):
        return {"case": "3", "title": "State-Aware Hybrid Search Orchestration under Strict Computational Budgets", "reason": "FP is consistently worse; retain as a negative result."}
    return {"case": "2_OR_MIXED", "title": "State-Aware Hybrid Search Orchestration under Strict Computational Budgets",
            "reason": "No stable supported independent FP advantage. Treat FP as auxiliary; equivalence is not established without a prespecified margin."}


def write_csv(path, records):
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    metadata = json.loads((args.folder / "protocol.json").read_text())
    paths = sorted(args.folder.glob("*.json"))
    rows = []
    for path in paths:
        if path.name in ("protocol.json", "analysis.json", "freeze.json"):
            continue
        payload = json.loads(path.read_text())
        candidates = payload if isinstance(payload, list) else payload.get("rows", [payload])
        for row in candidates:
            if isinstance(row, dict) and "actual_queries" in row:
                rows.append(row)
    if not rows:
        raise SystemExit("No Protocol Q result rows; no inference produced.")
    hashes = {r["protocol_hash"] for r in rows}
    if len(hashes) != 1:
        raise ValueError("Mixed protocol/source hashes")
    if metadata.get("protocol_hash") and hashes != {metadata["protocol_hash"]}:
        raise ValueError("Result hash does not match the frozen protocol")
    for row in rows:
        audit_row(row, row["bks"])
        audit_controls(row, metadata)
    budgets = sorted({int(r["budget"]) for r in rows})
    instances = sorted({r["instance"] for r in rows})
    comparisons, per_instance, common = paired_analysis(rows, budgets, instances, repetitions=args.bootstrap)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance"], row["policy"], row["budget"]].append(row)
    benchmark = []
    for (name, policy, budget), group in sorted(grouped.items()):
        benchmark.append({"instance": name, "policy": policy, "budget": budget, "bks": group[0]["bks"],
                          "runs": len(group), "best_makespan": min(r["best_makespan"] for r in group),
                          "mean_makespan": float(np.mean([r["best_makespan"] for r in group])),
                          **{f"mean_{key}": float(np.mean([r[key] for r in group])) for key in (*METRICS, "actual_decodes", "vnd_queries", "ls_queries", "terminal_queries")},
                          "natural_early_stops": sum(r["actual_queries"] < r["budget"] for r in group)})
    requested_instances = metadata.get("instances", instances)
    requested_budgets = metadata.get("budgets", budgets)
    requested_policies = metadata.get("policies", sorted({r["policy"] for r in rows}))
    requested_runs = int(metadata.get("runs", 0))
    expected = len(requested_instances) * len(requested_budgets) * len(requested_policies) * requested_runs
    expected_groups = {(n, p, int(b)) for n in requested_instances
                       for p in requested_policies for b in requested_budgets}
    complete = (expected > 0 and expected == len(rows)
                and set(grouped) == expected_groups
                and all(len(g) == requested_runs for g in grouped.values()))
    final_test = metadata.get("split") == "final_test"
    output = {"analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol_hash": next(iter(hashes)), "split": metadata.get("split"), "rows": len(rows),
              "expected_rows": expected, "complete": complete, "instances": instances, "budgets": budgets,
              "difference_direction": "comparator minus Agent-FP; positive Gap/AUC favors Agent-FP",
              "statistical_unit": "instance means; hierarchical resampling preserves paired seeds",
              "multiplicity": "Holm across all budgets and comparators within each core/ablation metric family",
              "comparisons": comparisons, "benchmark": benchmark,
              "title_decision": title_decision(comparisons, complete and final_test),
              "limitation": "Validation/smoke results cannot establish a final-test contribution. Nonsignificance does not prove equivalence."}
    (args.folder / "analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(args.folder / "paired_comparisons.csv", comparisons)
    write_csv(args.folder / "paired_per_instance.csv", per_instance)
    write_csv(args.folder / "common_query_positions.csv", common)
    write_csv(args.folder / "benchmark.csv", benchmark)
    print(json.dumps({k: output[k] for k in ("rows", "expected_rows", "complete", "title_decision")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

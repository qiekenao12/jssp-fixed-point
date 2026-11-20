#!/usr/bin/env python3
"""Read-only M2 decision audit; no solver, fitting, target selection, or locked tests.

Saved B/E probabilities are replayed, not refitted. Two branch repetitions are
averaged before a state decision. Oracle labels are descriptive future outcomes
and are never passed to a selector. All new intervals are exploratory.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ACTIONS = ("GA", "LS", "VND")
TOL = 1e-12
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return float(np.mean(values))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def best_set(gains):
    return [a for a, gain in zip(ACTIONS, gains) if gain >= max(gains) - TOL]


def hierarchical_interval(states, field, seed=20261019, repetitions=10000):
    """Instance -> paired trajectory resampling; all states stay in trajectory."""
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in states:
        groups[row["instance"]][row["trajectory_id"]].append(row[field])
    values = [np.asarray([mean(rows) for _, rows in sorted(trajectories.items())])
              for _, trajectories in sorted(groups.items())]
    instance_means = np.asarray([mean(value) for value in values])
    rng = np.random.default_rng(seed)
    instance_draws = rng.integers(len(values), size=(repetitions, len(values)))
    sampled = np.empty_like(instance_draws, dtype=float)
    for slot in range(len(values)):
        for index, value in enumerate(values):
            mask = instance_draws[:, slot] == index
            n = int(mask.sum())
            if n:
                sampled[mask, slot] = np.mean(value[rng.integers(len(value), size=(n, len(value)))], axis=1)
    draws = sampled.mean(axis=1)
    return {"mean": mean(instance_means), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "instance_win_tie_loss": [int(np.sum(instance_means > TOL)),
                                      int(np.sum(np.abs(instance_means) <= TOL)),
                                      int(np.sum(instance_means < -TOL))],
            "instances": len(values), "repetitions": repetitions,
            "unit": "state fraction" if field == "VND_strictly_nonoptimal" else "gain/s0",
            "inference": "exploratory paired instance->trajectory bootstrap; no states/repeats treated independent"}


def collect(split):
    analysis_path = ROOT / "projects/analysis" / f"fp_v2_{split}_analysis.json"
    analysis = json.loads(analysis_path.read_text())
    selected = (analysis["configurations"][analysis["selected_config_index"]]
                if split == "development" else analysis)
    predictions = {(r["state_id"], r["action"], r["repeat"]): r for r in selected["predictions"]}
    states, branches, sources = [], [], {str(analysis_path.relative_to(ROOT)): sha(analysis_path)}
    for path in sorted((ROOT / "results" / f"fp_v2_{split}").glob("*.json")):
        doc = json.loads(path.read_text())
        if "records" not in doc:
            continue
        sources[str(path.relative_to(ROOT))] = sha(path)
        groups = collections.defaultdict(dict)
        for branch in doc["records"]:
            key = (branch["state_id"], branch["action"], branch["repeat"])
            pred = predictions.pop(key)
            expected = (branch["start_makespan"] - branch["end_makespan"]) / branch["s0"]
            assert abs(branch["gain"] - expected) < TOL
            assert abs(pred["gain"] - expected) < TOL
            assert bool(branch["improved"]) == bool(expected > TOL)
            groups[branch["state_id"]][branch["action"], branch["repeat"]] = branch, pred
            branches.append({"split": split, **{k: branch[k] for k in (
                "instance", "trajectory_id", "state_id", "run", "generation", "action", "repeat",
                "s0", "start_makespan", "end_makespan", "gain", "improved", "queries", "decodes", "wall_seconds")},
                "absolute_improvement": branch["start_makespan"] - branch["end_makespan"],
                "B_probability": pred["probabilities"]["B"], "E_probability": pred["probabilities"]["E"]})
        for state_id, records in sorted(groups.items()):
            assert set(records) == {(a, r) for a in ACTIONS for r in (0, 1)}
            original = records["GA", 0][0]
            row = {"split": split, "instance": doc["instance"], "trajectory_id": doc["trajectory_id"],
                   "state_id": state_id, "run": doc["run"], "generation": original["generation"],
                   "s0": doc["s0"], "start_makespan": original["start_makespan"],
                   "stagnant_generations": original["real_stagnant_generations"], "phase": original["phase"],
                   "recent_main_gain_mean": mean(original["recent_real_gains"]),
                   **{f"x{i}": value for i, value in enumerate(original["start_x"])}}
            gains = np.asarray([[records[a, repeat][0]["gain"] for a in ACTIONS] for repeat in (0, 1)])
            avg = gains.mean(axis=0)
            sets = [best_set(g) for g in gains]
            avg_set = best_set(avg)
            for ai, action in enumerate(ACTIONS):
                rr = [records[action, repeat][0] for repeat in (0, 1)]
                pp = [records[action, repeat][1] for repeat in (0, 1)]
                for repeat in (0, 1):
                    row[f"{action}_gain_r{repeat}"] = rr[repeat]["gain"]
                    row[f"{action}_absolute_r{repeat}"] = rr[repeat]["start_makespan"] - rr[repeat]["end_makespan"]
                row.update({f"{action}_gain_mean": float(avg[ai]),
                            f"{action}_absolute_mean": float(avg[ai] * doc["s0"]),
                            f"{action}_repeat_gain_difference_r1_minus_r0": float(gains[1, ai] - gains[0, ai]),
                            f"{action}_repeat_gain_abs_difference": float(abs(gains[1, ai] - gains[0, ai])),
                            f"{action}_empirical_improvement_probability": mean([r["improved"] for r in rr]),
                            f"{action}_B_probability": mean([p["probabilities"]["B"] for p in pp]),
                            f"{action}_E_probability": mean([p["probabilities"]["E"] for p in pp])})
                for cost in ("queries", "decodes", "wall_seconds"):
                    row[f"{action}_{cost}_mean"] = mean([r[cost] for r in rr])
            oracle = float(max(avg))
            row.update({"oracle_action": avg_set[0], "oracle_best_set": "|".join(avg_set),
                        "oracle_unique": int(len(avg_set) == 1), "oracle_gain": oracle,
                        "oracle_absolute": oracle * doc["s0"], "fixed_VND_gain": float(avg[2]),
                        "fixed_VND_absolute": float(avg[2] * doc["s0"]),
                        "oracle_minus_VND": float(oracle - avg[2]),
                        "any_action_improves": int(oracle > TOL),
                        "VND_strictly_nonoptimal": int("VND" not in avg_set),
                        "VND_uniquely_best": int(avg_set == ["VND"]),
                        "VND_tied_best": int("VND" in avg_set and len(avg_set) > 1),
                        "all_action_means_tied": int(len(avg_set) == 3),
                        "all_action_means_zero": int(oracle <= TOL),
                        "repeat_oracle_r0": sets[0][0], "repeat_oracle_r1": sets[1][0],
                        "repeat_oracle_argmax_agrees": int(sets[0][0] == sets[1][0]),
                        "repeat_best_sets_overlap": int(bool(set(sets[0]) & set(sets[1]))),
                        "repeat_both_have_positive_outcomes": int(min(gains.max(axis=1)) > TOL),
                        "repeat_same_unique_oracle": int(len(sets[0]) == len(sets[1]) == 1 and sets[0] == sets[1]),
                        "alternative_beats_VND_in_both_repeats": int(any(np.all(gains[:, i] > gains[:, 2] + TOL) for i in (0, 1)))})
            cross = mean([gains[1, ACTIONS.index(sets[0][0])], gains[0, ACTIONS.index(sets[1][0])]])
            row["cross_repeat_oracle_gain"] = cross
            row["cross_repeat_oracle_minus_VND"] = cross - float(avg[2])
            # An optimistic VND-preserving tie rule isolates tie damage from repeat-selection noise.
            tie_vnd_choices = ["VND" if "VND" in s else s[0] for s in sets]
            cross_preserve = mean([gains[1, ACTIONS.index(tie_vnd_choices[0])], gains[0, ACTIONS.index(tie_vnd_choices[1])]])
            row["cross_repeat_oracle_VND_ties_gain"] = cross_preserve
            row["cross_repeat_oracle_VND_ties_minus_VND"] = cross_preserve - float(avg[2])
            probability = np.asarray([row[f"{a}_empirical_improvement_probability"] for a in ACTIONS])
            probability_set = best_set(probability)
            row.update({"empirical_probability_best_set": "|".join(probability_set),
                        "empirical_probability_choice": probability_set[0],
                        "empirical_probability_choice_gain": float(avg[ACTIONS.index(probability_set[0])]),
                        "empirical_probability_gain_regret": oracle - float(avg[ACTIONS.index(probability_set[0])]),
                        "empirical_probability_set_excludes_gain_oracle": int(not(set(probability_set) & set(avg_set))),
                        "empirical_probability_tied_but_gain_differs": int(len(probability_set) > 1 and np.ptp(avg[[ACTIONS.index(a) for a in probability_set]]) > TOL),
                        "any_probability_gain_pair_inversion": int(any((probability[i] - probability[j]) * (avg[i] - avg[j]) < -TOL for i in range(3) for j in range(i)))})
            for variant in ("B", "E"):
                scores = [row[f"{a}_{variant}_probability"] for a in ACTIONS]
                assert all(abs(records[a, 0][1]["probabilities"][variant] - records[a, 1][1]["probabilities"][variant]) < TOL for a in ACTIONS)
                chosen = int(np.argmax(scores))
                row.update({f"{variant}_action": ACTIONS[chosen], f"{variant}_gain": float(avg[chosen]),
                            f"{variant}_absolute": float(avg[chosen] * doc["s0"]),
                            f"{variant}_regret": oracle - float(avg[chosen]),
                            f"{variant}_is_optimal": int(ACTIONS[chosen] in avg_set),
                            f"{variant}_VND_score_margin": float(scores[2] - max(scores[:2]))})
            row.update({"E_minus_B": row["E_gain"] - row["B_gain"],
                        "B_minus_VND": row["B_gain"] - row["fixed_VND_gain"],
                        "E_minus_VND": row["E_gain"] - row["fixed_VND_gain"],
                        "E_changes_B_action": int(row["E_action"] != row["B_action"]),
                        "E_corrects_B_wrong": int(not row["B_is_optimal"] and row["E_is_optimal"]),
                        "E_damages_B_right": int(row["B_is_optimal"] and not row["E_is_optimal"]),
                        "E_improves_B_gain": int(row["E_gain"] > row["B_gain"] + TOL),
                        "E_reduces_B_gain": int(row["E_gain"] < row["B_gain"] - TOL)})
            states.append(row)
    assert not predictions
    assert len(states) == {"development": 2880, "validation": 2160}[split]
    assert len({r["instance"] for r in states}) == {"development": 16, "validation": 12}[split]
    for variant in ("B", "E"):
        assert abs(mean([r[f"{variant}_gain"] for r in states]) - selected["summary"]["macro_instance_trajectory_metrics"][variant]["selected_gain"]) < TOL
    return states, branches, sources


AGG_FIELDS = ("oracle_gain", "fixed_VND_gain", "oracle_minus_VND", "B_gain", "E_gain", "B_regret", "E_regret",
              "B_minus_VND", "E_minus_VND", "E_minus_B", "cross_repeat_oracle_gain", "cross_repeat_oracle_minus_VND",
              "cross_repeat_oracle_VND_ties_gain", "cross_repeat_oracle_VND_ties_minus_VND",
              "VND_strictly_nonoptimal", "VND_uniquely_best", "VND_tied_best", "all_action_means_zero",
              "repeat_oracle_argmax_agrees", "repeat_best_sets_overlap", "repeat_same_unique_oracle",
              "alternative_beats_VND_in_both_repeats", "E_changes_B_action", "E_corrects_B_wrong", "E_damages_B_right",
              "E_improves_B_gain", "E_reduces_B_gain", "empirical_probability_gain_regret",
              "empirical_probability_set_excludes_gain_oracle", "empirical_probability_tied_but_gain_differs",
              "any_probability_gain_pair_inversion")


def aggregate(states, keys):
    groups = collections.defaultdict(list)
    for row in states:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, rows in sorted(groups.items()):
        r = dict(zip(keys, key)); r["states"] = len(rows)
        r.update({field: mean([s[field] for s in rows]) for field in AGG_FIELDS})
        for action in ACTIONS:
            for variant in ("B", "E"):
                r[f"{variant}_chooses_{action}"] = sum(s[f"{variant}_action"] == action for s in rows)
            for field in ("gain_mean", "absolute_mean", "repeat_gain_abs_difference", "empirical_improvement_probability", "queries_mean", "decodes_mean", "wall_seconds_mean"):
                r[f"{action}_{field}"] = mean([s[f"{action}_{field}"] for s in rows])
        out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="projects/analysis/fp_decision_oracle.json")
    args = parser.parse_args()
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    all_states, all_branches, sources, summaries = [], [], {}, {}
    for split in ("development", "validation"):
        states, branches, inputs = collect(split)
        sources.update(inputs)
        summary = aggregate(states, ("split",))[0]
        summary["instances"] = len({r["instance"] for r in states})
        summary["trajectories"] = len({r["trajectory_id"] for r in states})
        summary["branches"] = len(branches)
        summary["intervals"] = {field: hierarchical_interval(states, field) for field in
                                ("oracle_minus_VND", "cross_repeat_oracle_minus_VND", "cross_repeat_oracle_VND_ties_minus_VND",
                                 "E_minus_B", "B_minus_VND", "E_minus_VND", "VND_strictly_nonoptimal")}
        summary["action_costs"] = {}
        for action in ACTIONS:
            selected = [r for r in branches if r["action"] == action]
            summary["action_costs"][action] = {
                "improvement_event_rate": mean([r["improved"] for r in selected]),
                "mean_gain": mean([r["gain"] for r in selected]),
                "mean_absolute_improvement": mean([r["absolute_improvement"] for r in selected]),
                "conditional_positive_mean_gain": mean([r["gain"] for r in selected if r["improved"]]) if any(r["improved"] for r in selected) else None,
                **{cost: {"mean": mean([r[cost] for r in selected]), "median": float(np.median([r[cost] for r in selected])),
                          "p95": float(np.quantile([r[cost] for r in selected], .95)), "min": min(r[cost] for r in selected),
                          "max": max(r[cost] for r in selected)} for cost in ("queries", "decodes", "wall_seconds")}}
        summary["selection_counts"] = {v: dict(collections.Counter(r[f"{v}_action"] for r in states)) for v in ("B", "E")}
        summary["absolute_improvement_means"] = {field: mean([r[field] for r in states]) for field in
                                                 ("oracle_absolute", "fixed_VND_absolute", "B_absolute", "E_absolute")}
        summary["empirical_probability_choice_gain"] = mean([r["empirical_probability_choice_gain"] for r in states])
        summary["saved_selector_probabilities"] = {
            v: {"mean_by_action": {a: mean([r[f"{a}_{v}_probability"] for r in states]) for a in ACTIONS},
                "VND_minus_best_alternative_score_quantiles": dict(zip(("min", "q25", "median", "q75", "max"),
                    np.quantile([r[f"{v}_VND_score_margin"] for r in states], [0, .25, .5, .75, 1]).tolist()))}
            for v in ("B", "E")}
        summary["counts"] = {field: sum(r[field] for r in states) for field in
                              ("VND_strictly_nonoptimal", "VND_uniquely_best", "VND_tied_best", "all_action_means_zero",
                               "E_changes_B_action", "E_corrects_B_wrong", "E_damages_B_right", "E_improves_B_gain", "E_reduces_B_gain",
                               "empirical_probability_set_excludes_gain_oracle", "empirical_probability_tied_but_gain_differs",
                               "any_probability_gain_pair_inversion", "alternative_beats_VND_in_both_repeats")}
        summary["counts"].update({f"{v}_wrong": sum(not r[f"{v}_is_optimal"] for r in states) for v in ("B", "E")})
        summary["E_correction_rate_given_B_wrong"] = summary["counts"]["E_corrects_B_wrong"] / summary["counts"]["B_wrong"] if summary["counts"]["B_wrong"] else None
        summary["oracle_counts_fixed_tie_break"] = dict(collections.Counter(r["oracle_action"] for r in states))
        summary["oracle_best_sets"] = dict(collections.Counter(r["oracle_best_set"] for r in states))
        summary["E_B_change_states"] = [{k: r[k] for k in ("instance", "trajectory_id", "state_id", "B_action", "E_action", "B_gain", "E_gain", "oracle_best_set", "E_minus_B")} for r in states if r["E_changes_B_action"]]
        positive = [r for r in states if r["any_action_improves"]]
        both_positive = [r for r in states if r["repeat_both_have_positive_outcomes"]]
        summary["nonzero_state_count"] = len(positive)
        summary["nonzero_state_repeat_argmax_agreement"] = mean([r["repeat_oracle_argmax_agrees"] for r in positive]) if positive else None
        summary["both_repeats_positive_state_count"] = len(both_positive)
        summary["both_positive_repeat_best_sets_overlap"] = mean([r["repeat_best_sets_overlap"] for r in both_positive]) if both_positive else None
        summaries[split] = summary
        all_states.extend(states); all_branches.extend(branches)
    outputs = {}
    for suffix, rows in (("states", all_states), ("branches", all_branches),
                         ("trajectories", aggregate(all_states, ("split", "instance", "trajectory_id"))),
                         ("instances", aggregate(all_states, ("split", "instance")))):
        output = path.with_name(path.stem + "_" + suffix + ".csv")
        write_csv(output, rows)
        outputs[str(output.relative_to(ROOT))] = {"rows": len(rows), "sha256": sha(output)}
    # Exhaustive ordinary-state quartiles are descriptive, never a selector or a filter.
    bins = []
    development = [r for r in all_states if r["split"] == "development"]
    for feature in ("x0", "x1", "x2", "x3", "stagnant_generations", "phase", "recent_main_gain_mean"):
        edges = sorted(set(np.quantile([r[feature] for r in development], [.25, .5, .75]).tolist()))
        for split in ("development", "validation"):
            for bin_index in range(len(edges) + 1):
                selected = [r for r in all_states if r["split"] == split and int(np.searchsorted(edges, r[feature], side="right")) == bin_index]
                if selected:
                    row = aggregate(selected, ("split",))[0]
                    row.update({"feature": feature, "bin_index": bin_index, "development_edges": json.dumps(edges)})
                    bins.append(row)
    output = path.with_name(path.stem + "_state_strata.csv")
    write_csv(output, bins); outputs[str(output.relative_to(ROOT))] = {"rows": len(bins), "sha256": sha(output)}
    result = {"purpose": "Final decision mechanism diagnosis A/B using saved M2 only", "status": "COMPLETE",
              "data_roles": {"development": "development and descriptive diagnosis", "validation": "previously exposed: descriptive diagnosis only; not fresh confirmation", "locked_test": "NOT_READ_OR_RUN"},
              "units": {"gain": "(start_makespan-end_makespan)/s0", "absolute": "start_makespan-end_makespan", "state_reward": "mean of two independent diagnostic repeats", "oracle": "max of same-state two-repeat sample means: optimistic and unavailable at decision time"},
              "tie_break": "GA,LS,VND fixed order; optimal correctness accepts every tied maximum; VND-preserving cross-repeat sensitivity also reported",
              "interpretation_limits": ["Only two repeats cannot identify statewise true expected action values.", "Cross-repeat oracle transfer is a noise sensitivity diagnostic, not a learned or deployed policy.", "Actual atomic action lengths/costs differ; cost fields are audit only and never determine action selection.", "State quartiles are exhaustive descriptive summaries with development-only edges, not selected subgroups or learned decision rules.", "No new classifiers, optimizer changes, search actions, tuning, or locked-test access."],
              "bootstrap": {"seed": 20261019, "repetitions": 10000, "grouping": "instance then complete paired trajectories", "correction": "None: exploratory diagnosis, no confirmatory GO claim"},
              "summaries": summaries, "input_sha256": sources, "outputs": outputs,
              "script_sha256": sha(__file__)}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({split: {k: result["summaries"][split][k] for k in ("states", "oracle_minus_VND", "cross_repeat_oracle_minus_VND", "cross_repeat_oracle_VND_ties_minus_VND", "selection_counts")} for split in ("development", "validation")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

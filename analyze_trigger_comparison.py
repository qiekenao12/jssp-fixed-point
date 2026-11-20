#!/usr/bin/env python3
"""Audit and summarize controlled trigger results without pooling states as independent trials."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from run_trigger_comparison import TRIGGERS


def ranks(values):
    _, inverse, counts = np.unique(np.asarray(values), return_inverse=True, return_counts=True)
    rank_values = np.cumsum(counts) - (counts - 1) / 2
    return rank_values[inverse]


def spearman(left, right):
    if len(left) < 3:
        return None
    left, right = ranks(left), ranks(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def bootstrap_mean(values, seed=20260916):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    generator = np.random.default_rng(seed)
    means = np.mean(generator.choice(values, size=(10000, len(values))), axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_bootstrap(matrix):
    generator = np.random.default_rng(20260916)
    array = np.asarray(matrix)
    count_instances, count_runs = array.shape
    instance_indices = generator.integers(count_instances, size=(10000, count_instances, 1))
    run_indices = generator.integers(count_runs, size=(10000, count_instances, count_runs))
    means = np.mean(array[instance_indices, run_indices], axis=(1, 2))
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def sign_flip(values):
    values = np.asarray(values, dtype=float)
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(values))))
    observed = abs(float(values.mean()))
    return float(np.mean(np.abs(np.mean(signs * values, axis=1)) >= observed - 1e-12))


def aggregate(rows):
    events = [event for row in rows for event in row["events"]]
    count, successes = len(events), sum(event["effective"] for event in events)
    return {
        "runs": len(rows), "avg_gap": float(np.mean([row["gap"] for row in rows])),
        "std_gap_across_runs_and_instances": float(np.std([row["gap"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "vnd_calls_total": count, "vnd_calls_mean": count / len(rows),
        "vnd_effective_total": successes,
        "vnd_effectiveness": successes / count if count else None,
        "mean_gain_bks_pp_per_call": float(np.mean([event["gain_bks_pp"] for event in events])) if count else None,
        "avg_vnd_queries": float(np.mean([row["phase_queries"].get("vnd", 0) for row in rows])),
        "avg_elapsed": float(np.mean([row["elapsed"] for row in rows])),
        "avg_actual_decodes": float(np.mean([row["decodes"] for row in rows])),
        "capped_call_rate": float(np.mean([event["capped"] for event in events])) if count else None,
    }


def summarize_probe(rows, instances):
    events = [event for row in rows for event in row["events"]]
    fitted = [event for event in events if event["fp_residual"] is not None]
    reliable = [event for event in fitted if event["fp_reliable"]]
    per_run = []
    for row in rows:
        eligible = [event for event in row["events"] if event["fp_reliable"]]
        record = {"instance": row["instance"], "run": row["run"], "states": len(eligible)}
        for feature in ("fp_residual", "state_drift", "fitness_change", "stagnant_generations"):
            record[feature] = spearman(
                [event[feature] for event in eligible],
                [event["gain_bks_pp"] for event in eligible],
            )
        per_run.append(record)
    correlations = {}
    for feature in ("fp_residual", "state_drift", "fitness_change", "stagnant_generations"):
        instance_means = {}
        for name in instances:
            values = [record[feature] for record in per_run if record["instance"] == name and record[feature] is not None]
            if values:
                instance_means[name] = float(np.mean(values))
        correlations[feature] = {
            "pooled_descriptive_rho": spearman([event[feature] for event in reliable], [event["gain_bks_pp"] for event in reliable]),
            "per_instance_mean_rho": instance_means,
            "macro_mean_rho": float(np.mean(list(instance_means.values()))) if instance_means else None,
            "instance_bootstrap_95ci": bootstrap_mean(list(instance_means.values())),
            "valid_runs": sum(record[feature] is not None for record in per_run),
            "undefined_runs": sum(record[feature] is None for record in per_run),
        }
    selected_metrics = {}
    for trigger in TRIGGERS:
        selected = [event for event in reliable if event["recommendations"][trigger]]
        positives = sum(event["effective"] for event in reliable)
        selected_metrics[trigger] = {
            "selected_states": len(selected), "available_states": len(reliable),
            "precision": float(np.mean([event["effective"] for event in selected])) if selected else None,
            "recall": sum(event["effective"] for event in selected) / positives if positives else None,
            "avg_gain_bks_pp": float(np.mean([event["gain_bks_pp"] for event in selected])) if selected else None,
        }
    quantile_bins = []
    if reliable:
        boundaries = np.unique(np.quantile([event["fp_residual"] for event in reliable], [0, .25, .5, .75, 1]))
        for index in range(len(boundaries) - 1):
            lower, upper = boundaries[index:index + 2]
            selected = [event for event in reliable if event["fp_residual"] >= lower and
                        (event["fp_residual"] <= upper if index == len(boundaries) - 2 else event["fp_residual"] < upper)]
            quantile_bins.append({
                "lower": float(lower), "upper": float(upper), "states": len(selected),
                "effectiveness": float(np.mean([event["effective"] for event in selected])) if selected else None,
                "mean_gain_bks_pp": float(np.mean([event["gain_bks_pp"] for event in selected])) if selected else None,
            })
    return {
        "states": len(events), "fitted_states": len(fitted), "reliable_states": len(reliable),
        "correlations": correlations, "per_run_correlations": per_run,
        "recommendation_precision": selected_metrics, "residual_quantile_bins": quantile_bins,
        "effective_states": sum(event["effective"] for event in reliable),
        "scope": "Detached probes on neutral GA+LS states, not solely FP-selected states. Precision is conditional on common fitted/reliable states, without deployment cooldown.",
    }


def audit_groups(folder, metadata):
    rows = []
    for name in metadata["instances"]:
        for run in range(metadata["runs"]):
            group = json.loads((folder / f"{name}_{run:02d}.json").read_text())
            assert {row["trigger"] for row in group} == set((*TRIGGERS, "probe"))
            assert len(group) == 5 and len({row["initial_fingerprint"] for row in group}) == 1
            warmup_traces = [[
                {key: value for key, value in observation.items() if key != "observation_elapsed"}
                for observation in row["observations"]
                if observation["generation"] < metadata["protocol"]["warmup"]
            ] for row in group]
            assert all(trace == warmup_traces[0] for trace in warmup_traces)
            for row in group:
                assert row["instance"] == name and row["run"] == run
                assert row["queries"] == metadata["protocol"]["queries"]
                assert sum(row["phase_queries"].values()) == row["queries"]
                assert row["phase_queries"].get("vnd", 0) <= metadata["protocol"]["vnd_queries"]
                assert row["vnd_calls"] == len(row["events"])
                assert row["vnd_effective"] == sum(event["gain"] > 0 for event in row["events"])
                assert all(0 <= event["queries"] <= metadata["protocol"]["call_queries"] for event in row["events"])
                assert all(abs(event["before"] - event["after"] - event["gain"]) < 1e-10 for event in row["events"])
            rows.extend(group)
    return rows


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plots(folder, summary, probe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 220})
    labels = ["Fixed", "Stagnation", "State drift", "FP residual"]
    colors = ["#5975a4", "#cc8963", "#5f9e6e", "#b55d60"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for axis, key, title in zip(axes, ("vnd_calls_mean", "vnd_effectiveness", "avg_gap"),
                               ("VND calls per run", "Effective VND calls (%)", "Final average Gap (%)")):
        values = [summary[trigger][key] * (100 if key == "vnd_effectiveness" else 1) for trigger in TRIGGERS]
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, fmt="%.2f", fontsize=9)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(0, max(values) * 1.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Trigger-only control: equal objective-query budgets")
    figure.savefig(folder / "trigger_comparison.png")
    figure.savefig(folder / "trigger_comparison.pdf")
    plt.close(figure)
    bins = probe["residual_quantile_bins"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = [f"Q{index + 1}\nn={item['states']}" for index, item in enumerate(bins)]
    axes[0].bar(labels, [100 * item["effectiveness"] for item in bins], color="#5975a4")
    axes[0].set_ylabel("VND effectiveness (%)")
    axes[0].set_xlabel("FP residual quantiles: low to high")
    means = probe["correlations"]["fp_residual"]["per_instance_mean_rho"]
    axes[1].barh(list(means), list(means.values()), color="#b55d60")
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].set_xlabel("Mean within-run Spearman rho (residual vs gain)")
    figure.suptitle("Detached probes: lower residual predicts gain only if rho < 0")
    figure.savefig(folder / "fp_residual_probe.png")
    figure.savefig(folder / "fp_residual_probe.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.folder / "protocol.json").read_text())
    rows = audit_groups(args.folder, metadata)
    instances = metadata["instances"]
    summary = {trigger: aggregate([row for row in rows if row["trigger"] == trigger]) for trigger in TRIGGERS}
    by_instance = {name: {trigger: aggregate([row for row in rows if row["trigger"] == trigger and row["instance"] == name])
                          for trigger in TRIGGERS} for name in instances}
    lookup = {(row["instance"], row["run"], row["trigger"]): row for row in rows}
    comparisons = {}
    for baseline in TRIGGERS[:-1]:
        matrix = [[lookup[name, run, baseline]["gap"] - lookup[name, run, "fp_residual"]["gap"]
                   for run in range(metadata["runs"])] for name in instances]
        instance_effects = np.mean(matrix, axis=1)
        comparisons[baseline] = {
            "baseline_minus_fp_gap_pp": float(np.mean(matrix)),
            "hierarchical_bootstrap_95ci": paired_bootstrap(matrix),
            "instance_sign_flip_two_sided_p": sign_flip(instance_effects),
            "fp_win_tie_loss_instances": [int(np.sum(instance_effects > 1e-10)), int(np.sum(abs(instance_effects) <= 1e-10)), int(np.sum(instance_effects < -1e-10))],
        }
    ordered = sorted(comparisons, key=lambda key: comparisons[key]["instance_sign_flip_two_sided_p"])
    adjusted = 0
    for index, baseline in enumerate(ordered):
        adjusted = max(adjusted, min(1, (len(ordered) - index) * comparisons[baseline]["instance_sign_flip_two_sided_p"]))
        comparisons[baseline]["holm_adjusted_p"] = adjusted
    probe = summarize_probe([row for row in rows if row["trigger"] == "probe"], instances)
    result = {"metadata": metadata, "summary": summary, "by_instance": by_instance,
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "paired_comparisons": comparisons, "probe_analysis": probe,
              "audit": "Passed all paired initialization, query budget, event and gain consistency checks."}
    (args.folder / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    write_csv(args.folder / "summary.csv", [{"trigger": trigger, **values} for trigger, values in summary.items()])
    write_csv(args.folder / "per_instance.csv", [{"instance": name, "trigger": trigger, **values}
                                               for name, methods in by_instance.items() for trigger, values in methods.items()])
    write_csv(args.folder / "events.csv", [
        {"instance": row["instance"], "run": row["run"], "trigger": row["trigger"],
         **{key: value for key, value in event.items() if key != "recommendations"},
         **{f"signal_{key}": value for key, value in event["recommendations"].items()}}
        for row in rows for event in row["events"]
    ])
    report = [
        "# FP 与简单触发器：受控实验报告",
        "",
        "本报告是新增实验记录，不修改论文，也不使用先前存在 VND 强度混杂的试跑结果。",
        "",
        "## 实验协议",
        "",
        f"- 实例：{', '.join(instances)}；每实例 {metadata['runs']} 个配对种子。",
        f"- 每组总预算 {metadata['protocol']['queries']} 次目标函数查询（缓存命中也计数），VND 子预算上限 {metadata['protocol']['vnd_queries']} 次；实际解码次数另记。该预算不等同于墙钟时间。",
        f"- 种群 {metadata['protocol']['population']}，固定 GA 参数；每 10 代对最优个体执行相同 LS；每 5 代观测、85 代后开放触发、共同冷却 10 代。",
        f"- VND 对当前最优个体调用一次算一次；相同 max_iter=100，每次最多 {metadata['protocol']['call_queries']} 次查询，严格改善才有效。所有组都使用截断时保留最好已评估解的相同封装。",
        "- Fixed 每 20 代；Stagnation 最优值连续 15 代不改进；State Drift 同一归一化四维状态的一步加权 L2 差 ≤0.012；FP 使用可靠仿射算子残差 ≤0.012。",
        "- FP 至少 16 个同间隔转移样本，拟合误差 ≤0.03，最多 48 个转移，ridge=1e-6；本实验隔离残差信号，不继承旧 Agent/阶段/停滞联合门控。",
        "- 终局精化所有组统一关闭，避免终局 SA/VND 遮蔽触发收益；GA/LS 可消费剩余总预算，因此各组最终代数允许不同。",
        "- 所有组计算相同观测量用于诊断，故时间包含 FP 观测开销，不能用该时间比较原生触发器的检测成本。初始化、各代 GA/LS/VND 使用配对且隔离的随机流。",
        "- 阈值预先固定，未按最终排名重新调优；结果只适用于该协议，不证明算法不可替代。",
        "",
        "## 主结果",
        "",
        "| Trigger | VND调用总数 | 平均调用/运行 | 有效率 | 最终Avg Gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for trigger, values in summary.items():
        report.append(f"| {trigger} | {values['vnd_calls_total']} | {values['vnd_calls_mean']:.2f} | {values['vnd_effectiveness']*100:.2f}% | {values['avg_gap']:.3f}% |")
    report += ["", "有效率 = 改善当前最优 makespan 的 VND 次数 / 实际 VND 调用次数。跨实例 Gap 等权平均；有效率用合计成功数除以合计调用数，不平均百分比。",
               "", "![四触发器对照](trigger_comparison.png)", "", "## 配对比较",
               "", "差值为 baseline − FP 的 Gap（百分点），正值才表示 FP 更好。95% 区间在实例、实例内配对种子两层重采样；p 值在实例均值层面做双侧精确符号翻转并进行三重 Holm 校正。",
               "", "| 对照 | Gap差 | 95%区间 | Holm p | FP实例胜/平/负 |", "|---|---:|---|---:|---|"]
    for baseline, values in comparisons.items():
        lower, upper = values["hierarchical_bootstrap_95ci"]
        report.append(f"| {baseline} | {values['baseline_minus_fp_gap_pp']:+.3f} | [{lower:+.3f}, {upper:+.3f}] | {values['holm_adjusted_p']:.4f} | {values['fp_win_tie_loss_instances']} |")
    report += ["", "## 残差与调用收益：独立探测", "",
               f"额外 {len(instances)*metadata['runs']} 条 GA+LS 轨迹，在所有候选时刻进行不回写的 VND 探测；共 {probe['states']} 个状态，其中 {probe['reliable_states']} 个满足相同 FP 可靠性门槛。探测有独立随机流、缓存和预算，不计入四组部署预算，也不改变搜索轨迹。",
               "", "相关性先在每条轨迹内计算，再在实例内、实例间等权平均。收益恒零或信号恒定的轨迹相关系数不可定义，单独报告，不能填成零或当成支持证据。区间基于实例重采样；状态不是独立实验单位。",
               "", "| 信号 | 平均轨迹Spearman ρ | 实例Bootstrap 95%区间 | 有定义/无定义轨迹 |",
               "|---|---:|---|---|"]
    for feature, values in probe["correlations"].items():
        interval = values["instance_bootstrap_95ci"]
        interval_text = "不可定义" if interval is None else f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"
        mean_text = "不可定义" if values["macro_mean_rho"] is None else f"{values['macro_mean_rho']:+.3f}"
        report.append(f"| {feature} | {mean_text} | {interval_text} | {values['valid_runs']}/{values['undefined_runs']} |")
    report += ["", "下表在同一批可靠的中性轨迹状态上比较四种触发条件的选择准确性。Precision 是被选状态中 VND 有效的比例；Recall 是所有有效状态中被选中的比例。此处比较原始条件，不应用部署冷却或剩余预算门控；各阈值选择率不同，不能把它当成等调用次数对照。",
               "", "| 条件 | 选择状态数 | Precision | Recall | 平均收益（Gap百分点） |",
               "|---|---:|---:|---:|---:|"]
    for trigger, values in probe["recommendation_precision"].items():
        precision_text = "不可定义" if values["precision"] is None else f"{values['precision']*100:.2f}%"
        recall_text = "不可定义" if values["recall"] is None else f"{values['recall']*100:.2f}%"
        gain_text = "不可定义" if values["avg_gain_bks_pp"] is None else f"{values['avg_gain_bks_pp']:.4f}"
        report.append(f"| {trigger} | {values['selected_states']} | {precision_text} | {recall_text} | {gain_text} |")
    report += ["", "FP 假设要求残差与收益呈负相关。Spearman 是关联证据，不是因果证明；分桶图是描述性结果，阶段和实例仍可能影响关系。",
               "", "![独立探测分析](fp_residual_probe.png)", "", "## 结论与边界", ""]
    fp_best = all(summary["fp_residual"]["avg_gap"] < summary[baseline]["avg_gap"] for baseline in TRIGGERS[:-1])
    report.append("FP 的平均 Gap 在这组设置下最低，但仍需结合区间和有效率判断。" if fp_best
                  else "FP 的平均 Gap 并非四组最低，当前实验不支持“FP 优于所有简单触发器”的主张。")
    correlation = probe["correlations"]["fp_residual"]
    interval = correlation["instance_bootstrap_95ci"]
    report.append("残差—收益相关区间完全低于零，支持该设置下的负向关联；仍不证明不可替代。"
                  if interval and interval[1] < 0 else
                  "残差—收益区间没有完全落在零以下，当前结果不足以确立“残差越小，VND 收益越高”。")
    report += [
        "",
        "这是固定单个查询预算下的触发器消融，不代表原完整 Agent-FP 的性能复现。需要更多预算点、阈值敏感性、独立验证实例和更强简单基线才能讨论稳健性；本次不追调阈值以制造胜出。",
        "",
        "查询次数包含缓存命中，因此计算公平性限定为逻辑目标函数查询额度，不能等同于相同墙钟时间或完全相同真实解码工作量。summary.csv 同时列出实际解码次数。探测执行截断版 VND，结论不自动外推到无限制完整 VND。",
        "",
        "## 复现",
        "",
        "运行：OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_trigger_comparison.py --runs 20 --workers 4 --output results/trigger_controlled_20runs",
        "",
        "分析：python3 analyze_trigger_comparison.py results/trigger_controlled_20runs",
        "",
        "protocol.json 保存参数、种子规则所需基础种子、源码及实例文件哈希；逐实例逐种子 JSON 支持断点续跑；events.csv 保存逐次触发/探测日志，summary.csv 与 per_instance.csv 保存统计。",
        "",
    ]
    (args.folder / "report.md").write_text("\n".join(report), encoding="utf-8")
    plots(args.folder, summary, probe)
    print(json.dumps({"summary": summary, "paired": comparisons, "fp_correlation": probe["correlations"]["fp_residual"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Agent-FP消融实验
验证各个核心组件的有效性
"""
import sys, os, time, json, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tools.jssp_core import JSSPInstance, JSSPUtils, JSSPEncoder, JSSPEvaluator
from tools.meta_scheduler import MetaScheduler
from tools.controller import Controller as ControllerV8  # v10.1: 使用优化版控制器
from tools.benchmark_data import BKS, BKS_SOURCE, BKS_VERSION, validate_makespan

BASE_SEED = int(os.getenv("JSSP_SEED", "20260813"))


def seed_run(instance_name, run_index):
    """为每个实例和运行生成稳定种子，便于消融配置做配对比较。"""
    seed = BASE_SEED + sum(ord(ch) for ch in instance_name)
    seed += 9176 * run_index
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

# 按小、中、大规模选择实例
TEST_INSTANCES = [
    "ft06",   # 小规模: 6x6 = 36 ops
    "la04",   # 小规模: 10x5 = 50 ops
    "ft10",   # 中规模: 10x10 = 100 ops
    "orb01",  # 中规模: 10x10 = 100 ops
    "la21",   # 大规模: 15x10 = 150 ops
    "la26",   # 大规模: 20x10 = 200 ops
    "la40",   # 超大规模: 15x15 = 225 ops
    "yn1",    # 超大规模: 20x20 = 400 ops
    "ta31",   # 超大规模: 30x15 = 450 ops
    "ta51",   # 超大规模: 50x15 = 750 ops
]

# 实验配置
NUM_RUNS = int(os.getenv('JSSP_NUM_RUNS', '20'))  # 正式实验默认20次，可用环境变量试跑


def result_suffix():
    """Optional suffix keeps model-validation outputs separate."""
    tag = ''.join(
        char for char in os.getenv('JSSP_RESULT_TAG', '').strip()
        if char.isalnum() or char in ('-', '_')
    )
    return f'_{tag}' if tag else ''


def parse_jobshop1(filepath):
    """解析OR-Library的jobshop1.txt文件"""
    instances = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('instance') and len(line.split()) == 2:
            name = line.split()[1]
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith('+')):
                i += 1
            if i < len(lines) and not lines[i][0].isdigit():
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines): break
            parts = lines[i].strip().split()
            if len(parts) < 2: 
                i += 1; 
                continue
            try:
                num_jobs = int(parts[0])
                num_machines = int(parts[1])
            except:
                i += 1; 
                continue
            i += 1
            processing_times, machine_sequences = [], []
            for _ in range(num_jobs):
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i >= len(lines): break
                values = list(map(int, lines[i].strip().split()))
                if len(values) >= 2 * num_machines:
                    machines = values[0::2]
                    times = values[1::2]
                    machine_sequences.append(machines[:num_machines])
                    processing_times.append(times[:num_machines])
                i += 1
            if len(processing_times) == num_jobs and len(machine_sequences) == num_jobs:
                lb = JSSPUtils.compute_lower_bound(processing_times, machine_sequences, num_machines)
                ub = max(lb * 2, BKS.get(name, lb * 2))
                instances[name] = {
                    "num_jobs": num_jobs, "num_machines": num_machines,
                    "processing_times": processing_times, "machine_sequences": machine_sequences,
                    "upper_bound": ub
                }
        else:
            i += 1
    return instances


# ============ 消融实验配置 ============
ABLATION_CONFIGS = {
    'Full': {
        'adaptive_params': True,
        'agent_commander': True,
        'fixed_point': True,
        'local_search': True,
        'vnd': True,
        'terminal_refinement': True,
        'description': 'Agent管控全部组件 (FP不动点模型+局部搜索+终局精化)'
    },
    'No Agent': {
        'adaptive_params': False,
        'agent_commander': False,
        'fixed_point': True,
        'local_search': True,
        'vnd': True,
        'terminal_refinement': True,
        'description': '无Agent，全部组件固定开启 (公平基线)'
    },
    'No FP': {
        'adaptive_params': True,
        'agent_commander': True,
        'fixed_point': False,
        'local_search': True,
        'vnd': False,
        'terminal_refinement': True,
        'description': '关闭FP不动点判断，Agent仍管控局部搜索和终局精化'
    },
    'No LocalSearch': {
        'adaptive_params': True,
        'agent_commander': True,
        'fixed_point': True,
        'local_search': False,
        'vnd': False,
        'terminal_refinement': True,
        'description': '关闭局部搜索和VND，Agent管控FP与终局精化'
    },
    'No Terminal Refinement': {
        'adaptive_params': True,
        'agent_commander': True,
        'fixed_point': True,
        'local_search': True,
        'vnd': True,
        'terminal_refinement': False,
        'description': '关闭终局多起点精化，Agent管控FP、局部搜索和VND'
    },
    'No Components': {
        'adaptive_params': False,
        'agent_commander': False,
        'fixed_point': False,
        'local_search': False,
        'vnd': False,
        'terminal_refinement': False,
        'description': 'Agent、FP、局部搜索和终局精化全部关闭，仅保留基础GA'
    },
}


def run_ablation(instance, config_name, config, num_runs=1, instance_name=None, bks=None):
    """运行消融实验，多次运行取平均值"""
    all_makespans = []
    all_times = []
    component_runs = []
    
    for run in range(num_runs):
        seed_run(
            instance_name=instance_name or f"{instance.num_jobs}x{instance.num_machines}",
            run_index=run,
        )
        start = time.time()
        scheduler = MetaScheduler(instance)

        # 使用Agent控制器替换默认Controller（仅当Agent启用时）
        if config['adaptive_params']:
            scheduler.controller = ControllerV8(instance.num_jobs, instance.num_machines)
            scheduler.controller.set_max_generations(scheduler.max_generations)
            scheduler.controller.set_early_stop_stagnant(scheduler.early_stop_stagnant)  # v8.2

        # 新的API：直接通过参数控制；Agent负责调度，FP负责不动点判断。
        result = scheduler.optimize(
            use_adaptive_params=config['adaptive_params'],
            use_fixed_point=config['fixed_point'],
            use_local_search=config.get('local_search', True),
            use_vnd=config.get('vnd', True),
            use_agent_commander=config.get(
                'agent_commander', config.get('ap_commander', False)
            ),
            use_terminal_refinement=config.get('terminal_refinement', True),
        )
        elapsed = time.time() - start
        validate_makespan(instance_name, result.best_makespan, bks)
        all_makespans.append(result.best_makespan)
        all_times.append(elapsed)
        controller_summary = scheduler.controller.get_controller_summary()
        component_runs.append({
            'safe_rescue_attempts': controller_summary.get('safe_rescue_attempts', 0),
            'safe_rescue_accepted': controller_summary.get('safe_rescue_accepted', 0),
            'safe_rescue_best_improvements': controller_summary.get(
                'safe_rescue_best_improvements', 0
            ),
            'fp_vnd_executions': scheduler.fp_detector.vnd_executed_count,
            'fp_vnd_success_rate': scheduler.fp_detector.vnd_success_rate,
            'final_refinement_starts': scheduler.final_refinement_start_count,
            'final_archive_starts': scheduler.final_archive_start_count,
            'commander_directives': controller_summary.get(
                'commander_directive_counts', {}
            ),
            'component_feedback': controller_summary.get(
                'component_feedback', {}
            ),
            'component_roi': controller_summary.get('component_roi', {}),
            'orchestration': result.orchestration,
        })
    
    avg_makespan = sum(all_makespans) / len(all_makespans)
    return {
        'makespans': all_makespans,
        'times': all_times,
        'makespan': avg_makespan,
        'best_makespan': min(all_makespans),
        'worst_makespan': max(all_makespans),
        'std_makespan': float(np.std(all_makespans)),
        'hit_rate': (
            sum(m == bks for m in all_makespans) / len(all_makespans)
            if bks is not None else None
        ),
        'generations': result.total_generations,
        'time': sum(all_times) / len(all_times),
        'component_stats': {
            'runs': component_runs,
            'avg_safe_rescue_attempts': float(np.mean([
                item['safe_rescue_attempts'] for item in component_runs
            ])),
            'avg_safe_rescue_accepted': float(np.mean([
                item['safe_rescue_accepted'] for item in component_runs
            ])),
            'avg_safe_rescue_best_improvements': float(np.mean([
                item['safe_rescue_best_improvements'] for item in component_runs
            ])),
            'avg_fp_vnd_executions': float(np.mean([
                item['fp_vnd_executions'] for item in component_runs
            ])),
            'avg_fp_vnd_success_rate': float(np.mean([
                item['fp_vnd_success_rate'] for item in component_runs
            ])),
            # 反馈按所有配对运行累加，不能只取最后一次运行。
            'component_feedback_total': _aggregate_component_feedback(
                component_runs
            ),
            'component_feedback_avg': _average_component_feedback(
                component_runs
            ),
            # 论文中同时报告调用效率，避免把“调用更多”误认为“调度更好”。
            'efficiency': _component_efficiency(component_runs),
            'orchestration': _aggregate_orchestration(component_runs),
        },
    }


def _aggregate_orchestration(component_runs):
    """Aggregate Agent budget/action telemetry without mixing run averages."""
    numeric = ('consumed', 'gain', 'requested', 'rejected')
    totals = {key: {} for key in numeric}
    allocated_total = 0.0
    modes = {}
    action_count = 0
    rejected_actions = 0
    for run in component_runs:
        data = run.get('orchestration', {}) or {}
        modes[str(data.get('mode', 'unknown'))] = (
            modes.get(str(data.get('mode', 'unknown')), 0) + 1
        )
        action_count += int(data.get('actions', 0) or 0)
        rejected_actions += int(data.get('rejected_actions', 0) or 0)
        allocated_total += float(data.get('allocated', 0.0) or 0.0)
        for key in numeric:
            for component, value in (data.get(key, {}) or {}).items():
                totals[key][component] = totals[key].get(component, 0.0) + float(value)
    count = max(1, len(component_runs))
    return {
        'runs': len(component_runs),
        'modes': modes,
        'actions': action_count,
        'rejected_actions': rejected_actions,
        'avg_allocated_total': allocated_total / count,
        'avg_requested': {k: v / count for k, v in totals['requested'].items()},
        'avg_consumed': {k: v / count for k, v in totals['consumed'].items()},
        'avg_gain': {k: v / count for k, v in totals['gain'].items()},
        'avg_rejected': {k: v / count for k, v in totals['rejected'].items()},
    }


def _aggregate_component_feedback(component_runs):
    """汇总组件反馈总量，字段含义与Controller保持一致。"""
    totals = {}
    for run in component_runs:
        for component, metrics in run.get('component_feedback', {}).items():
            target = totals.setdefault(
                component,
                {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                 'work_calls': 0.0, 'elapsed': 0.0},
            )
            for metric in target:
                target[metric] += float(metrics.get(metric, 0.0))
    return totals


def _average_component_feedback(component_runs):
    """汇总每次运行的组件反馈均值，便于跨配置比较。"""
    totals = _aggregate_component_feedback(component_runs)
    run_count = max(1, len(component_runs))
    return {
        component: {
            metric: value / run_count for metric, value in metrics.items()
        }
        for component, metrics in totals.items()
    }


def _component_efficiency(component_runs):
    """计算有效调用率和单位调用收益。"""
    totals = _aggregate_component_feedback(component_runs)
    result = {}
    for component, metrics in totals.items():
        calls = float(metrics.get('calls', 0.0))
        work_calls = float(metrics.get('work_calls', 0.0))
        improvements = float(metrics.get('improvements', 0.0))
        gain = float(metrics.get('gain', 0.0))
        denominator = work_calls if work_calls > 0 else calls
        result[component] = {
            'calls': calls,
            'work_calls': work_calls,
            'elapsed': float(metrics.get('elapsed', 0.0)),
            'improvements': improvements,
            'gain': gain,
            'improvement_rate': improvements / max(denominator, 1.0),
            'gain_per_call': gain / max(denominator, 1.0),
            'gain_per_second': gain / max(float(metrics.get('elapsed', 0.0)), 1e-9),
        }
    return result


def main():
    filepath = os.path.join(os.path.dirname(__file__), 'jobshop1.txt')
    if not os.path.exists(filepath):
        print("Error: jobshop1.txt not found.")
        return
    
    print("="*70)
    print(" Agent-FP ABLATION EXPERIMENT")
    print(" Main metric: AvgGap (lower is better); all configurations use paired seeds.")
    print(" Full: Agent schedules local search, FP checking, Agent-authorized VND, and terminal refinement.")
    print("="*70)
    print()
    
    all_instances = parse_jobshop1(filepath)
    
    # 运行消融实验
    all_results = {name: [] for name in ABLATION_CONFIGS}
    
    for idx, name in enumerate(TEST_INSTANCES):
        if name not in all_instances:
            continue
            
        data = all_instances[name]
        bks = BKS[name]
        nj, nm = data['num_jobs'], data['num_machines']
        
        print("="*70)
        print(f" [{idx+1}/{len(TEST_INSTANCES)}] Instance: {name}  ({nj}x{nm}, BKS={bks})")
        print("-"*70)
        
        instance = JSSPInstance(
            num_jobs=nj, num_machines=nm,
            processing_times=data['processing_times'],
            machine_sequences=data['machine_sequences'],
            upper_bound=data['upper_bound']
        )
        
        for config_name, config in ABLATION_CONFIGS.items():
            print(f"  Running {config_name} [{config['description']}] (paired x{NUM_RUNS})...", end=" ", flush=True)
            start = time.time()
            # 临时重定向stdout以抑制MetaScheduler的详细输出
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                result = run_ablation(
                    instance,
                    config_name,
                    config,
                    num_runs=NUM_RUNS,
                    instance_name=name,
                    bks=bks,
                )
            finally:
                sys.stdout = old_stdout
            elapsed = time.time() - start
            
            avg_makespan = result['makespan']  # 使用平均结果
            best_makespan = result['best_makespan']
            avg_gap = (avg_makespan - bks) / bks * 100 if bks > 0 else 0
            best_gap = (best_makespan - bks) / bks * 100 if bks > 0 else 0
            component_stats = result['component_stats']
            feedback_avg = component_stats['component_feedback_avg']
            ls_feedback = feedback_avg.get('local_search', {})
            fp_feedback = feedback_avg.get('fp_detection', {})
            vnd_feedback = feedback_avg.get('vnd', {})
            terminal_feedback = feedback_avg.get('terminal', {})
            print(
                f"Best={best_makespan}, Avg={avg_makespan:.1f}, "
                f"AvgGap={avg_gap:.1f}%, BestGap={best_gap:.1f}%, Time={elapsed:.1f}s "
                f"| LS={ls_feedback.get('improvements', 0.0):.1f}/"
                f"{ls_feedback.get('calls', 0.0):.1f}, "
                f"FP-check={fp_feedback.get('improvements', 0.0):.1f}/"
                f"{fp_feedback.get('calls', 0.0):.1f}, "
                f"VND={vnd_feedback.get('improvements', 0.0):.1f}/"
                f"{vnd_feedback.get('calls', 0.0):.1f}, "
                f"Terminal={terminal_feedback.get('improvements', 0.0):.1f}/"
                f"{terminal_feedback.get('calls', 0.0):.1f}"
            )
            
            all_results[config_name].append({
                'name': name, 'bks': bks,
                'makespans': result['makespans'],
                'times': result['times'],
                'avg_makespan': avg_makespan,
                'best_makespan': best_makespan,
                'worst_makespan': result['worst_makespan'],
                'avg_gap': avg_gap,
                'best_gap': best_gap,
                'std_makespan': result['std_makespan'],
                'bks_hit_rate': result['hit_rate'],
                'time': result['time'],
                'component_stats': result['component_stats'],
            })
    
    # 输出汇总表格
    print("\n" + "="*90)
    print(f" ABLATION RESULTS (Average Gap %) - {NUM_RUNS}轮配对运行，AvgGap越低越好")
    print("="*90)
    print("| Instance | BKS |", end="")
    for config_name in ABLATION_CONFIGS:
        print(f" {config_name} |", end="")
    print()
    print("|----------|-----|" + "--------|" * len(ABLATION_CONFIGS))
    
    for i, name in enumerate(TEST_INSTANCES):
        if name not in all_instances:
            continue
        bks = BKS[name]
        print(f"| {name.ljust(8)} | {bks:3d} |", end="")
        for config_name in ABLATION_CONFIGS:
            r = all_results[config_name][i]
            print(f" {r['avg_gap']:5.1f} |", end="")
        print()
    
    # 平均Gap
    print("\n| Average  |     |", end="")
    for config_name in ABLATION_CONFIGS:
        gaps = [r['avg_gap'] for r in all_results[config_name]]
        avg = sum(gaps) / len(gaps) if gaps else 0
        print(f" {avg:5.1f} |", end="")
    print()
    
    # 最佳Gap表（参考）
    print("\n" + "="*90)
    print(f" BEST Gap REFERENCE ({NUM_RUNS}轮中的单次最佳值，仅作补充参考)")
    print("="*90)
    print("| Instance | BKS |", end="")
    for config_name in ABLATION_CONFIGS:
        print(f" {config_name} |", end="")
    print()
    print("|----------|-----|" + "--------|" * len(ABLATION_CONFIGS))
    
    for i, name in enumerate(TEST_INSTANCES):
        if name not in all_instances:
            continue
        bks = BKS[name]
        print(f"| {name.ljust(8)} | {bks:3d} |", end="")
        for config_name in ABLATION_CONFIGS:
            r = all_results[config_name][i]
            print(f" {r['best_gap']:5.1f} |", end="")
        print()
    
    print("\n| Average  |     |", end="")
    for config_name in ABLATION_CONFIGS:
        gaps = [r['best_gap'] for r in all_results[config_name]]
        avg = sum(gaps) / len(gaps) if gaps else 0
        print(f" {avg:5.1f} |", end="")
    print()
    
    # 组件贡献度分析 (基于平均Gap)
    print("\n组件贡献度分析 (基于平均Gap，越低越好):")
    print("-"*70)
    full_gaps = [r['avg_gap'] for r in all_results['Full']]
    full_avg = sum(full_gaps) / len(full_gaps) if full_gaps else 0
    
    baseline_name = 'No Agent'
    baseline_gaps = [r['avg_gap'] for r in all_results[baseline_name]]
    baseline_avg = sum(baseline_gaps) / len(baseline_gaps) if baseline_gaps else 0
    
    impact_results = {}
    for config_name, results in all_results.items():
        if config_name == baseline_name:
            continue
        gaps = [r['avg_gap'] for r in results]
        avg = sum(gaps) / len(gaps) if gaps else 0
        # 相对无Agent全组件基线的改进量（越低越好，所以用基线减当前）。
        improvement_vs_baseline = baseline_avg - avg  # 正值=比无Agent基线好
        diff_vs_full = avg - full_avg  # 正值=比Full差
        # 判断：比无Agent全组件基线好（改进量>0）即协同有效。
        status = "有效" if improvement_vs_baseline > 0 else ("有害" if improvement_vs_baseline < 0 else "中性")
        impact_results[config_name] = {
            'avg_gap': avg,
            'improvement_vs_baseline': improvement_vs_baseline,
            'diff_vs_full': diff_vs_full,
            'status': status
        }
        print(f"  {config_name}: AvgGap={avg:.2f}%, 比基线({baseline_name})改进={improvement_vs_baseline:+.2f}%, vsFull差异={diff_vs_full:+.2f}% → {status}")
    
    # 额外分析
    print("\n  结论:")
    print(f"    - 基线({baseline_name}): AvgGap={baseline_avg:.2f}%")
    print(f"    - Full(Agent+全部组件): AvgGap={full_avg:.2f}%, 比无Agent基线改进={baseline_avg - full_avg:+.2f}%")
    
    # 单独分析Agent和FP不动点模型的贡献
    no_terminal = 'No Terminal Refinement'
    no_fp = 'No FP'
    no_local_search = 'No LocalSearch'
    if no_terminal in impact_results:
        terminal_gain = impact_results[no_terminal]['avg_gap'] - full_avg
        print(f"    - 终局精化协同收益: 关闭后AvgGap变化={terminal_gain:+.2f}%")
    if no_fp in impact_results:
        fp_gain = impact_results[no_fp]['avg_gap'] - full_avg
        print(f"    - FP协同收益: 关闭FP后AvgGap变化={fp_gain:+.2f}%")
    if no_local_search in impact_results:
        ls_gain = impact_results[no_local_search]['avg_gap'] - full_avg
        print(f"    - 局部搜索协同收益: 关闭局部搜索后AvgGap变化={ls_gain:+.2f}%")

    full_component_stats = [r.get('component_stats', {}) for r in all_results['Full']]
    if full_component_stats:
        avg_rescue_attempts = np.mean([
            item.get('avg_safe_rescue_attempts', 0.0) for item in full_component_stats
        ])
        avg_rescue_accepted = np.mean([
            item.get('avg_safe_rescue_accepted', 0.0) for item in full_component_stats
        ])
        avg_vnd_executions = np.mean([
            item.get('avg_fp_vnd_executions', 0.0) for item in full_component_stats
        ])
        avg_final_starts = np.mean([
            np.mean([run.get('final_refinement_starts', 0) for run in item.get('runs', [])])
            for item in full_component_stats
        ])
        print(
            f"    - 组件执行诊断: Agent安全救援={avg_rescue_accepted:.2f}/"
            f"{avg_rescue_attempts:.2f} 次/实例/运行, VND={avg_vnd_executions:.2f} 次/实例/运行, "
            f"终局多起点={avg_final_starts:.2f} 个/运行"
        )
    
    # ============ 保存结果到文件 ============
    save_results(all_results, impact_results, NUM_RUNS)


def save_results(all_results, impact_results, num_runs):
    """保存实验结果到文件"""
    # 创建结果目录
    result_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    # 准备保存的数据
    save_data = {
        'experiment': 'ablation',
        'num_runs': num_runs,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'bks_source': BKS_SOURCE,
        'bks_version': BKS_VERSION,
        'fp_model_mode': os.getenv('FP_MODEL_MODE', 'residual_shadow'),
        'instances': TEST_INSTANCES,
        'configs': ABLATION_CONFIGS,
        'results': all_results,
        'impact_analysis': impact_results,
        'bks': BKS
    }
    
    # 保存JSON文件
    suffix = result_suffix()
    json_path = os.path.join(
        result_dir, f'ablation_results_{num_runs}runs{suffix}.json'
    )
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {json_path}")
    
    # ============ 保存CSV文件（包含详细数据和汇总） ============
    csv_path = os.path.join(
        result_dir, f'ablation_results_{num_runs}runs{suffix}.csv'
    )
    with open(csv_path, 'w', encoding='utf-8') as f:
        # 第一部分：详细数据（每一轮）
        f.write('=== DETAIL DATA (每一轮) ===\n')
        f.write('Instance,BKS,Config,Run,Makespan,Gap(%),Time(s)\n')
        
        for i, name in enumerate(TEST_INSTANCES):
            bks = BKS[name]
            for config_name in ABLATION_CONFIGS:
                result = all_results[config_name][i]
                makespans = result.get('makespans', [])
                times = result.get('times', [])
                for run_idx, (m, t) in enumerate(zip(makespans, times)):
                    gap = (m - bks) / bks * 100 if bks > 0 else 0
                    f.write(f'{name},{bks},{config_name},{run_idx+1},{m},{gap:.2f},{t:.4f}\n')
        
        # 分隔线
        f.write('\n=== SUMMARY (平均值) ===\n')
        
        # 汇总表头
        f.write('Instance,BKS')
        for config in ABLATION_CONFIGS:
            f.write(
                f',{config}_AvgMakespan,{config}_AvgGap(%),'
                f'{config}_BestGap(%),{config}_StdMakespan,'
                f'{config}_BksHitRate,{config}_AvgTime(s)'
            )
        f.write('\n')
        
        # 汇总数据行
        for i, name in enumerate(TEST_INSTANCES):
            bks = BKS[name]
            f.write(f'{name},{bks}')
            for config in ABLATION_CONFIGS:
                result = all_results[config][i]
                avg_makespan = result['avg_makespan']
                avg_gap = result['avg_gap']
                best_makespan = result['best_makespan']
                best_gap = result['best_gap']
                std_makespan = result['std_makespan']
                bks_hit_rate = result['bks_hit_rate']
                avg_time = result['time']
                f.write(
                    f',{avg_makespan:.1f},{avg_gap:.2f},{best_gap:.2f},'
                    f'{std_makespan:.2f},{bks_hit_rate:.3f},{avg_time:.2f}'
                )
            f.write('\n')
        
        # 平均行
        f.write('Average,')
        for config in ABLATION_CONFIGS:
            gaps = [r['avg_gap'] for r in all_results[config]]
            avg = sum(gaps) / len(gaps) if gaps else 0
            stds = [r.get('std_makespan', 0.0) for r in all_results[config]]
            hit_rates = [r.get('bks_hit_rate', 0.0) for r in all_results[config]]
            times = [r.get('time', 0.0) for r in all_results[config]]
            avg_std = sum(stds) / len(stds) if stds else 0.0
            avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
            avg_time = sum(times) / len(times) if times else 0.0
            f.write(f',,{avg:.2f},,{avg_std:.2f},{avg_hit_rate:.3f},{avg_time:.2f}')
        f.write('\n')
    
    print(f"CSV文件已保存到: {csv_path}")


if __name__ == '__main__':
    main()

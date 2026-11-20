"""
Meta-Scheduler: 调度 Agent-FP 全流程
整合Observer, Controller, Executor, FixedPoint-Detector

核心设计原则：
1. 协调机制：Agent、FixedPoint模型和Executor分工明确
   - Agent：负责观察、调度组件和分析执行反馈
   - FixedPoint模型：只负责收敛判断，不执行优化
   - Executor：执行局部搜索和VND
2. 优先级策略：Agent根据FixedPoint模型的状态调度VND
3. 操作互斥：同一时间只执行一种类型的干预操作
4. 状态感知：基于当前收敛状态决定执行哪个组件的建议
"""
import json
import math
import os
import random
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass, field, asdict

from tools.jssp_core import JSSPInstance, JSSPUtils, JSSPEncoder
from tools.observer import Observer
from tools.controller import Controller, GAParams  # v9.1: 增强版控制器
from tools.executor import Executor, Population, CrossoverOperator, MutationOperator
from tools.fixed_point_detector import FixedPointDetector, FixedPointStatus, ConvergenceLevel
from tools.search_orchestration import BudgetLedger, SearchAction
from tools.fp_signals import FPObservationTracker


@dataclass
class FinalResult:
    best_makespan: int
    best_chromosome: list
    best_schedule: list
    convergence_history: list
    total_generations: int
    fixed_point_generation: int
    final_diversity: float
    final_params: GAParams
    orchestration: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d['best_makespan'] = int(self.best_makespan)
        d['total_generations'] = int(self.total_generations)
        d['fixed_point_generation'] = int(self.fixed_point_generation)
        d['final_diversity'] = float(self.final_diversity)
        d['convergence_history'] = [int(v) for v in self.convergence_history]
        sched = []
        for item in self.best_schedule:
            sched.append({
                "job_id": int(item["job_id"]),
                "op_index": int(item["op_index"]),
                "machine": int(item["machine"]),
                "start_time": int(item["start_time"]),
                "end_time": int(item["end_time"]),
                "processing_time": int(item["processing_time"])
            })
        d['best_schedule'] = sched
        d['best_chromosome'] = [int(g) for g in self.best_chromosome]
        return d


class MetaScheduler:
    """
    统筹Observer、Agent、FixedPoint不动点模型和Executor。
    FixedPoint模型只输出收敛状态，Agent下发执行计划。
    """
    def __init__(self, instance: JSSPInstance):
        self.instance = instance
        self.observer = Observer()
        self.controller = Controller(instance.num_jobs, instance.num_machines)
        self.executor = Executor(instance)
        total_ops = instance.num_jobs * instance.num_machines

        # ===== v6.5: 实例形状感知，设置FP检测器 =====
        aspect_ratio = max(instance.num_jobs, instance.num_machines) / min(instance.num_jobs, instance.num_machines)
        is_asymmetric = aspect_ratio > 1.3

        # 动态确定进化参数。质量优先：中大型实例需要足够代数让
        # 初始化的派工解、交叉和终局精化能够协同，而不是过早停止在
        # 随机初始解附近。
        if total_ops <= 30:
            self.max_generations = 260
            self.early_stop_stagnant = 85
            self.pop_size = 120
        elif total_ops <= 50:
            self.max_generations = 360
            self.early_stop_stagnant = 105
            self.pop_size = 145
        elif total_ops <= 100:
            self.max_generations = 500
            self.early_stop_stagnant = 135
            self.pop_size = 175
        elif total_ops <= 200:
            self.max_generations = 620
            self.early_stop_stagnant = 165
            self.pop_size = 200
        else:
            self.max_generations = 760
            self.early_stop_stagnant = 200
            self.pop_size = 220

        self.fp_detector = FixedPointDetector(problem_size=total_ops, max_generations=self.max_generations)
        self.fp_detector.set_asymmetric(is_asymmetric)  # v6.5: 设置实例形状

        # v8.2: 向Controller传递早停阈值，用于兼容自适应模式的阈值保护
        self.controller.set_max_generations(self.max_generations)
        self.controller.set_early_stop_stagnant(self.early_stop_stagnant)
        
        # 协调状态
        self.controller_paused = False
        self.last_fp_action_gen = -50
        self.fp_action_cooldown = 0
        self.final_refinement_start_count = 0
        self.final_archive_start_count = 0
        self.agent_trace = []
        self.ap_trace = self.agent_trace  # 旧诊断结果兼容别名
        
        # SA冷却参数（用于局部搜索的SA接受准则）
        self.sa_initial_temp = 5.0
        self.sa_cooling_rate = 0.995

    def _init_population(self, size: int):
        """
        使用多种启发式规则初始化种群（增强版）
        包含：随机初始化、MWKR、LPT、SPT、关键路径优先、混合策略
        
        改进：
        1. 提高随机比例到50%，增加初始多样性
        2. 添加关键路径优先策略
        3. 添加混合策略（从多个启发式中组合）
        """
        num_jobs = self.instance.num_jobs
        num_machines = self.instance.num_machines
        population = []
        
        # 随机个体用于探索；其余个体由真实派工规则构造。旧版本将同一
        # 工件连续排布，几乎没有利用机器约束，导致初始种群质量偏低。
        random_ratio = 0.30
        mwkr_ratio = 0.18
        lpt_ratio = 0.16
        spt_ratio = 0.14
        hybrid_ratio = 0.22
        
        n_random = max(1, int(size * random_ratio))
        n_mwkr = max(1, int(size * mwkr_ratio))
        n_lpt = max(1, int(size * lpt_ratio))
        n_spt = max(1, int(size * spt_ratio))
        n_hybrid = size - n_random - n_mwkr - n_lpt - n_spt
        
        # 1. 随机初始化（增加多样性）
        for _ in range(n_random):
            chrom = JSSPUtils.generate_random_chromosome(num_jobs, num_machines)
            population.append(chrom)
        
        # 2-5. 每一步按当前工件/机器可用时间选择下一工序，因此生成的
        # 染色体与派工策略对应的可行调度一致。
        for _ in range(n_mwkr):
            chrom = self._generate_dispatch_chromosome('mwkr', randomness=0.18)
            population.append(chrom)
        
        # 3. LPT启发式：优先加工需要总加工时间最长的工件
        for _ in range(n_lpt):
            chrom = self._generate_dispatch_chromosome('lpt', randomness=0.16)
            population.append(chrom)
        
        # 4. SPT启发式：优先加工需要总加工时间最短的工件
        for _ in range(n_spt):
            chrom = self._generate_dispatch_chromosome('spt', randomness=0.16)
            population.append(chrom)
        
        # 5. 混合策略：从两个启发式中各取一半工件
        for _ in range(n_hybrid):
            chrom = self._generate_dispatch_chromosome('hybrid', randomness=0.24)
            population.append(chrom)
        
        # 打乱种群顺序增加多样性
        random.shuffle(population)
        return population

    def _fixed_vnd_plan(self, fp_status, generation: int,
                        allow_terminal_refine: bool = True):
        """无Agent基线的固定VND计划。

        FP只提供不动点收敛报告；无Agent基线不调用Agent，而是使用固定
        的调度规则模拟组件授权。
        强度相近的固定规则。真正的VND仍由下面的Executor执行。
        """
        empty = {
            'enabled': False,
            'stage': None,
            'apply_vnd': False,
            'top_k_ratio': 0.0,
            'local_search_iterations': 0,
        }
        if fp_status is None:
            return empty
        if getattr(fp_status, 'signal_v2', None) is not None:
            raise ValueError('Semantic FP status requires the typed Agent path, not the legacy fixed helper')
        progress = (generation + 1) / max(self.max_generations, 1)
        stagnant = int(getattr(fp_status, 'stagnant_generations', 0))
        level = getattr(
            getattr(fp_status, 'convergence_level', None), 'value', ''
        )
        converged = bool(getattr(fp_status, 'is_fixed_point', False))
        sign = level in ('early_sign', 'partial', 'full')
        if (allow_terminal_refine and progress >= 0.70
                and (converged or sign)):
            stage, top_k, iterations = 'final', 0.04, 360
        elif progress >= 0.45 and (converged or sign) and stagnant >= 8:
            stage, top_k, iterations = 'mid', 0.04, 180
        elif progress >= 0.25 and sign and stagnant >= 6:
            stage, top_k, iterations = 'early', 0.06, 120
        else:
            return empty
        return {
            'enabled': True,
            'stage': stage,
            'apply_vnd': True,
            'top_k_ratio': top_k,
            'local_search_iterations': iterations,
            'fp_guided': True,
            'fp_gate_reason': f'fixed_baseline_{level}',
            'fp_convergence_level': level,
            'fp_stagnant_generations': stagnant,
        }

    def _generate_dispatch_chromosome(self, rule: str, randomness: float = 0.0) -> np.ndarray:
        """构造基于活动调度的工序序列。

        每次只从各工件的下一道可行工序中选择一个。评分同时考虑机器和
        工件的最早可开工时间，避免了把一个工件的全部工序连续堆入染色体
        造成的长等待。随机受限候选表保留多样性而不退化为纯随机初始化。
        """
        num_jobs = self.instance.num_jobs
        num_machines = self.instance.num_machines
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        chromosome = []

        while len(chromosome) < num_jobs * num_machines:
            candidates = []
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= num_machines:
                    continue
                machine = self.instance.machine_sequences[job_id][op_idx]
                duration = self.instance.processing_times[job_id][op_idx]
                start = max(job_ready[job_id], machine_ready[machine])
                completion = start + duration
                remaining_work = sum(self.instance.processing_times[job_id][op_idx:])
                remaining_ops = num_machines - op_idx

                if rule == 'spt':
                    score = (duration, start, completion, remaining_work)
                elif rule == 'lpt':
                    score = (start, -duration, completion, -remaining_work)
                elif rule == 'mwkr':
                    score = (start, -remaining_work, completion, -duration)
                else:  # hybrid: earliest completion with remaining-work pressure
                    score = (completion - 0.30 * remaining_work,
                             start, -remaining_ops, -duration)
                candidates.append((score, job_id, machine, completion))

            candidates.sort(key=lambda item: item[0])
            # 不能向下取整：0.16-0.24 的配置此前都会得到大小为1的候选
            # 集，导致同一派工规则反复生成完全相同的染色体。
            rcl_size = min(
                len(candidates),
                1 + int(math.ceil(randomness * min(5, len(candidates))))
            )
            _, job_id, machine, completion = random.choice(candidates[:max(1, rcl_size)])
            chromosome.append(job_id)
            next_op[job_id] += 1
            job_ready[job_id] = completion
            machine_ready[machine] = completion

        return np.array(chromosome, dtype=int)

    def _update_elite_archive(self, candidates, fitness_values, max_size: int = 8):
        """维护不参与进化的多样性精英档案。"""
        pool = list(getattr(self, '_elite_archive', []))
        pool.extend((float(cost), individual.copy())
                    for individual, cost in zip(candidates, fitness_values))

        unique = {}
        for cost, individual in pool:
            key = tuple(int(gene) for gene in individual)
            if key not in unique or cost < unique[key][0]:
                unique[key] = (cost, individual.copy())

        ranked = sorted(unique.values(), key=lambda item: item[0])
        archive = []
        min_distance = 0.08
        for cost, individual in ranked:
            if not archive:
                archive.append((cost, individual.copy()))
            else:
                distance = min(
                    JSSPUtils.hamming_distance(individual, archived) / len(individual)
                    for _, archived in archive
                )
                if distance >= min_distance:
                    archive.append((cost, individual.copy()))
            if len(archive) >= max_size:
                break

        self._elite_archive = archive

    def _try_adaptive_elite_rescue(self, diversity: float) -> tuple[bool, bool, float]:
        """兼容旧版自适应参数模式：生成候选并写入独立档案，不改变主种群。"""
        if len(self.population) < 4:
            return False, False, 0.0

        ranked = np.argsort(np.array(self.fitness_values))
        best_before = self.fitness_values[ranked[0]]
        parent1 = self.population[ranked[0]].copy()
        parent2 = self.population[ranked[min(3, len(ranked) - 1)]].copy()
        crossover_name, mutation_name = self.controller.get_safe_rescue_combo(diversity)
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        try:
            child1, child2 = CrossoverOperator.crossover_by_name(
                parent1, parent2, crossover_name
            )
            candidates = [
                MutationOperator.mutation_by_name(child1, mutation_name),
                MutationOperator.mutation_by_name(child2, mutation_name),
            ]
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)

        candidate_fitness = self.executor.evaluate_population(candidates)
        candidate_index = int(np.argmin(candidate_fitness))
        candidate_cost = candidate_fitness[candidate_index]
        candidate_key = tuple(int(gene) for gene in candidates[candidate_index])
        self._update_elite_archive(candidates, candidate_fitness)
        accepted = any(
            tuple(int(gene) for gene in archived) == candidate_key
            for _, archived in self._elite_archive
        )

        improvement_ratio = (
            (best_before - candidate_cost) / max(abs(best_before), 1.0)
        )
        return accepted, candidate_cost < best_before, improvement_ratio

    def _select_diverse_refinement_indices(self, max_count: int) -> list[int]:
        """选择终局精化起点：兼顾适应度排名和染色体结构差异。"""
        if not self.population:
            return []

        fitness = np.asarray(self.fitness_values, dtype=float)
        ranked = np.argsort(fitness).tolist()
        max_count = min(max(1, int(max_count)), len(ranked))
        pool = ranked[:min(len(ranked), max_count * 4)]
        selected = [pool[0]]

        while len(selected) < max_count:
            remaining = [idx for idx in pool if idx not in selected]
            if not remaining:
                break
            quality_denominator = max(len(pool) - 1, 1)
            best_candidate = None
            best_score = -float('inf')
            for idx in remaining:
                rank = pool.index(idx)
                quality = 1.0 - rank / quality_denominator
                distance = min(
                    JSSPUtils.hamming_distance(self.population[idx], self.population[chosen])
                    / max(len(self.population[idx]), 1)
                    for chosen in selected
                )
                # 质量占主导，距离用于避免所有起点落在同一局部盆地。
                score = 0.70 * quality + 0.30 * distance
                if score > best_score:
                    best_score = score
                    best_candidate = idx
            selected.append(best_candidate)
        return selected
    
    def _generate_hybrid_chromosome(self, num_jobs: int, num_machines: int) -> np.ndarray:
        """
        混合策略：从MWKR取前半部分工件，剩余从SPT中选取
        确保所有工件都出现且只出现一次
        """
        # 计算每个工件的总加工时间
        job_total_times = []
        for job_id in range(num_jobs):
            total_time = sum(self.instance.processing_times[job_id])
            job_total_times.append((job_id, total_time))
        
        # MWKR：按总时间降序
        mwkr_order = sorted(job_total_times, key=lambda x: x[1], reverse=True)
        # SPT：按总时间升序
        spt_order = sorted(job_total_times, key=lambda x: x[1])
        
        # 从MWKR取前一半，剩余从SPT中选取（不重复）
        mid = num_jobs // 2
        selected_jobs = set()
        
        # 先从MWKR选前半
        hybrid_order = []
        for job_id, _ in mwkr_order[:mid]:
            hybrid_order.append(job_id)
            selected_jobs.add(job_id)
        
        # 再从SPT中选剩余的（排除已选的）
        for job_id, _ in spt_order:
            if job_id not in selected_jobs:
                hybrid_order.append(job_id)
                selected_jobs.add(job_id)
                if len(hybrid_order) >= num_jobs:
                    break
        
        # 生成染色体
        chromosome = []
        for job_id in hybrid_order:
            chromosome.extend([job_id] * num_machines)
        
        # 添加随机扰动
        if random.random() < 0.4:
            for _ in range(random.randint(2, 4)):
                idx1, idx2 = random.sample(range(len(chromosome)), 2)
                chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        
        return np.array(chromosome)
    
    def _generate_mwkr_chromosome(self, num_jobs: int, num_machines: int) -> np.ndarray:
        """
        MWKR (Most Work Remaining First) 启发式
        优先加工剩余总工作量最多的工件
        """
        # 计算每个工件的总加工时间
        job_total_times = []
        for job_id in range(num_jobs):
            total_time = sum(self.instance.processing_times[job_id])
            job_total_times.append((job_id, total_time))
        
        # 按总加工时间降序排序
        job_total_times.sort(key=lambda x: x[1], reverse=True)
        job_order = [job_id for job_id, _ in job_total_times]
        
        # 生成染色体：按工件顺序依次排列，每个工件的所有工序连续
        chromosome = []
        for job_id in job_order:
            chromosome.extend([job_id] * num_machines)
        
        # 添加一些随机扰动
        if random.random() < 0.3:
            for _ in range(random.randint(1, 3)):
                idx1, idx2 = random.sample(range(len(chromosome)), 2)
                chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        
        return np.array(chromosome)
    
    def _generate_lpt_chromosome(self, num_jobs: int, num_machines: int) -> np.ndarray:
        """
        LPT (Longest Processing Time First) 启发式
        基于每个工件的单个最长工序时间排序
        """
        # 计算每个工件的最长工序时间
        job_max_times = []
        for job_id in range(num_jobs):
            max_time = max(self.instance.processing_times[job_id])
            job_max_times.append((job_id, max_time))
        
        # 按最长工序时间降序排序
        job_max_times.sort(key=lambda x: x[1], reverse=True)
        job_order = [job_id for job_id, _ in job_max_times]
        
        # 生成染色体
        chromosome = []
        for job_id in job_order:
            chromosome.extend([job_id] * num_machines)
        
        # 添加随机扰动
        if random.random() < 0.3:
            for _ in range(random.randint(1, 3)):
                idx1, idx2 = random.sample(range(len(chromosome)), 2)
                chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        
        return np.array(chromosome)
    
    def _generate_spt_chromosome(self, num_jobs: int, num_machines: int) -> np.ndarray:
        """
        SPT (Shortest Processing Time First) 启发式
        优先加工总加工时间最短的工件
        """
        # 计算每个工件的总加工时间
        job_total_times = []
        for job_id in range(num_jobs):
            total_time = sum(self.instance.processing_times[job_id])
            job_total_times.append((job_id, total_time))
        
        # 按总加工时间升序排序（最短优先）
        job_total_times.sort(key=lambda x: x[1])
        job_order = [job_id for job_id, _ in job_total_times]
        
        # 生成染色体
        chromosome = []
        for job_id in job_order:
            chromosome.extend([job_id] * num_machines)
        
        # 添加随机扰动
        if random.random() < 0.3:
            for _ in range(random.randint(1, 3)):
                idx1, idx2 = random.sample(range(len(chromosome)), 2)
                chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        
        return np.array(chromosome)

    def optimize(self,
                 use_adaptive_params: bool = True,
                 use_fixed_point: bool = True,
                 use_local_search: bool = True,
                 use_vnd: bool = True,
                 use_terminal_refinement: bool = True,
                 use_ap_commander: bool = True,
                 collect_ap_diagnostics: bool = False,
                 use_agent_commander: Optional[bool] = None,
                 verbose: Optional[bool] = None,
                 budget_mode: Optional[str] = None,
                 search_budget: Optional[float] = None,
                 vnd_schedule: Optional[str] = None,
                 paper_q=None,
                 fp_semantic_mode: str = 'legacy') -> FinalResult:
        """
        主优化循环
        
        Args:
            use_adaptive_params: 是否启用Controller自适应参数
            use_fixed_point: 是否启用FixedPointDetector收敛检测
            use_local_search: 是否启用局部搜索
            use_vnd: 是否启用由Agent根据FP状态授权的VND；默认开启
            use_terminal_refinement: 是否启用终局强化、终局多起点VND和最终SA
            use_ap_commander: 旧版兼容参数，等价于use_agent_commander
            collect_ap_diagnostics: 是否记录逐代Agent诊断轨迹（默认关闭，参数名保留兼容）
            use_agent_commander: Agent是否统一下发局部搜索、FP检查、VND
                和终局精化计划
            verbose: 是否输出逐代日志；默认读取JSSP_VERBOSE，关闭日志不改变搜索轨迹。
            budget_mode: 搜索预算模式，shadow只记录，enforce拒绝超预算动作。
            search_budget: 统一组件工作预算；为空时读取AGENT_SEARCH_BUDGET。
            vnd_schedule: 无FP且无Agent时的可选VND基线，支持fixed或stagnation；
                默认none，保持历史消融语义。
            fp_semantic_mode: 显式选择legacy、semantic_nofp或semantic_original_fp。
                新模式只修复信号时间单位，不改变底层算子。legacy保持冻结行为。
        """
        if fp_semantic_mode not in ('legacy', 'semantic_nofp', 'semantic_original_fp'):
            raise ValueError('Unknown fp_semantic_mode')
        if fp_semantic_mode != 'legacy':
            commander_enabled = (use_ap_commander if use_agent_commander is None
                                 else bool(use_agent_commander))
            if not use_fixed_point or not use_adaptive_params or not commander_enabled:
                raise ValueError('semantic modes require shared FP observation and Agent framework')
            if paper_q is not None:
                raise ValueError('New semantic modes require a separate output protocol from frozen Q')
        # Resetting the opt-in mode makes repeated explicit calls unambiguous;
        # the default remains the historical detector/controller path.
        self.fp_detector.semantic_mode = (
            'legacy' if fp_semantic_mode == 'legacy' else 'generation_v2'
        )
        self.fp_detector.signal_tracker = FPObservationTracker()
        if verbose is None:
            verbose = os.getenv('JSSP_VERBOSE', '0').lower() in ('1', 'true', 'yes')
        self.termination_reason = 'max_generations'
        self.current_generation = -1
        self.controller.feedback_adaptation_enabled = (
            paper_q.feedback if paper_q is not None else True
        )

        def set_query_phase(phase):
            meter = getattr(self.executor, 'query_meter', None)
            if meter is not None:
                meter.phase = phase
        if use_agent_commander is not None:
            use_ap_commander = bool(use_agent_commander)
        if budget_mode is None:
            budget_mode = os.getenv('AGENT_BUDGET_MODE', 'shadow')
        if search_budget is None:
            raw_budget = os.getenv('AGENT_SEARCH_BUDGET', '')
            search_budget = float(raw_budget) if raw_budget else None
        if vnd_schedule is None:
            vnd_schedule = os.getenv('AGENT_VND_SCHEDULE', 'none')
        self.budget_ledger = BudgetLedger(search_budget, budget_mode)
        # 保存本次运行的初始随机状态。终局会用它复现基准SA的候选链，
        # 然后与GA/VND候选做精英选择；不影响前面FP-AGA的正常探索。
        initial_python_rng_state = random.getstate()
        initial_numpy_rng_state = np.random.get_state()
        self.agent_trace = []
        self.ap_trace = self.agent_trace  # 旧诊断结果兼容别名
        self.final_refinement_start_count = 0
        self.final_archive_start_count = 0
        self.agent_directive_trace = []
        self.ap_directive_trace = self.agent_directive_trace  # 旧字段兼容

        # 初始化种群
        set_query_phase('initialization')
        self.population = self._init_population(self.pop_size)
        self.fitness_values = self.executor.evaluate_population(self.population)
        self._elite_archive = []
        self._update_elite_archive(self.population, self.fitness_values)

        best_makespan = int(min(self.fitness_values))
        best_chromosome = self.population[int(np.argmin(self.fitness_values))].copy()

        convergence = [best_makespan]
        stale_count = 0
        fp_detected_gen = -1
        
        # 终局强化参数
        final_intensification_start = int(self.max_generations * 0.78)
        final_intensification_done = False
        terminal_authorized = not (use_adaptive_params and use_ap_commander)
        terminal_called = False
        terminal_pre_best = None
        terminal_elapsed = 0.0
        terminal_work_calls = 0.0
        terminal_action_registered = False
        terminal_started = time.perf_counter()

        def authorize_action(payload, component, enabled=True, budget=None,
                            target='population', target_size=1.0,
                            mode='default', reason='scheduler'):
            """Register one structured action and optionally enforce its budget."""
            if payload:
                action = SearchAction(**payload)
                if not enabled and action.enabled:
                    action = SearchAction(
                        component=action.component, enabled=False,
                        target=action.target, target_size=action.target_size,
                        budget=action.budget, budget_unit=action.budget_unit,
                        mode=action.mode, reason=action.reason,
                        expected_roi=action.expected_roi,
                        metadata=action.metadata,
                    )
            else:
                action = SearchAction(
                    component=component, enabled=bool(enabled),
                    target=target, target_size=float(target_size),
                    budget=float(budget or 0.0), budget_unit='work_calls',
                    mode=mode, reason=reason,
                )
            return self.budget_ledger.authorize(action)

        def authorize_terminal_once():
            """Register the terminal plan once, even when early-stop also fires."""
            nonlocal terminal_action_registered
            if terminal_action_registered:
                return True
            terminal_action_registered = True
            total_ops = self.instance.num_jobs * self.instance.num_machines
            n_final_top = (
                min(22, max(12, int(self.pop_size * 0.16)))
                if total_ops > 400 else
                min(16, max(6, int(self.pop_size * 0.12)))
            )
            final_starts = (
                min(14, max(10, int(self.pop_size * 0.06)))
                if total_ops > 400 else
                min(10, max(6, int(self.pop_size * 0.04)))
            )
            archive_starts = 8 if total_ops > 400 else 5
            # Reserve the two possible terminal passes: the normal terminal
            # window and the final post-loop multi-start pass. Actual work is
            # recorded separately and is usually lower than this cap.
            estimated_work = float(
                n_final_top + final_starts + archive_starts +
                (3 if total_ops > 400 else 2) + 2
            )
            return authorize_action(
                None, 'terminal', enabled=True, budget=estimated_work,
                target='elite_archive', target_size=1.0,
                mode='multi_start', reason='terminal',
            )

        for generation in range(self.max_generations):
            self.current_generation = generation
            set_query_phase('ga')
            # --- Observer: 种群状态观察 ---
            obs = self.observer.observe(
                self.population, self.fitness_values, generation
            )

            # --- FixedPoint-Detector: 首先进行收敛检测 ---
            # v8.0建议1：FP检测频率自适应化
            # 早期(0-40%)：每5代检测（减少开销）
            # 中期(40-80%)：每3代检测
            # 后期(80-100%)：每代检测（保证收敛精度）
            fp_status = None
            vnd_plan = {
                'enabled': False,
                'stage': None,
                'apply_vnd': False,
                'top_k_ratio': 0.0,
                'local_search_iterations': 0,
            }
            exec_params = {
                'inject_random': False,
                'inject_ratio': 0.0,
                'boost_mutation': False,
                'pm_multiplier': 1.0,
                'local_search_on_best': False,
                'local_search_iterations': 0,
                'apply_vnd': False,
            }
            fp_action_type = 'none'
            fp_check_called = False
            vnd_called = False
            vnd_improved = False
            vnd_gain = 0.0
            vnd_elapsed = 0.0
            vnd_work_calls = 0.0
            vnd_fp_guided = False
            vnd_fp_reason = None
            progress_ratio = generation / max(self.max_generations, 1)

            # 指挥模式下，Agent先基于只读状态授权执行组件；它不再接触GA参数、
            # 算子选择、随机注入或救援候选。
            # 诊断模式也采集无Agent基线的组件执行反馈；这只更新统计对象，
            # 不参与无Agent路径的任何搜索或参数决策。
            if use_adaptive_params and use_ap_commander:
                ap_directive = self.controller.decide_execution_directive({
                    'diversity': obs.state.diversity,
                    'stagnant_count': obs.state.stagnant_generations,
                    'generation': generation,
                    'max_generations': self.max_generations,
                    'fp_vnd_count': self.fp_detector.vnd_executed_count,
                    'component_feedback': self.controller.component_feedback,
                    'budget_remaining': (
                        max(0, self.executor.query_meter.budget - self.executor.query_meter.queries)
                        if paper_q is not None else self.budget_ledger.remaining
                    ),
                })
                terminal_authorized = bool(
                    ap_directive['allow_terminal_refinement']
                )
            else:
                ap_directive = {
                    'directive': 'legacy' if use_adaptive_params else 'disabled',
                    'allow_fp_action': True,
                    'allow_terminal_refinement': True,
                    'allow_generation_local_search': True,
                    'local_search_plan': None,
                    'fp_plan': {
                        'allow_action': True,
                        'check_interval': None,
                    },
                    'components': {
                        'local_search_enabled': True,
                        'fixed_point_enabled': True,
                        'terminal_refinement_enabled': True,
                        'local_search_plan': None,
                        'fixed_point_plan': None,
                    },
                    'progress': progress_ratio,
                }
                terminal_authorized = True

            component_plan = dict(ap_directive['components'])
            # 实验配置是组件能力边界；Agent只能在已启用的组件集合内调度。
            component_plan['local_search_enabled'] = bool(
                component_plan.get('local_search_enabled', True)
                and use_local_search
            )
            component_plan['fixed_point_enabled'] = bool(
                component_plan.get('fixed_point_enabled', True)
                and use_fixed_point
            )
            component_plan['terminal_refinement_enabled'] = bool(
                component_plan.get('terminal_refinement_enabled', True)
                and use_terminal_refinement
            )
            terminal_authorized = (
                terminal_authorized
                and component_plan['terminal_refinement_enabled']
            )

            # Agent动作进入统一预算账本。shadow模式只记录动作，不改变
            # 既有性能路径；enforce模式下超预算的局部搜索计划会被拒绝。
            ls_payload = (ap_directive.get('actions') or {}).get('local_search')
            ls_budget_ok = authorize_action(
                ls_payload, 'local_search',
                enabled=bool(component_plan['local_search_enabled']
                              and use_local_search),
                # Non-Agent baselines have no action payload. Estimate their
                # fixed elite-search call count so enforce-mode comparisons
                # use the same ledger as Agent scheduling.
                budget=(15.0 if use_local_search else 0.0),
                target='elite', target_size=15.0,
                mode='baseline', reason='fixed_schedule',
            )
            if not ls_budget_ok and self.budget_ledger.mode == 'enforce':
                component_plan['local_search_enabled'] = False
                if component_plan.get('local_search_plan'):
                    component_plan['local_search_plan'] = dict(
                        component_plan['local_search_plan']
                    )
                    component_plan['local_search_plan']['enabled'] = False

            # v8.0建议1：自适应FP检测频率
            if use_adaptive_params and use_ap_commander:
                fp_detect_interval = ap_directive['fp_plan']['check_interval']
            elif progress_ratio < 0.40:
                fp_detect_interval = 5  # 早期：每5代
            elif progress_ratio < 0.80:
                fp_detect_interval = 3  # 中期：每3代
            else:
                fp_detect_interval = 1  # 后期：每代

            fp_budget_ok = True
            if (use_fixed_point and component_plan['fixed_point_enabled']
                    and generation % fp_detect_interval == 0):
                fp_payload = (ap_directive.get('actions') or {}).get('fp_detection')
                fp_budget_ok = authorize_action(
                    fp_payload, 'fp_detection', enabled=True, budget=1.0,
                    target='population_state', target_size=1.0,
                    mode='convergence_check', reason='fixed_schedule',
                )

            if (use_fixed_point and component_plan['fixed_point_enabled']
                    and fp_budget_ok
                    and generation % fp_detect_interval == 0):
                fp_status, _ = self.fp_detector.detect(
                    self.population, self.fitness_values,
                    self.controller, generation,
                    allow_final_refine=(
                        use_terminal_refinement and terminal_authorized
                    ),
                    allow_actions=bool(ap_directive['allow_fp_action']),
                    action_policy=ap_directive.get('fp_plan'),
                    **({'observed_stagnant_generations': int(obs.state.stagnant_generations)}
                       if fp_semantic_mode != 'legacy' else {}),
                )
                fp_check_called = True
                
                if fp_status.is_fixed_point and fp_detected_gen < 0:
                    fp_detected_gen = generation
                
            if (fp_check_called or (paper_q is not None and paper_q.policy in ('fixed', 'stagnation'))):
                # FP只报告不动点收敛状态；VND计划由Agent或无Agent固定基线生成。
                if use_adaptive_params and use_ap_commander:
                    vnd_observation = {
                            'generation': generation,
                            'max_generations': self.max_generations,
                            'directive': ap_directive['directive'],
                            'fp_vnd_authorized': bool(
                                ap_directive['allow_fp_action']
                            ),
                            'component_feedback': self.controller.component_feedback,
                        }
                    if fp_semantic_mode == 'semantic_nofp':
                        # Detector still runs at identical observation epochs,
                        # but no FP status or residual reaches this decision.
                        vnd_plan = self.controller.decide_vnd_plan(
                            vnd_observation, non_fp_signal={
                                'eligible': obs.state.stagnant_generations >= 8,
                                'stagnant_generations': int(obs.state.stagnant_generations),
                                'reason': 'semantic_observer_stagnation',
                            },
                        )
                    elif paper_q is not None and paper_q.policy != 'agent_fp':
                        non_fp_signal = paper_q.non_fp_signal(
                            generation, obs.state.stagnant_generations
                        )
                        if paper_q.policy in ('fixed', 'stagnation'):
                            vnd_observation['fp_vnd_authorized'] = True
                        vnd_plan = self.controller.decide_vnd_plan(
                            vnd_observation, non_fp_signal=non_fp_signal,
                        )
                    else:
                        vnd_plan = self.controller.decide_vnd_plan(
                            vnd_observation, fp_status,
                            **({'fp_signal': fp_status.signal_v2}
                               if fp_semantic_mode == 'semantic_original_fp' else {}),
                        )
                else:
                    vnd_plan = self._fixed_vnd_plan(
                        fp_status,
                        generation,
                        allow_terminal_refine=use_terminal_refinement,
                    )
                vnd_fp_guided = bool(vnd_plan.get('fp_guided', False))
                vnd_fp_reason = vnd_plan.get('fp_gate_reason')
                fp_action_type = fp_status.convergence_level.value if fp_status else 'none'

                if use_vnd and vnd_plan.get('enabled', False):
                    # One unit is one candidate sent to VND. The internal
                    # neighborhood depth remains an execution parameter.
                    estimated_vnd_calls = max(
                        1,
                        int(self.pop_size * vnd_plan.get('top_k_ratio', 0.0))
                    )
                    vnd_budget = float(estimated_vnd_calls)
                    vnd_action = SearchAction(
                        component='vnd', enabled=True, target='elite',
                        target_size=float(vnd_plan.get('top_k_ratio', 0.0)),
                        budget=vnd_budget, mode=str(vnd_plan.get('stage', 'mid')),
                        budget_unit='work_calls',
                        reason=str(vnd_plan.get('fp_gate_reason', 'fp')), 
                        expected_roi=self.controller.component_roi_snapshot().get(
                            'vnd', 0.0
                        ),
                    )
                    vnd_budget_ok = self.budget_ledger.authorize(vnd_action)
                    if (not vnd_budget_ok
                            and self.budget_ledger.mode == 'enforce'):
                        vnd_plan = {
                            'enabled': False, 'apply_vnd': False,
                            'stage': None, 'top_k_ratio': 0.0,
                            'local_search_iterations': 0,
                            'budget_rejected': True,
                        }
                        vnd_fp_guided = False

            # Optional controlled baseline: schedule VND from a fixed clock or
            # phenotype stagnation, without consulting FP. This branch is
            # opt-in and is intentionally outside the normal ablation path.
            if (use_vnd and not use_fixed_point and not use_agent_commander
                    and vnd_schedule in ('fixed', 'stagnation')):
                should_schedule = (
                    generation % 10 == 0 if vnd_schedule == 'fixed' else
                    obs.state.stagnant_generations >= 8
                )
                if should_schedule:
                    vnd_plan = {
                        'enabled': True, 'apply_vnd': True,
                        'stage': ('fixed_schedule' if vnd_schedule == 'fixed'
                                  else 'stagnation_schedule'),
                        'top_k_ratio': (0.06 if self.controller.total_ops <= 100
                                        else 0.04),
                        'local_search_iterations': 180,
                        'fp_guided': False,
                        'fp_gate_reason': vnd_schedule,
                    }
                    estimated_vnd_calls = max(
                        1, int(self.pop_size * vnd_plan['top_k_ratio'])
                    )
                    vnd_budget_ok = self.budget_ledger.authorize(SearchAction(
                        component='vnd', enabled=True, target='elite',
                        target_size=vnd_plan['top_k_ratio'],
                        budget=float(estimated_vnd_calls),
                        budget_unit='work_calls', mode=vnd_plan['stage'],
                        reason=vnd_schedule,
                    ))
                    if not vnd_budget_ok and self.budget_ledger.mode == 'enforce':
                        vnd_plan['enabled'] = False
                        vnd_plan['apply_vnd'] = False
            
            # --- Controller: FP感知的分阶段参数调整 ---
            if use_adaptive_params and use_ap_commander:
                # Agent指挥模式保持基础GA参数固定，搜索动作由局部搜索/VND执行。
                params = GAParams(
                    pc=0.85,
                    pm=0.05,
                    n=2,
                    selection_method="tournament",
                    tournament_size=3,
                    population_size=self.pop_size,
                )
                self.controller_paused = False
                ap_crisis_active = False
            elif use_adaptive_params:
                # v9.0: 构建包含FP状态的观测字典（双向感知）
                obs_dict = {
                    'diversity': obs.state.diversity,
                    'improvement_rate': obs.state.improvement_rate,
                    'stagnant_count': obs.state.stagnant_generations,
                    'best_fitness': float(np.min(self.fitness_values)),
                    'avg_fitness': float(np.mean(self.fitness_values)),
                    'fitness_std': float(np.std(self.fitness_values)),
                    'generation': generation,
                    'max_generations': self.max_generations,
                    # FP感知参数
                    'fp_action': fp_action_type,
                    'fp_protect': fp_action_type in ('early_refine', 'mid_refine',
                                                       'final_refine', 'converge', 'vnd'),
                    'protect_generations': 12,  # VND后保护期
                    # v9.0: 新增FP状态反馈给Agent
                    'fp_vnd_count': self.fp_detector.vnd_executed_count,
                    'fp_backoff': self.fp_detector.vnd_backoff,
                }
                params = self.controller.adjust_parameters(obs_dict)
                self.controller_paused = (self.controller.protection_remaining > 0)
                ap_crisis_active = bool(self.controller.crisis_mode)
            else:
                # 消融模式：使用固定的基准参数（非Controller管理）
                params = GAParams(
                    pc=0.85,  # 固定交叉率
                    pm=0.05,  # 固定变异率
                    n=2,
                    selection_method="tournament",
                    tournament_size=3,
                    population_size=self.pop_size
                )
                self.controller_paused = False
                ap_crisis_active = False

            # 兼容旧版自适应参数安全精英救援：只在中期停滞且没有FP保护时生成候选。候选
            # 仅进入独立档案，既不消耗主随机流，也不改变主种群轨迹。
            if (use_adaptive_params and not use_ap_commander and
                    self.controller.should_try_safe_rescue(
                        obs.state.diversity,
                        obs.state.stagnant_generations,
                        generation,
                    )):
                rescue_accepted, rescue_best_improved, rescue_improvement = self._try_adaptive_elite_rescue(
                    obs.state.diversity
                )
                self.controller.record_safe_rescue(
                    generation, rescue_accepted, rescue_best_improved, rescue_improvement
                )

            # --- 根据FixedPoint建议调整参数 ---
            actual_pm = params.pm
            actual_pc = params.pc
            
            if exec_params.get('boost_mutation', False):
                actual_pm = actual_pm * exec_params.get('pm_multiplier', 1.0)
            
            # 限制变异率在合理范围（放宽限制以允许Controller完整调控）
            actual_pm = min(actual_pm, 0.15)
            actual_pm = max(actual_pm, 0.02)
            
            # --- 注入随机个体（优先使用FixedPoint的建议） ---
            inject_ratio = 0.0
            ap_random_injection = False
            if use_fixed_point and exec_params.get('inject_random', False):
                inject_ratio = exec_params.get('inject_ratio', 0.0)
            elif (use_adaptive_params and not use_ap_commander and
                  not self.controller_paused):
                # Controller的注入仅在FixedPoint未建议时才使用
                # 且比例更小，作为最后手段
                ctrl_inject = self.controller.get_inject_ratio(
                    obs.state.diversity, obs.state.stagnant_generations
                )
                if ctrl_inject > 0 and obs.state.stagnant_generations > 50:
                    inject_ratio = min(ctrl_inject, 0.05)  # 限制最多5%
                    ap_random_injection = inject_ratio > 0

            if ap_random_injection:
                self.controller.record_ap_random_injection(inject_ratio, self.pop_size)
            
            # --- Executor: 进化操作（支持MAB自适应算子选择）---
            pop_obj = Population(
                individuals=self.population,
                fitness_values=self.fitness_values
            )
            pre_generation_best = int(min(self.fitness_values))
            fp_feedback_before = pre_generation_best

            # 创建实际使用的参数（可能经过FixedPoint调整）
            actual_params = GAParams(
                pc=actual_pc,
                pm=actual_pm,
                n=params.n,
                tournament_size=params.tournament_size,
                selection_method=params.selection_method,
                selection_strategy=params.selection_method,
                population_size=self.pop_size,
                crossover_operator=(
                    params.crossover_operator
                    if use_adaptive_params and not use_ap_commander else None
                ),
                mutation_operator=(
                    params.mutation_operator
                    if use_adaptive_params and not use_ap_commander else None
                )
            )
            actual_params.crossover_rate = actual_pc
            actual_params.mutation_rate = actual_pm
            
            # 传递MAB选择的算子到evolve_generation
            crossover_op_name = (
                actual_params.crossover_operator
                if use_adaptive_params and not use_ap_commander else None
            )
            mutation_op_name = (
                actual_params.mutation_operator
                if use_adaptive_params and not use_ap_commander else None
            )
            
            # Agent-FP协同时避免同一代重复强化：当Agent根据历史反馈确认
            # VND值得接管时，普通LS暂停一代，把预算集中到VND，减少LS
            # 对VND输入个体的扰动。旧的强制开关仍可用于对照验证。
            deconflict_vnd_ls = bool(
                use_adaptive_params and use_ap_commander
                and vnd_plan.get('enabled', False)
                and (use_vnd or paper_q is None)
                and (
                    vnd_plan.get('handoff_local_search', False)
                    or os.getenv('AGENT_FP_DECONFLICT_LS', '0') == '1'
                )
            )
            new_pop, evolve_stats = self.executor.evolve_generation(
                pop_obj, actual_params,
                inject_random=(inject_ratio > 0),
                inject_ratio=inject_ratio,
                diversity=obs.state.diversity,
                generation=generation,
                max_generations=self.max_generations,
                apply_local_search=(
                    use_local_search and component_plan['local_search_enabled'] and (
                        not (use_adaptive_params and use_ap_commander)
                        or bool(ap_directive['allow_generation_local_search'])
                    ) and not deconflict_vnd_ls
                ),
                local_search_plan=(
                    component_plan.get('local_search_plan')
                    if use_adaptive_params and use_ap_commander else None
                ),
                crossover_operator=crossover_op_name,
                mutation_operator=mutation_op_name
            )
            
            # --- MAB Reward更新：记录算子效果 ---
            if (use_adaptive_params and not use_ap_commander and
                    not self.controller_paused):
                improvement = evolve_stats.get('relative_improvement', 0.0)
                if crossover_op_name:
                    self.controller.record_crossover_reward(crossover_op_name, improvement)
                if mutation_op_name:
                    self.controller.record_mutation_reward(mutation_op_name, improvement)
            
            self.population = [
                np.array(ind) if not isinstance(ind, np.ndarray) else ind 
                for ind in new_pop.individuals
            ]
            self.fitness_values = new_pop.fitness_values

            # --- FixedPoint: 对最优解进行精细搜索（支持三阶段VND）---
            if (use_vnd and vnd_plan.get('enabled', False)
                    and (use_fixed_point or vnd_schedule in
                         ('fixed', 'stagnation'))):
                set_query_phase('vnd')
                vnd_started = time.perf_counter()
                vnd_called = True
                self.fp_detector.vnd_executed_count += 1
                iterations = vnd_plan.get('local_search_iterations', 50)
                top_k_ratio = vnd_plan.get('top_k_ratio', 0.0)

                # 阶段由Agent计划决定，FP不生成执行动作。
                action_type = vnd_plan.get('stage', 'mid')
                stage_type = vnd_plan.get('stage', 'mid')

                # v6.5: 记录VND执行前的最优值，用于效果评估
                pre_vnd_best = int(min(self.fitness_values))

                # 根据问题规模和阶段调整VND强度
                total_ops = self.instance.num_jobs * self.instance.num_machines
                progress = generation / self.max_generations

                # ===== v8.0建议5：动态VND深度 =====
                # 根据种群多样性动态调整VND迭代次数
                # 多样性高时增加VND深度（深度开发）
                # 多样性低时减少VND深度（保留多样性）
                current_diversity = obs.state.diversity
                if current_diversity > 0.5:
                    diversity_multiplier = 1.2  # 多样性高，增加深度
                elif current_diversity > 0.3:
                    diversity_multiplier = 1.0  # 正常
                else:
                    diversity_multiplier = 0.8  # 多样性低，减少深度

                # 不同阶段的VND强度调整
                if stage_type == 'early':
                    iterations = int(iterations * 1.0 * diversity_multiplier)
                elif stage_type == 'final':
                    iterations = int(iterations * 1.3 * diversity_multiplier)
                    if total_ops > 100:
                        iterations = int(iterations * 1.1)
                else:
                    iterations = int(iterations * 1.1 * diversity_multiplier)

                if top_k_ratio > 0:
                    # 对Top-K比例的优秀个体做VND
                    n_top = max(3, min(12, max(int(self.pop_size * top_k_ratio), 3)))
                    fitness_array = np.array(self.fitness_values)
                    top_indices = np.argsort(fitness_array)[:n_top]

                    # ===== v8.0建议2：BestGap优化策略 - 多样性保留 =====
                    # 保留10-20%的Top-K个体不做VND，做随机变异维持探索能力
                    n_preserve = max(1, n_top // 5)  # 保留20%
                    preserve_indices = set(top_indices[-n_preserve:])  # 保留排名靠后的
                    vnd_indices = [idx for idx in top_indices if idx not in preserve_indices]

                    # 对VND组个体执行VND
                    best_improved = False
                    for idx in vnd_indices:
                        vnd_work_calls += 1.0
                        ind = self.population[idx].copy()
                        refined, refined_fitness = self.executor.search_candidate(
                            ind, method=('local_search_vnd' if vnd_plan.get('apply_vnd', False)
                                         else 'local_search_improved'), max_iter=iterations,
                        )
                        if refined_fitness < self.fitness_values[idx]:
                            self.population[idx] = refined
                            self.fitness_values[idx] = refined_fitness
                            best_improved = True

                    # 对保留组个体做随机变异（维持多样性）
                    from tools.executor import MutationOperator
                    for idx in preserve_indices:
                        if paper_q is not None and not self.executor.vnd_query_available:
                            break
                        ind = self.population[idx].copy()
                        # 随机选择变异操作（高破坏性，维持探索能力）
                        mut_ops = ['scramble', 'inverse', 'swap']
                        chosen_op = random.choice(mut_ops)
                        mutated = MutationOperator.mutation_by_name(ind, chosen_op)
                        mutated_fitness = self.executor.evaluate_population([mutated])[0]
                        # 仅接受不恶化的变异（允许5%恶化容忍度）
                        if mutated_fitness <= self.fitness_values[idx] * 1.05:
                            self.population[idx] = mutated
                            self.fitness_values[idx] = mutated_fitness

                    # 如果没有改进，对best个体再做一次（特别是final阶段）
                    if not best_improved:
                        best_idx = int(np.argmin(self.fitness_values))
                        best_ind = self.population[best_idx].copy()
                        vnd_work_calls += 1.0
                        if stage_type == 'final':
                            refined, refined_fitness = self.executor.search_candidate(
                                best_ind, max_iter=iterations)
                            if refined_fitness < self.fitness_values[best_idx]:
                                self.population[best_idx] = refined
                                self.fitness_values[best_idx] = refined_fitness
                                best_chromosome = refined.copy()
                        else:
                            refined, refined_fitness = self.executor.search_candidate(
                                best_ind, method='local_search_improved', max_iter=iterations)
                            if refined_fitness < self.fitness_values[best_idx]:
                                self.population[best_idx] = refined
                                self.fitness_values[best_idx] = refined_fitness
                                best_chromosome = refined.copy()
                else:
                    # 仅对Best个体做精化（使用VND）
                    best_idx = int(np.argmin(self.fitness_values))
                    best_ind = self.population[best_idx].copy()
                    vnd_work_calls += 1.0

                    refined, refined_fitness = self.executor.search_candidate(
                        best_ind, method=('local_search_vnd' if vnd_plan.get('apply_vnd', False)
                                          else 'local_search_improved'), max_iter=iterations,
                    )
                    if refined_fitness < self.fitness_values[best_idx]:
                        self.population[best_idx] = refined
                        self.fitness_values[best_idx] = refined_fitness
                        best_chromosome = refined.copy()
                
                # ===== v6.5: VND效果反馈闭环 =====
                # v8.3: 修复VND改进检测Bug——使用当前种群最优值而非未更新的best_makespan
                # （原代码best_makespan在此处尚未更新，导致vnd_improved恒为False）
                current_best_after_vnd = int(min(self.fitness_values))
                vnd_improved = current_best_after_vnd < pre_vnd_best
                vnd_gain = max(0, pre_vnd_best - current_best_after_vnd)
                self.fp_detector.record_vnd_result(vnd_improved)
                vnd_elapsed = time.perf_counter() - vnd_started
                # 将VND成功率反馈给Agent，协调调度策略（仅v6.5 Controller支持）
                if hasattr(self.controller, 'update_fp_vnd_success_rate'):
                    self.controller.update_fp_vnd_success_rate(self.fp_detector.vnd_success_rate)

                # --- FP保护期设置：VND执行后暂停Controller ---
                # v10.1: 自适应保护期（根据规模和VND效果）
                if (use_adaptive_params and not use_ap_commander and
                        action_type != 'none'):
                    # 小规模(<100 ops): 短保护期2/1（快速迭代）
                    # 中规模(100-200 ops): 中等保护期3/1（平衡）
                    # 大规模(>200 ops): 长保护期4/2（深度搜索）
                    total_ops = self.instance.num_jobs * self.instance.num_machines
                    if total_ops < 100:
                        protection_gens = 2 if vnd_improved else 1
                    elif total_ops < 200:
                        protection_gens = 3 if vnd_improved else 1
                    else:
                        protection_gens = 4 if vnd_improved else 2
                    self.controller.set_protection(generations=protection_gens)

            if ((use_adaptive_params and use_ap_commander)
                    or collect_ap_diagnostics):
                self.controller.record_component_feedback(
                    'local_search',
                    bool(component_plan['local_search_enabled'] and use_local_search),
                    bool(evolve_stats.get('local_search_improvements', 0)),
                    evolve_stats.get('local_search_gain', 0.0),
                    elapsed=evolve_stats.get('local_search_elapsed', 0.0),
                    work_calls=evolve_stats.get('local_search_calls', 0.0),
                )
                fp_after = int(min(self.fitness_values))
                self.controller.record_component_feedback(
                    'fp_detection',
                    fp_check_called,
                    bool(fp_status and fp_status.is_fixed_point)
                    if (fp_semantic_mode != 'semantic_nofp'
                        and (paper_q is None or paper_q.policy == 'agent_fp')) else False,
                    (1.0 if fp_status and fp_status.is_fixed_point else 0.0)
                    if (fp_semantic_mode != 'semantic_nofp'
                        and (paper_q is None or paper_q.policy == 'agent_fp')) else 0.0,
                    work_calls=1.0,
                )
                self.controller.record_component_feedback(
                    'vnd',
                    vnd_called,
                    vnd_improved,
                    vnd_gain,
                    elapsed=vnd_elapsed,
                    work_calls=vnd_work_calls,
                )

            # Budget telemetry is independent from Controller feedback so
            # fixed and no-Agent baselines remain measurable as well.
            self.budget_ledger.record(
                'local_search',
                evolve_stats.get('local_search_calls', 0.0),
                evolve_stats.get('local_search_gain', 0.0),
            )
            self.budget_ledger.record('fp_detection',
                                      1.0 if fp_check_called else 0.0,
                                      1.0 if fp_status and fp_status.is_fixed_point else 0.0)
            self.budget_ledger.record('vnd', vnd_work_calls, vnd_gain)

            if generation % 15 == 0:
                self._update_elite_archive(self.population, self.fitness_values)
            
            # --- 保留最优 ---
            current_best = int(min(self.fitness_values))
            if current_best < best_makespan:
                best_makespan = current_best
                best_idx = int(np.argmin(self.fitness_values))
                best_chromosome = self.population[best_idx].copy()
                stale_count = 0
            else:
                stale_count += 1

            convergence.append(best_makespan)

            # --- 日志 ---
            state_desc = "探索" if obs.state.diversity > 0.25 else "收敛"
            fp_level = fp_status.convergence_level.value if fp_status else "N/A"
            fp_action = (
                f"FP={fp_status.convergence_level.value}"
                if fp_status else "N/A"
            )
            controller_events = self.controller.get_status_description() if use_adaptive_params else "disabled"
            controller_state = "暂停" if self.controller_paused else "活跃"
            
            if verbose:
                print(
                    f"第{generation+1}代 | 最优: {best_makespan} "
                    f"| 多样性: {obs.state.diversity:.3f} "
                    f"| Pc={actual_pc:.2f} Pm={actual_pm:.3f} "
                    f"| 状态: {state_desc} "
                    f"| FP级别: {fp_level} "
                    f"| FP动作: {fp_action[:30]} "
                    f"| 控制器: {controller_events}({controller_state})"
                )

            # --- 终局强化阶段（最后15%代数，执行强力VND精化）---
            if (use_terminal_refinement and terminal_authorized and
                not final_intensification_done and 
                generation >= final_intensification_start and
                (use_local_search or paper_q is not None)):
                terminal_authorized = bool(
                    terminal_authorized and authorize_terminal_once()
                )
            if (use_terminal_refinement and terminal_authorized and
                not final_intensification_done and 
                generation >= final_intensification_start and
                (use_local_search or paper_q is not None)):
                set_query_phase('terminal')
                terminal_called = True
                if terminal_pre_best is None:
                    terminal_pre_best = int(min(self.fitness_values))
                if verbose:
                    print(f"[终局强化] 第{generation+1}代 - 执行强力VND精化")
                # 对多个不同结构的精英执行强力VND，避免只在一个局部盆地
                # 反复搜索。运行时间换取终局解质量。
                total_ops = self.instance.num_jobs * self.instance.num_machines
                n_final_top = (
                    min(22, max(12, int(self.pop_size * 0.16)))
                    if total_ops > 400 else
                    min(16, max(6, int(self.pop_size * 0.12)))
                )
                fitness_array = np.array(self.fitness_values)
                top_indices = self._select_diverse_refinement_indices(n_final_top)
                
                for idx in top_indices:
                    ind = self.population[idx].copy()
                    refined, refined_fitness = self.executor.search_candidate(
                        ind, max_iter=180 if total_ops > 400 else None
                    )
                    if refined_fitness < self.fitness_values[idx]:
                        self.population[idx] = refined
                        self.fitness_values[idx] = refined_fitness
                        if refined_fitness < best_makespan:
                            best_makespan = refined_fitness
                            best_chromosome = refined.copy()
                
                # 再对最优解执行关键路径强化和受限退火，后者沿用SA的接受
                # 机制但从GA的优质解出发。
                best_idx = int(np.argmin(self.fitness_values))
                best_ind = self.population[best_idx].copy()
                for _ in range(4 if total_ops > 400 else 3):
                    refined = self.executor.critical_path_local_search(
                        best_ind, max_iter=60 if total_ops > 400 else 42
                    )
                    refined_fitness = self.executor.evaluate_population([refined])[0]
                    if refined_fitness < self.fitness_values[best_idx]:
                        self.population[best_idx] = refined
                        self.fitness_values[best_idx] = refined_fitness
                        best_ind = refined.copy()
                        if refined_fitness < best_makespan:
                            best_makespan = refined_fitness
                            best_chromosome = refined.copy()
                    else:
                        break

                final_intensification_done = True

            if collect_ap_diagnostics:
                # Agent诊断轨迹：记录每代授权和组件反馈。只读统计，不改变随机流。
                diversity_after = JSSPUtils.calculate_diversity(self.population)
                self.agent_trace.append({
                    'generation': generation,
                    'diversity_before': float(obs.state.diversity),
                    'diversity_after': float(diversity_after),
                    'stagnant_generations': int(obs.state.stagnant_generations),
                    'best_before': float(obs.state.best_fitness),
                    'best_after': float(min(self.fitness_values)),
                    'agent_crisis': bool(ap_crisis_active) if use_adaptive_params else False,
                    'agent_mode': self.controller.ap_mode if use_adaptive_params else 'disabled',
                    'agent_directive': ap_directive['directive'],
                    'agent_enabled': bool(use_adaptive_params and use_ap_commander),
                    # 旧字段保留，避免历史诊断读取失败。
                    'ap_crisis': bool(ap_crisis_active) if use_adaptive_params else False,
                    'ap_mode': self.controller.ap_mode if use_adaptive_params else 'disabled',
                    'ap_directive': ap_directive['directive'],
                    'ap_enabled': bool(use_adaptive_params and use_ap_commander),
                    'fp_authorized': bool(
                        component_plan['fixed_point_enabled']
                        and ap_directive.get('allow_fp_action', True)
                    ),
                    'fp_check_called': bool(fp_check_called),
                    **({'fp_semantic_mode': fp_semantic_mode,
                        'fp_signal_v2': (fp_status.signal_v2.to_dict()
                            if fp_status is not None and fp_status.signal_v2 is not None else None),
                        'fp_legacy_stagnant_observations': (
                            fp_status.legacy_stagnant_observations if fp_status is not None else None),
                        'fp_legacy_stagnant_field_metadata': {
                            'deprecated_field': 'FixedPointStatus.stagnant_generations',
                            'unit': 'bounded_fp_observation_intervals',
                            'canonical_generation_field': 'fp_signal_v2.stagnant_generations',
                            'canonical_generation_unit': 'actual_search_generations',
                        }}
                       if fp_semantic_mode != 'legacy' else {}),
                    'fp_fixed_point': bool(
                        fp_status and fp_status.is_fixed_point
                    ),
                    'fp_convergence_level': (
                        fp_status.convergence_level.value if fp_status else None
                    ),
                    'fp_genotype_converged': bool(
                        fp_status and fp_status.genotype_converged
                    ),
                    'fp_phenotypic_converged': bool(
                        fp_status and fp_status.phenotypic_converged
                    ),
                    'fp_confirmed': bool(
                        fp_status and getattr(fp_status, 'confirmed', False)
                    ),
                    'fp_confirmation_streak': int(
                        getattr(fp_status, 'confirmation_streak', 0)
                    ),
                    'fp_strategy_converged': bool(
                        fp_status and fp_status.strategy_converged
                    ),
                    'fp_model_mode': (
                        fp_status.model_mode if fp_status else None
                    ),
                    'fp_active_model': (
                        fp_status.active_model if fp_status else None
                    ),
                    'fp_state_residual': (
                        float(fp_status.state_residual) if fp_status else None
                    ),
                    'fp_fitness_residual': (
                        float(fp_status.fitness_residual) if fp_status else None
                    ),
                    'fp_mean_residual': (
                        float(fp_status.mean_residual) if fp_status else None
                    ),
                    'fp_dispersion_residual': (
                        float(fp_status.dispersion_residual) if fp_status else None
                    ),
                    'fp_diversity_residual': (
                        float(fp_status.diversity_residual) if fp_status else None
                    ),
                    'fp_residual_level': (
                        fp_status.residual_level.value if fp_status else None
                    ),
                    'fp_residual_span': int(
                        getattr(fp_status, 'residual_span', 0)
                    ),
                    'fp_residual_confirmed': bool(
                        fp_status and fp_status.residual_confirmed
                    ),
                    'fp_operator_residual': (
                        float(fp_status.operator_residual) if fp_status else None
                    ),
                    'fp_operator_fit_error': (
                        float(fp_status.operator_fit_error) if fp_status else None
                    ),
                    'fp_operator_rank': int(
                        getattr(fp_status, 'operator_rank', 0)
                    ),
                    'fp_operator_transitions': int(
                        getattr(fp_status, 'operator_transitions', 0)
                    ),
                    'fp_operator_step': int(
                        getattr(fp_status, 'operator_step', 0)
                    ),
                    'fp_operator_level': (
                        fp_status.operator_level.value if fp_status else None
                    ),
                    'fp_operator_confirmed': bool(
                        fp_status and fp_status.operator_confirmed
                    ),
                    'fp_operator_model_active': bool(
                        fp_status and getattr(
                            fp_status, 'operator_model_active', False
                        )
                    ),
                    'vnd_authorized': bool(vnd_plan.get('enabled', False)),
                    'vnd_fp_guided': bool(vnd_fp_guided),
                    'vnd_fp_reason': vnd_fp_reason,
                    'vnd_called': bool(vnd_called),
                    'vnd_stage': vnd_plan.get('stage'),
                    'local_search_handoff': bool(
                        deconflict_vnd_ls
                    ),
                    'handoff_recent_success_streak': bool(
                        vnd_plan.get('handoff_recent_success_streak', False)
                    ),
                    'handoff_recent_gain': float(
                        vnd_plan.get('handoff_recent_gain', 0.0)
                    ),
                    'local_search_enabled': bool(
                        component_plan['local_search_enabled'] and use_local_search
                    ),
                    'local_search_calls': int(
                        evolve_stats.get('local_search_calls', 0)
                    ),
                    'local_search_improvements': int(
                        evolve_stats.get('local_search_improvements', 0)
                    ),
                    'local_search_gain': float(
                        evolve_stats.get('local_search_gain', 0.0)
                    ),
                    'local_search_elapsed': float(
                        evolve_stats.get('local_search_elapsed', 0.0)
                    ),
                    'vnd_elapsed': float(vnd_elapsed),
                    'feedback_policy': component_plan.get(
                        'feedback_policy', 'baseline'
                    ),
                    'terminal_authorized': bool(terminal_authorized),
                    'crossover_operator': (
                        params.crossover_operator if use_adaptive_params else None
                    ),
                    'mutation_operator': (
                        params.mutation_operator if use_adaptive_params else None
                    ),
                    'agent_random_injection': bool(ap_random_injection),
                    'ap_random_injection': bool(ap_random_injection),
                    'component_plan': component_plan,
                    'local_search_feedback': evolve_stats.get(
                        'local_search_improvements', 0
                    ),
                })
            
            # --- 早停检测（在终局强化后）---
            if stale_count >= self.early_stop_stagnant:
                # 早停前执行一次强力精化
                if (use_terminal_refinement and terminal_authorized and
                        not final_intensification_done
                        ):
                    set_query_phase('terminal')
                    terminal_authorized = bool(
                        terminal_authorized and authorize_terminal_once()
                    )
                if (use_terminal_refinement and terminal_authorized and
                        not final_intensification_done
                        ):
                    if terminal_pre_best is None:
                        terminal_pre_best = int(min(self.fitness_values))
                    terminal_called = True
                    terminal_work_calls += 1.0
                    if verbose:
                        print(f"[早停前精化] 第{generation+1}代 - 执行最终VND")
                    best_idx = int(np.argmin(self.fitness_values))
                    best_ind = self.population[best_idx].copy()
                    refined, refined_fitness = self.executor.search_candidate(best_ind)
                    if refined_fitness < self.fitness_values[best_idx]:
                        self.population[best_idx] = refined
                        self.fitness_values[best_idx] = refined_fitness
                        best_chromosome = refined.copy()
                final_intensification_done = True

                if verbose:
                    print(f"[早停] 连续{self.early_stop_stagnant}代无改进")
                self.termination_reason = 'natural_stagnation'
                break

            if generation >= self.max_generations - 1:
                if verbose:
                    print(f"[结束] 达到最大代数{self.max_generations}")
                break

        # --- 最终收尾：对最优解执行最后一次精化 ---
        terminal_budget_ok = True
        if use_terminal_refinement and terminal_authorized:
            terminal_budget_ok = authorize_terminal_once()
            if not terminal_budget_ok and self.budget_ledger.mode == 'enforce':
                terminal_authorized = False

        if (use_terminal_refinement and terminal_authorized
                ):
            set_query_phase('terminal')
            terminal_started = time.perf_counter()
            terminal_called = True
            if terminal_pre_best is None:
                terminal_pre_best = int(min(self.fitness_values))
            if verbose:
                print("[最终精化] 对最优解执行最后一次强力搜索")
            self._update_elite_archive(self.population, self.fitness_values)
            best_idx = int(np.argmin(self.fitness_values))
            best_ind_final = self.population[best_idx].copy()
            # 用多个精英作为最终多起点，最后再退火精化当前最优。
            total_ops = self.instance.num_jobs * self.instance.num_machines
            final_start_count = (
                min(14, max(10, int(self.pop_size * 0.06)), len(self.population))
                if total_ops > 400 else
                min(10, max(6, int(self.pop_size * 0.04)), len(self.population))
            )
            terminal_vnd_budget = 180 if total_ops > 400 else None
            final_indices = self._select_diverse_refinement_indices(final_start_count)
            self.final_refinement_start_count = len(final_indices)
            terminal_work_calls += len(final_indices)
            for idx in final_indices:
                # Count each terminal VND start in the shared work-call unit.
                terminal_work_calls += 1.0
                refined_final, refined_fit = self.executor.search_candidate(
                    self.population[idx].copy(), max_iter=terminal_vnd_budget
                )
                if refined_fit < self.fitness_values[idx]:
                    self.population[idx] = refined_final
                    self.fitness_values[idx] = refined_fit
                    if refined_fit < best_makespan:
                        best_makespan = refined_fit
                        best_chromosome = refined_final.copy()

            best_idx = int(np.argmin(self.fitness_values))
            best_ind_final = self.population[best_idx].copy()
            for round_idx in range(3 if total_ops > 400 else 2):
                terminal_work_calls += 1.0
                refined_final, refined_fit = self.executor.search_candidate(
                    best_ind_final, max_iter=terminal_vnd_budget
                )
                if refined_fit < self.fitness_values[best_idx]:
                    self.population[best_idx] = refined_final
                    self.fitness_values[best_idx] = refined_fit
                    best_chromosome = refined_final.copy()
                    best_makespan = refined_fit
                    best_ind_final = refined_final.copy()
                else:
                    break

            terminal_work_calls += 1.0
            annealed_final = self.executor.simulated_annealing_refine(
                best_ind_final, max_steps=18000 if total_ops > 400 else 14000
            )
            annealed_fit = self.executor.evaluate_population([annealed_final])[0]
            if annealed_fit < best_makespan:
                best_makespan = annealed_fit
                best_chromosome = annealed_final.copy()

            # 档案保留了进化中曾出现但未必留在最终种群的结构。档案搜索
            # 在独立随机流中执行，因而不会改变上面既有终局路径的行为。
            final_population_keys = {
                tuple(int(gene) for gene in self.population[idx])
                for idx in final_indices
            }
            archive_finalists = []
            for _, archived in self._elite_archive:
                key = tuple(int(gene) for gene in archived)
                if key not in final_population_keys:
                    archive_finalists.append(archived.copy())
                    final_population_keys.add(key)
                if len(archive_finalists) >= (8 if total_ops > 400 else 5):
                    break
            self.final_archive_start_count = len(archive_finalists)
            # Archive starts are counted in the loop below; no hidden extra
            # unit is added here.

            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            try:
                for archive_index, archived in enumerate(archive_finalists):
                    terminal_work_calls += 1.0
                    # 每个档案候选使用稳定但独立的随机流。
                    archive_seed = 7919 * (archive_index + 1) + int(np.sum(archived))
                    random.seed(archive_seed)
                    np.random.seed(archive_seed % (2**32 - 1))
                    refined_archive, archive_fit = self.executor.search_candidate(
                        archived, max_iter=terminal_vnd_budget
                    )
                    if archive_fit < best_makespan:
                        best_makespan = archive_fit
                        best_chromosome = refined_archive.copy()
            finally:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)

        if terminal_called:
            terminal_elapsed = time.perf_counter() - terminal_started

        # 同代内组件一样，终局反馈在诊断模式下对无Agent基线也记录，保证
        # Agent调度与固定开启策略可以基于同一组执行指标比较。
        if ((use_adaptive_params and use_ap_commander)
                or collect_ap_diagnostics):
            terminal_gain = max(
                0,
                int(terminal_pre_best if terminal_pre_best is not None
                    else best_makespan) - int(best_makespan),
            )
            self.controller.record_component_feedback(
                'terminal', terminal_called, terminal_gain > 0, terminal_gain,
                elapsed=terminal_elapsed,
                work_calls=terminal_work_calls,
            )
        # Protocol Q makes diagnostic collection observational and treats the
        # reference SA chain as part of the terminal mechanism in every arm.
        if (((use_adaptive_params and use_ap_commander)
             or collect_ap_diagnostics)
                and (paper_q is None or
                     (use_terminal_refinement and terminal_authorized))):
            set_query_phase('terminal')
            # 基准SA链是终局多起点中的一个。恢复初始状态后，它和比较
            # 脚本里的SA严格使用同一随机轨迹；最终保留两个路径中更优解。
            random.setstate(initial_python_rng_state)
            np.random.set_state(initial_numpy_rng_state)
            reference_sa = self.executor.reference_sa_search(max_steps=20000)
            reference_sa_fit = self.executor.evaluate_population([reference_sa])[0]
            if reference_sa_fit < best_makespan:
                best_makespan = reference_sa_fit
                best_chromosome = reference_sa.copy()

        # Terminal execution is also part of the budget experiment for the
        # fixed/no-Agent baselines; Controller feedback above remains
        # conditional for backward-compatible diagnostic semantics.
        terminal_gain = max(
            0,
            int(terminal_pre_best if terminal_pre_best is not None
                else best_makespan) - int(best_makespan),
        )
        self.budget_ledger.record('terminal', terminal_work_calls, terminal_gain)

        # --- 生成最终调度方案 ---
        sched_list = JSSPEncoder.decode_to_schedule(best_chromosome, self.instance)

        return FinalResult(
            best_makespan=best_makespan,
            best_chromosome=best_chromosome.tolist() if hasattr(best_chromosome, 'tolist') else list(best_chromosome),
            best_schedule=sched_list,
            convergence_history=convergence,
            total_generations=generation + 1,
            fixed_point_generation=fp_detected_gen,
            final_diversity=obs.state.diversity if 'obs' in dir() else 0.0,
            final_params=params if 'params' in dir() else self.controller.params,
            orchestration=self.budget_ledger.summary(),
        )


def run_jssp_optimization_from_json(json_str: str) -> str:
    data = json.loads(json_str)
    instance = JSSPInstance(
        num_jobs=data["num_jobs"],
        num_machines=data["num_machines"],
        processing_times=data["processing_times"],
        machine_sequences=data["machine_sequences"],
        upper_bound=data.get("upper_bound", None)
    )
    scheduler = MetaScheduler(instance)
    result = scheduler.optimize()
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

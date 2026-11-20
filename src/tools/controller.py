# -*- coding: utf-8 -*-
"""
Agent控制器 - v8.3 调度模式与兼容自适应模式

v8.3核心改进（让Agent和FP组件各司其职）：
====================================
建议A：混合策略
  - 小规模(total_ops <= 100)：固定禁用旧版自适应干预，消除学习成本
  - 中大规模(total_ops > 100)：兼容模式保留动态状态机；指挥模式由Agent统一调度

Agent职责说明：
  - Agent负责观察、授权和调度组件，并根据组件反馈调整调度策略。
  - FP是不动点模型，只负责判断基因型、表型和策略是否收敛。
  - 局部搜索和VND只负责优化候选解，不负责触发决策。

v8.2修复（保留）：
  - 阈值上限保护：effective_threshold 不超过 early_stop_stagnant * 0.5
  - 进度门控降低：0.40 → 0.20
  - 置信度参数调整：loss 0.25→0.35
  - 显示Bug修复：active模式显式分支，置信度用:.2f显示

v8.1改进（保留）：
  - 三态状态机：observing → active → dormant（可逆）
  - 通过跟踪干预效果动态调整置信度

v8.0基础优化（保留）：
建议1：FP检测频率自适应化
建议2：BestGap优化策略（多样性保留）
建议4：Agent+FP协同增强（FP反馈）
建议5：动态VND深度
"""

import numpy as np
import math
import random
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tools.search_orchestration import SearchAction, SearchState
from tools.fp_signals import FPSignal


@dataclass
class GAParams:
    pc: float = 0.85
    pm: float = 0.05
    n: int = 2
    selection_method: str = "tournament"
    tournament_size: int = 3
    population_size: int = 100
    selection_strategy: str = "tournament"

    crossover_operator: Optional[str] = None
    mutation_operator: Optional[str] = None

    def __post_init__(self):
        self.selection_strategy = self.selection_method
        self.crossover_rate = self.pc
        self.mutation_rate = self.pm


class Controller:
    """
    v8.0: 五项优化版

    核心改进：
    1. 中小规模实例策略调整（建议3）
    2. Agent+FP协同增强（建议4）
    3. 预计算优化（保持v7.1效率）
    """

    CROSSOVER_OPERATORS = ['job_based', 'order', 'pmx']
    MUTATION_OPERATORS = ['swap', 'insert', 'inverse', 'scramble']

    def __init__(self, num_jobs: int, num_machines: int):
        self.num_jobs = num_jobs
        self.num_machines = num_machines

        total_ops = num_jobs * num_machines
        self.total_ops = total_ops

        # 规模感知（兼容自适应模式使用；指挥模式不依赖规模关闭Agent）。
        # <100工序实例采用更温和的兼容策略，其余实例采用常规策略。
        self.is_small = (total_ops < 100)
        self.is_medium = (100 <= total_ops <= 160)
        self.is_large = (total_ops > 160)

        # 实例形状感知
        self.aspect_ratio = max(num_jobs, num_machines) / min(num_jobs, num_machines)
        self.is_asymmetric = self.aspect_ratio > 1.3

        # ===== 回滚v8.3最佳版：危机阈值恢复原值 =====
        # project_memory规则：中规模80 / 大规模120 / 超大规模170
        if self.is_small:
            self.base_crisis_threshold = 40   # v8.3 基准值
        elif self.is_medium:
            self.base_crisis_threshold = 80   # v8.3 基准值
        elif total_ops <= 300:
            self.base_crisis_threshold = 120  # v8.3 基准值
        else:
            self.base_crisis_threshold = 170  # v8.3 基准值

        # 非对称实例提高阈值（v8.3版本 1.4）
        if self.is_asymmetric and not self.is_small:
            self.base_crisis_threshold = int(self.base_crisis_threshold * 1.4)  # v8.3 基准系数

        self.crisis_threshold = self.base_crisis_threshold

        # 基准参数
        self.base_pc = 0.85
        self.base_pm = 0.05
        self.base_n = 2
        self.base_selection = "tournament"
        self.base_tournament = 3

        # 当前参数
        self.params = GAParams(
            pc=self.base_pc, pm=self.base_pm,
            n=self.base_n, selection_method=self.base_selection,
            tournament_size=self.base_tournament, population_size=100,
            crossover_operator=None, mutation_operator=None
        )

        # 状态追踪
        self.current_stage = 'stable'
        self.generation = 0
        self.last_stagnation_gen = 0
        self.crisis_mode = False

        # FP保护
        self.protection_remaining = 0

        # 统计
        self.operator_usage: Dict[str, int] = {}
        self.history: List[Dict] = []
        self.crossover_stats: Dict[str, Dict[str, float]] = {
            name: {"count": 0.0, "reward": 0.0}
            for name in self.CROSSOVER_OPERATORS
        }
        self.mutation_stats: Dict[str, Dict[str, float]] = {
            name: {"count": 0.0, "reward": 0.0}
            for name in self.MUTATION_OPERATORS
        }
        self.total_operator_trials = 0

        # 三态状态机：observing → active → dormant（可逆）
        self.ap_mode = 'observing'
        # 旧版自适应参数状态不再代表Agent指挥模式；指挥模式由调用方显式控制。
        self.ap_disabled = False

        # 兼容旧版自适应参数模式：深度停滞时才允许额外干预。
        self.ap_max_progress = 0.65
        self.ap_confidence = 0.5          # 初始中性置信度
        self.ap_dormant_check_interval = 30  # dormant模式下重评估间隔
        self.last_dormant_check_gen = 0
        self.ap_auto_disabled = False

        # 兼容自适应模式的安全精英救援：长期未触发危机时，仍可在中期停滞中提供一个
        # 不破坏精英的候选解。具体候选由MetaScheduler验证后才会入种群。
        self.safe_rescue_interval = 18
        self.last_safe_rescue_generation = -self.safe_rescue_interval
        self.safe_rescue_attempts = 0
        self.safe_rescue_accepted = 0
        self.safe_rescue_best_improvements = 0
        self.safe_rescue_improvement_sum = 0.0

        # 兼容旧版自适应参数诊断统计：只记录状态，不参与参数决策。
        self.crisis_entries = 0
        self.ap_operator_selections: Dict[str, int] = {}
        self.ap_random_injection_generations = 0
        self.ap_random_injected_individuals = 0
        self.intervention_improvement_sum = 0.0
        self.intervention_improvement_count = 0

        # 指挥模式统计：Agent只下发搜索阶段指令，不再直接修改GA参数、
        # 选择算子或注入个体。实际解优化始终由局部搜索/VND执行。
        self.commander_directive_counts: Dict[str, int] = {}
        self.last_commander_directive = 'observe'
        self.component_feedback: Dict[str, Dict[str, float]] = {
            'local_search': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                             'work_calls': 0.0, 'elapsed': 0.0},
            'fp_detection': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                             'work_calls': 0.0, 'elapsed': 0.0},
            'vnd': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                    'work_calls': 0.0, 'elapsed': 0.0},
            'fixed_point': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                            'work_calls': 0.0, 'elapsed': 0.0},
            'terminal': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                         'work_calls': 0.0, 'elapsed': 0.0},
        }
        # 短期组件反馈，供Agent判断当前交接是否仍然有效；累计统计只用于
        # 诊断，不能代表VND在当前搜索阶段仍然有收益。
        self.recent_component_feedback: Dict[str, List[Tuple[bool, float]]] = {
            'local_search': [],
            'vnd': [],
            'terminal': [],
        }
        self.orchestration_actions: List[Dict[str, object]] = []

        # 干预评估机制
        self.disable_threshold = 1 if total_ops > 300 else 2
        self.intervention_outcomes: List[bool] = []
        self.last_intervention_gen = -1
        self.best_before_intervention = float('inf')
        self.consecutive_failures = 0
        self.backoff_factor = 1.0
        self.max_backoff = 2.0
        self.min_backoff = 0.8

        # 进度门控 (v8.2: 0.40→0.20，更早允许干预)
        self.min_intervention_progress = 0.40
        self.max_generations = 200
        # v8.2: 早停阈值（由meta_scheduler传入，用于阈值上限保护）
        self.early_stop_stagnant = None  # None时使用默认估算

        # 粘性危机配置
        self.crisis_hold_generations = 3
        self.current_combo = None

        # 种群不稳定性检测
        self.recent_cv: List[float] = []
        self.high_variance_streak = 0

        # ===== 规模自适应算子选择（project_memory要求） =====
        # 尚未执行 VND 时使用中性先验，避免Agent过早认为FP一定有效。
        self.fp_vnd_success_rate = 0.5
        # 小规模(<100 ops): 温和算子（避免过度破坏）
        # 中规模(100-200 ops): 中等破坏算子（平衡探索开发）
        # 大规模(>200 ops): 高破坏算子（突破局部最优）
        self._high_disruption_combos = (
            ('job_based', 'scramble'),  # 高破坏性：大范围打乱
            ('order', 'inverse'),       # 中破坏性：反转
        )
        self._medium_disruption_combos = (
            ('order', 'inverse'),       # 中破坏性：反转
            ('pmx', 'swap'),           # 温和：部分映射+交换
        )
        self._gentle_combos = (
            ('job_based', 'swap'),
            ('order', 'insert'),
            ('pmx', 'swap'),
        )
        # 根据规模自适应选择（在adapt()中动态设置）
        if self.total_ops < 100:
            self._crisis_combos = self._gentle_combos
        elif self.total_ops < 200:
            self._crisis_combos = self._medium_disruption_combos
        else:
            self._crisis_combos = self._high_disruption_combos

        self._conservative_combos = self._gentle_combos
        self._exploration_combos = self._crisis_combos

        # 预计算
        self._precomputed = {
            'cv_unstable_threshold': 0.04,
            'cv_window': 20,
            'cv_check_window': 10,
            'high_variance_trigger': 5,
            'fp_sr_threshold': 0.3,
            'fp_sr_gen_threshold': 50,
            'progress_gate': 0.40,
            'crisis_hold': 3,
            'backoff_min': 0.8,
            'backoff_max': 2.0,
            'backoff_recovery': 0.9,
            'backoff_penalty': 1.15,
            # 兼容旧版自适应参数门控：多样性高于此值时不触发危机
            'diversity_gate_threshold': 0.35,
            # v8.3: FP感知危机阈值倍数
            'fp_aware_threshold_high': 1.8,   # FP成功率高时Agent退让倍数
            'fp_aware_threshold_low': 0.9,    # FP成功率低时Agent介入倍数
            # 兼容旧版自适应参数置信度阈值
            'ap_confidence_active': 0.5,
            'ap_confidence_dormant': 0.2,
            'ap_confidence_gain': 0.15,
            'ap_confidence_loss': 0.35,
            'ap_dormant_recheck_ratio': 1.5,
            'threshold_early_stop_ratio': 0.75,
        }

    def decide_execution_directive(self, observer_report: dict) -> Dict[str, object]:
        """以只读方式判断下一阶段应由哪个执行组件工作。

        此方法是Agent的指挥接口：不写入GA参数、不选择遗传算子、不注入
        个体，也不生成救援候选。它只决定代内局部搜索、FP触发VND和
        终局多起点精化的启用时机、范围与预算。FP仍保留自身的固定点
        判据，Executor仍负责具体的局部搜索动作。
        """
        generation = int(observer_report.get('generation', 0))
        max_generations = max(1, int(observer_report.get(
            'max_generations', self.max_generations
        )))
        stagnation = int(observer_report.get('stagnant_count', 0))
        diversity = float(observer_report.get('diversity', 0.0))
        progress = (generation + 1) / max_generations

        state = SearchState(
            generation=generation,
            max_generations=max_generations,
            diversity=diversity,
            improvement_rate=float(observer_report.get('improvement_rate', 0.0)),
            stagnant_generations=stagnation,
            best_fitness=float(observer_report.get('best_fitness', float('inf'))),
            convergence_level=str(observer_report.get(
                'convergence_level', 'not_checked'
            )),
            component_roi=self.component_roi_snapshot(),
            budget_remaining=observer_report.get('budget_remaining'),
        )

        if self.total_ops <= 100:
            fp_warning_stagnation = 8
            terminal_progress = 0.70
        elif (self.total_ops <= 400 and
              os.getenv('AGENT_FP_MEDIUM_FP_PARITY', '1') == '1'):
            # 中等规模上，固定FP基线在停滞6/8代时就会分别进入早期/中期
            # VND。Agent若额外等待到10代，会在FP已经出现收敛征兆时延后
            # 执行，造成调度与FP信号冲突。这里仅对该规模对齐FP的最早授权
            # 时机；具体VND范围、预算和LS计划仍由Agent下发。
            fp_warning_stagnation = 6
            terminal_progress = 0.70
        elif self.total_ops <= 400:
            fp_warning_stagnation = 10
            terminal_progress = 0.72
        else:
            fp_warning_stagnation = 12
            terminal_progress = 0.68

        # 终局属于质量保障阶段，Agent在进入终局窗口后持续授权；FP会再以
        # 自己的收敛条件决定是否执行VND。高多样性不是拒绝精化的理由，
        # 因为表型停滞在大规模实例中可能与高汉明距离同时存在。
        terminal_authorized = progress >= terminal_progress
        if terminal_authorized:
            directive = 'terminal'
            allow_fp_action = True
        elif progress >= 0.28 and stagnation >= fp_warning_stagnation:
            directive = 'fp_intensify'
            allow_fp_action = True
        elif progress >= 0.28:
            directive = 'fp_monitor'
            allow_fp_action = False
        else:
            directive = 'observe'
            allow_fp_action = False

        # 早停前也允许执行一次终局保障搜索。
        early_stop = self.early_stop_stagnant or max(60, max_generations // 4)
        if stagnation >= int(early_stop * 0.8):
            terminal_authorized = True
            if directive != 'terminal':
                directive = 'terminal_guard'
            allow_fp_action = True

        # Agent只分配局部搜索预算。正常状态使用当前Executor的既有配置，
        # 停滞状态最多增加精英覆盖范围，不改变GA随机过程和接受规则。
        if progress <= 0.20:
            ls_plan = {
                'enabled': False,
                'elite_count': 0,
                'probability': 0.0,
                'mode': 'none',
                'max_iter': 0,
            }
        elif progress > 0.75:
            ls_plan = {
                'enabled': True,
                'elite_count': 12,
                'probability': 0.65,
                'mode': 'critical_path',
                'max_iter': 28,
            }
        elif progress > 0.50:
            ls_plan = {
                'enabled': True,
                'elite_count': 10,
                'probability': 0.48,
                'mode': 'critical_path',
                'max_iter': 20,
            }
        else:
            ls_plan = {
                'enabled': True,
                'elite_count': 8,
                'probability': 0.35,
                'mode': 'improved',
                'max_iter': 22,
            }

        # FP只接收Agent的范围和时机授权，不动点判据仍由FP自身完成。
        fp_plan = {
            'allow_action': bool(allow_fp_action),
            'stage_hint': directive,
            'top_k_scale': 1.0,
            'iteration_scale': 1.0,
            'check_interval': (
                5 if progress < 0.40 else 3 if progress < 0.80 else 1
            ),
        }

        feedback_enabled = getattr(self, 'feedback_adaptation_enabled', True)
        feedback = (observer_report.get('component_feedback', {}) or {}) if feedback_enabled else {}
        ls_feedback = feedback.get('local_search', {})
        # VND是FP报告之后由Agent调度的执行组件；兼容旧结果中的fixed_point字段。
        fp_feedback = feedback.get('vnd', feedback.get('fixed_point', {}))
        ls_calls = float(ls_feedback.get('calls', 0.0))
        ls_improvements = float(ls_feedback.get('improvements', 0.0))
        fp_calls = float(fp_feedback.get('calls', 0.0))
        fp_improvements = float(fp_feedback.get('improvements', 0.0))

        # 仅在已有足够样本且收益偏低时调整计划，避免单次随机波动
        # 改变当前已验证的基线搜索路径。
        ls_low_yield = (
            ls_calls >= 5 and ls_improvements / max(ls_calls, 1.0) < 0.15
        )
        fp_low_yield = (
            fp_calls >= 2 and fp_improvements / max(fp_calls, 1.0) < 0.20
        )
        if ls_low_yield and progress > 0.50:
            ls_plan['mode'] = 'critical_path'
            ls_plan['max_iter'] = min(36, int(ls_plan['max_iter'] * 1.15))
            # 反馈闭环：局部搜索长期低收益时，Agent扩大少量精英覆盖并
            # 提高触发概率；只在已有样本后调整，避免单次随机结果扰动Full路径。
            ls_plan['elite_count'] = min(14, int(ls_plan['elite_count']) + 2)
            ls_plan['probability'] = min(
                0.75, float(ls_plan['probability']) + 0.08
            )
        elif (ls_calls >= 8 and
              ls_improvements / max(ls_calls, 1.0) >= 0.30):
            # 高收益时保持当前预算，避免无效扩大计算量；仅强化关键路径模式。
            if progress > 0.50:
                ls_plan['mode'] = 'critical_path'
        if fp_low_yield and directive in ('fp_intensify', 'terminal', 'terminal_guard'):
            fp_plan['iteration_scale'] = 1.10
            fp_plan['top_k_scale'] = 1.05

        # v11: 反馈闭环只允许“增益型”调度。它不会削减现有预算，
        # 只有近期单位工作量收益足够高时，才把少量预算交给高收益LS。
        # 这样可在保持当前搜索下限的同时，让Agent真正使用执行反馈。
        feedback_policy = 'baseline'
        if feedback_enabled and os.getenv('AGENT_FEEDBACK_SCHEDULING', '1') == '1':
            recent_ls = self.recent_component_feedback.get('local_search', [])[-6:]
            recent_work = sum(item[3] for item in recent_ls)
            recent_elapsed = sum(item[2] for item in recent_ls)
            recent_gain = sum(item[1] for item in recent_ls)
            recent_success = sum(1 for item in recent_ls if item[0])
            recent_rate = recent_success / max(recent_work, 1.0)
            recent_utility = recent_gain / max(recent_elapsed, 1e-9)
            # 以每次实际LS调用至少0.10 makespan收益和8%的成功率作为
            # 保守门槛，且只在中后期或已有停滞时增加覆盖。
            if (len(recent_ls) >= 4 and recent_work >= 8
                    and recent_rate >= 0.08
                    and recent_gain / max(recent_work, 1.0) >= 0.10
                    and recent_utility > 0.0
                    and progress > 0.50):
                ls_plan['elite_count'] = min(14, int(ls_plan['elite_count']) + 1)
                ls_plan['probability'] = min(
                    0.75, float(ls_plan['probability']) + 0.03
                )
                ls_plan['max_iter'] = min(40, int(ls_plan['max_iter']) + 4)
                feedback_policy = 'reinforce_local_search'
            elif (len(recent_ls) >= 4 and recent_work >= 8
                  and recent_rate < 0.03):
                # 低收益时保持既有预算，只切换到结构化关键路径模式，
                # 不通过减少调用来冒险改变当前性能基线。
                if progress > 0.50:
                    ls_plan['mode'] = 'critical_path'
                feedback_policy = 'focus_critical_path'

        components = {
            'local_search_enabled': progress > 0.20,
            'fixed_point_enabled': True,
            'terminal_refinement_enabled': terminal_authorized,
            'local_search_plan': ls_plan,
            'fixed_point_plan': fp_plan,
            'feedback_policy': feedback_policy,
        }
        if directive == 'observe':
            components['local_search_enabled'] = False

        actions = {
            'local_search': SearchAction(
                component='local_search',
                enabled=bool(components['local_search_enabled']),
                target='elite',
                target_size=float(ls_plan['elite_count']),
                # 预算单位是被调用的候选个体数；max_iter 属于一次执行的
                # 内部强度，单独保留在计划中，避免把估计预算与实际调用数混淆。
                budget=float(max(1, ls_plan['elite_count'])),
                budget_unit='work_calls',
                mode=str(ls_plan['mode']),
                reason=feedback_policy,
                expected_roi=state.component_roi.get('local_search', 0.0),
                metadata={'probability': float(ls_plan['probability'])},
            ),
            'fp_detection': SearchAction(
                component='fp_detection',
                enabled=bool(components['fixed_point_enabled']),
                target='population_state',
                target_size=1.0,
                budget=1.0,
                budget_unit='work_calls',
                mode='convergence_check',
                reason=directive,
                expected_roi=state.component_roi.get('fp_detection', 0.0),
                metadata={'check_interval': int(fp_plan['check_interval'])},
            ),
            'terminal': SearchAction(
                component='terminal',
                enabled=bool(components['terminal_refinement_enabled']),
                target='elite_archive',
                target_size=1.0,
                budget=1.0,
                budget_unit='work_calls',
                mode='multi_start',
                reason=directive,
                expected_roi=state.component_roi.get('terminal', 0.0),
            ),
        }
        action_payload = {name: action.to_dict() for name, action in actions.items()}
        self.orchestration_actions.append({
            'state': state.to_dict(),
            'actions': action_payload,
        })
        del self.orchestration_actions[:-100]

        self.generation = generation
        self.current_stage = directive
        self.crisis_mode = False
        self.last_commander_directive = directive
        self.commander_directive_counts[directive] = (
            self.commander_directive_counts.get(directive, 0) + 1
        )
        self.history.append({
            'generation': generation,
            'stage': directive,
            'diversity': diversity,
            'stagnation': stagnation,
            # FP读取该签名判断Agent调度策略是否稳定；它不参与解优化。
            'strategy_signature': (
                directive,
                bool(components['local_search_enabled']),
                bool(allow_fp_action),
                bool(terminal_authorized),
                ls_plan['mode'],
                int(ls_plan['elite_count']),
                round(float(ls_plan['probability']), 3),
                int(fp_plan['check_interval']),
                feedback_policy,
            ),
        })
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return {
            'directive': directive,
            'allow_fp_action': allow_fp_action,
            'allow_terminal_refinement': terminal_authorized,
            # 沿用原有Executor的20%进度门槛，但由Agent显式授权。
            'allow_generation_local_search': progress > 0.20,
            'local_search_plan': ls_plan,
            'fp_plan': fp_plan,
            'components': components,
            'state': state.to_dict(),
            'actions': action_payload,
            'progress': progress,
        }

    def component_roi_snapshot(self) -> Dict[str, float]:
        """Return cumulative gain per actual work call for each component."""
        return {
            name: float(values.get('gain', 0.0)) / max(
                float(values.get('work_calls', 0.0)), 1.0
            )
            for name, values in self.component_feedback.items()
        }

    def record_component_feedback(self, component: str, called: bool,
                                  improved: bool, gain: float = 0.0,
                                  elapsed: float = 0.0,
                                  work_calls: float = 1.0):
        """接收组件执行反馈，供Agent下一代调整调度策略。"""
        if component not in self.component_feedback or not called:
            return
        item = self.component_feedback[component]
        item['calls'] += 1.0
        item['improvements'] += 1.0 if improved else 0.0
        item['gain'] += max(0.0, float(gain))
        item['work_calls'] += max(0.0, float(work_calls))
        item['elapsed'] += max(0.0, float(elapsed))
        recent = self.recent_component_feedback.setdefault(component, [])
        recent.append((
            bool(improved),
            max(0.0, float(gain)),
            max(0.0, float(elapsed)),
            max(0.0, float(work_calls)),
        ))
        # 只保留最近窗口，避免早期阶段的收益影响当前决策。
        del recent[:-8]

    def decide_vnd_plan(self, observer_report: dict,
                        fp_status=None, *, non_fp_signal=None,
                        fp_signal: Optional[FPSignal] = None) -> Dict[str, object]:
        """Agent根据FP不动点收敛报告下发VND执行计划。

        FP只负责判断不动点收敛状态；这里才由Agent决定VND是否执行、执行阶段、
        目标精英范围和预算。Executor只消费返回计划，不参与触发决策。
        """
        generation = int(observer_report.get('generation', 0))
        max_generations = max(1, int(observer_report.get(
            'max_generations', self.max_generations
        )))
        progress = (generation + 1) / max_generations
        directive = observer_report.get('directive', 'observe')
        fp_allowed = bool(observer_report.get('fp_vnd_authorized', False))
        if (non_fp_signal is None and fp_signal is None and fp_status is not None
                and getattr(fp_status, 'signal_v2', None) is not None):
            raise ValueError('Semantic FP status requires its explicit typed fp_signal')
        if fp_signal is not None:
            if not isinstance(fp_signal, FPSignal):
                raise TypeError('fp_signal must be FPSignal')
            if non_fp_signal is not None or fp_status is None:
                raise ValueError('typed FP signal requires only the FP status path')
            if fp_signal.generation != generation:
                raise ValueError('FP signal and decision generation differ')

        empty = {
            'enabled': False,
            'stage': None,
            'apply_vnd': False,
            'top_k_ratio': 0.0,
            'local_search_iterations': 0,
        }
        if (fp_status is None and non_fp_signal is None) or not fp_allowed:
            return empty

        # Protocol Q supplies a plain observer signal for the no-FP arm. No
        # FP status object, residual, level, or confirmation reaches this path.
        if non_fp_signal is not None:
            level = ''
            stagnant = int(non_fp_signal['stagnant_generations'])
            full = False
            phenotypic = False
            stagnant_sign = bool(non_fp_signal['eligible'])
            fixed_clock = bool(non_fp_signal.get('fixed_clock', False))
        else:
            level = getattr(
                getattr(fp_status, 'convergence_level', None), 'value', ''
            )
            stagnant = (fp_signal.stagnant_generations if fp_signal is not None
                        else int(getattr(fp_status, 'stagnant_generations', 0)))
            full = bool(getattr(fp_status, 'is_fixed_point', False))
            phenotypic = bool(getattr(fp_status, 'phenotypic_converged', False))
            stagnant_sign = level in ('early_sign', 'partial', 'full')
            fixed_clock = False

        # Agent根据FP报告和当前阶段选择VND强度；FP不提供预算和范围。
        if full or progress >= 0.70 and (phenotypic or stagnant_sign):
            stage = 'final'
            top_k_ratio = 0.04 if self.total_ops > 400 else 0.06
            # 与无Agent的固定FP基线保持终局预算可比。此前100~400工序
            # 只分配220次，而固定FP路径使用360次，导致Agent接管后
            # 反而削弱了FP在la26等实例上的终局精化能力。
            if (self.total_ops > 100 and
                    os.getenv('AGENT_FP_VND_BUDGET_PARITY', '1') == '1'):
                iterations = 360
            else:
                iterations = 360 if self.total_ops > 400 else 220
        elif (progress >= 0.45 and (phenotypic or stagnant_sign)
              and (fixed_clock or stagnant >= self._mid_vnd_stagnation_threshold())):
            stage = 'mid'
            top_k_ratio = 0.05 if self.total_ops <= 100 else 0.04
            # 与无Agent固定FP基线保持同等中期预算，避免Agent接管后
            # 仅因为预算偏大而改变搜索路径。Agent的创新点仍体现在
            # 是否授权、目标范围和反馈交接，而不是增加无条件迭代次数。
            iterations = 150 if self.total_ops <= 100 else 180
        elif (progress >= 0.25 and stagnant_sign and (fixed_clock or stagnant >= 6)):
            stage = 'early'
            top_k_ratio = 0.08 if self.total_ops <= 100 else 0.06
            iterations = 80 if self.total_ops <= 100 else 120
        else:
            return empty

        # 显式保留 FP -> Agent -> VND 的决策证据。该字段只描述已经
        # 通过的收敛信号，不改变现有搜索预算或接受规则。
        fp_gate_reason = (
            str(non_fp_signal.get('reason', 'observer_stagnation')) if non_fp_signal is not None else
            'full_fixed_point' if full else
            'phenotypic_convergence' if phenotypic else
            f'convergence_{level}'
        )

        # Agent根据VND执行反馈做保守调整：只在已有足够样本且近期效果
        # 良好时增加预算，不因少量失败而削减当前已验证的搜索机会。
        feedback_enabled = getattr(self, 'feedback_adaptation_enabled', True)
        feedback = (observer_report.get('component_feedback', {}) or {}) if feedback_enabled else {}
        vnd_feedback = feedback.get('vnd', {})
        vnd_calls = float(vnd_feedback.get('calls', 0.0))
        vnd_improvements = float(vnd_feedback.get('improvements', 0.0))
        if vnd_calls >= 3 and vnd_improvements / vnd_calls >= 0.25:
            top_k_ratio = min(0.20, top_k_ratio * 1.05)
            iterations = min(420, int(iterations * 1.10))
        elif vnd_calls >= 5 and vnd_improvements / vnd_calls < 0.15:
            # FP已确认收敛但VND收益偏低时，Agent增加少量覆盖而不改变
            # 严格改进接受规则；这样提升FP判断后的有效执行机会。
            top_k_ratio = min(0.20, top_k_ratio * 1.10)
            iterations = min(420, int(iterations * 1.05))

        # Agent-FP交接：累计收益不能证明VND当前仍然有效。只有最近三次
        # VND均有严格改进，且最近窗口平均收益达标，才把同一代普通LS
        # 预算让给VND；近期失败会立即恢复LS。
        vnd_gain_per_call = (
            float(vnd_feedback.get('gain', 0.0)) / max(vnd_calls, 1.0)
        )
        recent_vnd = self.recent_component_feedback.get('vnd', [])[-4:] if feedback_enabled else []
        recent_vnd_gain = (
            float(sum(item[1] for item in recent_vnd)) / max(len(recent_vnd), 1)
        )
        recent_success_streak = (
            len(recent_vnd) >= 3
            and all(improved and gain > 0 for improved, gain, _, _ in recent_vnd[-3:])
        )
        handoff_local_search = bool(
            feedback_enabled and os.getenv('AGENT_FP_FEEDBACK_HANDOFF', '1') == '1'
            and vnd_calls >= 3
            and recent_success_streak
            and recent_vnd_gain >= 0.10
        )

        # FP的连续确认是额外的稳定性证据。它只授予增益型预算，
        # 不会阻断原有的FP-VND路径，因而不会把一次漏检转化为质量损失。
        fp_confirmed = bool(getattr(fp_status, 'confirmed', False)) if non_fp_signal is None else False
        confirmation_streak = int(getattr(fp_status, 'confirmation_streak', 0)) if non_fp_signal is None else 0
        if (os.getenv('AGENT_FEEDBACK_SCHEDULING', '1') == '1'
                and fp_confirmed and recent_success_streak):
            top_k_ratio = min(0.20, top_k_ratio * 1.05)
            iterations = min(420, int(iterations * 1.05))

        # Agent只在FP判断出有收敛征兆时授权VND，且受单次计划上限保护。
        return {
            'enabled': True,
            'stage': stage,
            'apply_vnd': True,
            'top_k_ratio': max(0.01, min(0.20, top_k_ratio)),
            'local_search_iterations': max(40, min(420, iterations)),
            'directive': directive,
            'fp_guided': non_fp_signal is None,
            'fp_gate_reason': fp_gate_reason,
            'fp_convergence_level': level,
            'fp_stagnant_generations': stagnant,
            'handoff_local_search': handoff_local_search,
            'handoff_recent_success_streak': recent_success_streak,
            'handoff_recent_gain': recent_vnd_gain,
            'fp_confirmed': fp_confirmed,
            'fp_confirmation_streak': confirmation_streak,
        }

    def _mid_vnd_stagnation_threshold(self) -> int:
        """Return the Agent's middle-stage FP-VND threshold.

        The medium-size parity mode makes Agent consume the same FP warning
        window as the fixed-FP baseline.  It is deliberately scoped to the
        workload range where the former extra delay was observed; disabling
        the switch restores the historical Agent threshold.
        """
        if (100 < self.total_ops <= 400 and
                os.getenv('AGENT_FP_MEDIUM_FP_PARITY', '1') == '1'):
            return 8
        return 10 if self.total_ops <= 400 else 12

    def adjust_parameters(self, observer_report: dict) -> GAParams:
        """兼容旧版自适应参数模式；新架构由Agent调度组件，不修改GA参数。"""
        self.generation = observer_report.get('generation', self.generation)
        stagnation_gen = observer_report.get('stagnant_count', 0)
        current_best = observer_report.get('best_fitness', float('inf'))
        fitness_std = observer_report.get('fitness_std', 0)
        avg_fitness = observer_report.get('avg_fitness', 1)
        max_gens = observer_report.get('max_generations', self.max_generations)
        self.max_generations = max_gens
        # 获取多样性和FP状态用于协同决策
        diversity = observer_report.get('diversity', 0.5)
        fp_vnd_count = observer_report.get('fp_vnd_count', 0)
        fp_backoff = observer_report.get('fp_backoff', 1.0)

        # 记录上一轮策略，供 FixedPoint-Detector 判断策略是否稳定。
        self.history.append({
            "generation": self.generation,
            "params": {
                "pc": float(self.params.pc),
                "pm": float(self.params.pm),
                "n": int(self.params.n),
            },
            "stage": self.current_stage,
            "diversity": float(diversity),
            "stagnation": int(stagnation_gen),
        })
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # 基于多样性和停滞程度调整基础 GA 速率。
        # 这里只做小幅调整，避免控制器本身成为新的随机扰动源。
        self.params.pc = self.base_pc
        self.params.pm = self.base_pm
        self.params.n = self.base_n
        if diversity < 0.16 and stagnation_gen >= max(12, self.base_crisis_threshold // 2):
            self.params.pc = min(0.95, self.base_pc + 0.05)
            self.params.pm = min(0.12, self.base_pm + 0.04)

        # 兼容自适应模式的动态门控（休眠期定期重激活）
        if self.ap_mode == 'dormant':
            # 休眠模式：不干预，定期检查是否应重新激活
            if (self.generation - self.last_dormant_check_gen >= self.ap_dormant_check_interval):
                self.last_dormant_check_gen = self.generation
                # 如果停滞超过阈值的1.5倍，尝试重新激活
                effective_threshold_check = int(self.base_crisis_threshold * self.backoff_factor)
                if stagnation_gen >= effective_threshold_check * self._precomputed['ap_dormant_recheck_ratio']:
                    self.ap_mode = 'observing'
                    self.ap_confidence = 0.3  # 低置信度重启
            # dormant模式下不干预
            self.params.crossover_operator = None
            self.params.mutation_operator = None
            self.current_stage = 'dormant'
            return self.params

        # 种群不稳定性检测
        normalized_cv = fitness_std / (abs(avg_fitness) + 1e-10)
        self.recent_cv.append(normalized_cv)
        if len(self.recent_cv) > self._precomputed['cv_window']:
            self.recent_cv = self.recent_cv[-self._precomputed['cv_window']:]

        if len(self.recent_cv) >= self._precomputed['cv_check_window']:
            avg_recent_cv = np.mean(self.recent_cv[-self._precomputed['cv_check_window']:])
            if avg_recent_cv > self._precomputed['cv_unstable_threshold']:
                self.high_variance_streak += 1
            else:
                self.high_variance_streak = max(0, self.high_variance_streak - 1)

        # v8.1: 粘性危机评估 + 置信度状态机
        if self.last_intervention_gen >= 0:
            gens_since = self.generation - self.last_intervention_gen
            if gens_since >= self._precomputed['crisis_hold']:
                improved = current_best < self.best_before_intervention - 1e-6
                self.intervention_outcomes.append(improved)
                intervention_gain = (
                    (self.best_before_intervention - current_best)
                    / max(abs(self.best_before_intervention), 1.0)
                )
                self.intervention_improvement_sum += intervention_gain
                self.intervention_improvement_count += 1

                if improved:
                    # 干预成功：提升置信度
                    self.consecutive_failures = 0
                    self.backoff_factor = max(
                        self._precomputed['backoff_min'],
                        self.backoff_factor * self._precomputed['backoff_recovery']
                    )
                    self.ap_confidence = min(1.0, self.ap_confidence + self._precomputed['ap_confidence_gain'])
                    if self.ap_confidence >= self._precomputed['ap_confidence_active']:
                        self.ap_mode = 'active'
                else:
                    # 干预失败：降低置信度
                    self.consecutive_failures += 1
                    self.backoff_factor = min(
                        self._precomputed['backoff_max'],
                        self.backoff_factor * self._precomputed['backoff_penalty']
                    )
                    self.ap_confidence = max(0.0, self.ap_confidence - self._precomputed['ap_confidence_loss'])
                    # 置信度低于阈值时进入休眠
                    if self.ap_confidence < self._precomputed['ap_confidence_dormant']:
                        self.ap_mode = 'dormant'
                        self.last_dormant_check_gen = self.generation
                        self.crisis_mode = False
                        self.current_combo = None
                        self.last_intervention_gen = -1
                        self.params.crossover_operator = None
                        self.params.mutation_operator = None
                        self.current_stage = 'dormant'
                        return self.params
                    elif self.ap_confidence < self._precomputed['ap_confidence_active']:
                        self.ap_mode = 'observing'

                self.crisis_mode = False
                self.current_combo = None
                self.last_intervention_gen = -1

        # 兼容自适应模式的进度门控和工作窗口
        progress = self.generation / max(self.max_generations, 1)

        # 早期门控（前20%）
        if progress < self._precomputed['progress_gate']:
            self.crisis_mode = False
            self.params.crossover_operator = None
            self.params.mutation_operator = None
            self.current_stage = 'stable'
            return self.params

        # 兼容自适应模式在70%进度后逐步退出，让FP主导终局
        if progress > self.ap_max_progress:
            if not self.crisis_mode:
                # 不再触发新危机
                self.params.crossover_operator = None
                self.params.mutation_operator = None
                self.current_stage = 'handover_to_fp'
                return self.params
            # 已有危机继续完成

        # 计算有效危机阈值
        effective_threshold = int(self.base_crisis_threshold * self.backoff_factor)
        effective_threshold = int(effective_threshold * max(0.8, min(2.0, fp_backoff)))

        # 高方差时额外提高阈值30%
        if self.high_variance_streak >= self._precomputed['high_variance_trigger']:
            effective_threshold = int(effective_threshold * 1.3)

        # FP状态感知——兼容自适应模式根据FP活跃度调整干预阈值
        # FP执行次数多：说明搜索已获得较多精化，兼容模式适度退让
        # FP执行次数少：说明FP不活跃，兼容模式可以更积极
        if fp_vnd_count > 6:
            # FP已多次执行，兼容模式减少干预
            effective_threshold = int(effective_threshold * 1.3)
        elif fp_vnd_count < 3 and progress > 0.4:
            # FP尚未触发时保留GA探索，避免兼容模式抢占开发预算。
            effective_threshold = int(effective_threshold * 1.1)

        # FP成功率反馈（辅助调整）
        if self.fp_vnd_success_rate >= 0.6:
            # FP效果很好，兼容模式轻度退让
            effective_threshold = int(effective_threshold * 1.2)
        elif self.fp_vnd_success_rate < 0.3:
            # FP效果暂弱时也只轻微放宽救援阈值，避免破坏精英结构。
            effective_threshold = int(effective_threshold * 1.05)

        # v8.2: 阈值上限保护——确保AP在早停前有干预机会
        if self.early_stop_stagnant is not None:
            threshold_cap = int(self.early_stop_stagnant * self._precomputed['threshold_early_stop_ratio'])
            if effective_threshold > threshold_cap:
                effective_threshold = threshold_cap
        else:
            estimated_early_stop = max(60, int(self.max_generations * 0.25))
            threshold_cap = int(estimated_early_stop * self._precomputed['threshold_early_stop_ratio'])
            if effective_threshold > threshold_cap:
                effective_threshold = threshold_cap

        # 保护期不调整
        if self.protection_remaining > 0:
            self.protection_remaining -= 1
            self.current_stage = 'protect'
            self.params.crossover_operator = None
            self.params.mutation_operator = None
            return self.params

        # 规模自适应多样性门控
        # 小规模(<100 ops): 0.30（更早触发AP，因为搜索空间小）
        # 中规模(100-200 ops): 0.35（平衡）
        # 大规模(>200 ops): 0.40（更晚触发AP，给GA更多自由探索空间）
        if self.total_ops < 100:
            diversity_threshold = 0.30
        elif self.total_ops < 200:
            diversity_threshold = 0.35
        else:
            diversity_threshold = 0.40

        if diversity > diversity_threshold:
            self.crisis_mode = False
            self.current_combo = None
            self.params.crossover_operator = None
            self.params.mutation_operator = None
            self.current_stage = 'stable'
            return self.params

        # 粘性危机模式
        if self.crisis_mode and self.current_combo is not None:
            self.params.crossover_operator = self.current_combo[0]
            self.params.mutation_operator = self.current_combo[1]
            self.current_stage = 'crisis'
            return self.params

        # 判断是否进入危机模式
        if stagnation_gen >= effective_threshold:
            self.crisis_mode = True
            self.crisis_entries += 1
            self.last_stagnation_gen = stagnation_gen
            self.crisis_attempt_count = getattr(self, 'crisis_attempt_count', 0) + 1

            self.best_before_intervention = current_best
            self.last_intervention_gen = self.generation

            # 兼容自适应模式在危机阶段使用探索算子注入多样性。
            self.current_combo = self._select_crisis_combo()
            combo_key = f"{self.current_combo[0]}+{self.current_combo[1]}"
            self.ap_operator_selections[combo_key] = (
                self.ap_operator_selections.get(combo_key, 0) + 1
            )
            self.params.crossover_operator = self.current_combo[0]
            self.params.mutation_operator = self.current_combo[1]
            self.current_stage = 'crisis'
        else:
            self.crisis_mode = False
            self.current_combo = None
            self.params.crossover_operator = None
            self.params.mutation_operator = None
            self.current_stage = 'stable'

        # 记录使用
        op_name = self.params.crossover_operator or 'adaptive(default)'
        self.operator_usage[op_name] = self.operator_usage.get(op_name, 0) + 1
        mut_name = self.params.mutation_operator or 'adaptive(default)'
        self.operator_usage[mut_name] = self.operator_usage.get(mut_name, 0) + 1

        return self.params

    def _select_crisis_combo(self) -> Tuple[str, str]:
        """兼容模式使用 UCB 风格评分选择危机算子组合。

        小规模：温和算子（避免过度破坏）
        中规模：中等破坏算子（平衡探索开发）
        大规模：高破坏算子（突破局部最优）
        """
        if not self._crisis_combos:
            return self._gentle_combos[0]

        def score(operator_name: str, stats: Dict[str, Dict[str, float]]) -> float:
            item = stats[operator_name]
            if item["count"] == 0:
                return float("inf")
            exploration = math.sqrt(
                2.0 * math.log(max(1.0, self.total_operator_trials)) / item["count"]
            )
            return item["reward"] / item["count"] + exploration

        scored = []
        for crossover_name, mutation_name in self._crisis_combos:
            pair_score = score(crossover_name, self.crossover_stats)
            pair_score += score(mutation_name, self.mutation_stats)
            scored.append((pair_score, crossover_name, mutation_name))

        _, crossover_name, mutation_name = max(scored, key=lambda item: item[0])
        return crossover_name, mutation_name

    def record_crossover_reward(self, operator_name: str, improvement: float):
        if operator_name not in self.crossover_stats:
            return
        reward = max(0.0, min(1.0, float(improvement)))
        stats = self.crossover_stats[operator_name]
        stats["count"] += 1.0
        stats["reward"] += reward
        self.total_operator_trials += 1

    def record_mutation_reward(self, operator_name: str, improvement: float):
        if operator_name not in self.mutation_stats:
            return
        reward = max(0.0, min(1.0, float(improvement)))
        stats = self.mutation_stats[operator_name]
        stats["count"] += 1.0
        stats["reward"] += reward

    def get_selected_crossover(self) -> str:
        return self.params.crossover_operator

    def get_selected_mutation(self) -> str:
        return self.params.mutation_operator

    def get_inject_ratio(self, diversity: float, stagnation_gen: int) -> float:
        if diversity >= 0.25 or stagnation_gen < 30:
            return 0.0
        if diversity < 0.10 and stagnation_gen >= 80:
            return 0.05
        if diversity < 0.18 and stagnation_gen >= 50:
            return 0.03
        return 0.01

    def should_try_safe_rescue(self, diversity: float, stagnation_gen: int,
                               generation: int) -> bool:
        """兼容模式下判定是否尝试一次不破坏精英的候选。"""
        progress = generation / max(self.max_generations, 1)
        if self.ap_mode == 'dormant' or self.protection_remaining > 0 or self.crisis_mode:
            return False
        if progress < 0.45 or progress > self.ap_max_progress:
            return False
        if generation - self.last_safe_rescue_generation < self.safe_rescue_interval:
            return False
        # 当前GA在部分实例上保持较高汉明距离，却连续多代没有改善最优值。
        # 这属于结构性停滞，不能再用“低多样性”作为唯一前提。
        return stagnation_gen >= 12 and diversity < 0.90

    @staticmethod
    def get_safe_rescue_combo(diversity: float) -> Tuple[str, str]:
        """低多样性时扩大邻域，其余情况以保结构的算子为主。"""
        if diversity < 0.20:
            return 'order', 'inverse'
        return 'job_based', 'insert'

    def record_safe_rescue(self, generation: int, accepted: bool,
                           improved_best: bool, improvement_ratio: float = 0.0):
        self.last_safe_rescue_generation = generation
        self.safe_rescue_attempts += 1
        if accepted:
            self.safe_rescue_accepted += 1
        if improved_best:
            self.safe_rescue_best_improvements += 1
        self.safe_rescue_improvement_sum += float(improvement_ratio)
        self.operator_usage['safe_rescue'] = self.operator_usage.get('safe_rescue', 0) + 1

    def record_ap_random_injection(self, inject_ratio: float, population_size: int):
        """兼容旧版接口：记录自适应参数模式注入的随机个体数量。"""
        self.ap_random_injection_generations += 1
        self.ap_random_injected_individuals += max(
            1, int(round(float(inject_ratio) * population_size))
        )

    def set_protection(self, generations: int = 3):
        self.protection_remaining = generations

    def update_fp_vnd_success_rate(self, success_rate: float):
        self.fp_vnd_success_rate = success_rate

    def set_max_generations(self, max_gens: int):
        self.max_generations = max_gens

    def set_early_stop_stagnant(self, early_stop: int):
        """v8.2: 设置早停阈值，用于阈值上限保护"""
        self.early_stop_stagnant = early_stop

    def get_controller_summary(self) -> Dict:
        return {
            'stage': self.current_stage,
            'crisis_mode': self.crisis_mode,
            'ap_mode': self.ap_mode,
            # Agent是新架构的正式名称；ap_*字段仅保留给旧结果读取。
            'agent_mode': self.ap_mode,
            'agent_confidence': round(self.ap_confidence, 2),
            'ap_confidence': round(self.ap_confidence, 2),
            'ap_disabled': self.ap_disabled,
            'ap_auto_disabled': self.ap_auto_disabled,
            'operator_usage': self.operator_usage.copy(),
            'problem_scale': 'small' if self.is_small else ('medium' if self.is_medium else 'large'),
            'is_asymmetric': self.is_asymmetric,
            'backoff_factor': round(self.backoff_factor, 2),
            'intervention_success_rate': (
                sum(self.intervention_outcomes) / len(self.intervention_outcomes)
                if self.intervention_outcomes else 1.0
            ),
            'intervention_count': len(self.intervention_outcomes),
            'consecutive_failures': self.consecutive_failures,
            'fp_vnd_success_rate': round(self.fp_vnd_success_rate, 2),
            'safe_rescue_attempts': self.safe_rescue_attempts,
            'safe_rescue_accepted': self.safe_rescue_accepted,
            'safe_rescue_best_improvements': self.safe_rescue_best_improvements,
            'safe_rescue_avg_improvement_ratio': (
                self.safe_rescue_improvement_sum / self.safe_rescue_attempts
                if self.safe_rescue_attempts else 0.0
            ),
            'crisis_entries': self.crisis_entries,
            'ap_operator_selections': self.ap_operator_selections.copy(),
            'ap_operator_selection_count': sum(self.ap_operator_selections.values()),
            'agent_operator_selections': self.ap_operator_selections.copy(),
            'agent_operator_selection_count': sum(self.ap_operator_selections.values()),
            'ap_random_injection_generations': self.ap_random_injection_generations,
            'ap_random_injected_individuals': self.ap_random_injected_individuals,
            'agent_random_injection_generations': self.ap_random_injection_generations,
            'agent_random_injected_individuals': self.ap_random_injected_individuals,
            'intervention_avg_improvement_ratio': (
                self.intervention_improvement_sum / self.intervention_improvement_count
                if self.intervention_improvement_count else 0.0
            ),
            'intervention_improvement_count': self.intervention_improvement_count,
            'commander_directive_counts': self.commander_directive_counts.copy(),
            'last_commander_directive': self.last_commander_directive,
            'component_feedback': {
                name: values.copy()
                for name, values in self.component_feedback.items()
            },
            'component_roi': self.component_roi_snapshot(),
            'orchestration_actions': list(self.orchestration_actions),
        }

    def get_status_description(self) -> str:
        # 显示Agent模式状态
        if self.ap_mode == 'dormant':
            return f"[dormant 置信{self.ap_confidence:.2f}] X:adaptive M:adaptive (Agent休眠中)"

        if self.protection_remaining > 0:
            status = f"protect [保护{self.protection_remaining}代]"
        elif self.crisis_mode:
            status = "crisis"
        elif self.ap_mode == 'active':
            status = f"active 置信{self.ap_confidence:.2f}"
        elif self.ap_mode == 'observing':
            status = f"observe 置信{self.ap_confidence:.2f}"
        else:
            status = "stable"

        x_op = self.params.crossover_operator or 'adaptive'
        m_op = self.params.mutation_operator or 'adaptive'

        # v8.3: 显示FP反馈状态
        fp_feedback = ""
        if self.fp_vnd_success_rate >= 0.5:
            fp_feedback = " FP优"
        elif self.fp_vnd_success_rate < 0.3:
            fp_feedback = " FP差"

        backoff_info = ""
        if self.backoff_factor > 1.1:
            backoff_info = f" ⚠退避{self.backoff_factor:.1f}x"
        elif self.high_variance_streak >= 5:
            backoff_info = f" ⚠不稳定"

        if self.consecutive_failures > 0:
            backoff_info += f" 失败{self.consecutive_failures}/{self.disable_threshold}"

        return f"[{status}{backoff_info}{fp_feedback}] X:{x_op} M:{m_op}"

    def reset(self):
        self.current_stage = 'stable'
        self.generation = 0
        self.crisis_mode = False
        self.protection_remaining = 0
        self.operator_usage = {}
        self.history = []
        self.intervention_outcomes = []
        self.last_intervention_gen = -1
        self.consecutive_failures = 0
        self.backoff_factor = 1.0
        self.recent_cv = []
        self.high_variance_streak = 0
        self.fp_vnd_success_rate = 0.5
        self.crossover_stats = {
            name: {"count": 0.0, "reward": 0.0}
            for name in self.CROSSOVER_OPERATORS
        }
        self.mutation_stats = {
            name: {"count": 0.0, "reward": 0.0}
            for name in self.MUTATION_OPERATORS
        }
        self.total_operator_trials = 0
        # v8.3: 保留disabled模式（小规模固定禁用）
        self.ap_disabled = False
        self.ap_mode = 'observing'
        self.ap_confidence = 0.5
        self.last_safe_rescue_generation = -self.safe_rescue_interval
        self.safe_rescue_attempts = 0
        self.safe_rescue_accepted = 0
        self.safe_rescue_best_improvements = 0
        self.safe_rescue_improvement_sum = 0.0
        self.crisis_entries = 0
        self.ap_operator_selections = {}
        self.ap_random_injection_generations = 0
        self.ap_random_injected_individuals = 0
        self.intervention_improvement_sum = 0.0
        self.intervention_improvement_count = 0
        self.commander_directive_counts = {}
        self.last_commander_directive = 'observe'
        self.component_feedback = {
            'local_search': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                             'work_calls': 0.0, 'elapsed': 0.0},
            'fp_detection': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                             'work_calls': 0.0, 'elapsed': 0.0},
            'vnd': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                    'work_calls': 0.0, 'elapsed': 0.0},
            'fixed_point': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                            'work_calls': 0.0, 'elapsed': 0.0},
            'terminal': {'calls': 0.0, 'improvements': 0.0, 'gain': 0.0,
                         'work_calls': 0.0, 'elapsed': 0.0},
        }
        self.recent_component_feedback = {
            'local_search': [], 'vnd': [], 'terminal': []
        }
        self.orchestration_actions = []
        self.last_dormant_check_gen = 0
        self.current_combo = None
        self.crisis_attempt_count = 0


# 新架构正式名称；Controller保留为历史代码的兼容入口。
AgentController = Controller

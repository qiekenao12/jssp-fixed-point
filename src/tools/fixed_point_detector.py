"""Fixed-point operator residual convergence detector.

FP observes the search process and reports convergence. It neither changes
individuals nor schedules local search. The residual model is independent of
Agent decisions, avoiding the old Agent -> strategy -> FP -> Agent cycle.
"""
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from tools.jssp_core import JSSPUtils
from tools.controller import Controller, GAParams
from tools.fp_signals import FPSignal, FPObservationTracker


class ConvergenceLevel(Enum):
    NO_CONVERGENCE = "no_convergence"
    EARLY_SIGN = "early_sign"
    PARTIAL_CONVERGENCE = "partial"
    FULL_CONVERGENCE = "full"
    PREMATURE_CONVERGENCE = "premature"


@dataclass
class FixedPointCriteria:
    genotype_epsilon: float = 0.10
    phenotypic_stagnant_generations: int = 15
    strategy_threshold: float = 0.015
    stable_generations: int = 8
    residual_window: int = 5
    residual_early_threshold: float = 0.030
    residual_partial_threshold: float = 0.012
    residual_full_threshold: float = 0.004
    best_weight: float = 0.35
    mean_weight: float = 0.30
    dispersion_weight: float = 0.15
    diversity_weight: float = 0.20
    # Use substantially more transitions than affine coefficients to reduce
    # small-window overfitting; ridge keeps rank-deficient data well-defined.
    operator_min_transitions: int = 16
    operator_ridge: float = 1e-6
    operator_fit_threshold: float = 0.030

    def adapt_to_problem_size(self, total_ops: int):
        if total_ops <= 50:
            self.genotype_epsilon = 0.12
            self.phenotypic_stagnant_generations = 12
            self.stable_generations = 6
            self.residual_window = 4
        elif total_ops <= 100:
            self.genotype_epsilon = 0.10
            self.phenotypic_stagnant_generations = 15
            self.stable_generations = 8
            self.residual_window = 5
        else:
            self.genotype_epsilon = 0.07
            self.phenotypic_stagnant_generations = 18
            self.stable_generations = 10
            self.residual_window = 6


@dataclass
class FixedPointAction:
    action_type: str
    description: str
    params: Dict = field(default_factory=dict)


@dataclass
class FixedPointStatus:
    is_fixed_point: bool
    convergence_level: ConvergenceLevel
    genotype_converged: bool
    phenotypic_converged: bool
    strategy_converged: bool
    genotype_score: float
    phenotypic_variance: float
    phenotypic_delta: float
    strategy_delta: float
    action: FixedPointAction
    # Deprecated compatibility alias: ALWAYS bounded FP observation intervals,
    # in every mode. Actual generations exist only in the typed signal_v2.
    stagnant_generations: int = 0
    diversity_estimate: float = 0.0
    # 连续确认只作为Agent的附加置信信号，不改变FP的基本收敛判定。
    confirmation_streak: int = 0
    confirmed: bool = False
    # Compatibility fields above keep old result readers working. Strategy
    # convergence is diagnostic only and is not part of the residual model.
    model_mode: str = 'residual_shadow'
    active_model: str = 'legacy'
    state_residual: float = 1.0
    fitness_residual: float = 1.0
    mean_residual: float = 1.0
    dispersion_residual: float = 1.0
    diversity_residual: float = 1.0
    residual_threshold: float = 0.004
    residual_window: int = 0
    residual_span: int = 0
    residual_level: ConvergenceLevel = ConvergenceLevel.NO_CONVERGENCE
    residual_confirmation_streak: int = 0
    residual_confirmed: bool = False
    operator_residual: float = 1.0
    operator_fitness_residual: float = 1.0
    operator_mean_residual: float = 1.0
    operator_dispersion_residual: float = 1.0
    operator_diversity_residual: float = 1.0
    operator_fit_error: float = 1.0
    operator_rank: int = 0
    operator_transitions: int = 0
    operator_step: int = 0
    operator_level: ConvergenceLevel = ConvergenceLevel.NO_CONVERGENCE
    operator_confirmation_streak: int = 0
    operator_confirmed: bool = False
    operator_model_active: bool = False
    # New typed contract is opt-in. The older scalar fields remain compatible
    # with frozen historical readers and retain their documented legacy units.
    signal_v2: Optional[FPSignal] = None

    @property
    def legacy_stagnant_observations(self) -> int:
        """Explicit unit-safe name for the frozen bounded compatibility count."""
        return self.stagnant_generations


class FixedPointDetector:
    """不动点收敛检测器。

    FP 只读取种群和适应度历史并报告收敛状态；它不决定、不执行
    局部搜索或 VND。FP只报告不动点收敛状态；Agent负责下发局部搜索和
    VND的触发时机、目标范围与预算。
    
    三个阶段的触发配置（基于问题规模自适应）：
    - 早期微精化：进度达到30%+且有短期停滞（5-8代）即触发
    - 中期VND：进度达到50%+且有中期停滞（8-12代）触发
    - 终局强化：进度达到75%+且有长期停滞（6-10代）或完全收敛触发
    """

    VALID_MODEL_MODES = {
        'legacy', 'residual_shadow', 'residual',
        'operator_shadow', 'operator',
    }

    def __init__(self, problem_size: int = 100, max_generations: int = 200,
                 model_mode: Optional[str] = None, *, semantic_mode: str = 'legacy'):
        if semantic_mode not in ('legacy', 'generation_v2'):
            raise ValueError('semantic_mode must be legacy or generation_v2')
        self.semantic_mode = semantic_mode
        self.signal_tracker = FPObservationTracker()
        self.criteria = FixedPointCriteria()
        self.criteria.adapt_to_problem_size(problem_size)
        
        self.history = {
            'avg_hamming': [],
            'best_fitness': [],
            'convergence_levels': [],
            'improvement_rates': [],
            'diversity_estimates': [],
            'search_states': [],
            'state_residuals': [],
            'residual_levels': [],
            'operator_residuals': [],
            'operator_levels': [],
        }

        requested_mode = (model_mode or os.getenv(
            'FP_MODEL_MODE', 'residual_shadow'
        )).strip().lower()
        if requested_mode not in self.VALID_MODEL_MODES:
            valid = ', '.join(sorted(self.VALID_MODEL_MODES))
            raise ValueError(f'FP_MODEL_MODE must be one of: {valid}')
        self.model_mode = requested_mode
        self.fitness_scale = None
        
        self.stable_counter = 0
        self.generation_count = 0
        self.problem_size = problem_size
        self.max_generations = max_generations
        
        # 收敛稳定性要求
        self.stability_required = 5
        
        # ===== v10.0 建议1: 回滚FP参数到v8.3 =====
        if problem_size <= 50:
            # 小规模问题
            self.stage1_progress = 0.25
            self.stage2_progress = 0.45
            self.stage3_progress = 0.70
            self.stage1_stagnant = 8       # v8.3参数
            self.stage2_stagnant = 8
            self.stage3_stagnant = 5
            self.stage1_top_k = 0.08       # v8.3参数
            self.stage2_top_k = 0.05
            self.stage3_top_k = 0.03
            self.stage1_max_iter = 80      # v8.3参数
            self.stage2_max_iter = 120
            self.stage3_max_iter = 200
        elif problem_size <= 100:
            # 中规模问题
            self.stage1_progress = 0.28
            self.stage2_progress = 0.50
            self.stage3_progress = 0.73
            self.stage1_stagnant = 11      # v8.3参数
            self.stage2_stagnant = 10
            self.stage3_stagnant = 7
            self.stage1_top_k = 0.07       # v8.3参数
            self.stage2_top_k = 0.04
            self.stage3_top_k = 0.03
            self.stage1_max_iter = 100     # v8.3参数
            self.stage2_max_iter = 150
            self.stage3_max_iter = 220
        elif problem_size <= 400:
            # 大规模问题（保持原有150~400工序档位）
            self.stage1_progress = 0.30
            self.stage2_progress = 0.55
            self.stage3_progress = 0.75
            self.stage1_stagnant = 14      # v8.3参数
            self.stage2_stagnant = 12
            self.stage3_stagnant = 9
            self.stage1_top_k = 0.06       # v8.3参数
            self.stage2_top_k = 0.04
            self.stage3_top_k = 0.03
            self.stage1_max_iter = 120     # v8.3参数
            self.stage2_max_iter = 180
            self.stage3_max_iter = 250
        else:
            # 超大规模问题：更早进入中后期精化，并提高单次搜索预算。
            # 仅在停滞成立时触发，避免用VND替代大规模实例的探索阶段。
            self.stage1_progress = 0.32
            self.stage2_progress = 0.50
            self.stage3_progress = 0.68
            self.stage1_stagnant = 18
            self.stage2_stagnant = 14
            self.stage3_stagnant = 7
            self.stage1_top_k = 0.05
            self.stage2_top_k = 0.04
            self.stage3_top_k = 0.04
            self.stage1_max_iter = 180
            self.stage2_max_iter = 260
            self.stage3_max_iter = 360
        
        # VND执行次数限制（增加到12次，提高干预机会）
        self.max_vnd_executions = 20 if problem_size > 400 else 16
        self.vnd_executed_count = 0
        
        # 冷却期（缩短到5代，提高响应速度）
        self.vnd_cooldown = 5
        self.action_cooldown = 0
        
        # 记录
        self.last_action_type = None
        self.improvement_count = 0
        self.no_improvement_count = 0
        
        # 收敛检测计数器
        self.consecutive_levels = []
        self.confirmation_streak = 0
        self.last_positive_level = None
        self.residual_confirmation_streak = 0
        self.last_residual_positive_level = None
        self.operator_confirmation_streak = 0
        self.last_operator_positive_level = None

        # ===== v6.5: VND效果追踪与自适应强度 =====
        # v8.3: 增强退避机制——失败时更激进退避，成功时更快恢复
        self.vnd_success_count = 0     # VND成功改进次数
        self.vnd_failure_count = 0     # VND失败次数
        self.vnd_success_rate = 1.0    # VND成功率
        self.vnd_backoff = 1.0         # VND退避因子（乘到停滞阈值上）
        self.max_vnd_backoff = 2.5     # v8.3: 1.8→2.5，失败时更激进退避
        self.min_vnd_backoff = 0.8     # 最小退避倍数（成功时降低阈值）

        # ===== v6.5: 实例形状感知 =====
        # 检测是否为非对称实例（如15×10），这类实例关键路径长，VND需更保守
        self.is_asymmetric = False  # 由MetaScheduler设置

    def detect(self,
                       population: List[np.ndarray],
                       fitness_values: List[float],
                       controller: Controller,
                       generation: int,
                       allow_final_refine: bool = True,
                       allow_actions: bool = True,
                       action_policy: Optional[Dict] = None,
                       *, observed_stagnant_generations: Optional[int] = None
                       ) -> Tuple[FixedPointStatus, Dict]:
        """
        FP 只执行收敛判断。保留旧参数以兼容可视化和外部调用，但
        allow_actions/action_policy 不再改变返回结果，也不会产生 VND 计划。
        """
        if self.semantic_mode == 'generation_v2':
            if (isinstance(observed_stagnant_generations, bool)
                    or not isinstance(observed_stagnant_generations, int)
                    or observed_stagnant_generations < 0):
                raise ValueError('generation_v2 requires Observer actual stagnant generations')
        self.generation_count = generation + 1
        progress_ratio = self.generation_count / self.max_generations
        # controller is read only for legacy compatibility diagnostics. The
        # residual model never reads Agent actions or strategy signatures.
        geno_result = self._check_genotype_convergence(population)
        pheno_result = self._check_phenotypic_stagnation(
            fitness_values, generation,
            observed_stagnant_generations=(observed_stagnant_generations
                if self.semantic_mode == 'generation_v2' else None),
        )
        search_state = self._build_search_state(
            fitness_values, geno_result['avg_hamming'], generation
        )
        residual_result = self._check_fixed_point_residual(search_state)
        operator_result = self._check_operator_fixed_point(search_state)
        signal_v2 = self._build_semantic_signal(
            search_state, operator_result, residual_result,
            observed_stagnant_generations,
        ) if self.semantic_mode == 'generation_v2' else None
        
        # 早期(<阶段1进度)：仅记录，不干预
        if progress_ratio < self.stage1_progress:
            self._record_history(
                geno_result['avg_hamming'], np.min(fitness_values),
                ConvergenceLevel.NO_CONVERGENCE, 0.0,
                geno_result['avg_hamming'] * 2,
                search_state=search_state,
                residual_result=residual_result,
                operator_result=operator_result,
            )
            action = FixedPointAction(action_type="none", description="早期仅观察", params={})
            return FixedPointStatus(
                is_fixed_point=False,
                convergence_level=ConvergenceLevel.NO_CONVERGENCE,
                genotype_converged=False,
                phenotypic_converged=False,
                strategy_converged=False,
                genotype_score=0.5,
                phenotypic_variance=1.0,
                phenotypic_delta=1.0,
                strategy_delta=1.0,
                action=action,
                stagnant_generations=0,
                diversity_estimate=0.5,
                confirmation_streak=0,
                confirmed=False,
                signal_v2=signal_v2,
                **self._residual_status_fields(residual_result),
                **self._operator_status_fields(operator_result),
            ), self._empty_params()
        
        # Legacy strategy stability is retained only to read historical files.
        # It is not an input to the residual fixed-point model.
        strat_result = self._check_strategy_convergence(controller)
        
        diversity_est = geno_result.get('avg_hamming', 0.5) * 2
        self._update_improvement_count(pheno_result)
        
        consecutive_stagnant = pheno_result.get('stagnant_generations', 0)
        pheno_conv = pheno_result['converged']
        pheno_stagnant = pheno_result.get('stagnant', False)
        
        # Legacy classification is kept as a migration baseline.
        if pheno_conv and strat_result['converged'] and geno_result['converged']:
            legacy_level = ConvergenceLevel.FULL_CONVERGENCE
        elif pheno_conv:
            legacy_level = ConvergenceLevel.PARTIAL_CONVERGENCE
        elif pheno_stagnant:
            legacy_level = ConvergenceLevel.EARLY_SIGN
        else:
            legacy_level = ConvergenceLevel.NO_CONVERGENCE

        active_level = (
            operator_result['level']
            if self.model_mode == 'operator'
            else residual_result['level']
            if self.model_mode == 'residual'
            else legacy_level
        )

        # v11: 对收敛征兆增加连续确认和滞回。只有当前检测等级不低于
        # 上一次正向等级时才累积确认，避免一次短暂停滞直接被当成稳定状态。
        # 该信号仅供Agent决定是否追加少量预算，原有VND触发条件保持不变。
        level_rank = {
            ConvergenceLevel.NO_CONVERGENCE: 0,
            ConvergenceLevel.EARLY_SIGN: 1,
            ConvergenceLevel.PARTIAL_CONVERGENCE: 2,
            ConvergenceLevel.FULL_CONVERGENCE: 3,
        }
        if active_level == ConvergenceLevel.NO_CONVERGENCE:
            self.confirmation_streak = 0
            self.last_positive_level = None
        elif (self.last_positive_level is not None
              and level_rank[active_level] >= level_rank[self.last_positive_level]):
            self.confirmation_streak += 1
            self.last_positive_level = active_level
        else:
            self.confirmation_streak = 1
            self.last_positive_level = active_level
        confirmed = (
            active_level == ConvergenceLevel.FULL_CONVERGENCE
            or self.confirmation_streak >= 2
        )
        
        # 记录历史
        improvement = self._calc_improvement_rate(fitness_values)
        self._record_history(
            geno_result['avg_hamming'],
            np.min(fitness_values),
            active_level,
            improvement,
            diversity_est,
            search_state=search_state,
            residual_result=residual_result,
            operator_result=operator_result,
        )
        
        is_fp = active_level == ConvergenceLevel.FULL_CONVERGENCE
        
        status = FixedPointStatus(
            is_fixed_point=is_fp,
            convergence_level=active_level,
            genotype_converged=geno_result['converged'],
            phenotypic_converged=pheno_result['converged'],
            strategy_converged=strat_result['converged'],
            genotype_score=geno_result['avg_hamming'],
            phenotypic_variance=pheno_result.get('variance', 0),
            phenotypic_delta=pheno_result.get('delta', 0),
            strategy_delta=strat_result['delta'],
            action=FixedPointAction(
                action_type='none',
                description='FP仅完成不动点收敛判断，VND由Agent调度',
                params={},
            ),
            stagnant_generations=consecutive_stagnant,
            diversity_estimate=diversity_est,
            confirmation_streak=self.confirmation_streak,
            confirmed=confirmed,
            signal_v2=signal_v2,
            **self._residual_status_fields(residual_result),
            **self._operator_status_fields(operator_result),
        )
        
        return status, self._empty_params()

    def _build_semantic_signal(self, state, operator, residual, stagnant_generations):
        """Expose units and missing evidence without promoting training fit."""
        if self.model_mode in ('operator', 'operator_shadow'):
            ready = operator['transitions'] >= self.criteria.operator_min_transitions
            value = float(operator['operator_residual']) if ready else None
            fit_error = float(operator['fit_error']) if ready else None
            source = 'unified_affine_operator'
        else:
            ready = residual['span'] > 0
            value = float(residual['state_residual']) if ready else None
            fit_error = None
            source = 'window_state_displacement'
        return self.signal_tracker.observe(
            generation=state['generation'],
            best_fitness=state['best'] * self.fitness_scale,
            stagnant_generations=stagnant_generations,
            residual=value, model_ready=ready, source_model=source,
            internal_model_stagnant_generations=self._residual_stagnant_generations(state['best']),
            training_fit_error=fit_error,
            # No grouped held-out error exists in the legacy online estimator.
            prediction_error=None, reliability=0.0,
        )

    # 兼容旧版调用方；新代码应使用 detect()，FP不再执行任何动作。
    detect_and_act = detect

    def _build_search_state(self, fitness_values: List[float],
                            diversity: float, generation: int) -> Dict:
        """Build the Agent-independent normalized state x_t."""
        values = np.asarray(fitness_values, dtype=float)
        best = float(np.min(values))
        if self.fitness_scale is None:
            self.fitness_scale = max(abs(best), 1.0)
        scale = self.fitness_scale
        return {
            'generation': int(generation),
            'best': best / scale,
            'mean': float(np.mean(values)) / scale,
            'dispersion': float(np.std(values)) / scale,
            'diversity': float(np.clip(diversity, 0.0, 1.0)),
        }

    def _check_fixed_point_residual(self, current: Dict) -> Dict:
        """Estimate the weighted residual between x_t and its windowed image."""
        reference = None
        min_generation = current['generation'] - self.criteria.residual_window
        for candidate in reversed(self.history['search_states']):
            if candidate['generation'] <= min_generation:
                reference = candidate
                break

        if reference is None:
            result = {
                'state_residual': 1.0,
                'fitness_residual': 1.0,
                'mean_residual': 1.0,
                'dispersion_residual': 1.0,
                'diversity_residual': 1.0,
                'span': 0,
                'level': ConvergenceLevel.NO_CONVERGENCE,
            }
            self._update_residual_confirmation(result)
            return result

        components = {
            'fitness_residual': abs(current['best'] - reference['best']),
            'mean_residual': abs(current['mean'] - reference['mean']),
            'dispersion_residual': abs(
                current['dispersion'] - reference['dispersion']
            ),
            'diversity_residual': abs(
                current['diversity'] - reference['diversity']
            ),
        }
        residual = (
            self.criteria.best_weight * components['fitness_residual']
            + self.criteria.mean_weight * components['mean_residual']
            + self.criteria.dispersion_weight
            * components['dispersion_residual']
            + self.criteria.diversity_weight
            * components['diversity_residual']
        )
        stagnant = self._residual_stagnant_generations(current['best'])
        low_diversity = current['diversity'] <= self.criteria.genotype_epsilon

        if (residual <= self.criteria.residual_full_threshold
                and low_diversity
                and stagnant >= self.criteria.stable_generations):
            level = ConvergenceLevel.FULL_CONVERGENCE
        elif (residual <= self.criteria.residual_partial_threshold
              and stagnant >= max(3, self.criteria.stable_generations // 2)):
            level = ConvergenceLevel.PARTIAL_CONVERGENCE
        elif (residual <= self.criteria.residual_early_threshold
              or stagnant >= max(2, self.criteria.stable_generations // 3)):
            level = ConvergenceLevel.EARLY_SIGN
        else:
            level = ConvergenceLevel.NO_CONVERGENCE

        result = {
            **components,
            'state_residual': float(residual),
            'span': current['generation'] - reference['generation'],
            'level': level,
        }
        self._update_residual_confirmation(result)
        return result

    def _check_operator_fixed_point(self, current: Dict) -> Dict:
        """Estimate the fixed point of an empirical search transition operator.

        Consecutive observed states provide transitions (x_i, x_{i+1}). We
        fit the conditional-mean operator F_hat(x)=A x+c using ridge
        regression and evaluate the actual fixed-point equation
        x_t=F_hat(x_t). This is different from measuring x_t-x_{t-w}: the
        fitted operator can model a changing trajectory and still expose its
        fixed point.
        """
        states = self.history['search_states']
        min_transitions = self.criteria.operator_min_transitions
        if len(states) < min_transitions + 1:
            return self._empty_operator_result()

        # FP checks are scheduled at different intervals (early/mid/late).
        # A single F_tau must not mix transitions with different tau values.
        current_step = current['generation'] - states[-1]['generation']
        if current_step <= 0:
            return self._empty_operator_result()
        transitions = []
        for previous, current_state in zip(states[:-1], states[1:]):
            step = current_state['generation'] - previous['generation']
            if step != current_step:
                continue
            transitions.append((previous, current_state))
        transitions = transitions[-(min_transitions + 32):]
        if len(transitions) < min_transitions:
            return self._empty_operator_result()

        source = np.asarray([
            [previous['best'], previous['mean'], previous['dispersion'],
             previous['diversity']]
            for previous, _ in transitions
        ], dtype=float)
        target = np.asarray([
            [next_state['best'], next_state['mean'], next_state['dispersion'],
             next_state['diversity']]
            for _, next_state in transitions
        ], dtype=float)

        # The intercept is part of the operator. Ridge regularization is
        # applied to A but not to the intercept so constants remain exact.
        design = np.column_stack([source, np.ones(len(source))])
        regularizer = np.diag([self.criteria.operator_ridge] * 4 + [0.0])
        try:
            coefficients = np.linalg.solve(
                design.T @ design + regularizer,
                design.T @ target,
            )
        except np.linalg.LinAlgError:
            return self._empty_operator_result()

        predicted = design @ coefficients
        fit_errors = predicted - target
        weights = np.sqrt(np.array([
            self.criteria.best_weight,
            self.criteria.mean_weight,
            self.criteria.dispersion_weight,
            self.criteria.diversity_weight,
        ]))
        fit_error = float(np.mean(np.linalg.norm(fit_errors * weights, axis=1)))
        current_vector = np.asarray([
            current['best'], current['mean'], current['dispersion'],
            current['diversity'],
        ], dtype=float)
        predicted_current = np.r_[current_vector, 1.0] @ coefficients
        fixed_point_error = current_vector - predicted_current
        component_names = (
            'operator_fitness_residual', 'operator_mean_residual',
            'operator_dispersion_residual', 'operator_diversity_residual',
        )
        components = {
            name: float(abs(value))
            for name, value in zip(component_names, fixed_point_error)
        }
        operator_residual = float(np.linalg.norm(fixed_point_error * weights))
        low_diversity = current['diversity'] <= self.criteria.genotype_epsilon
        stagnant = self._residual_stagnant_generations(current['best'])
        reliable = fit_error <= self.criteria.operator_fit_threshold

        if (reliable and operator_residual <= self.criteria.residual_full_threshold
                and low_diversity
                and stagnant >= self.criteria.stable_generations):
            level = ConvergenceLevel.FULL_CONVERGENCE
        elif (reliable
              and operator_residual <= self.criteria.residual_partial_threshold
              and stagnant >= max(3, self.criteria.stable_generations // 2)):
            level = ConvergenceLevel.PARTIAL_CONVERGENCE
        elif (reliable
              and operator_residual <= self.criteria.residual_early_threshold):
            level = ConvergenceLevel.EARLY_SIGN
        else:
            level = ConvergenceLevel.NO_CONVERGENCE

        result = {
            **components,
            'operator_residual': operator_residual,
            'fit_error': fit_error,
            'rank': int(np.linalg.matrix_rank(design)),
            'transitions': int(len(source)),
            'step': int(current_step),
            'level': level,
        }
        self._update_operator_confirmation(result)
        return result

    def _empty_operator_result(self) -> Dict:
        return {
            'operator_residual': 1.0,
            'operator_fitness_residual': 1.0,
            'operator_mean_residual': 1.0,
            'operator_dispersion_residual': 1.0,
            'operator_diversity_residual': 1.0,
            'fit_error': 1.0,
            'rank': 0,
            'transitions': 0,
            'step': 0,
            'level': ConvergenceLevel.NO_CONVERGENCE,
            'confirmation_streak': 0,
            'confirmed': False,
        }

    def _update_operator_confirmation(self, result: Dict):
        level = result['level']
        rank = {
            ConvergenceLevel.NO_CONVERGENCE: 0,
            ConvergenceLevel.EARLY_SIGN: 1,
            ConvergenceLevel.PARTIAL_CONVERGENCE: 2,
            ConvergenceLevel.FULL_CONVERGENCE: 3,
        }
        if level == ConvergenceLevel.NO_CONVERGENCE:
            self.operator_confirmation_streak = 0
            self.last_operator_positive_level = None
        elif (self.last_operator_positive_level is not None
              and rank[level] >= rank[self.last_operator_positive_level]):
            self.operator_confirmation_streak += 1
            self.last_operator_positive_level = level
        else:
            self.operator_confirmation_streak = 1
            self.last_operator_positive_level = level
        result['confirmation_streak'] = self.operator_confirmation_streak
        result['confirmed'] = (
            level == ConvergenceLevel.FULL_CONVERGENCE
            or self.operator_confirmation_streak >= 2
        )

    def _operator_status_fields(self, result: Dict) -> Dict:
        return {
            'operator_residual': result['operator_residual'],
            'operator_fitness_residual': result['operator_fitness_residual'],
            'operator_mean_residual': result['operator_mean_residual'],
            'operator_dispersion_residual': result['operator_dispersion_residual'],
            'operator_diversity_residual': result['operator_diversity_residual'],
            'operator_fit_error': result['fit_error'],
            'operator_rank': result['rank'],
            'operator_transitions': result['transitions'],
            'operator_step': result['step'],
            'operator_level': result['level'],
            'operator_confirmation_streak': result['confirmation_streak'],
            'operator_confirmed': result['confirmed'],
            'operator_model_active': self.model_mode == 'operator',
        }

    def _residual_stagnant_generations(self, current_best: float) -> int:
        count = 0
        tolerance = 1e-12
        for state in reversed(self.history['search_states']):
            if current_best < state['best'] - tolerance:
                break
            count = self.generation_count - 1 - state['generation']
        return max(0, int(count))

    def _update_residual_confirmation(self, result: Dict):
        level = result['level']
        rank = {
            ConvergenceLevel.NO_CONVERGENCE: 0,
            ConvergenceLevel.EARLY_SIGN: 1,
            ConvergenceLevel.PARTIAL_CONVERGENCE: 2,
            ConvergenceLevel.FULL_CONVERGENCE: 3,
        }
        if level == ConvergenceLevel.NO_CONVERGENCE:
            self.residual_confirmation_streak = 0
            self.last_residual_positive_level = None
        elif (self.last_residual_positive_level is not None
              and rank[level] >= rank[self.last_residual_positive_level]):
            self.residual_confirmation_streak += 1
            self.last_residual_positive_level = level
        else:
            self.residual_confirmation_streak = 1
            self.last_residual_positive_level = level
        result['confirmation_streak'] = self.residual_confirmation_streak
        result['confirmed'] = (
            level == ConvergenceLevel.FULL_CONVERGENCE
            or self.residual_confirmation_streak >= 2
        )

    def _residual_status_fields(self, result: Dict) -> Dict:
        return {
            'model_mode': self.model_mode,
            'active_model': (
                'operator' if self.model_mode == 'operator'
                else 'residual' if self.model_mode == 'residual'
                else 'legacy'
            ),
            'state_residual': result['state_residual'],
            'fitness_residual': result['fitness_residual'],
            'mean_residual': result['mean_residual'],
            'dispersion_residual': result['dispersion_residual'],
            'diversity_residual': result['diversity_residual'],
            'residual_threshold': self.criteria.residual_full_threshold,
            'residual_window': self.criteria.residual_window,
            'residual_span': result['span'],
            'residual_level': result['level'],
            'residual_confirmation_streak': result['confirmation_streak'],
            'residual_confirmed': result['confirmed'],
        }

    def _update_improvement_count(self, pheno_result: Dict):
        if pheno_result.get('improving', False):
            self.improvement_count += 1
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        self.improvement_count = min(self.improvement_count, 50)
        self.no_improvement_count = min(self.no_improvement_count, 100)

    def _empty_params(self) -> Dict:
        return {
            'inject_random': False,
            'inject_ratio': 0.0,
            'boost_mutation': False,
            'pm_multiplier': 1.0,
            'local_search_on_best': False,
            'local_search_iterations': 0,
            'apply_vnd': False,
            'top_k_ratio': 0.0,
            'stage': None,
        }

    def _check_genotype_convergence(self, population: List[np.ndarray]) -> Dict:
        if len(population) < 2:
            return {'converged': True, 'avg_hamming': 0.0}

        avg_hamming = self._calculate_average_hamming(population)
        max_possible = len(population[0])
        normalized = avg_hamming / max_possible if max_possible > 0 else 0.0

        converged = normalized < self.criteria.genotype_epsilon

        return {
            'converged': converged,
            'avg_hamming': normalized,
            'raw_hamming': avg_hamming
        }

    def _calculate_average_hamming(self, population: List[np.ndarray]) -> float:
        """优化：使用numpy向量化替代双重Python循环"""
        n = len(population)
        if n < 2:
            return 0.0
        pop_matrix = np.array(population)
        total = 0
        for i in range(n - 1):
            total += np.sum(pop_matrix[i] != pop_matrix[i+1:])
        count = n * (n - 1) // 2
        return total / count if count > 0 else 0.0

    def _check_phenotypic_stagnation(self, fitness_values: List[float], generation: int,
                                    *, observed_stagnant_generations: Optional[int] = None) -> Dict:
        arr = np.array(fitness_values)
        variance = float(np.var(arr))
        current_best = np.min(fitness_values)
        
        recent_window = min(10, len(self.history['best_fitness']))
        if recent_window < 3:
            return {
                'converged': False,
                'stagnant': False,
                'improving': True,
                'variance': variance,
                'delta': 0.0,
                'stagnant_generations': 0,
                'improvement_ratio': 1.0,
            }
        
        # 复制窗口，避免 append 修改历史列表本身。
        recent_bests = list(self.history['best_fitness'][-recent_window:])
        recent_bests.append(current_best)
        
        improvements = 0
        # 只统计从窗口末端开始的连续停滞，更符合“当前是否停滞”的语义。
        current_stagnant = 0

        for i in range(1, len(recent_bests)):
            if recent_bests[i] < recent_bests[i-1] - 1e-6:
                improvements += 1
                current_stagnant = 0
            else:
                current_stagnant += 1
        
        total_gaps = len(recent_bests) - 1
        improvement_ratio = improvements / max(total_gaps, 1)
        # current_stagnant remains a bounded observation count in all modes;
        # never overwrite it with an actual-generation quantity.
        
        improving = improvement_ratio > 0.40
        stagnant_threshold = max(4, self.criteria.phenotypic_stagnant_generations // 3)
        converged_threshold = max(8, self.criteria.phenotypic_stagnant_generations * 2 // 3)
        if observed_stagnant_generations is None:
            # Frozen legacy predicates operate on their historical observation
            # count; retained solely for reproducible legacy comparisons.
            stagnant = current_stagnant >= stagnant_threshold
            converged = (improvement_ratio < 0.10) and (current_stagnant >= converged_threshold)
        else:
            # Explicit semantic predicates consume actual Observer generations.
            stagnant = observed_stagnant_generations >= stagnant_threshold
            converged = (improvement_ratio < 0.10) and (observed_stagnant_generations >= converged_threshold)
        
        if recent_bests[0] > 0:
            total_improvement = (recent_bests[0] - recent_bests[-1]) / recent_bests[0] * 100
        else:
            total_improvement = 0.0
        
        return {
            'converged': converged,
            'stagnant': stagnant,
            'improving': improving,
            'variance': variance,
            'delta': abs(current_best - recent_bests[-2]) / (recent_bests[-2] + 1e-10),
            'stagnant_generations': current_stagnant,
            'improvement_ratio': improvement_ratio,
            'total_improvement': total_improvement,
        }

    def _check_strategy_convergence(self, controller: Controller) -> Dict:
        if len(controller.history) < 8:
            return {'converged': False, 'delta': 1.0}

        recent = controller.history[-10:]
        signatures = [h.get('strategy_signature') for h in recent]
        if all(signature is not None for signature in signatures):
            # Agent模式下直接比较实际调度签名，避免把固定GA参数误判为
            # 策略收敛。delta表示相邻调度策略发生变化的比例。
            changes = sum(
                signatures[i] != signatures[i - 1]
                for i in range(1, len(signatures))
            )
            delta = changes / max(len(signatures) - 1, 1)
            return {
                'converged': delta < self.criteria.strategy_threshold,
                'delta': delta,
            }

        pc_vals = [h.get('params', {}).get('pc', controller.params.pc) for h in recent]
        pm_vals = [h.get('params', {}).get('pm', controller.params.pm) for h in recent]
        
        pc_range = max(pc_vals) - min(pc_vals)
        pm_range = max(pm_vals) - min(pm_vals)
        
        delta = (pc_range + pm_range) / 2
        converged = delta < self.criteria.strategy_threshold
        
        return {'converged': converged, 'delta': delta}

    def _calc_improvement_rate(self, fitness_values: List[float]) -> float:
        if len(self.history['best_fitness']) < 3:
            return 0.0
        
        current_best = np.min(fitness_values)
        prev_best = self.history['best_fitness'][-1]
        if prev_best > 0:
            return (prev_best - current_best) / prev_best * 100
        return 0.0

    def _record_history(self,
                       avg_hamming: float,
                       best_fitness: float,
                       convergence_level: ConvergenceLevel,
                       improvement: float,
                       diversity: float = 0.5,
                       search_state: Optional[Dict] = None,
                       residual_result: Optional[Dict] = None,
                       operator_result: Optional[Dict] = None):
        self.history['avg_hamming'].append(avg_hamming)
        self.history['best_fitness'].append(best_fitness)
        self.history['convergence_levels'].append(convergence_level.value)
        self.history['improvement_rates'].append(improvement)
        self.history['diversity_estimates'].append(diversity)
        if search_state is not None:
            self.history['search_states'].append(dict(search_state))
        if residual_result is not None:
            self.history['state_residuals'].append(
                float(residual_result['state_residual'])
            )
            self.history['residual_levels'].append(
                residual_result['level'].value
            )
        if operator_result is not None:
            self.history['operator_residuals'].append(
                float(operator_result['operator_residual'])
            )
            self.history['operator_levels'].append(
                operator_result['level'].value
            )
        
        for key in self.history:
            if len(self.history[key]) > 80:
                self.history[key] = self.history[key][-80:]

    def get_convergence_summary(self) -> Dict:
        if len(self.history['convergence_levels']) < 3:
            return {'status': 'insufficient_data'}
        
        recent_levels = self.history['convergence_levels'][-5:]
        recent_improvements = self.history['improvement_rates'][-5:]
        
        avg_improvement = np.mean([x for x in recent_improvements if x > 0]) if any(x > 0 for x in recent_improvements) else 0.0
        
        if all(l == 'full' for l in recent_levels[-3:]):
            trend = 'fully_converged'
        elif all(l in ['partial', 'full'] for l in recent_levels[-3:]):
            trend = 'converging'
        elif all(l in ['no_convergence', 'early_sign'] for l in recent_levels[-3:]):
            trend = 'exploring'
        else:
            trend = 'mixed'
        
        return {
            'trend': trend,
            'avg_improvement': float(avg_improvement),
            'recent_levels': recent_levels[-10:],
            'generations_tracked': len(self.history['best_fitness']),
            'improvement_count': self.improvement_count,
            'no_improvement_count': self.no_improvement_count,
            'vnd_executed': self.vnd_executed_count,
            'vnd_remaining': self.max_vnd_executions - self.vnd_executed_count,
            'vnd_success_rate': round(self.vnd_success_rate, 2),
            'vnd_backoff': round(self.vnd_backoff, 2),
            'fp_model_mode': self.model_mode,
            'active_fp_model': (
                'operator' if self.model_mode == 'operator'
                else 'residual' if self.model_mode == 'residual'
                else 'legacy'
            ),
            'recent_state_residuals': self.history['state_residuals'][-5:],
            'recent_residual_levels': self.history['residual_levels'][-5:],
            'recent_operator_residuals': self.history['operator_residuals'][-5:],
            'recent_operator_levels': self.history['operator_levels'][-5:],
        }

    def record_vnd_result(self, improved: bool):
        """
        v6.5: 记录VND执行结果，自适应调整VND触发强度
        v8.3: 更激进的退避/恢复——失败时更快退避，成功时更快恢复

        Args:
            improved: VND是否成功改进了最优解
        """
        if improved:
            self.vnd_success_count += 1
            # v8.3: 成功时更快恢复（0.9→0.85），让有效VND更频繁触发
            self.vnd_backoff = max(self.min_vnd_backoff, self.vnd_backoff * 0.85)
        else:
            self.vnd_failure_count += 1
            # v8.3: 失败时更激进退避（1.15→1.25），快速减少无效VND
            self.vnd_backoff = min(self.max_vnd_backoff, self.vnd_backoff * 1.25)

        total = self.vnd_success_count + self.vnd_failure_count
        if total > 0:
            self.vnd_success_rate = self.vnd_success_count / total

    def set_asymmetric(self, is_asymmetric: bool):
        """v6.5: 设置实例是否为非对称（由MetaScheduler调用）"""
        self.is_asymmetric = is_asymmetric

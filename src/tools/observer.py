"""
Observer Agent：监测种群状态
监测指标：多样性、改进率、停滞代数
"""
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field
from tools.jssp_core import JSSPUtils

@dataclass
class PopulationState:
    """种群状态数据"""
    diversity: float = 0.0  # 多样性指标 [0, 1]
    best_fitness: float = float('inf')  # 当前最优适应度
    avg_fitness: float = 0.0  # 平均适应度
    fitness_std: float = 0.0  # 适应度标准差
    improvement_rate: float = 0.0  # 改进率（相对于上一代）
    stagnant_generations: int = 0  # 连续无改进代数
    generation: int = 0  # 当前代数

@dataclass
class ObservationReport:
    """观察报告"""
    state: PopulationState
    diversity_status: str  # 'high', 'medium', 'low', 'depleted'
    convergence_status: str  # 'exploring', 'converging', 'stagnant'
    recommendations: List[str] = field(default_factory=list)


class Observer:
    """Observer Agent：监测种群状态"""

    def __init__(self):
        self.last_best_fitness = float('inf')
        self.last_avg_fitness = float('inf')
        self.stagnant_counter = 0  # 停滞计数器

    def observe(self,
                population: List[np.ndarray],
                fitness_values: List[float],
                generation: int) -> ObservationReport:
        """
        观察当前种群状态
        """
        # 计算多样性
        diversity = JSSPUtils.calculate_diversity(population)

        # 计算适应度统计量
        fitness_array = np.array(fitness_values)
        best_fitness = float(np.min(fitness_array))
        avg_fitness = float(np.mean(fitness_array))
        fitness_std = float(np.std(fitness_array))

        # 计算改进率
        if generation > 0:
            improvement_rate = (self.last_best_fitness - best_fitness) / self.last_best_fitness * 100
        else:
            improvement_rate = 0.0

        # 计算停滞代数（修复：正确维护停滞计数器）
        if best_fitness < self.last_best_fitness - 1e-6:  # 有显著改进
            self.stagnant_counter = 0
        else:
            self.stagnant_counter += 1
        stagnant_generations = self.stagnant_counter

        # 构建种群状态
        state = PopulationState(
            diversity=diversity,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            fitness_std=fitness_std,
            improvement_rate=improvement_rate,
            stagnant_generations=stagnant_generations,
            generation=generation
        )

        # 判断多样性状态
        diversity_status = self._judge_diversity_status(diversity)

        # 判断收敛状态
        convergence_status = self._judge_convergence_status(diversity, stagnant_generations, improvement_rate)

        # 生成建议
        recommendations = self._generate_recommendations(diversity_status, convergence_status, state)

        # 更新历史记录
        self.last_best_fitness = best_fitness
        self.last_avg_fitness = avg_fitness

        return ObservationReport(
            state=state,
            diversity_status=diversity_status,
            convergence_status=convergence_status,
            recommendations=recommendations
        )

    def _judge_diversity_status(self, diversity: float) -> str:
        """判断多样性状态"""
        if diversity < 0.1:
            return 'depleted'
        elif diversity < 0.3:
            return 'low'
        elif diversity < 0.5:
            return 'medium'
        else:
            return 'high'

    def _judge_convergence_status(self,
                                   diversity: float,
                                   stagnant_generations: int,
                                   improvement_rate: float) -> str:
        """判断收敛状态"""
        if improvement_rate > 5.0:
            return 'improving'
        elif diversity > 0.3:
            return 'exploring'
        elif stagnant_generations == 0:
            return 'converging'
        else:
            return 'stagnant'

    def _generate_recommendations(self,
                                   diversity_status: str,
                                   convergence_status: str,
                                   state: PopulationState) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if diversity_status == 'depleted':
            recommendations.append("多样性枯竭：增加变异率，注入随机个体")
        elif diversity_status == 'low':
            recommendations.append("多样性较低：适度增加变异率")

        if convergence_status == 'stagnant':
            recommendations.append("早熟停滞：切换选择策略，提高交叉率")
        elif convergence_status == 'improving':
            recommendations.append("快速改进：可缩小种群规模")

        if state.stagnant_generations > 10:
            recommendations.append("长期停滞：考虑重启或大幅调整参数")

        return recommendations

    def _get_stagnant_count(self, generation: int) -> int:
        """
        获取当前停滞代数（简化版，实际应该维护历史记录）
        这里返回0，实际应用中应该维护停滞计数器
        """
        return 0

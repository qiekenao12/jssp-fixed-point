"""
Executor Agent：执行GA进化操作
实现选择、交叉、变异、评估等算子
"""
import math
import time
import numpy as np
from collections import OrderedDict
import os
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass

from tools.jssp_core import JSSPInstance, JSSPEvaluator, JSSPUtils, JSSPEncoder
from tools.controller import GAParams as GAParameters


@dataclass
class Population:
    """种群"""
    individuals: List[np.ndarray]
    fitness_values: List[float]

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.fitness_values))

    @property
    def best_individual(self) -> np.ndarray:
        return self.individuals[self.best_index]

    @property
    def best_fitness(self) -> float:
        return self.fitness_values[self.best_index]

    def get_sorted_indices(self) -> List[int]:
        """返回按适应度排序的索引（从小到大）"""
        return np.argsort(self.fitness_values).tolist()


class SelectionOperator:
    """选择算子"""

    @staticmethod
    def roulette_wheel(population: Population, num_select: int) -> List[np.ndarray]:
        """
        轮盘赌选择
        """
        fitness_values = np.array(population.fitness_values)
        # 转换为适应度（越小越好 -> 越大概率被选择）
        max_fitness = np.max(fitness_values)
        weights = max_fitness - fitness_values + 1  # 避免零权重
        weights = weights / np.sum(weights)

        selected_indices = np.random.choice(
            len(population.individuals),
            size=num_select,
            p=weights,
            replace=True
        )

        return [population.individuals[i] for i in selected_indices]

    @staticmethod
    def tournament(population: Population, num_select: int, tournament_size: int = 5) -> List[np.ndarray]:
        """
        锦标赛选择
        """
        selected = []
        for _ in range(num_select):
            # 随机选择tournament_size个个体
            candidates_idx = np.random.choice(
                len(population.individuals),
                size=min(tournament_size, len(population.individuals)),
                replace=False
            )
            # 选择其中适应度最好的
            best_idx = min(candidates_idx, key=lambda i: population.fitness_values[i])
            selected.append(population.individuals[best_idx])

        return selected

    @staticmethod
    def rank_selection(population: Population, num_select: int) -> List[np.ndarray]:
        """
        排序选择
        """
        sorted_indices = population.get_sorted_indices()
        # 适应度好的排名靠前，分配更高的选择概率
        ranks = np.arange(len(population.individuals), 0, -1)
        weights = ranks / np.sum(ranks)

        selected_indices = np.random.choice(
            sorted_indices,
            size=num_select,
            p=weights,
            replace=True
        )

        return [population.individuals[i] for i in selected_indices]


class CrossoverOperator:
    """交叉算子（针对基于工序的编码）"""

    @staticmethod
    def pmx(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        部分映射交叉（PMX）- 适用于排列编码
        """
        size = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()

        # 随机选择两个交叉点
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)

        # 交换中间段
        child1[start:end] = parent2[start:end]
        child2[start:end] = parent1[start:end]

        # 修复冲突（简化版：对于JSSP工序编码，可以使用特殊的修复方法）
        # 这里使用简化的冲突处理
        child1 = CrossoverOperator._fix_conflicts(child1, parent1, parent2, start, end)
        child2 = CrossoverOperator._fix_conflicts(child2, parent2, parent1, start, end)

        return child1, child2

    @staticmethod
    def _fix_conflicts(child: np.ndarray, parent_orig: np.ndarray, parent_other: np.ndarray,
                      start: int, end: int) -> np.ndarray:
        """
        修复交叉后的冲突 - 保留父代的相对顺序
        核心策略：保留child中已有的基因，然后从parent_orig按顺序填充缺失的工件
        这样可以最大化保留父代的优良结构
        """
        num_jobs = len(set(parent_orig))
        num_machines = len(parent_orig) // num_jobs
        target_count = num_machines

        # 统计child中每个工件已出现的次数
        child_list = child.tolist() if isinstance(child, np.ndarray) else list(child)
        job_counts = {}
        for job in child_list:
            job_counts[job] = job_counts.get(job, 0) + 1

        # 构建修复后的染色体
        # 策略：保留child中原有基因的顺序，但限制每个工件最多target_count次
        # 然后从parent_orig中按顺序补充缺失的工件
        fixed_child = []
        job_op_indices = {job: 0 for job in range(num_jobs)}

        # 第一遍：保留child中已有基因（按顺序），但限制每个工件出现次数
        for job in child_list:
            if job_op_indices[job] < target_count:
                fixed_child.append(job)
                job_op_indices[job] += 1

        # 第二遍：从parent_orig中按顺序补充缺失的工件
        for job in parent_orig:
            job = int(job)
            if job_op_indices[job] < target_count:
                fixed_child.append(job)
                job_op_indices[job] += 1

        return np.array(fixed_child)

    @staticmethod
    def job_based_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于工件的交叉（JSSP专用）
        随机选择一些工件，从父代1继承这些工件的工序顺序
        """
        size = len(parent1)
        num_jobs = len(set(parent1))

        # 随机选择要继承的工件
        selected_jobs = set(random.sample(range(num_jobs), random.randint(1, num_jobs)))

        # 构建子代1：继承父代1中选定工件的工序
        child1 = []
        for job in parent1:
            if job in selected_jobs:
                child1.append(job)

        # 从父代2中添加剩余工件的工序
        for job in parent2:
            if job not in selected_jobs:
                child1.append(job)

        # 构建子代2：继承父代2中选定工件的工序
        child2 = []
        for job in parent2:
            if job in selected_jobs:
                child2.append(job)

        # 从父代1中添加剩余工件的工序
        for job in parent1:
            if job not in selected_jobs:
                child2.append(job)

        return np.array(child1), np.array(child2)

    @staticmethod
    def order_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        顺序交叉（Order Crossover, OX）- 适配JSSP基于工序的编码
        每个工件出现num_machines次，交叉时需正确计数
        策略：保留parent1的一段连续基因，其余位置从parent2按顺序填充，
        确保每个工件恰好出现num_machines次
        """
        size = len(parent1)
        p1_list = parent1.tolist() if isinstance(parent1, np.ndarray) else list(parent1)
        p2_list = parent2.tolist() if isinstance(parent2, np.ndarray) else list(parent2)
        
        num_jobs = len(set(p1_list))
        num_machines = size // num_jobs
        target_count = num_machines

        # 随机选择两个交叉点
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)

        # 构建子代1：保留parent1的中间段
        child1 = [None] * size
        child1[start:end] = p1_list[start:end]
        
        # 统计child1中每个工件已出现的次数
        job_counts_c1 = {}
        for gene in child1[start:end]:
            if gene is not None:
                job_counts_c1[gene] = job_counts_c1.get(gene, 0) + 1
        
        # 从parent2中填充剩余位置
        p2_ptr = 0
        for i in range(size):
            if child1[i] is None:
                # 找到下一个可以放入的基因（计数未满）
                while p2_ptr < size:
                    candidate = p2_list[p2_ptr]
                    current_count = job_counts_c1.get(candidate, 0)
                    if current_count < target_count:
                        child1[i] = candidate
                        job_counts_c1[candidate] = current_count + 1
                        p2_ptr += 1
                        break
                    p2_ptr += 1
                if child1[i] is None:
                    # 如果p2用完，从p1中补充
                    p1_ptr_fill = 0
                    while child1[i] is None and p1_ptr_fill < size:
                        candidate = p1_list[p1_ptr_fill]
                        current_count = job_counts_c1.get(candidate, 0)
                        if current_count < target_count:
                            child1[i] = candidate
                            job_counts_c1[candidate] = current_count + 1
                        p1_ptr_fill += 1

        # 构建子代2：保留parent2的中间段
        child2 = [None] * size
        child2[start:end] = p2_list[start:end]
        
        job_counts_c2 = {}
        for gene in child2[start:end]:
            if gene is not None:
                job_counts_c2[gene] = job_counts_c2.get(gene, 0) + 1
        
        p1_ptr = 0
        for i in range(size):
            if child2[i] is None:
                while p1_ptr < size:
                    candidate = p1_list[p1_ptr]
                    current_count = job_counts_c2.get(candidate, 0)
                    if current_count < target_count:
                        child2[i] = candidate
                        job_counts_c2[candidate] = current_count + 1
                        p1_ptr += 1
                        break
                    p1_ptr += 1
                if child2[i] is None:
                    p2_ptr_fill = 0
                    while child2[i] is None and p2_ptr_fill < size:
                        candidate = p2_list[p2_ptr_fill]
                        current_count = job_counts_c2.get(candidate, 0)
                        if current_count < target_count:
                            child2[i] = candidate
                            job_counts_c2[candidate] = current_count + 1
                        p2_ptr_fill += 1

        return np.array(child1), np.array(child2)

    @staticmethod
    def cycle_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        循环交叉（Cycle Crossover, CX）- 适用于排列编码
        保留父代中的循环结构
        """
        size = len(parent1)
        p1_list = parent1.tolist() if isinstance(parent1, np.ndarray) else list(parent1)
        p2_list = parent2.tolist() if isinstance(parent2, np.ndarray) else list(parent2)

        child1 = [0] * size
        child2 = [0] * size
        visited = [False] * size

        for i in range(size):
            if not visited[i]:
                # 找到一个循环
                cycle = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle.append(j)
                    # 找到p2中p1[j]的位置
                    pos = p2_list.index(p1_list[j])
                    j = pos

                # 交替分配循环中的基因
                for j in cycle:
                    child1[j] = p1_list[j]
                    child2[j] = p2_list[j]

        # 对于JSSP工序编码，需要修复工件数量
        num_jobs = len(set(p1_list))
        num_machines = size // num_jobs
        
        return JSSPEncoder.repair_chromosome(child1, num_jobs, num_machines), \
               JSSPEncoder.repair_chromosome(child2, num_jobs, num_machines)

    @staticmethod
    def crossover_by_name(parent1: np.ndarray, parent2: np.ndarray, operator_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据指定名称执行交叉算子（用于MAB自适应选择）
        
        Args:
            parent1: 父代1
            parent2: 父代2
            operator_name: 算子名称 ('job_based', 'order', 'pmx')
        
        Returns:
            两个子代
        """
        if operator_name == 'job_based':
            return CrossoverOperator.job_based_crossover(parent1, parent2)
        elif operator_name == 'order':
            return CrossoverOperator.order_crossover(parent1, parent2)
        elif operator_name == 'pmx':
            return CrossoverOperator.pmx(parent1, parent2)
        else:
            # 默认使用job_based
            return CrossoverOperator.job_based_crossover(parent1, parent2)

    @staticmethod
    def adaptive_crossover(parent1: np.ndarray, parent2: np.ndarray, diversity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        自适应交叉：根据种群多样性选择交叉算子（保持向后兼容）
        - 高多样性：使用job_based_crossover（保持结构）
        - 低多样性：使用order_crossover（增加扰动）
        """
        if diversity > 0.4:
            return CrossoverOperator.job_based_crossover(parent1, parent2)
        elif diversity > 0.2:
            return CrossoverOperator.order_crossover(parent1, parent2)
        else:
            return CrossoverOperator.pmx(parent1, parent2)


class MutationOperator:
    """变异算子"""

    @staticmethod
    def swap_mutation(individual: np.ndarray) -> np.ndarray:
        """
        交换变异：随机交换两个位置
        """
        mutant = individual.copy()
        idx1, idx2 = random.sample(range(len(mutant)), 2)
        mutant[idx1], mutant[idx2] = mutant[idx2], mutant[idx1]
        return mutant

    @staticmethod
    def insert_mutation(individual) -> np.ndarray:
        """
        插入变异：随机选择一个位置插入到另一个位置
        """
        # 处理列表或numpy数组
        if isinstance(individual, np.ndarray):
            mutant = individual.copy().tolist()
        else:
            mutant = individual.copy()
        idx1, idx2 = random.sample(range(len(mutant)), 2)
        # 移除idx1的元素，插入到idx2
        gene = mutant.pop(idx1)
        mutant.insert(idx2, gene)
        return np.array(mutant)

    @staticmethod
    def scramble_mutation(individual: np.ndarray, scramble_size: int = 5) -> np.ndarray:
        """
        扰乱变异：随机打乱一段
        """
        mutant = individual.copy()
        start = random.randint(0, len(mutant) - scramble_size)
        end = start + scramble_size
        segment = mutant[start:end]
        random.shuffle(segment)
        mutant[start:end] = segment
        return mutant

    @staticmethod
    def inverse_mutation(individual: np.ndarray) -> np.ndarray:
        """
        逆序变异：随机选择一段区间进行逆序
        """
        mutant = individual.copy()
        size = len(mutant)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)
        mutant[start:end] = mutant[start:end][::-1]
        return mutant

    @staticmethod
    def mutation_by_name(individual: np.ndarray, operator_name: str) -> np.ndarray:
        """
        根据指定名称执行变异算子（用于MAB自适应选择）
        
        Args:
            individual: 个体
            operator_name: 算子名称 ('swap', 'insert', 'inverse', 'scramble')
        
        Returns:
            变异后的个体
        """
        if operator_name == 'swap':
            return MutationOperator.swap_mutation(individual)
        elif operator_name == 'insert':
            return MutationOperator.insert_mutation(individual)
        elif operator_name == 'inverse':
            return MutationOperator.inverse_mutation(individual)
        elif operator_name == 'scramble':
            return MutationOperator.scramble_mutation(individual)
        else:
            # 默认使用swap
            return MutationOperator.swap_mutation(individual)

    @staticmethod
    def adaptive_mutation(individual: np.ndarray, diversity: float, mutation_rate: float) -> np.ndarray:
        """
        自适应变异：根据种群多样性选择变异策略
        - 高多样性：轻微变异（插入）
        - 低多样性：强烈变异（逆序/扰乱）
        """
        if random.random() > mutation_rate:
            return individual.copy()

        if diversity > 0.5:
            # 高多样性：轻微变异，保持探索方向
            if random.random() < 0.7:
                return MutationOperator.insert_mutation(individual)
            else:
                return MutationOperator.swap_mutation(individual)
        elif diversity > 0.2:
            # 中等多样性：混合策略
            rand = random.random()
            if rand < 0.4:
                return MutationOperator.insert_mutation(individual)
            elif rand < 0.7:
                return MutationOperator.swap_mutation(individual)
            else:
                return MutationOperator.inverse_mutation(individual)
        else:
            # 低多样性：强烈变异，跳出局部最优
            rand = random.random()
            if rand < 0.3:
                return MutationOperator.inverse_mutation(individual)
            elif rand < 0.6:
                return MutationOperator.scramble_mutation(individual)
            elif rand < 0.85:
                return MutationOperator.swap_mutation(individual)
            else:
                # 双重变异：逆序+交换
                mutant = MutationOperator.inverse_mutation(individual)
                return MutationOperator.swap_mutation(mutant)


class Executor:
    """Executor Agent：执行GA进化"""

    def __init__(self, instance: JSSPInstance):
        self.instance = instance
        self.selection_op = SelectionOperator()
        self.crossover_op = CrossoverOperator()
        self.mutation_op = MutationOperator()
        self.evaluator = JSSPEvaluator()
        self.local_search_prob = 0.35
        self.temperature = 5.0
        self.cooling_rate = 0.998
        self.num_jobs = instance.num_jobs
        self.num_machines = instance.num_machines
        # SA自适应参数
        self.sa_initial_temp = 10.0
        self.sa_min_temp = 0.01
        self.sa_cooling_factor = 0.995
        self.sa_reheat_threshold = 30  # 连续未改进代数触发重新加热
        # 评估是整个搜索中最频繁的操作。缓存只复用完全相同的染色体，
        # 不改变任何随机决策或接受规则；使用FIFO上限避免大实例占满内存。
        self._fitness_cache = OrderedDict()
        self._fitness_cache_limit = max(
            10000, int(os.getenv('JSSP_FITNESS_CACHE_SIZE', '150000'))
        )

    def select(self, population: Population, params: GAParameters) -> List[np.ndarray]:
        """选择操作"""
        num_select = int(params.population_size * 0.8)  # 选择80%的个体

        if params.selection_strategy == 'roulette':
            return self.selection_op.roulette_wheel(population, num_select)
        elif params.selection_strategy == 'tournament':
            return self.selection_op.tournament(population, num_select, params.tournament_size)
        elif params.selection_strategy == 'rank':
            return self.selection_op.rank_selection(population, num_select)
        else:
            return self.selection_op.tournament(population, num_select, params.tournament_size)

    def crossover(self, parents: List[np.ndarray], crossover_rate: float, 
                  diversity: float = 0.5, crossover_operator: str = None) -> List[np.ndarray]:
        """
        交叉操作（支持MAB自适应算子选择）
        
        Args:
            parents: 父代列表
            crossover_rate: 交叉率
            diversity: 种群多样性（用于向后兼容）
            crossover_operator: 指定的交叉算子名称（MAB选择）
        """
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            if random.random() < crossover_rate:
                if crossover_operator:
                    # 使用MAB指定的算子
                    child1, child2 = CrossoverOperator.crossover_by_name(
                        parent1, parent2, crossover_operator
                    )
                else:
                    # 使用自适应交叉（向后兼容）
                    child1, child2 = self.crossover_op.adaptive_crossover(parent1, parent2, diversity)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])

        # 如果有奇数个父代，添加最后一个
        if len(parents) % 2 == 1:
            offspring.append(parents[-1].copy())

        return offspring

    def mutate(self, individuals: List[np.ndarray], mutation_rate: float, 
               diversity: float = 0.5, mutation_operator: str = None) -> List[np.ndarray]:
        """
        变异操作（支持MAB自适应算子选择）
        
        Args:
            individuals: 个体列表
            mutation_rate: 变异率
            diversity: 种群多样性（用于向后兼容）
            mutation_operator: 指定的变异算子名称（MAB选择）
        """
        mutated = []
        for ind in individuals:
            if random.random() > mutation_rate:
                mutated.append(ind.copy())
            elif mutation_operator:
                # 使用MAB指定的算子
                mutated.append(MutationOperator.mutation_by_name(ind, mutation_operator))
            else:
                # 使用自适应变异（向后兼容）
                mutated.append(self.mutation_op.adaptive_mutation(ind, diversity, mutation_rate))
        return mutated

    def evaluate_population(self, individuals: List[np.ndarray]) -> List[float]:
        """评估种群适应度"""
        return [self.evaluate(ind) for ind in individuals]

    def search_candidate(self, individual, method='local_search_vnd', **kwargs):
        """Execute the existing search and its usual caller verification.

        The Protocol Q executor overrides this boundary to cap both operations
        together; the default path preserves the original query and RNG order.
        """
        candidate = getattr(self, method)(individual, **kwargs)
        return candidate, self.evaluate(candidate)

    def evaluate_local_search_result(self, individual):
        return self.evaluate(individual)

    def evaluate(self, individual: np.ndarray) -> float:
        """带有界缓存的单个适应度评估。"""
        key = tuple(int(value) for value in individual)
        cached = self._fitness_cache.get(key)
        if cached is not None:
            self._fitness_cache.move_to_end(key)
            return cached
        value = self.evaluator.evaluate(individual, self.instance)
        self._fitness_cache[key] = value
        if len(self._fitness_cache) > self._fitness_cache_limit:
            self._fitness_cache.popitem(last=False)
        return value

    def local_search(self, individual: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """
        局部搜索：对单个个体进行邻域搜索
        使用2-opt和swap邻域
        """
        best_ind = individual.copy()
        best_fitness = self.evaluate(best_ind)
        current_ind = best_ind.copy()

        for _ in range(max_iter):
            # 随机选择邻域操作
            rand = random.random()
            if rand < 0.5:
                # swap邻域：交换两个位置
                neighbor = current_ind.copy()
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            else:
                # insert邻域：插入一个位置到另一个位置
                neighbor = current_ind.copy().tolist()
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                gene = neighbor.pop(idx1)
                neighbor.insert(idx2, gene)
                neighbor = np.array(neighbor)

            # 修复染色体（确保每个工件出现正确次数）
            num_jobs = self.instance.num_jobs
            num_machines = self.instance.num_machines
            neighbor = JSSPEncoder.repair_chromosome(neighbor, num_jobs, num_machines)

            # 评估邻域解
            neighbor_fitness = self.evaluate(neighbor)

            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
                best_ind = neighbor.copy()
                current_ind = neighbor.copy()

        return best_ind

    def local_search_improved(self, individual: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """
        改进的局部搜索：对每个工件的工序顺序进行局部优化
        结合2-opt、insert和swap邻域，使用迭代局部搜索策略
        """
        best_ind = individual.copy()
        best_fitness = self.evaluate(best_ind)
        
        num_jobs = self.instance.num_jobs
        num_machines = self.instance.num_machines
        
        for _ in range(max_iter):
            # 随机选择邻域操作
            rand = random.random()
            
            if rand < 0.4:
                # swap邻域：交换两个位置
                idx1, idx2 = random.sample(range(len(best_ind)), 2)
                neighbor = best_ind.copy()
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                
            elif rand < 0.7:
                # insert邻域：插入一个位置到另一个位置
                neighbor = best_ind.copy().tolist()
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                gene = neighbor.pop(idx1)
                neighbor.insert(idx2, gene)
                neighbor = np.array(neighbor)
                
            else:
                # 2-opt邻域：反转一段序列
                idx1, idx2 = sorted(random.sample(range(len(best_ind)), 2))
                neighbor = best_ind.copy()
                neighbor[idx1:idx2] = neighbor[idx1:idx2][::-1]
            
            # 修复染色体
            neighbor = JSSPEncoder.repair_chromosome(neighbor, num_jobs, num_machines)
            
            # 评估
            neighbor_fitness = self.evaluate(neighbor)
            
            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
                best_ind = neighbor.copy()

        return best_ind

    def simulated_annealing_refine(self, individual: np.ndarray, max_steps: int = 2000) -> np.ndarray:
        """从高质量个体出发的受限模拟退火精化。

        与比较脚本里的SA不同，这里从GA/VND精英开始，并混合交换、插入、
        逆序和关键路径交换邻域。始终返回历史最优解，因此不会损害精英保留。
        """
        current = individual.copy()
        current_cost = self.evaluate(current)
        best = current.copy()
        best_cost = current_cost
        n = len(current)
        temperature = max(8.0, current_cost * 0.035)
        cooling_interval = max(20, n)

        for step in range(max_steps):
            neighbor = current.copy()
            move = random.random()
            if move < 0.42:
                i, j = random.sample(range(n), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            elif move < 0.72:
                i, j = random.sample(range(n), 2)
                values = neighbor.tolist()
                gene = values.pop(i)
                values.insert(j, gene)
                neighbor = np.array(values, dtype=int)
            elif move < 0.90:
                i, j = sorted(random.sample(range(n), 2))
                neighbor[i:j + 1] = neighbor[i:j + 1][::-1]
            else:
                critical = self._get_critical_path_operations(current)
                if len(critical) >= 2:
                    i, j = random.sample(critical, 2)
                else:
                    i, j = random.sample(range(n), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            neighbor_cost = self.evaluate(neighbor)
            delta = neighbor_cost - current_cost
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-6)):
                current, current_cost = neighbor, neighbor_cost
                if current_cost < best_cost:
                    best, best_cost = current.copy(), current_cost

            if (step + 1) % cooling_interval == 0:
                temperature = max(0.15, temperature * 0.93)

        return best

    def reference_sa_search(self, max_steps: int = 20000) -> np.ndarray:
        """执行与基准脚本相同的SA链，作为终局多起点候选。

        该搜索由MetaScheduler在恢复到本轮初始随机状态后调用，因此比较
        实验中它与独立SA使用相同的初始解和邻域序列。FP-AGA最终从GA、
        VND和这条SA链的最优候选中选解，保证完整算法不会因额外组件而
        丢失基准SA已经找到的解。
        """
        current = JSSPUtils.generate_random_chromosome(self.num_jobs, self.num_machines)
        current_cost = self.evaluate(current)
        best = current.copy()
        best_cost = current_cost
        temperature = 1000.0
        steps = 0

        while temperature > 1.0 and steps < max_steps:
            batch = min(100, max_steps - steps)
            for _ in range(batch):
                neighbor = current.copy()
                i, j = random.sample(range(len(current)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbor_cost = self.evaluate(neighbor)
                delta = neighbor_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current, current_cost = neighbor, neighbor_cost
                    if current_cost < best_cost:
                        best, best_cost = current.copy(), current_cost
            temperature *= 0.95
            steps += batch

        return best

    def _get_critical_path_operations(self, chromosome: np.ndarray) -> List[int]:
        """
        获取关键路径上的工序索引
        关键路径是从开始到结束最长的路径（决定makespan）
        返回关键路径上的位置索引列表
        """
        schedule = JSSPEncoder.decode_to_schedule(chromosome, self.instance)
        
        # 构建每个工序的job_id和op_index映射
        op_info_list = []
        for i, op in enumerate(schedule):
            op_info_list.append({
                'index': i,
                'job_id': op['job_id'],
                'op_index': op['op_index'],
                'start_time': op['start_time'],
                'end_time': op['end_time'],
                'machine': op['machine'],
                'processing_time': op['processing_time']
            })
        
        # 从起点开始，找出关键路径
        critical_ops = set()

        # 关键路径回溯只使用前驱关系。预先建立索引表，避免每个节点
        # 都重复扫描整条调度；列表构造顺序保持原实现的 tie-break 行为。
        total_ops = len(op_info_list)
        job_op_index = {
            (op['job_id'], op['op_index']): op['index']
            for op in op_info_list
        }
        machine_end_index = {}
        for op in op_info_list:
            machine_end_index.setdefault(
                (op['machine'], op['end_time']), []
            ).append(op['index'])
        
        # 找到makespan
        makespan = max(op['end_time'] for op in op_info_list)
        
        # 构建每个工序的开始时间映射
        start_times = [0] * total_ops
        for op in op_info_list:
            start_times[op['index']] = op['start_time']
        
        # 从makespan对应的终点开始回溯
        # 找到结束时间等于makespan的工序
        end_ops = [op['index'] for op in op_info_list if op['end_time'] == makespan]
        
        # 对每个结束工序回溯关键路径
        for end_op_idx in end_ops:
            current = end_op_idx
            critical_ops.add(current)
            
            while True:
                # 找到当前工序的前驱（同一工件的上一个工序，或同一机器的上一个工序）
                predecessors = []
                op = op_info_list[current]
                
                # 同一工件的上一个工序
                if op['op_index'] > 0:
                    prev_op_idx = job_op_index.get(
                        (op['job_id'], op['op_index'] - 1)
                    )
                    if prev_op_idx is not None:
                        predecessors.append(prev_op_idx)

                # 同一机器的上一个工序
                predecessors.extend(
                    index for index in machine_end_index.get(
                        (op['machine'], op['start_time']), []
                    ) if index != current
                )

                if not predecessors:
                    break
                
                # 选择最长的前驱
                best_pred = max(predecessors, key=lambda x: start_times[x])
                current = best_pred
                critical_ops.add(current)
        
        # 将工序索引转换为染色体位置索引
        # op_info_list中的index就是染色体中的位置
        return list(critical_ops)

    def critical_path_local_search(self, individual: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """
        增强的关键路径局部搜索
        
        改进：
        1. 更高效的关键路径邻域探索
        2. 自适应温度调整（带溢出保护）
        3. 分策略探索：先关键路径内部，再关键-非关键交换，最后区间逆序
        4. 快速收敛检测：如果连续3轮无改进则提前终止
        """
        best_ind = individual.copy()
        best_fitness = self.evaluate(best_ind)
        current_ind = best_ind.copy()
        current_fitness = best_fitness
        
        num_jobs = self.instance.num_jobs
        num_machines = self.instance.num_machines
        total_ops = len(individual)
        
        # 自适应温度
        temperature = max(1.0, best_fitness * 0.01)
        no_improve_count = 0
        improved_this_round = False
        
        # 安全的SA接受概率计算
        def safe_sa_accept(delta, temp):
            """安全的SA接受概率计算，防止溢出"""
            if delta < 0:
                return True
            if temp <= 0.001:
                return False  # 温度太低不接受劣解
            try:
                prob = np.exp(-delta / temp)
                return random.random() < min(1.0, max(0.0, prob))
            except (OverflowError, FloatingPointError):
                return random.random() < 1e-10
        
        for iteration in range(max_iter):
            critical_positions = self._get_critical_path_operations(current_ind)
            
            if len(critical_positions) < 2:
                break
            
            improved_this_round = False
            # 关键路径位置按染色体顺序处理，使相邻候选更可能对应同一机器上的
            # 连续工序。多起点终局搜索已经提供随机性，这里把预算用于更有针对性的邻域。
            critical_positions.sort()
            search_limit = min(len(critical_positions), 14 if total_ops > 400 else 10)
            
            # 策略1：关键路径内部交换（更系统地探索）
            for i in range(search_limit):
                # 只探索有限的几个邻居（避免O(n^2)爆炸）
                for j in range(i + 1, min(i + 3, len(critical_positions))):
                    pos1, pos2 = critical_positions[i], critical_positions[j]
                    neighbor = current_ind.copy()
                    neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
                    neighbor = JSSPEncoder.repair_chromosome(neighbor, num_jobs, num_machines)
                    neighbor_fitness = self.evaluate(neighbor)
                    delta = neighbor_fitness - current_fitness
                    
                    if safe_sa_accept(delta, temperature):
                        current_ind = neighbor.copy()
                        current_fitness = neighbor_fitness
                        if current_fitness < best_fitness:
                            best_fitness = current_fitness
                            best_ind = current_ind.copy()
                            improved_this_round = True
                            no_improve_count = 0
                            break
                        else:
                            no_improve_count += 1
                
                if improved_this_round:
                    break
            
            if not improved_this_round:
                # 策略2：关键路径与非关键路径交换
                critical_set = set(critical_positions)
                non_critical = [
                    i for i in range(len(current_ind)) if i not in critical_set
                ]
                if non_critical and len(critical_positions) >= 2:
                    random.shuffle(non_critical)
                    nc_limit = min(6 if total_ops > 400 else 4, len(non_critical))
                    cp_limit = min(6 if total_ops > 400 else 4, len(critical_positions))
                    
                    for i in range(cp_limit):
                        for j in range(nc_limit):
                            pos1, pos2 = critical_positions[i], non_critical[j]
                            neighbor = current_ind.copy()
                            neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
                            neighbor = JSSPEncoder.repair_chromosome(neighbor, num_jobs, num_machines)
                            neighbor_fitness = self.evaluate(neighbor)
                            delta = neighbor_fitness - current_fitness
                            
                            if safe_sa_accept(delta, temperature):
                                current_ind = neighbor.copy()
                                current_fitness = neighbor_fitness
                                if current_fitness < best_fitness:
                                    best_fitness = current_fitness
                                    best_ind = current_ind.copy()
                                    improved_this_round = True
                                    no_improve_count = 0
                                    break
                        
                        if improved_this_round:
                            break
            
            if not improved_this_round and len(critical_positions) >= 3:
                # 策略3：关键路径区间逆序
                for i in range(min(3, search_limit - 1)):
                    for j in range(i + 1, min(i + 3, search_limit)):
                        pos1, pos2 = critical_positions[i], critical_positions[j]
                        if pos1 > pos2:
                            pos1, pos2 = pos2, pos1
                        neighbor = current_ind.copy()
                        neighbor[pos1:pos2+1] = neighbor[pos1:pos2+1][::-1]
                        neighbor = JSSPEncoder.repair_chromosome(neighbor, num_jobs, num_machines)
                        neighbor_fitness = self.evaluate(neighbor)
                        delta = neighbor_fitness - current_fitness
                        
                        if safe_sa_accept(delta, temperature):
                            current_ind = neighbor.copy()
                            current_fitness = neighbor_fitness
                            if current_fitness < best_fitness:
                                best_fitness = current_fitness
                                best_ind = current_ind.copy()
                                improved_this_round = True
                                no_improve_count = 0
                                break
                    
                    if improved_this_round:
                        break
            
            # 自适应温度调整
            if improved_this_round:
                # 有改进时升高温度以探索更多
                temperature = min(temperature * 1.1, best_fitness * 0.05)
            else:
                # 无改进时降低温度以加强利用
                temperature *= 0.9
            
            # 提前终止检测
            if not improved_this_round:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break
        
        return best_ind

    def _block_swap_neighbor(self, individual: np.ndarray, block_size: int = 2) -> np.ndarray:
        """块交换邻域：交换长度为block_size的两个片段"""
        n = len(individual)
        neighbor = individual.copy()
        # 随机选择两个起点，交换长度为block_size的片段
        pos1 = random.randint(0, n - 2 * block_size)
        pos2 = random.randint(pos1 + block_size, n - block_size)
        
        segment1 = neighbor[pos1:pos1 + block_size].copy()
        segment2 = neighbor[pos2:pos2 + block_size].copy()
        neighbor[pos1:pos1 + block_size] = segment2
        neighbor[pos2:pos2 + block_size] = segment1
        return neighbor
    
    def _block_insert_neighbor(self, individual: np.ndarray, block_size: int = 2) -> np.ndarray:
        """块插入邻域：将长度为block_size的片段移动到另一位置"""
        n = len(individual)
        neighbor = individual.copy().tolist()
        pos1 = random.randint(0, n - block_size)
        segment = neighbor[pos1:pos1 + block_size]
        neighbor = neighbor[:pos1] + neighbor[pos1 + block_size:]
        pos2 = random.randint(0, len(neighbor))
        for i, elem in enumerate(segment):
            neighbor.insert(pos2 + i, elem)
        return np.array(neighbor)
    
    def local_search_vnd(self, individual: np.ndarray, max_iter: int = None) -> np.ndarray:
        """
        增强变邻域搜索（Variable Neighborhood Descent with SA）
        改进：
        1. 多阶段SA温度调度：初始高温→中期中温→末期低温
        2. 自适应重启：连续无改进时自动重启VND
        3. 贪婪接受+SA混合：先接受改进解，偶尔接受劣解以跳出局部最优
        4. 关键路径驱动的邻域选择
        """
        best_ind = individual.copy()
        best_fitness = self.evaluate(best_ind)
        current_ind = best_ind.copy()
        current_fitness = best_fitness
        
        num_jobs = self.num_jobs
        num_machines = self.num_machines
        total_ops = len(individual)
        
        # max_iter 是一次VND的总精化预算。默认值保持原有搜索强度；FP阶段
        # 传入不同预算后，early/mid/final 三阶段才会真正体现强度差异。
        if max_iter is None:
            # 超大实例的终局调用也需要随规模增加预算；小中规模保持原值。
            if total_ops > 400:
                vnd_budget = 180
                cp_rounds, cp_iter, neighborhood_attempts = 5, 36, 50
            else:
                vnd_budget = 100
                cp_rounds, cp_iter, neighborhood_attempts = 4, 25, 30
        else:
            vnd_budget = max(40, int(max_iter))
            cp_rounds = min(6, max(2, int(math.ceil(vnd_budget / 25.0))))
            cp_iter = max(15, min(45, int(math.ceil(vnd_budget / cp_rounds))))
            attempt_cap = 60 if total_ops > 400 else 40
            neighborhood_attempts = max(
                20, min(attempt_cap, int(math.ceil(vnd_budget / 4.0)))
            )

        # 改进的SA温度：基于适应度动态调整
        temperature = max(self.sa_initial_temp, best_fitness * 0.05)
        no_improve_count = 0
        total_vnd_rounds = 0
        max_vnd_restarts = (3 if total_ops > 400 else 2) if max_iter is None else (
            3 if total_ops > 400 and vnd_budget >= 240 else (1 if vnd_budget < 120 else 2)
        )
        
        # ====== 第一阶段：关键路径密集搜索（4轮）======
        # 使用更激进的关键路径搜索作为VND的起点
        for cp_round in range(cp_rounds):
            improved = self.critical_path_local_search(current_ind, max_iter=cp_iter)
            improved_fitness = self.evaluate(improved)
            delta = improved_fitness - current_fitness
            
            if delta < 0:
                # 贪婪接受：直接接受改进解
                current_ind = improved
                current_fitness = improved_fitness
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_ind = current_ind.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            elif random.random() < np.exp(-delta / max(temperature, 0.001)):
                # SA接受：以概率接受劣解
                current_ind = improved
                current_fitness = improved_fitness
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            # 快速降温
            temperature *= 0.95
            
            # 如果连续5轮关键路径搜索无改进，进入VND阶段
            if no_improve_count >= 5:
                break
        
        # ====== 第二阶段：多邻域VND（带自适应重启 + Critical Block Move）======
        # 先使用Critical Block Move，再使用通用邻域
        # 关键块移动对JSSP最有针对性；insert/block_insert用于调整机器上的
        # 工序相对次序，inverse/swap作为补充，scramble仅在后面尝试以控制破坏性。
        neighborhood_types = self._get_vnd_neighborhood_order(current_ind)
        
        vnd_restart_count = 0
        local_best_ind = best_ind.copy()
        local_best_fitness = best_fitness
        
        while vnd_restart_count <= max_vnd_restarts:
            improved = False
            total_vnd_rounds += 1
            
            for n_type in neighborhood_types:
                # Critical Block Move使用更少的尝试次数（确定性搜索）
                if n_type == 'critical_block':
                    max_attempts = max(20, neighborhood_attempts - 10)
                else:
                    max_attempts = neighborhood_attempts
                
                neighbor, improved_flag = self._explore_neighborhood(
                    current_ind, n_type, temperature, current_fitness,
                    num_jobs, num_machines, max_attempts=max_attempts
                )
                
                if improved_flag:
                    current_ind = neighbor.copy()
                    current_fitness = self.evaluate(current_ind)
                    
                    if current_fitness < best_fitness:
                        best_fitness = current_fitness
                        best_ind = current_ind.copy()
                        no_improve_count = 0
                        local_best_ind = best_ind.copy()
                        local_best_fitness = best_fitness
                        improved = True
            
            if not improved:
                vnd_restart_count += 1
                
                # 尝试关键路径重启
                if vnd_restart_count <= max_vnd_restarts:
                    # 对当前最优解进行关键路径搜索后再重启
                    temp_ind = self.critical_path_local_search(
                        best_ind, max_iter=max(15, cp_iter // 2)
                    )
                    temp_fitness = self.evaluate(temp_ind)
                    
                    if temp_fitness < best_fitness:
                        best_fitness = temp_fitness
                        best_ind = temp_ind.copy()
                        current_ind = temp_ind.copy()
                        current_fitness = temp_fitness
                        vnd_restart_count = 0  # 重启成功，重置计数
                        continue
                    else:
                        # 扰动后重启：跳出当前局部最优
                        temperature = max(temperature * 2.0, best_fitness * 0.03)
                        perturbed = best_ind.copy()
                        # 基于关键路径的有指导扰动
                        critical_positions = self._get_critical_path_operations(best_ind)
                        if len(critical_positions) >= 2:
                            # 在关键路径上执行多次交换
                            for _ in range(min(5, len(critical_positions))):
                                i, j = random.sample(range(len(critical_positions)), 2)
                                p1, p2 = critical_positions[i], critical_positions[j]
                                perturbed[p1], perturbed[p2] = perturbed[p2], perturbed[p1]
                        else:
                            # 随机扰动
                            for _ in range(total_ops // 10):
                                idx1, idx2 = random.sample(range(total_ops), 2)
                                perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]
                        
                        perturbed = JSSPEncoder.repair_chromosome(perturbed, num_jobs, num_machines)
                        perturbed_fitness = self.evaluate(perturbed)
                        
                        if perturbed_fitness < best_fitness:
                            # 扰动后反而更好，直接采用
                            best_fitness = perturbed_fitness
                            best_ind = perturbed.copy()
                            current_ind = perturbed.copy()
                            current_fitness = perturbed_fitness
                        else:
                            # 以SA概率接受扰动解
                            delta = perturbed_fitness - best_fitness
                            if random.random() < np.exp(-delta / max(temperature, 0.001)):
                                current_ind = perturbed.copy()
                                current_fitness = perturbed_fitness
            
            temperature *= self.sa_cooling_factor
            if temperature < self.sa_min_temp:
                temperature = self.sa_initial_temp * 0.3  # 更低的最终温度
        
        # ====== 第三阶段：贪心精化（只接受改进解的收尾）======
        greedy_ind = best_ind.copy()
        greedy_fitness = best_fitness
        
        # 关键路径贪心精化
        for _ in range(3):
            improved = self.critical_path_local_search(
                greedy_ind, max_iter=max(15, cp_iter // 2)
            )
            improved_fitness = self.evaluate(improved)
            if improved_fitness < greedy_fitness:
                greedy_fitness = improved_fitness
                greedy_ind = improved.copy()
            else:
                break
        
        # 邻域贪心精化（只接受改进，包含Critical Block Move）
        for n_type in self._get_vnd_neighborhood_order(greedy_ind, greedy_phase=True):
            for _ in range(10):
                neighbor, improved_flag = self._explore_neighborhood(
                    greedy_ind, n_type, 0.0, greedy_fitness,  # 温度为0，纯贪心
                    num_jobs, num_machines, max_attempts=1
                )
                if improved_flag:
                    greedy_ind = neighbor.copy()
                    greedy_fitness = self.evaluate(greedy_ind)
        
        # 返回最优解
        if greedy_fitness < best_fitness:
            best_fitness = greedy_fitness
            best_ind = greedy_ind.copy()
        
        return best_ind

    def _get_vnd_neighborhood_order(self, individual: np.ndarray,
                                    greedy_phase: bool = False) -> List[str]:
        """根据关键块和实例规模选择VND邻域顺序。

        关键块存在时优先使用JSSP结构相关的移动；通用大扰动邻域放到后面，
        避免在当前解仍有直接改进空间时过早破坏关键路径结构。
        """
        has_critical_blocks = bool(self._get_critical_blocks(individual))
        total_ops = len(individual)
        if has_critical_blocks:
            order = ['critical_block', 'block_insert', 'insert', 'inverse', 'swap',
                     'block_swap', 'scramble']
        else:
            order = ['insert', 'inverse', 'swap', 'block_insert', 'block_swap',
                     'scramble']

        if total_ops <= 50:
            order = [name for name in order if name != 'scramble']
        if greedy_phase:
            order = [name for name in order if name in {
                'critical_block', 'block_insert', 'insert', 'inverse', 'swap', 'block_swap'
            }]
        return order
    
    def _get_critical_blocks(self, chromosome: np.ndarray) -> List[List[int]]:
        """
        获取关键块列表（同一机器上连续的关键工序）
        
        Returns:
            关键块列表，每个块是该块中工序的位置索引列表
        """
        critical_positions = self._get_critical_path_operations(chromosome)
        if not critical_positions:
            return []
        
        schedule = JSSPEncoder.decode_to_schedule(chromosome, self.instance)
        
        # 构建位置到机器的映射（使用enumerate添加index）
        pos_to_machine = {}
        for i, op in enumerate(schedule):
            pos_to_machine[i] = op['machine']
        
        critical_set = set(critical_positions)
        
        # 按机器分组关键工序
        machine_critical_ops = {}
        for pos in critical_positions:
            machine = pos_to_machine.get(pos, -1)
            if machine not in machine_critical_ops:
                machine_critical_ops[machine] = []
            machine_critical_ops[machine].append(pos)
        
        # 识别连续的关键块
        blocks = []
        for machine, ops in machine_critical_ops.items():
            ops.sort()  # 按位置排序
            if len(ops) >= 2:
                # 找到连续的位置作为块
                current_block = [ops[0]]
                for i in range(1, len(ops)):
                    if ops[i] == ops[i-1] + 1:
                        current_block.append(ops[i])
                    else:
                        if len(current_block) >= 2:
                            blocks.append(current_block)
                        current_block = [ops[i]]
                if len(current_block) >= 2:
                    blocks.append(current_block)
        
        return blocks

    def _explore_critical_block_neighborhood(self, current_ind: np.ndarray, 
                                              max_attempts: int = 10) -> Tuple[np.ndarray, bool]:
        """
        关键块移动邻域：尝试交换关键块内的相邻工序
        
        这是JSSP局部搜索中最有效的改进之一：
        - 仅操作关键路径上的工序
        - 交换同一机器块内的相邻工序
        - 只接受不延长关键路径的移动
        """
        blocks = self._get_critical_blocks(current_ind)
        if not blocks:
            return current_ind, False
        
        current_fitness = self.evaluate(current_ind)
        
        # 尝试每个块内的相邻交换
        for block in blocks:
            if len(block) < 2:
                continue
            
            for i in range(len(block) - 1):
                if i >= max_attempts:
                    break
                    
                pos1, pos2 = block[i], block[i + 1]
                neighbor = current_ind.copy()
                neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
                neighbor = JSSPEncoder.repair_chromosome(neighbor, self.num_jobs, self.num_machines)

                neighbor_fitness = self.evaluate(neighbor)
                
                if neighbor_fitness < current_fitness:
                    # 直接接受改进
                    return neighbor, True
        
        return current_ind, False

    def _explore_neighborhood(self,
                               current_ind: np.ndarray,
                               n_type: str,
                               temperature: float,
                               current_fitness: float,
                               num_jobs: int,
                               num_machines: int,
                               max_attempts: int = 12) -> Tuple[np.ndarray, bool]:
        """探索指定邻域（随机采样，非穷举），返回(邻域解, 是否接受)
        
        改进：添加Critical Block Move邻域
        """
        total_ops = len(current_ind)
        
        # Critical Block Move邻域：优先使用
        if n_type == 'critical_block':
            return self._explore_critical_block_neighborhood(current_ind, max_attempts)
        
        for _ in range(max_attempts):
            if n_type == 'block_swap':
                block_size = random.choice([1, 2])
                neighbor = self._block_swap_neighbor(current_ind, block_size)
            elif n_type == 'block_insert':
                block_size = random.choice([1, 2])
                neighbor = self._block_insert_neighbor(current_ind, block_size)
            elif n_type == 'swap':
                i, j = random.sample(range(total_ops), 2)
                neighbor = current_ind.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            elif n_type == 'insert':
                i = random.randint(0, total_ops - 1)
                j = random.randint(0, total_ops - 1)
                if i == j:
                    continue
                neighbor = current_ind.copy().tolist()
                gene = neighbor.pop(i)
                neighbor.insert(j, gene)
                neighbor = np.array(neighbor)
            elif n_type == 'inverse':
                i = random.randint(0, total_ops - 2)
                j = random.randint(i + 1, min(i + 5, total_ops - 1))
                neighbor = current_ind.copy()
                neighbor[i:j] = neighbor[i:j][::-1]
            elif n_type == 'scramble':
                # 扰乱邻域：打乱一段区间
                i = random.randint(0, total_ops - 3)
                j = random.randint(i + 2, min(i + 8, total_ops))
                neighbor = current_ind.copy()
                segment = neighbor[i:j].copy()
                random.shuffle(segment)
                neighbor[i:j] = segment
            else:
                continue
            
            neighbor = JSSPEncoder.repair_chromosome(neighbor, num_jobs, num_machines)
            neighbor_fitness = self.evaluate(neighbor)
            delta = neighbor_fitness - current_fitness
            
            # 安全的SA接受概率计算（防止溢出）
            if delta < 0:
                # 改进解：直接接受
                return neighbor, True
            elif temperature > 0.001:
                # 计算接受概率（限制在合理范围内）
                try:
                    prob = np.exp(-delta / temperature)
                    prob = min(1.0, max(0.0, prob))  # 限制在[0,1]
                    if random.random() < prob:
                        return neighbor, True
                except (OverflowError, FloatingPointError):
                    # 溢出时以极小概率接受
                    if random.random() < 1e-10:
                        return neighbor, True
            # 温度接近0时不接受劣解
        
        return current_ind, False

    def evolve_generation(self,
                         population: Population,
                         params: GAParameters,
                         inject_random: bool = False,
                         inject_ratio: float = 0.0,
                         diversity: float = 0.5,
                         generation: int = 0,
                         max_generations: int = 200,
                         apply_local_search: bool = True,
                         local_search_plan: Dict = None,
                         crossover_operator: str = None,
                         mutation_operator: str = None) -> Tuple[Population, Dict]:
        """
        执行一代GA进化（支持MAB自适应算子选择）
        
        Args:
            population: 当前种群
            params: GA参数
            inject_random: 是否注入随机个体
            inject_ratio: 注入比例
            diversity: 种群多样性
            generation: 当前代数
            max_generations: 最大迭代代数，用于局部搜索阶段判断
            apply_local_search: 是否执行精英局部搜索
            crossover_operator: MAB选择的交叉算子名称
            mutation_operator: MAB选择的变异算子名称
        
        Returns:
            (新种群, 进化统计信息)
        """
        # 1. 选择
        selected = self.select(population, params)

        # 2. 交叉（支持MAB指定的算子）
        offspring = self.crossover(
            selected, params.crossover_rate, diversity, crossover_operator
        )

        # 3. 变异（支持MAB指定的算子）
        mutated = self.mutate(
            offspring, params.mutation_rate, diversity, mutation_operator
        )

        # 4. 注入随机个体（如果需要）
        if inject_random and inject_ratio > 0:
            num_inject = max(1, int(len(mutated) * inject_ratio))
            random_individuals = [
                JSSPUtils.generate_random_chromosome(self.instance.num_jobs, self.instance.num_machines)
                for _ in range(num_inject)
            ]
            mutated = mutated[:-num_inject] + random_individuals

        # 5. 精英保留（改进：保留前15%的最优个体）
        elite_count = max(3, int(params.population_size * 0.15))
        sorted_indices = population.get_sorted_indices()

        elites = []
        elite_fitness = []
        for i in range(min(elite_count, len(population.individuals))):
            idx = sorted_indices[i]
            elites.append(population.individuals[idx].copy())
            elite_fitness.append(population.fitness_values[idx])

        # 6. 局部搜索：对Top精英个体进行更积极的局部搜索
        progress = generation / max(max_generations, 1)
        local_search_calls = 0
        local_search_improvements = 0
        local_search_gain = 0.0
        local_search_elapsed = 0.0
        if apply_local_search and progress > 0.20:
            local_search_started = time.perf_counter()
            if local_search_plan is not None:
                # AP指挥模式：Executor只消费计划，不再自行判断触发时机
                # 或目标范围。计划值仍受种群精英数量约束。
                ls_elites_count = min(
                    max(0, int(local_search_plan.get('elite_count', 0))),
                    len(elites),
                )
                ls_probability = max(
                    0.0, min(1.0, float(local_search_plan.get('probability', 0.0)))
                )
                ls_mode = local_search_plan.get('mode', 'improved')
                ls_max_iter = max(1, int(local_search_plan.get('max_iter', 22)))
            elif progress > 0.75:
                ls_elites_count = min(12, len(elites))
                ls_probability = 0.65
                ls_mode = 'critical_path'
                ls_max_iter = 28
            elif progress > 0.50:
                ls_elites_count = min(10, len(elites))
                ls_probability = 0.48
                ls_mode = 'critical_path'
                ls_max_iter = 20
            else:
                ls_elites_count = min(8, len(elites))
                ls_probability = self.local_search_prob
                ls_mode = 'improved'
                ls_max_iter = 22
            for i in range(ls_elites_count):
                if random.random() < ls_probability:
                    local_search_calls += 1
                    elite = elites[i]
                    if ls_mode == 'critical_path':
                        improved = self.critical_path_local_search(
                            elite, max_iter=ls_max_iter
                        )
                    else:
                        improved = self.local_search_improved(
                            elite, max_iter=ls_max_iter
                        )
                    
                    improved_fitness = self.evaluate_local_search_result(improved)
                    if improved_fitness < elite_fitness[i]:
                        local_search_improvements += 1
                        local_search_gain += (
                            elite_fitness[i] - improved_fitness
                        )
                        elites[i] = improved
                        elite_fitness[i] = improved_fitness
            local_search_elapsed = time.perf_counter() - local_search_started

        # 7. 合并精英到新种群
        if len(mutated) >= len(elites):
            for i in range(len(elites)):
                mutated[-(i+1)] = elites[i]
        else:
            mutated.extend(elites)

        # 8. 评估新种群
        fitness_values = self.evaluate_population(mutated)
        
        # 9. 计算进化统计信息（用于MAB reward）
        # 计算交叉和变异的改进效果
        # 确保精英的适应度正确
        if len(mutated) >= len(elites):
            for i in range(len(elites)):
                fitness_values[-(i+1)] = elite_fitness[i]

        # 9. 确保种群大小符合要求
        if len(mutated) > params.population_size:
            sorted_indices = np.argsort(fitness_values)
            selected_indices = sorted_indices[:params.population_size]
            mutated = [mutated[i] for i in selected_indices]
            fitness_values = [fitness_values[i] for i in selected_indices]
        elif len(mutated) < params.population_size:
            while len(mutated) < params.population_size:
                new_ind = JSSPUtils.generate_random_chromosome(
                    self.instance.num_jobs, self.instance.num_machines
                )
                mutated.append(new_ind)
                fitness_values.append(self.evaluate(new_ind))

        # 使用种群平均适应度作为算子奖励，避免精英保留使 best 值长期不变。
        avg_improvement = 0.0
        relative_improvement = 0.0
        if fitness_values and population.fitness_values:
            current_best = min(fitness_values)
            prev_best = min(population.fitness_values)
            current_avg = float(np.mean(fitness_values))
            prev_avg = float(np.mean(population.fitness_values))
            avg_improvement = prev_best - current_best
            if prev_avg > 0:
                relative_improvement = max(0.0, (prev_avg - current_avg) / prev_avg)

        stats = {
            'avg_improvement': avg_improvement,
            'relative_improvement': relative_improvement,
            'crossover_operator': crossover_operator,
            'mutation_operator': mutation_operator,
            'local_search_calls': local_search_calls,
            'local_search_improvements': local_search_improvements,
            'local_search_gain': local_search_gain,
            'local_search_elapsed': local_search_elapsed,
        }

        return Population(individuals=mutated, fitness_values=fitness_values), stats

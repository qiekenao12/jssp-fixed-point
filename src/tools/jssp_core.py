"""
JSSP核心工具：编码、解码、makespan计算
使用基于工序的编码方式（Job-based encoding）
"""
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class JSSPInstance:
    """JSSP问题实例"""
    num_jobs: int
    num_machines: int
    processing_times: List[List[int]]  # n×m加工时间矩阵
    machine_sequences: List[List[int]]  # n×m机器顺序矩阵
    upper_bound: int = 1200

class JSSPEncoder:
    """JSSP编码器：基于工序的编码"""

    @staticmethod
    def encode(num_jobs: int, num_machines: int) -> np.ndarray:
        """
        生成初始染色体（基于工序的编码）
        每个工序用一个数字表示，数字范围[0, num_jobs-1]
        染色体长度 = num_jobs * num_machines
        每个工件的工序按顺序出现
        """
        chromosome = []
        for job in range(num_jobs):
            chromosome.extend([job] * num_machines)
        return np.array(chromosome)

    @staticmethod
    def repair_chromosome(chromosome, num_jobs, num_machines=None):
        """
        修复染色体：确保每个工件出现正确的次数（num_machines次）
        适用于基于工序的编码
        改进：尽量保留原有基因的相对顺序
        """
        if isinstance(chromosome, np.ndarray):
            chrom = chromosome.tolist()
        else:
            chrom = list(chromosome)
        
        if num_machines is None:
            num_machines = len(chrom) // num_jobs
        target_count = num_machines
        
        # 统计每个工件出现的次数
        counts = {}
        for gene in chrom:
            counts[gene] = counts.get(gene, 0) + 1
        
        # 如果染色体已经有效（每个工件恰好出现target_count次），直接返回
        if all(counts.get(job, 0) == target_count for job in range(num_jobs)):
            return np.array(chrom) if isinstance(chromosome, list) else chromosome.copy()
        
        # 修复：按原有顺序保留基因，限制每个工件最多target_count次
        repaired = []
        temp_counts = {job: 0 for job in range(num_jobs)}
        
        for gene in chrom:
            gene = int(gene)
            if temp_counts[gene] < target_count:
                repaired.append(gene)
                temp_counts[gene] += 1
        
        # 如果缺少某些工件，从原始染色体中按顺序补充（而非直接追加）
        for gene in chrom:
            gene = int(gene)
            if temp_counts[gene] < target_count:
                repaired.append(gene)
                temp_counts[gene] += 1
        
        # 最终兜底：确保所有工件都达到target_count
        for job in range(num_jobs):
            while temp_counts[job] < target_count:
                repaired.append(job)
                temp_counts[job] += 1
        
        return np.array(repaired)

    @staticmethod
    def swap_mutation(chromosome):
        """
        交换变异：随机交换两个位置的基因
        """
        if isinstance(chromosome, np.ndarray):
            mutant = chromosome.copy()
        else:
            mutant = list(chromosome).copy()
        
        idx1, idx2 = np.random.choice(len(mutant), 2, replace=False)
        mutant[idx1], mutant[idx2] = mutant[idx2], mutant[idx1]
        
        if isinstance(chromosome, np.ndarray):
            return np.array(mutant)
        return mutant

    @staticmethod
    def inverse_mutation(chromosome):
        """
        逆序变异：随机选择一段区间进行逆序
        """
        if isinstance(chromosome, np.ndarray):
            mutant = chromosome.copy()
        else:
            mutant = list(chromosome).copy()
        
        size = len(mutant)
        start = np.random.randint(0, size)
        end = np.random.randint(start + 1, size + 1)
        
        mutant[start:end] = reversed(mutant[start:end])
        
        if isinstance(chromosome, np.ndarray):
            return np.array(mutant)
        return mutant

    @staticmethod
    def decode_to_schedule(chromosome: np.ndarray, instance: JSSPInstance) -> List[Dict]:
        """
        将染色体解码为调度方案
        返回：[{job_id: int, op_index: int, machine: int, start_time: int, end_time: int}, ...]
        """
        # Imported lazily to keep the accounting layer out of normal execution
        # and avoid an Executor -> core -> accounting import cycle.
        from tools.paper_q import record_decode
        record_decode()
        num_jobs = instance.num_jobs
        num_machines = instance.num_machines

        # 跟踪每个工件的当前工序索引
        job_op_indices = [0] * num_jobs
        # 跟踪每台机器的完工时间（使用0-based索引）
        machine_end_times = [0] * num_machines
        # 跟踪每个工件的完工时间（用于工序先后约束）
        job_end_times = [0] * num_jobs

        schedule = []

        # 按染色体顺序安排工序
        for job_id in chromosome:
            job_id = int(job_id)
            op_idx = job_op_indices[job_id]
            machine = instance.machine_sequences[job_id][op_idx]
            proc_time = instance.processing_times[job_id][op_idx]
            
            # 机器编号已经是0-based，直接使用
            machine_idx = machine

            # 计算最早开始时间（考虑机器可用时间和工件前序工序完工时间）
            start_time = max(
                machine_end_times[machine_idx],
                job_end_times[job_id]
            )
            end_time = start_time + proc_time

            schedule.append({
                'job_id': job_id,
                'op_index': op_idx,
                'machine': machine,
                'start_time': start_time,
                'end_time': end_time,
                'processing_time': proc_time
            })

            # 更新状态（使用0-based索引）
            machine_end_times[machine_idx] = end_time
            job_end_times[job_id] = end_time
            job_op_indices[job_id] += 1

        return schedule


class JSSPEvaluator:
    """JSSP适应度评估器"""

    @staticmethod
    def calculate_makespan(schedule: List[Dict]) -> int:
        """
        计算调度方案的makespan（最大完工时间）
        """
        if not schedule:
            return 0
        return max(op['end_time'] for op in schedule)

    @staticmethod
    def evaluate(chromosome: np.ndarray, instance: JSSPInstance) -> int:
        """
        评估染色体的适应度（makespan）
        """
        schedule = JSSPEncoder.decode_to_schedule(chromosome, instance)
        return JSSPEvaluator.calculate_makespan(schedule)


class JSSPUtils:
    """JSSP工具函数"""

    @staticmethod
    def hamming_distance(ind1: np.ndarray, ind2: np.ndarray) -> int:
        """
        计算两个个体之间的Hamming距离（基于工序编码）
        """
        return np.sum(ind1 != ind2)

    @staticmethod
    def calculate_diversity(population: List[np.ndarray]) -> float:
        """
        计算种群多样性（平均Hamming距离的归一化值）
        优化：使用numpy向量化替代双重Python循环
        """
        if len(population) < 2:
            return 0.0

        n = len(population)
        # 转换为矩阵 (n, L)
        pop_matrix = np.array(population)

        # 向量化计算：每个个体与后续所有个体的Hamming距离
        total_distance = 0
        for i in range(n - 1):
            # pop_matrix[i] vs pop_matrix[i+1:] -> (n-i-1, L)的布尔矩阵
            total_distance += np.sum(pop_matrix[i] != pop_matrix[i+1:])

        count = n * (n - 1) // 2

        if count == 0:
            return 0.0

        avg_distance = total_distance / count
        max_possible = len(population[0])  # 最大可能距离

        return avg_distance / max_possible if max_possible > 0 else 0.0

    @staticmethod
    def compute_lower_bound(processing_times: List[List[int]], machine_sequences: List[List[int]], num_machines: int) -> int:
        """
        计算理论下界（基于机器负载和作业总工时）
        """
        n_jobs = len(processing_times)
        # 每个作业的总加工时间
        job_times = [sum(row) for row in processing_times]
        # 每台机器的总负载
        machine_load = [0] * num_machines
        for j in range(n_jobs):
            for op_idx, m in enumerate(machine_sequences[j]):
                machine_load[m] += processing_times[j][op_idx]
        return max(max(job_times), max(machine_load) // num_machines)

    @staticmethod
    def generate_random_chromosome(num_jobs: int, num_machines: int) -> np.ndarray:
        """
        生成随机染色体（打乱工序顺序）
        """
        chromosome = JSSPEncoder.encode(num_jobs, num_machines)
        np.random.shuffle(chromosome)
        return chromosome

    @staticmethod
    def schedule_to_timeline(schedule: List[Dict], num_machines: int) -> Dict:
        """
        将调度转换为机器时间线
        返回：{machine_id: [(start, end, job_id), ...]}
        """
        timeline = {m: [] for m in range(num_machines)}
        for op in schedule:
            timeline[op['machine']].append({
                'start': op['start_time'],
                'end': op['end_time'],
                'job_id': op['job_id']
            })
        return timeline

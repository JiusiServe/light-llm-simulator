# Copyright (c) 2025 Huawei. All rights reserved.
import itertools
import logging
from types import SimpleNamespace
from typing import Optional, Tuple, Any
import math
from dataclasses import dataclass

from ui.config.hw_config import HardwareTopology

logger = logging.getLogger(__name__)


# --- Refactoring: Introduced Dataclasses to group parameters ---

GB = 1e9
BILLION = 1e9
MILISECOND_FACTOR = 1000

@dataclass
class ParallelismConfig:
    """Configuration for model parallelism strategies."""
    chunk_size: int
    pp: int = 1  # Pipeline Parallelism Degree
    tp: int = 1  # Tensor Parallelism Degree
    ep: int = 1  # Expert Parallelism Degree
    dp: int = 1  # Data Parallelism Degree
    

@dataclass
class ChunkContext:
    """Context for processing a single chunk."""
    stage: str  # e.g., 'prefill' or 'decode'
    batch_size: int
    chunk_size_act: int
    chunk_size_kv: Optional[int] = 0
    chunk_idx: Optional[int] = None
    output_length: Optional[int] = 1
    pp_stage_idx: Optional[int] = 0


@dataclass
class PerformanceMetricsInput:
    """Input metrics for performance analysis of a single operation."""
    ops: float = 0.0
    m_k_n: tuple = (1, 1, 1)
    unit: str = "cube"
    load_weight: float = 0.0
    load_act: float = 0.0
    store_act: float = 0.0
    load_kv_cache: float = 0.0
    store_kv_cache: float = 0.0
    pp_comm_time: float = 0.0
    tp_comm_time: float = 0.0
    # kvp_comm_time: float = 0.0
    ep_comm_time: float = 0.0
    sync_comm_time: float = 0.0


class CostModel:
    def __init__(self, model_config: Any, hw_topology: HardwareTopology, vllm_config: Any):
        self.model = model_config
        self.hw = hw_topology
        self.pp = vllm_config.parallel_config.pipeline_parallel_size
        self.tp = vllm_config.parallel_config.tensor_parallel_size
        self.dp = vllm_config.parallel_config.data_parallel_size
        self.ep = 1 ## fix later
        self.chunk_size = vllm_config.scheduler_config.max_num_batched_tokens
        self.vllm_config = vllm_config
        self.workload_timing_cache = {}

    def estimate_cost_per_chunk_per_pp_stage(self, chunk_ctx: ChunkContext) -> dict:
        bandwidth_intra = self.hw.hw_conf.intra_node_bandwidth
        size_of_a = self.model.dtype.itemsize

        # -- Should it be self.model.max_seq_length?
        memory = self._estimate_hbm_memory_consumption(chunk_ctx.batch_size, chunk_ctx.chunk_size_act)

        # Assuming default execution config for this specific estimation
        chunk_ctx.chunk_size_kv = chunk_ctx.chunk_size_act + (chunk_ctx.chunk_idx + 1)
        results = self._analyzer_per_chunk(chunk_ctx)

        processing_time = 0
        for k in results.get(chunk_ctx.stage).keys():
            if 'lm_head' not in k and 'embedding' not in k:
                processing_time += results.get(chunk_ctx.stage)[k]['total_time']

        processing_time *= self.model.hf_text_config.num_hidden_layers / self.pp

        processing_time += results.get(chunk_ctx.stage)['embedding']['total_time'] * (
                chunk_ctx.pp_stage_idx == 1)
        processing_time += results.get(chunk_ctx.stage)['lm_head']['total_time'] * (
                chunk_ctx.pp_stage_idx == (self.pp - 1))

        pp_time = 0.
        if chunk_ctx.pp_stage_idx < self.pp - 1:
            pp_comm_per_npu = (chunk_ctx.batch_size * self.model.get_hidden_size() * chunk_ctx.chunk_size_act * size_of_a)
            pp_time = pp_comm_per_npu / bandwidth_intra

        return {
            "memory": memory / GB,
            "time": (pp_time + processing_time) * MILISECOND_FACTOR
        }

    def estimate_cost_per_chunk_per_pp_stage_deepseek(self, chunk_ctx: ChunkContext) -> dict:

        bandwidth_intra = self.hw.hw_conf.intra_node_bandwidth
        size_of_a = self.model.dtype.itemsize
        memory = self._get_deepseek_size_per_device()
        memory += self._get_deepseek_kvcache_size(chunk_ctx.batch_size,
                                                  (chunk_ctx.chunk_idx + 1) * chunk_ctx.chunk_size_act)

        # exec_config = ExecutionConfig(chunk_size=chunk_ctx.chunk_size_act, use_flash_attn=False,
        #                               qkv_weights_concat=True)

        chunk_ctx.chunk_size_kv = chunk_ctx.chunk_size_act + (chunk_ctx.chunk_idx + 1)
        processing_time = 0
        results = self._analyzer_per_chunk_deepseek(chunk_ctx)

        for k in results.get(chunk_ctx.stage).keys():
            if 'lm_head' not in k and 'embedding' not in k:
                processing_time += results.get(chunk_ctx.stage)[k]['total_time']
        processing_time *= self.model.hf_text_config.num_hidden_layers / self.pp

        processing_time += results.get(chunk_ctx.stage)['embedding']['total_time'] * (
                chunk_ctx.pp_stage_idx == 1)

        processing_time += results.get(chunk_ctx.stage)['lm_head']['total_time'] * (
                chunk_ctx.pp_stage_idx == (self.pp - 1))
        pp_time = 0.
        if chunk_ctx.pp_stage_idx < self.pp - 1:
            pp_comm_per_npu = (chunk_ctx.batch_size * self.model.get_hidden_size() * chunk_ctx.chunk_size_act * size_of_a)
            pp_time = pp_comm_per_npu / bandwidth_intra

        return {"memory": memory / GB, "time": (pp_time + processing_time) * MILISECOND_FACTOR}

    def estimate_cost_of(self, infer_config: ChunkContext) -> dict:
        """
        Estimates the end-to-end cost for a standard transformer model by
        orchestrating prefill and decode calculations.
        """
        error = None
        # --- 1. Memory and Hardware Validation ---
        prefill_memory = self._get_model_size_per_device()
        prefill_memory += self._get_kvcache_size(infer_config.batch_size, infer_config.chunk_size_act)
        if prefill_memory > self.hw.hw_conf.npu_memory:
            error = "cannot run prefill due to low hardware memory"

        # --- 2. Prefill Calculation ---
        prefill_time = self._calculate_prefill_time(infer_config)
        chunk_size = min(infer_config.chunk_size_act, self.chunk_size)
        prefill_time += self._compute_pp_comm_cost_total(infer_config.batch_size, chunk_size, self.pp,
                                                         is_prefill=True)

        # --- 3. Decode Calculation ---
        decode_memory = self._get_model_size_per_device()
        decode_memory += self._get_kvcache_size(infer_config.batch_size,
                                                infer_config.chunk_size_act + infer_config.output_length)
        if decode_memory > self.hw.hw_conf.npu_memory:
            error = "cannot run decode due to low hardware memory"

        total_decode_time, decode_time_per_token = self._calculate_decode_time(infer_config)
        decode_time_per_token += self._compute_pp_comm_cost_total(infer_config.batch_size, 1, self.pp,
                                                                  is_prefill=False)

        # --- 4. Assemble Results ---
        if error is not None:
            logger.error("Error: %s", error)

        return {
            "error": error,
            "memory_error": "" if error is None else error,
            "prefill_memory": prefill_memory / GB,
            "TTFT": prefill_time * MILISECOND_FACTOR,
            "throughput_prefill": infer_config.batch_size / prefill_time if prefill_time > 0 else 0,
            "decode_memory": decode_memory / GB,
            "TBT": decode_time_per_token * MILISECOND_FACTOR,
            "decode_time": total_decode_time * MILISECOND_FACTOR,
            "throughput_decode": infer_config.batch_size / decode_time_per_token if decode_time_per_token > 0 else 0,
        }

    def optimize_parallelism(self, infer_config: ChunkContext, evaluation_mode: bool = False) -> dict:
        available_npus = self.hw.npus_per_rank
        npu_available_mem = self.hw.hw_conf.npu_memory
        size_of_w = self.model.dtype.itemsize
        model_size = self.model.model_size_b * BILLION * size_of_w

        best_config = {}
        min_prefill_time = float('inf')
        min_decode_time = float('inf')

        chunk_sizes = [
            size
            for size in [512, 1024, 2048, 4096, 8192, 16384, 32768]
            if size <= self.model.max_seq_length
        ]

        parallel_degrees = [1, 2, 4, 8]

        for pp, chunk_size, tp in itertools.product(parallel_degrees, chunk_sizes, parallel_degrees
                                                         ):
            if pp * tp > available_npus or model_size > npu_available_mem * tp * pp:
                continue            

            result = self._evaluate_parallel_config(infer_config)

            if result['TTFT'] < min_prefill_time:
                min_prefill_time = result['TTFT']
                best_config['prefill'] = result['prefill']

            if result['TBT'] < min_decode_time:
                min_decode_time = result['TBT']
                best_config['decode'] = result['decode']

        if 'prefill' in best_config and 'decode' in best_config:
            best_config['total_time'] = best_config['prefill']['time'] + best_config['decode']['time']

        best_config.update(infer_config.__dict__)
        return best_config

    def estimate_cost_of_deepseek_prefill(self, infer_config: ChunkContext) -> dict:
        
        num_nodes = self.hw.number_of_ranks
        available_npus = self.hw.npus_per_rank
        npu_available_mem = self.hw.hw_conf.npu_memory
        model_size = self._get_deepseek_size()
        n_routed_experts = self.model.n_routed_experts

        error = None
        if (self.pp * self.tp * self.kvp * self.dp > num_nodes * available_npus or
                model_size > num_nodes * available_npus * npu_available_mem):
            error = 'cannot run this setup on this hardware'

        prefill_memory = self._get_deepseek_size_per_device()
        prefill_memory += self._get_deepseek_kvcache_size(infer_config.batch_size, infer_config.chunk_size_act)

        memory_error = ""
        if prefill_memory > self.hw.hw_conf.npu_memory:
            memory_error = "cannot run prefill due to low hardware memory"

        chunk_size = min(infer_config.chunk_size_act, self.chunk_size)
        rem = infer_config.chunk_size_act % chunk_size
        num_chunks = infer_config.chunk_size_act // chunk_size + (rem > 0)

        prefill_time = 0.0
        for i in range(num_chunks):
            chunk_ctx = ChunkContext(
                stage='prefill',
                batch_size=infer_config.batch_size,
                chunk_size_act=chunk_size,
                chunk_idx=i,
                chunk_size_kv=(i + 1) * chunk_size
            )
            results = self._analyzer_per_chunk_deepseek(chunk_ctx)
            chunk_time = self._accumulate_prefill_times(results, n_routed_experts, i, num_chunks)
            prefill_time += chunk_time

        prefill_time += self._compute_pp_comm_cost_total(
            infer_config.batch_size, chunk_size, self.pp, is_prefill=True
        )

        return {
            "error": error,
            "memory_error": memory_error,
            "prefill_memory": prefill_memory,
            "TTFT": prefill_time,
        }

    def estimate_cost_of_deepseek_decode(self, infer_config: ChunkContext) -> dict:
        """
        Estimates the decode cost for a DeepSeek model by orchestrating
        the analysis of the first and last generated tokens.
        """
        # --- 1. Memory Calculation and Validation ---
        decode_memory = self._get_deepseek_size_per_device()
        decode_memory += self._get_deepseek_kvcache_size(
            infer_config.batch_size,
            infer_config.chunk_size_act + infer_config.output_length
            
        )

        error = None
        if decode_memory > self.hw.hw_conf.npu_memory:
            error = "cannot run decode due to low hardware memory"

        # --- 2. Calculate Time for First and Last Tokens using the Helper ---    
        first_chunk_ctx = ChunkContext(stage='decode',
                                       batch_size=infer_config.batch_size,
                                       chunk_size_act=1,
                                       chunk_size_kv=infer_config.chunk_size_kv
                                       )
        first_decode_time = self._calculate_deepseek_single_token_time(first_chunk_ctx)

        last_chunk_ctx = ChunkContext(stage='decode',
                                      batch_size=infer_config.batch_size,
                                      chunk_size_act=1,
                                      chunk_size_kv=infer_config.chunk_size_kv + infer_config.output_length)
        last_decode_time = self._calculate_deepseek_single_token_time(last_chunk_ctx)

        # --- 3. Assemble Final Results ---
        if infer_config.output_length > 0:
            total_decode_time = (first_decode_time + last_decode_time) * infer_config.output_length / 2
            decode_time_per_token = total_decode_time / infer_config.output_length
        else:
            total_decode_time = 0.0
            decode_time_per_token = 0.0

        decode_time_per_token += self._compute_pp_comm_cost_total(
            infer_config.batch_size, 1, self.pp, is_prefill=False
        )

        if error is not None:
            logger.error("Error: %s", error)

        return {
            "error": error,
            "decode_memory": decode_memory,
            "TBT": decode_time_per_token,
            "decode_time": total_decode_time
        }

    def estimate_cost_of_deepseek(self, infer_config: ChunkContext) -> dict:
        res_prefill = self.estimate_cost_of_deepseek_prefill(infer_config)
        res_decode = self.estimate_cost_of_deepseek_decode(infer_config)

        error = None
        if (res_prefill['error'] is not None) or (res_decode['error'] is not None):
            error = f"prefill error: {res_prefill['error']} | decode error: {res_decode['error']}"

        throughput_prefill = self.dp / res_prefill['TTFT'] if res_prefill['TTFT'] > 0 else 0
        throughput_decode = self.dp / res_decode['TBT'] if res_decode['TBT'] > 0 else 0

        return {
            "error": error,
            "prefill_memory": res_prefill['prefill_memory'] / GB,
            "TTFT": res_prefill['TTFT'] * MILISECOND_FACTOR,
            "throughput_prefill": throughput_prefill,
            "decode_memory": res_decode['decode_memory'] / GB,
            "TBT": res_decode['TBT'] * MILISECOND_FACTOR,
            "decode_time": res_decode['decode_time'] * MILISECOND_FACTOR,
            "throughput_decode": throughput_decode,
        }

    def _calculate_deepseek_single_token_time(self, chunk_ctx: ChunkContext) -> float:
        """Calculates the processing time for a single DeepSeek decode token."""
        results = self._analyzer_per_chunk_deepseek(chunk_ctx)
        num_experts_per_tok = self.model.num_experts_per_tok

        attn_time = ffn_time = moe_time = 0.0
        for k, v in results['decode'].items():
            if 'lm_head' in k or 'embedding' in k:
                continue
            if 'ffn' in k:
                ffn_time += v.get('total_time', 0.0)
            elif 'moe' in k:
                # The MoE calculation differs slightly based on context length
                moe_inf_time = v.get('inference_time', 0.0)
                # if chunk_ctx.chunk_size_kv == infer_config.chunk_size_kv:  # Heuristic for first token
                moe_time += moe_inf_time * (self.dp * (num_experts_per_tok + 1) / self.ep)
                # else:
                #     moe_time += moe_inf_time * (self.dp * (num_experts_per_tok + 1))
                moe_time += v.get('communication_time', 0.0)
            else:
                attn_time += v.get('total_time', 0.0)

        first_k = getattr(self.model, "first_k_dense_replace", self.model.hf_text_config.num_hidden_layers)
        total_time = (attn_time * self.model.hf_text_config.num_hidden_layers +
                      moe_time * (self.model.hf_text_config.num_hidden_layers - first_k) +
                      ffn_time * first_k)

        # BUG FIX: Correctly gets lm_head time from the 'decode' results
        total_time += results.get('decode', {}).get('embedding', {}).get('total_time', 0.0)
        total_time += results.get('decode', {}).get('lm_head', {}).get('total_time', 0.0)

        return total_time

    def _evaluate_parallel_config(self, infer_config):
        result = self.estimate_cost_of(infer_config)

        prefill = {
            "cpp_degree": self.pp,
            "chunk_size": self.chunk_size,
            "tp_degree": self.tp,
            "time": result['TTFT'] / MILISECOND_FACTOR,
            "max_memory": result['prefill_memory'] * GB,
        }

        decode = {
            "cpp_degree": self.pp,
            "tp_degree": self.tp,
            "time_per_token": result['TBT'] / MILISECOND_FACTOR,
            "time": result['decode_time'] / MILISECOND_FACTOR,
            "max_memory": result['decode_memory'] * GB,
        }

        return {
            "TTFT": result['TTFT'],
            "TBT": result['TBT'],
            "prefill": prefill,
            "decode": decode,
        }

    def _evaluate_parallel_config_deepseek(self, infer_config):
        result = self.estimate_cost_of_deepseek(infer_config)
        prefill = {
            "cpp_degree": self.pp,
            "chunk_size": self.chunk_size,
            "tp_degree": self.tp,
            "time": result['TTFT'] / MILISECOND_FACTOR,
            "max_memory": result['prefill_memory'] * GB,
        }

        decode = {
            "cpp_degree": self.pp,
            "tp_degree": self.tp,
            "time_per_token": result['TBT'] / MILISECOND_FACTOR,
            "time": result['decode_time'] / MILISECOND_FACTOR,
            "max_memory": result['decode_memory'] * GB,
        }

        return {
            "TTFT": result['TTFT'],
            "TBT": result['TBT'],
            "prefill": prefill,
            "decode": decode,
        }

    def _accumulate_prefill_times(self, results, n_routed_experts, chunk_index, num_chunks):
        attn_time = 0.0
        ffn_time = 0.0
        moe_time = 0.0

        for k, v in results.get('prefill', {}).items():
            if 'lm_head' in k or 'embedding' in k:
                continue
            if 'ffn' in k:
                ffn_time += v['total_time']
            elif 'moe' in k:
                moe_time += v['inference_time'] * (self.dp * (n_routed_experts // self.ep + 1))
                moe_time += v['communication_time']
            else:
                attn_time += v['total_time']

        lm_head_time = results.get('prefill', {}).get('lm_head', {}).get('total_time', 0.0)
        embedding_time = results.get('prefill', {}).get('embedding', {}).get('total_time', 0.0)
        is_last_chunk = (chunk_index == num_chunks - 1)

        return (
                attn_time * self.model.hf_text_config.num_hidden_layers +
                moe_time * (self.model.hf_text_config.num_hidden_layers - self.model.first_k_dense_replace) +
                ffn_time * self.model.first_k_dense_replace +
                + embedding_time +
                (lm_head_time if is_last_chunk else 0.0)
        )

    def _analyze_to_results(self, name: str, metrics: PerformanceMetricsInput) -> dict:

        def optimize_tiling(inpsz_a, inpsz_b, outsz, L2_max_size):
            from math import ceil
            from scipy.optimize import minimize
            def memory_eq(t_):
                """memory usage in L2 cache with t_a and t_b tiles.
                we must satisfy:
                inpsz_a/t_a + inpsz_b/t_b + outsz/(t_a*t_b) <= L2_max_size
                where: t_a is the number of tiles for input a,
                    t_b is the number of tiles for input b, and outsz is the output size.
                equivalent to: L2_max_size * t_a * t_b - inpsz_a * t_b - inpsz_b * t_a - outsz >= 0
                """
                t_a, t_b = t_
                return L2_max_size * t_a * t_b - inpsz_a * t_b - inpsz_b * t_a - outsz

            def objective_f(t):
                """
                Objective function to minimize the product of t_a and t_b ==> minimize the number of tiling multiplications.
                minimum number of multiplications ==> large tiles ==> more computation intensity. 
                """
                t_a, t_b = t
                return t_a * t_b

            constraints = ({'type': 'ineq', 'fun': lambda t: t[0] - 1},  # t_a >= 1
                           {'type': 'ineq', 'fun': lambda t: t[1] - 1},  # t_b >=1 
                           {'type': 'ineq', 'fun': lambda t: L2_max_size - (inpsz_a * t[1] + inpsz_b * t[0] + outsz) / (
                                       t[0] * t[1])})  # L2_max_size >= inpsz_a/t_a + inpsz_b/t_b + outsz/(t_a*t_b) 

            # Initial guess
            initial_guess = [1, 1]

            # Minimize the function
            result = minimize(objective_f, initial_guess, constraints=constraints)

            # Print the minimum value and the corresponding x_a and x_b
            min_value = result.fun + L2_max_size  # Adjusted to reflect the actual minimum value
            optimal_t_a, optimal_t_b = result.x
            optimal_t_a = round(optimal_t_a * 1e8) / 1e8  # chop
            optimal_t_b = round(optimal_t_b * 1e8) / 1e8  # chop cases of 1.0000000001

            return ceil(optimal_t_a), ceil(optimal_t_b)

        def optimize_tiling_dims(m, k, n, L2_max_size, output_factor=None, byes_per_element=2):
            from scipy.optimize import minimize
            def memory_eq(t_):
                """memory usage in L2 cache with t_a and t_b tiles.
                we must satisfy:
                inpsz_a/t_a + inpsz_b/t_b + outsz/(t_a*t_b) <= L2_max_size
                where: t_a is the number of tiles for input a,
                    t_b is the number of tiles for input b, and outsz is the output size.
                equivalent to: L2_max_size * t_a * t_b - inpsz_a * t_b - inpsz_b * t_a - outsz >= 0
                """
                t_m, t_k, t_n = t_
                return L2_max_size * t_m * t_k * t_n - byes_per_element * m * k * t_n - byes_per_element * n * k * t_m - byes_per_element * m * n * t_k

            def objective_f(t):
                """
                Objective function to minimize the product of t_a and t_b ==> minimize the number of tiling multiplications.
                minimum number of multiplications ==> large tiles ==> more computation intensity. 
                """
                t_m, t_k, t_n = t
                return t_m * t_k * t_n

            output_factor = 1  # if output_factor is None else output_factor
            constraints = (
                {'type': 'ineq', 'fun': lambda t: t[0] - 1},  # x_a >= 1
                {'type': 'ineq', 'fun': lambda t: t[1] - 1},
                {'type': 'ineq', 'fun': lambda t: t[2] - 1},
                #    {'type': 'ineq', 'fun': lambda t: -m/t[0] +256},
                #    {'type': 'ineq', 'fun': lambda t: -k/t[1] +256},
                #    {'type': 'ineq', 'fun': lambda t: -n/t[2] +128},
                {'type': 'ineq', 'fun': lambda t: L2_max_size - byes_per_element * (
                            m * k * t[2] + n * k * t[0] + m * n * t[1] * output_factor) / (t[0] * t[1] * t[2])}
            )

            # Initial guess
            initial_guess = [1, 1, 1]

            # Minimize the function
            result = minimize(objective_f, initial_guess, constraints=constraints)

            # Print the minimum value and the corresponding x_a and x_b
            min_value = result.fun + L2_max_size  # Adjusted to reflect the actual minimum value
            optimal_t_m, optimal_t_k, optimal_t_n = result.x

            return round(optimal_t_m), round(optimal_t_k), round(optimal_t_n)

        size_of_a, size_of_w = self.model.dtype.itemsize, self.model.dtype.itemsize

        is_int8 = size_of_a <= 1 and size_of_w <= 1
        npu_flops = self.hw.hw_conf.npu_flops_int8 if is_int8 else self.hw.hw_conf.npu_flops_fp16
        if metrics.unit != 'cube':
            npu_flops = self.hw.hw_conf.vec_flops

        L2_cache_size = self.hw.hw_conf.onchip_buffer_size
        m, k, n = metrics.m_k_n
        if False:
            t_a, t_b = optimize_tiling(metrics.load_weight, metrics.load_act + metrics.load_kv_cache,
                                       metrics.store_act + metrics.store_kv_cache, L2_cache_size)
            memory_access = (metrics.load_weight + (metrics.load_act + metrics.load_kv_cache) * t_a +
                             metrics.store_kv_cache + metrics.store_act * 1)
        else:
            if metrics.m_k_n != (1, 1, 1):
                m, k, n = metrics.m_k_n
                t_m, t_k, t_n = optimize_tiling_dims(m, k, n, L2_cache_size)

                memory_access_1 = (metrics.load_weight * t_n + (metrics.load_act + metrics.load_kv_cache) * t_m +
                                   metrics.store_kv_cache + metrics.store_act * (1))
                memory_access_2 = (metrics.load_weight * 1 + (metrics.load_act + metrics.load_kv_cache) * t_m +
                                   metrics.store_kv_cache + metrics.store_act * (t_k))
                memory_access_3 = (metrics.load_weight * t_n + (metrics.load_act + metrics.load_kv_cache) * 1 +
                                   metrics.store_kv_cache + metrics.store_act * (t_k))

                memory_access = max(memory_access_1, memory_access_2, memory_access_3)
            else:
                memory_access = (metrics.load_weight + metrics.load_act + metrics.load_kv_cache +
                                 metrics.store_kv_cache + metrics.store_act)

        if memory_access == 0:
            return {"total_time": 0}  # Avoid division by zero

        large_dim = 4 * 1024
        # print(f"m,k,n: {m},{k},{n} | memory_access: {memory_access}")
        # if m > large_dim or n > large_dim or k > large_dim:           
        if False:  # (m + k + n) > large_dim:

            factor = max(1, (m + k + n) // large_dim)

            # if factor > 1:
            var_ = 0.
            # _is_double_large = sum([m>large_dim, k>large_dim, n>large_dim])
            # M_, m_ = (max(m,n), min(m,n))

            if max(m, k, n) > 16 * large_dim:
                var_ = 1 / factor
            metrics.ops = metrics.ops * (3.5 + var_)
            if factor > 1:
                metrics.ops = metrics.ops * (0.535 + var_)
            # else:
            #     metrics.ops *= 1.2
            # metrics.ops *= 1.2  # empirical adjustment for large matmuls
        # if m > 2*large_dim or n > 2*large_dim or k > 2*large_dim:           
        #     metrics.ops *= 1.2
        # if m > 3*large_dim or n > 3*large_dim or k > 3*large_dim:           
        #     metrics.ops *= 1.4
        # if m > 4*large_dim or n > 4*large_dim or k > 4*large_dim:           
        #     metrics.ops *= 1.07

        arithmetic_intensity = metrics.ops / memory_access
        performance = min(arithmetic_intensity * self.hw.hw_conf.local_memory_bandwidth, npu_flops)

        inference_time = metrics.ops / performance if performance > 0 else memory_access / self.hw.hw_conf.local_memory_bandwidth
        comm_time = metrics.tp_comm_time + metrics.ep_comm_time + metrics.pp_comm_time

        return {
            "OPs": metrics.ops, "memory_access": memory_access, "arithmetic_intensity": arithmetic_intensity,
            "performance": performance, "bound": 'compute' if performance == npu_flops else 'memory',
            "inference_time": inference_time, "communication_time": comm_time,
            "total_time": inference_time + comm_time
        }

    def _get_deepseek_size_per_device(self) -> float:
        # Unpack for readability
        hidden_size, q_lora_rank, kv_lora_rank = self.model.get_hidden_size(), self.model.q_lora_rank, self.model.kv_lora_rank
        moe_intermediate_size, qk_nope_head_dim = self.model.moe_intermediate_size, self.model.qk_nope_head_dim
        qk_rope_head_dim = self.model.qk_rope_head_dim
        num_heads, n_routed_experts = self.model.get_num_attention_heads(self.vllm_config.parallel_config), self.model.n_routed_experts
        intermediate_size = self.vllm_config.model_config.hf_config.intermediate_size
        num_layers, first_k_dense_replace = self.model.hf_text_config.num_hidden_layers, self.model.first_k_dense_replace
        vocab_size, size_of_w = self.vllm_config.model_config.get_vocab_size(), self.model.dtype.itemsize
        pp, tp, ep = self.pp, self.tp, self.ep

        attn = hidden_size * q_lora_rank + q_lora_rank * qk_nope_head_dim * num_heads / tp \
               + hidden_size * kv_lora_rank + kv_lora_rank * num_heads * qk_nope_head_dim / tp \
               + kv_lora_rank * num_heads * qk_nope_head_dim / tp + q_lora_rank * num_heads \
               * qk_rope_head_dim / tp + hidden_size * qk_rope_head_dim + qk_nope_head_dim \
               * num_heads * hidden_size / tp

        number_of_experts_per_device = (n_routed_experts // ep + 1)
        moe = hidden_size * moe_intermediate_size * 3 * number_of_experts_per_device + hidden_size * n_routed_experts
        ffn = intermediate_size * hidden_size * 3
        embeddings_and_head = vocab_size * hidden_size * 2
        rms_norm = hidden_size * 2

        return (first_k_dense_replace * ffn + (num_layers - first_k_dense_replace) * moe + num_layers * (
                attn + rms_norm) + embeddings_and_head) * size_of_w / pp

    def _get_deepseek_size(self):
        hidden_size = self.model.get_hidden_size()
        q_lora_rank = self.model.q_lora_rank
        kv_lora_rank = self.model.kv_lora_rank
        moe_intermediate_size = self.model.moe_intermediate_size
        qk_nope_head_dim = self.model.qk_nope_head_dim
        qk_rope_head_dim = self.model.qk_rope_head_dim
        num_heads = self.model.get_num_attention_heads(self.vllm_config.parallel_config)
        n_routed_experts = self.model.n_routed_experts
        intermediate_size = self.vllm_config.model_config.hf_config.intermediate_size
        num_layers = self.model.hf_text_config.num_hidden_layers
        first_k_dense_replace = self.model.first_k_dense_replace
        vocab_size = self.vllm_config.model_config.get_vocab_size()
        size_of_w = self.model.dtype.itemsize

        attn = hidden_size * q_lora_rank + q_lora_rank * qk_nope_head_dim * num_heads \
               + hidden_size * kv_lora_rank + kv_lora_rank * num_heads * qk_nope_head_dim \
               + kv_lora_rank * num_heads * qk_nope_head_dim + q_lora_rank * num_heads * qk_rope_head_dim \
               + hidden_size * qk_rope_head_dim + qk_nope_head_dim * num_heads * hidden_size

        moe = hidden_size * moe_intermediate_size * 3 * (n_routed_experts + 1) + \
              hidden_size * n_routed_experts

        ffn = intermediate_size * hidden_size * 3

        embeddings_and_head = vocab_size * hidden_size * 2

        rms_norm = hidden_size * 2

        return size_of_w * (first_k_dense_replace * ffn + (num_layers - first_k_dense_replace) * moe + num_layers * (
                attn + rms_norm) + embeddings_and_head)

    def _get_deepseek_kvcache_size(self, batch_size: int, seq_len: int) -> float:
        size_of_kv = self.model.dtype.itemsize
        numerator = self.model.hf_text_config.num_hidden_layers * batch_size * seq_len * (
                self.model.kv_lora_rank + self.model.qk_rope_head_dim) * size_of_kv
        denominator = self.pp
        return numerator / denominator if denominator > 0 else 0

    def _analyzer_per_chunk_deepseek(self, chunk_ctx: ChunkContext) -> dict:
        """
        Orchestrates the performance analysis of a single DeepSeek transformer layer.

        This method breaks the analysis into logical components (projections, attention, ffn, etc.),
        delegating the detailed calculations to specialized helper methods.
        """
        results = {'prefill': {}, 'decode': {}}
        stage = chunk_ctx.stage

        bw_intra = int(self.hw.hw_conf.intra_node_bandwidth)
        # todo check inter node bandwidth
        bw_inter = int(self.hw.hw_conf.inter_node_bandwidth)

        # Calculate common communication costs once
        hidden_size, size_of_a = self.model.get_hidden_size(), self.model.dtype.itemsize
        tp_comm_data = size_of_a * chunk_ctx.batch_size * chunk_ctx.chunk_size_act * hidden_size
        tp_comm = (
            2 * tp_comm_data / bw_intra if self.tp > 4 else tp_comm_data / bw_intra) if self.tp > 1 else 0.
        # kvp_comm = (
        #     2 * tp_comm_data / bw_intra if self.kvp > 4 else tp_comm_data / bw_intra) if self.kvp > 1 else 0.

        # --- Call helpers to analyze each part of the layer ---

        if False: # not exec_config.use_flash_attn:
            results.setdefault(stage, {}).update(
                self._analyze_deepseek_attention_projections(chunk_ctx, tp_comm)
            )
            results[stage].update(
                self._analyze_deepseek_attention_core(chunk_ctx, tp_comm))
        else:
            results[stage].update(self._analyze_deepseek_flashMLA(chunk_ctx, tp_comm))

        results[stage].update(self._analyze_deepseek_ffn(chunk_ctx))
        results[stage].update(self._analyze_deepseek_moe(chunk_ctx, tp_comm))
        results[stage].update(self._analyze_layer_residuals_and_norms(chunk_ctx))
        results[stage].update(self._analyze_lm_head(chunk_ctx))

        groups = {
            'attn': ['c_q_proj', 'c_kv_proj', 'q_rope', 'k_rope', 'q_proj', 'q_hat_proj', 'flash_attn', 'o_proj',
                     'attn_add'],   # rope
            'ffn': ['ffn_gate_proj', 'ffn_up_proj', 'ffn_down_proj', 'mlp_act', 'mlp_add'],
            'moe': ['moe_gate_proj', 'moe_up_proj', 'moe_down_proj', 'mlp_act', 'mlp_add'],
            'norm': ['attn_norm', 'mlp_norm'],
            # 'rotary': ['rotary'],
            'embedding': ['embedding'],
            'lm_head': ['lm_head'],

        }

        group_timings = {}

        for g in groups.keys():
            op_time = 0.
            for op in groups[g]:
                op_time += (results[stage][op]['total_time'])
            group_timings[g] = op_time * 1000

        return results, group_timings

    def _analyzer_per_chunk_deepseek_for_calibration(self, chunk_ctx: ChunkContext) -> dict:
        """
        Orchestrates the performance analysis of a single DeepSeek transformer layer.

        This method breaks the analysis into logical components (projections, attention, ffn, etc.),
        delegating the detailed calculations to specialized helper methods.
        """
        results = {'prefill': {}, 'decode': {}}
        stage = chunk_ctx.stage

        bw_intra = int(self.hw.hw_conf.intra_node_bandwidth)
        # todo check inter node bandwidth
        bw_inter = int(self.hw.hw_conf.inter_node_bandwidth)

        # Calculate common communication costs once
        hidden_size, size_of_a = self.model.get_hidden_size(), self.model.dtype.itemsize
        tp_comm_data = size_of_a * chunk_ctx.batch_size * chunk_ctx.chunk_size_act * hidden_size
        tp_comm = (
            2 * tp_comm_data / bw_intra if self.tp > 4 else tp_comm_data / bw_intra) if self.tp > 1 else 0.
        # kvp_comm = (
        #     2 * tp_comm_data / bw_intra if self.kvp > 4 else tp_comm_data / bw_intra) if self.kvp > 1 else 0.

        # --- Call helpers to analyze each part of the layer ---

        if False: #not exec_config.use_flash_attn:
            results.setdefault(stage, {}).update(
                self._analyze_deepseek_attention_projections(chunk_ctx, tp_comm)
            )
            results[stage].update(
                self._analyze_deepseek_attention_core(chunk_ctx, tp_comm))
        else:
            results[stage].update(self._analyze_deepseek_flashMLA(chunk_ctx, tp_comm))

        results[stage].update(self._analyze_layer_residuals_and_norms(chunk_ctx))
        results[stage].update(self._analyze_lm_head(chunk_ctx))

        results[stage].update(self._analyze_deepseek_ffn(chunk_ctx))
        results[stage].update(self._analyze_deepseek_moe_per_die_per_exp(chunk_ctx, tp_comm))

        groups = {
            'attn_1': ['mla_q_nope_dq', 'mla_cq_rms', 'mla_q_uq', 'mla_q_uk', 'mla_q_qr', 'mla_q_rope', 'mla_k_nope',
                       'mla_k_rope', 'flash_attn'],
            'attn_2': ['uv_proj', 'dynamic_quant', 'o_proj', 'routing'],
            'ffn': ['ffn_gate_proj', 'ffn_up_proj', 'ffn_down_proj', 'mlp_act', 'mlp_add'],
            'moe': ['moe_gate_proj', 'moe_up_proj', 'moe_down_proj', 'mlp_act', 'mlp_add'],
            'norm': ['attn_norm', 'mlp_norm'],
            # 'rotary': ['rotary'],
            'embedding': ['embedding'],
            'lm_head': ['lm_head'],

        }

        group_timings = {}

        for g in groups.keys():
            op_time = 0.
            for op in groups[g]:
                op_time += (results[stage][op]['inference_time'])
                # op_time +=(results[stage][op]['total_time'])
            group_timings[g] = op_time * 1e6  # in us

        return results, group_timings

    def _get_model_size_per_device(self) -> float:
        size_of_w, hidden_size, num_layers = self.model.dtype.itemsize, self.model.get_hidden_size(), self.model.hf_text_config.num_hidden_layers
        vocab_size = self.vllm_config.model_config.get_vocab_size()
        num_heads, kv_heads, head_size = self.model.get_num_attention_heads(self.vllm_config.parallel_config), self.model.get_num_kv_heads(self.vllm_config.parallel_config), self.model.get_head_size()
        intermediate_size = self.vllm_config.model_config.hf_config.intermediate_size

        ffn = intermediate_size * hidden_size * 3
        attn = num_heads * head_size * hidden_size * 2 + kv_heads * head_size * hidden_size * 2
        embeddings_and_head = vocab_size * hidden_size * 2
        rms_norm = hidden_size * 2

        param_memory = size_of_w * ((ffn + attn + rms_norm) * num_layers + embeddings_and_head)
        return param_memory / (self.tp * self.pp)

    def _get_kvcache_size(self, batch_size: int, seq_len: int) -> float:
        kv_heads, head_size, num_layers, size_of_kv = (self.model.get_num_kv_heads(self.vllm_config.parallel_config), self.model.get_head_size(),
                                                       self.model.hf_text_config.num_hidden_layers, self.model.dtype.itemsize)

        num_elements = 2 * batch_size * seq_len
        head_factors = kv_heads * head_size
        total_size = num_elements * head_factors * num_layers * size_of_kv
        denominator = self.tp * self.pp

        if total_size > 1e18:
            raise ValueError(f"KV cache size {total_size} too large.")

        return total_size / denominator if denominator > 0 else 0

    def _estimate_hbm_memory_consumption(self, batch_size: int, seq_len: int) -> float:
        param_memory = self._get_model_size_per_device(self.tp, self.pp)
        kv_cache_sz = self._get_kvcache_size(batch_size, seq_len)
        return param_memory + kv_cache_sz

    def _compute_pp_comm_cost_total(self, batch_size: int, chunk_size: int, pp_degree: int, is_prefill: bool) -> float:
        """Compute communication cost in seconds"""
        bandwidth_intra = self.hw.hw_conf.intra_node_bandwidth
        hidden_size = self.model.get_hidden_size()
        size_of_a = self.model.dtype.itemsize

        if pp_degree <= 1:
            return 0.0

        if is_prefill:
            pp_comm_per_npu = (batch_size * hidden_size * chunk_size * size_of_a)
        else:
            pp_comm_per_npu = (batch_size * hidden_size * size_of_a)

        pp_time = 0.0
        if pp_degree <= 4:
            pp_time += pp_comm_per_npu * (pp_degree - 1) / bandwidth_intra
        else:
            # Assuming a different communication pattern for larger degrees
            pp_time += pp_comm_per_npu * (pp_degree - 2) / bandwidth_intra + pp_comm_per_npu / bandwidth_intra

        return pp_time

    def _analyzer_per_chunk(self, chunk_ctx: ChunkContext) -> dict:
        """
        Orchestrates the performance analysis of a single standard transformer layer.
        This method delegates detailed calculations to specialized helper methods.
        """
        results = {'prefill': {}, 'decode': {}}
        stage = chunk_ctx.stage

        # Unpack variables needed by multiple helpers
        context_vars = {
            'batch_size': chunk_ctx.batch_size,
            'chunk_size_act': chunk_ctx.chunk_size_act,
            'chunk_size_kv': chunk_ctx.chunk_size_kv,
            'tp': self.tp,            
        }

        ctx = SimpleNamespace(**context_vars)

        # Call helpers to analyze each component of the layer takes tiling and blocks into accouunt 
        results[stage].update(self._analyze_std_attention_projections(ctx))
        results[stage].update(self._analyze_std_attention_core(ctx))
        results[stage].update(self._analyze_std_ffn_layer(ctx))

        # These helpers are generic and can be reused from the deepseek refactoring
        results[stage].update(self._analyze_layer_residuals_and_norms(chunk_ctx))
        results[stage].update(self._analyze_lm_head(chunk_ctx))

        groups = {
            'attn': ['qkv_proj', 'o_proj', 'flash_attn', 'attn_add',
                     'rotary_emb'] ,
            'mlp': ['gate_up_proj', 'down_proj', 'mlp_act', 'mlp_add'],
            'norm': ['attn_norm', 'mlp_norm'],
            'rotary': ['rotary'],
            'embedding': ['embedding'],
            'lm_head': ['lm_head'],

        }

        group_timings = {}

        for g in groups.keys():
            op_time = 0.
            for op in groups[g]:
                op_time += (results[stage][op]['total_time'])
            group_timings[g] = op_time * 1000

        return results, group_timings

    # def _analyzer_non_attn_per_chunk(self, chunk_ctx: ChunkContext, module_names: list) -> dict:
    #     """
    #     Orchestrates the performance analysis of a single standard transformer layer.
    #     This method delegates detailed calculations to specialized helper methods.
    #     """
    #     results = {'prefill': {}, 'decode': {}}
    #     stage = chunk_ctx.stage
    #     total_time = 0.0
    #     # Unpack variables needed by multiple helpers
    #     context_vars = {
    #         'batch_size': chunk_ctx.batch_size,
    #         'chunk_size_act': chunk_ctx.chunk_size_act,
    #         'chunk_size_kv': chunk_ctx.chunk_size_kv,
    #         'tp': self.tp,            
    #     }

    #     # Call helpers to analyze each component of the layer takes tiling and blocks into accouunt 
    #     results[stage].update(self._analyze_std_attention_projections(context_vars))
    #     # results[stage].update(self._analyze_std_attention_core(context_vars))
    #     results[stage].update(self._analyze_std_ffn_layer(context_vars))

    #     # These helpers are generic and can be reused from the deepseek refactoring
    #     results[stage].update(self._analyze_layer_residuals_and_norms(chunk_ctx))
    #     results[stage].update(self._analyze_lm_head(chunk_ctx))

    #     groups_set = {n.split('.')[0] for n in module_names }
    #     ops = [n.split('.')[1] for n in module_names if '.' in n and 'attn' not in n.split('.')[1]]
    #     for op in ops:
    #         total_time += results[stage][op]['inference_time']

    #     return results, total_time

    # # def _analyzer_attn_per_chunk(self, chunk_ctx_list: list[ChunkContext]) -> dict:
    #     """
    #     Orchestrates the performance analysis of a single standard transformer layer.
    #     This method delegates detailed calculations to specialized helper methods.
    #     """
    #     results = {'prefill': {}, 'decode': {}}
    #     total_time = 0.0
    #     for ctx in chunk_ctx_list:
    #         stage = ctx.stage

    #         # Unpack variables needed by multiple helpers
    #         context_vars = {
    #             'batch_size': ctx.batch_size,
    #             'chunk_size_act': ctx.chunk_size_act,
    #             'chunk_size_kv': ctx.chunk_size_kv,
    #             'tp': self.tp,            
    #         }

    #         attn_cost = self._analyze_std_attention_core(context_vars)
    #         # results[stage].update(attn_cost)
    #         total_time += attn_cost['flash_attn']['total_time']

    #     return total_time


    def _analyze_std_attention_projections(self, ctx) -> dict:
        """Analyzes Q, K, V, and Output projections for a standard transformer."""
        # Unpack params
        (b, s_act, tp) = (ctx.batch_size, ctx.chunk_size_act, self.tp)
        (h, h_kv, d_kv) = (self.model.get_hidden_size(), self.model.get_num_kv_heads(self.vllm_config.parallel_config), self.model.get_head_size())
        (w, a, kv) = (self.model.dtype.itemsize, self.model.dtype.itemsize, self.model.dtype.itemsize)
        bw_intra = self.hw.hw_conf.intra_node_bandwidth

        tp_comm_data = a * b * s_act * h
        tp_comm = (2 * tp_comm_data if tp > 4 else tp_comm_data) if tp > 1 else 0

        analysis = {}
        analysis['qkv_proj'] = self._analyze_to_results('qkv_proj', PerformanceMetricsInput(
            ops=3*2 * b * s_act * h * h / tp,
            m_k_n=(s_act, h, h // tp),
            load_weight=3*w * h * h / tp, load_act=a * b * s_act * h,
            store_act=3*a * b * s_act * h / tp))

        # in apply_rope, we load x, cos and sin, then split x and multiply by cos and sin then concat and store (do for both q and k)
        analysis['rotary_emb'] = self._analyze_to_results('rotary_emb', PerformanceMetricsInput(
            ops=b * s_act * h_kv * d_kv / tp,  # max model length
            unit='vector',
            load_weight=2 * w * s_act * d_kv,  # load cos and sin (each of size s_act * d_kv)
            load_act=2 * a * b * s_act * h / tp,
            # load q and k (each of size b * s_act * h) then load split q and k (each of size b * s_act * d_kv)
            store_act=2 * a * b * s_act * h / tp  # store q and k (after split and concat, each of size b * s_act * h)
        )
                                                    )
        analysis['rotary'] = self._analyze_to_results('rotary', PerformanceMetricsInput(
            ops=b * s_act * d_kv / tp,
            load_weight=2 * w * d_kv,  # load inv_freq_expanded
            load_act=2 * a * b * s_act,  # load position_ids_expanded in float32
            store_act=8 * a * b * s_act * d_kv  # see rotary code in qwen2
            )
        )

        analysis['o_proj'] = self._analyze_to_results('o_proj', PerformanceMetricsInput(
            ops=2 * b * s_act * h * h / tp, load_weight=w * h * h / tp, load_act=a * b * s_act * h / tp,
            m_k_n=(s_act, h, h // tp),
            store_act=a * b * s_act * h, tp_comm_time=tp_comm / bw_intra if bw_intra > 0 else 0))
        return analysis

    def _analyze_std_attention_core(self, ctx) -> dict:
        """Analyzes the core attention mechanism (Flash vs. Standard)."""
        # Unpack params
        (b, s_act, s_kv, tp) = (ctx.batch_size, ctx.chunk_size_act, ctx.chunk_size_kv, self.tp)
        (n_heads, kv_heads, d_kv) = (self.model.get_num_attention_heads(self.vllm_config.parallel_config), self.model.get_num_kv_heads(self.vllm_config.parallel_config), self.model.get_head_size())
        (a, kv) = (self.model.dtype.itemsize, self.model.dtype.itemsize)

        qk_ops = 2 * b * s_act * n_heads * d_kv * (s_act + s_kv) / tp
        sv_ops = 2 * b * s_act * n_heads * d_kv * (s_act + s_kv) / tp
        softmax_ops = 5 * b * n_heads / tp * s_act * (s_act + s_kv)

        analysis = {}
        if True:#exec_config.use_flash_attn:
            block_r = min(math.ceil(self.hw.hw_conf.onchip_buffer_size / (kv * d_kv)),
                          d_kv) if (kv * d_kv) > 0 else d_kv
            n_blocks_r = math.ceil(s_act / block_r) if block_r > 0 else 0
            analysis['flash_attn'] = self._analyze_to_results('flash_attn', PerformanceMetricsInput(
                ops=qk_ops + sv_ops + softmax_ops, load_act=a * s_act * d_kv * b * n_heads / tp,
                m_k_n=(s_act, d_kv, (s_act + s_kv)),
                store_act=2 * a * s_act * d_kv * b * n_heads / tp,  # q and k 
                load_kv_cache=kv * n_blocks_r * s_kv * d_kv * b * kv_heads / tp))
        else:
            metrics_qk = PerformanceMetricsInput(ops=qk_ops, load_act=a * b * s_act * n_heads * d_kv / tp,
                                                 m_k_n=(s_act, d_kv, (s_act + s_kv)),
                                                 load_kv_cache=kv * b * s_kv * kv_heads * d_kv / tp,
                                                 #  store_act=a * b * s_act * s_kv * n_heads / tp
                                                 )
            analysis['qk_matmul'] = self._analyze_to_results('qk_matmul', metrics_qk)

            metrics_sv = PerformanceMetricsInput(ops=sv_ops,
                                                 load_act=a * b * s_act * s_kv * n_heads / tp,
                                                 m_k_n=(s_act, (s_act + s_kv), d_kv),
                                                 load_kv_cache=kv * b * s_kv * kv_heads * d_kv / tp,
                                                 store_act=a * b * s_act * n_heads * d_kv / tp)  # h/tp = n_heads*d_kv/tp
            analysis['sv_matmul'] = self._analyze_to_results('sv_matmul', metrics_sv)

            # BUG FIX: Original used metrics_qk here. This now correctly uses metrics_softmax.
            metrics_softmax = PerformanceMetricsInput(ops=softmax_ops,
                                                      unit='vector',
                                                      load_act=a * b * n_heads / tp * s_act * s_kv,
                                                      store_act=a * b * n_heads / tp * s_act * s_kv)
            analysis['softmax'] = self._analyze_to_results('softmax', metrics_softmax)
        return analysis

    def _analyze_std_ffn_layer(self, ctx) -> dict:
        """Analyzes the FFN layer for a standard transformer."""
        # Unpack params
        (b, s_act, tp) = (ctx.batch_size, ctx.chunk_size_act, self.tp)
        (h, i) = (self.model.get_hidden_size(), self.vllm_config.model_config.hf_config.intermediate_size/tp)
        (w, a) = (self.model.dtype.itemsize, self.model.dtype.itemsize)
        bw_intra = self.hw.hw_conf.intra_node_bandwidth

        ffn_ops = 2 * b * s_act * h * i / tp
        ffn_weight = w * h * i / tp
        tp_comm = (2 * a * b * s_act * h if tp > 4 else a * b * s_act * h) if tp > 1 else 0

        analysis = {}
        metrics_gate_up = PerformanceMetricsInput(ops=ffn_ops, load_weight=ffn_weight,
                                                  #   m_k_n=(s_act, h, i / tp),
                                                  load_act=a * b * s_act * h, store_act=a * b * s_act * i / tp)
        analysis['gate_proj'] = self._analyze_to_results('gate_proj', metrics_gate_up)
        analysis['up_proj'] = self._analyze_to_results('up_proj', metrics_gate_up)
        
        analysis['gate_up_proj'] = self._analyze_to_results('gate_up_proj', PerformanceMetricsInput(ops=2*ffn_ops, load_weight=2*ffn_weight,
                                                  load_act=a * b * s_act * h, store_act=2*a * b * s_act * i / tp))

        metrics_down = PerformanceMetricsInput(ops=ffn_ops, load_weight=ffn_weight,
                                               # m_k_n=(s_act, i / tp, h),
                                               load_act=a * b * s_act * i / tp, store_act=a * b * s_act * h,
                                               tp_comm_time=tp_comm / bw_intra if bw_intra > 0 else 0)
        analysis['down_proj'] = self._analyze_to_results('down_proj', metrics_down)

        metrics_act = PerformanceMetricsInput(ops=4*b*s_act*h, load_weight=0, ## assuming silu
                                               load_act=b*s_act*h, store_act=b*s_act*h,
                                               )
        analysis['act_fn'] = self._analyze_to_results('act_fn', metrics_act)

        return analysis
        
    def _analyze_layer_residuals_and_norms(self, chunk_ctx: ChunkContext) -> dict:
        """Analyzes the cost of norms and residual connections."""
        (b, s_act) = (chunk_ctx.batch_size, chunk_ctx.chunk_size_act)
        (h, a) = (self.model.get_hidden_size(), self.model.dtype.itemsize)

        norm_ops = b * h * s_act * 7
        norm_act = a * b * h * s_act
        metrics_norm = PerformanceMetricsInput(ops=norm_ops, unit='vector', load_act=norm_act, store_act=norm_act)

        add_ops = b * h * s_act * 1
        add_act = a * b * h * s_act
        metrics_add = PerformanceMetricsInput(ops=add_ops, unit='vector', load_act=add_act, store_act=add_act)

        analysis = {}
        analysis['attn_norm'] = self._analyze_to_results('attn_norm', metrics_norm)
        analysis['mlp_norm'] = self._analyze_to_results('mlp_norm', metrics_norm)
        analysis['attn_add'] = self._analyze_to_results('attn_add', metrics_add)
        analysis['mlp_add'] = self._analyze_to_results('mlp_add', metrics_add)
        analysis['mlp_act'] = self._analyze_to_results('mlp_act', PerformanceMetricsInput(
            ops=b * h * s_act * 2, unit='vector', load_act=add_act * 2, store_act=add_act))
        return analysis

    def _analyze_lm_head(self, chunk_ctx: ChunkContext) -> dict:
        """Analyzes the final language model head projection."""
        (b, s, h, v) = (chunk_ctx.batch_size, chunk_ctx.chunk_size_act, self.model.get_hidden_size(), self.vllm_config.model_config.get_vocab_size())
        (w, a) = (self.model.dtype.itemsize, self.model.dtype.itemsize)

        metrics_lm = PerformanceMetricsInput(
            ops=2 * b * h * v, load_weight=w * h * v,
            load_act=a * b * h, store_act=a * b * v)

        metrics_embeddings = PerformanceMetricsInput(
            ops=0, load_weight=0,
            load_act=4 * a * b * s, store_act=2 * a * b * s * h)  #
        return {'lm_head': self._analyze_to_results('lm_head', metrics_lm),  # int64 
                'embedding': self._analyze_to_results('embedding', metrics_embeddings)}
    
    def get_workload_timing(self, workload: list, module_names: list):
        
        if f"{workload}" in self.workload_timing_cache.keys():
            return self.workload_timing_cache[f"{workload}"]
        
        # decode_work = [w for w in workload if w[1] == 1]
        # prefil_work = [w for w in workload if w[1] > 1]
        
        total_scheduled = sum([w[1] for w in workload])

        ctx_non_attn = ChunkContext(
            stage='na',
            batch_size=1,
            chunk_size_kv=0,
            chunk_size_act=total_scheduled            
        )
        total_time = 0.0

        # Unpack variables needed by multiple helpers       
        
        groups_set = {n.split('.')[0] for n in module_names }
        for g in groups_set:
            if 'self_attn' in g:
                if not hasattr(self, 'attn_ops'):
                    self.attn_ops = [n.split('.')[1] for n in module_names if 'self_attn.' in n and 'attn' not in n.split('.')[1]]
                attn_projs_performance = self._analyze_std_attention_projections(ctx_non_attn)
                for op in self.attn_ops:
                    total_time += attn_projs_performance[op]['inference_time']
                for w in workload:
                    ctx_attn = ChunkContext(
                                    stage='na',
                                    batch_size=1,
                                    chunk_size_kv=w[0],
                                    chunk_size_act=w[1] 
                                )                    
                    attn_performance = self._analyze_std_attention_core(ctx_attn)
                    total_time += attn_performance['flash_attn']['inference_time']

            if 'mlp' in g:
                if not hasattr(self, 'ffn_ops'):
                    self.ffn_ops = [n.split('.')[1] for n in module_names if 'mlp.' in n]
                
                ffn_performance = self._analyze_std_ffn_layer(ctx_non_attn)
                for op in self.ffn_ops:
                    total_time += ffn_performance[op]['inference_time']
            if 'norm' in g:                
                norm_performance = self._analyze_layer_residuals_and_norms(ctx_non_attn)
                total_time += norm_performance['attn_norm']['inference_time']
        
        self.workload_timing_cache[f"{workload}"] = total_time
        return total_time           

    def get_groups_timing(self, chunk_ctx: ChunkContext):
        # chunk_ctx = ChunkContext(
        #     stage='prefill',
        #     batch_size=infer_config.batch_size,
        #     chunk_size_act=infer_config.chunk_size_act,
        #     chunk_size_kv=infer_config.kv_length,
        # )

        _, group_timings = self._analyzer_per_chunk(chunk_ctx)

        return group_timings

    def get_groups_timing_deepseek(self, chunk_ctx: ChunkContext):
        # chunk_ctx = ChunkContext(
        #     stage='prefill',
        #     batch_size=infer_config.batch_size,
        #     chunk_size_act=infer_config.chunk_size_act,
        #     chunk_size_kv=infer_config.kv_length,
        # )

        _, group_timings = self._analyzer_per_chunk_deepseek_for_calibration(chunk_ctx)

        return group_timings

    def _calculate_prefill_time(self, infer_config: ChunkContext) -> float:
        """Calculates the total time required for the prefill stage."""
        # CORRECTED: Initialize the accumulator variable to zero before the loop.
        prefill_time = 0.0

        chunk_size = min(infer_config.chunk_size_act, self.chunk_size)
        if chunk_size == 0:
            return 0.0

        num_chunks = infer_config.chunk_size_act // chunk_size + (infer_config.chunk_size_act % chunk_size > 0)

        for i in range(num_chunks):
            chunk_ctx = ChunkContext(
                stage='prefill',
                batch_size=infer_config.batch_size,
                chunk_size_act=chunk_size,
                chunk_size_kv=(i + 1) * chunk_size,
            )
            results, group_timings = self._analyzer_per_chunk(chunk_ctx)

            attn_time = ffn_time = moe_time = 0.
            for k, v in results['prefill'].items():
                if 'lm_head' in k or 'embedding' in k:
                    continue
                if 'ffn' in k:
                    ffn_time += v['total_time']
                elif 'moe' in k:
                    moe_time += v['total_time']
                else:
                    attn_time += v['total_time']

            time_per_chunk = 0.0
            is_last_chunk = (i == (num_chunks - 1))
            first_k = getattr(self.model, "first_k_dense_replace", self.model.hf_text_config.num_hidden_layers)

            if is_last_chunk:
                # For the final chunk, we calculate the full layer cost and add the lm_head.
                time_per_chunk = (attn_time * self.model.hf_text_config.num_hidden_layers +
                                  moe_time * (self.model.hf_text_config.num_hidden_layers - first_k) +
                                  ffn_time * first_k +
                                  +  results.get('prefill', {}).get('embedding', {}).get('total_time', 0.0) +
                                  results.get('prefill', {}).get('lm_head', {}).get('total_time', 0.0))
            else:
                # For intermediate chunks, the layer cost is divided by the pipeline degree.
                time_per_chunk = (attn_time * self.model.hf_text_config.num_hidden_layers / self.pp +
                                  moe_time * (self.model.hf_text_config.num_hidden_layers - first_k) +
                                  +  results.get('prefill', {}).get('embedding', {}).get('total_time', 0.0) +
                                  ffn_time * first_k)

            prefill_time += time_per_chunk

        return prefill_time

    def _calculate_decode_time(self, infer_config: ChunkContext) -> Tuple[float, float]:
        """
        Calculates the total and per-token decode time by averaging the
        cost of the first and last generated tokens.
        """
        # Time for the first token after prompt
        first_chunk_ctx = ChunkContext('decode', infer_config.batch_size, 1, infer_config.chunk_size_kv)
        first_decode_time = self._calculate_single_token_decode_time(first_chunk_ctx)

        # Time for the last token to be generated
        last_chunk_ctx = ChunkContext('decode', infer_config.batch_size, 1,
                                      infer_config.chunk_size_kv + infer_config.output_length - 1)
        last_decode_time = self._calculate_single_token_decode_time(last_chunk_ctx)

        if infer_config.output_length == 0:
            return 0.0, 0.0

        total_decode_time = (first_decode_time + last_decode_time) * infer_config.output_length / 2
        decode_time_per_token = total_decode_time / infer_config.output_length

        return total_decode_time, decode_time_per_token

    def _calculate_single_token_decode_time(self, chunk_ctx: ChunkContext) -> float:
        """Calculates the processing time for a single decode token."""
        time = 0
        results, group_timing = self._analyzer_per_chunk(chunk_ctx)
        for k, v in results['decode'].items():
            if 'lm_head' not in k and 'embedding' not in k:
                time += v['total_time'] * self.model.hf_text_config.num_hidden_layers
            else:
                time += v['total_time']  # LM head
        return time

    def _analyze_deepseek_flashMLA(self, chunk_ctx: ChunkContext, tp_comm: float) -> dict:
        """Analyzes Q, K, V, and Output projections for a DeepSeek layer."""
        # Unpack params
        (b, s_act, s_kv) = (chunk_ctx.batch_size, chunk_ctx.chunk_size_act, chunk_ctx.chunk_size_kv)
        (h, d_c, d_R) = (self.model.get_hidden_size(), self.model.kv_lora_rank, self.model.qk_rope_head_dim)
        (d_q, n_heads, d_nope) = (self.model.q_lora_rank, self.model.get_num_attention_heads(self.vllm_config.parallel_config),
                                  self.model.qk_nope_head_dim)
        (w, a, kv) = (self.model.dtype.itemsize, self.model.dtype.itemsize, self.model.dtype.itemsize)

        analysis = {}

        ################### inputs and outputs
        x_sz = a * b * s_act * h
        q_n = a * b * s_act * d_c * n_heads / self.tp
        w_dq_uq_uk = w * (
                    h * d_q + d_q * n_heads * d_nope + n_heads * d_nope * d_c)  # can we absorbed W_UQ with W_UK? w * (h * d_q + d_q*d_c)
        w_dq = w * h * d_q
        w_uq = d_q * d_nope * n_heads / self.tp
        w_uk = d_c * d_nope * n_heads / self.tp
        w_qr = w * d_R * d_q * n_heads / self.tp
        c_q = a * b * s_act * d_q
        q_c = a * b * s_act * d_nope * n_heads / self.tp
        q_R = a * b * s_act * d_R * n_heads / self.tp
        w_dkv = w * d_c * h
        c_kv = kv * b * s_act * d_c
        w_kr = w * d_R * h
        k_R = kv * b * s_act * d_R
        o_hat = a * b * n_heads * s_act * d_c / self.tp
        w_uv = w * n_heads * d_nope * d_c / self.tp
        w_o = w * h * n_heads * d_nope / self.tp
        y = a * b * s_act * h
        o_h = a * b * s_act * d_nope * n_heads / self.tp
        w_router = w * h * self.model.n_routed_experts
        out = a * b * s_act * self.model.n_routed_experts
        #############
        RMS_norm_ops = 2 * b * s_act * d_q
        ROPE_ops = 2 * a * b * s_act * d_R * n_heads / self.tp
        ############## MLA Prolog 

        analysis['mla_q_nope_dq'] = self._analyze_to_results('mla_q_nope_dq', PerformanceMetricsInput(
            ops=2 * b * s_act * h * d_q,
            load_weight=w_dq,
            load_act=x_sz,
            store_act=c_q)
                                                             )

        analysis['mla_cq_rms'] = self._analyze_to_results('mla_cq_rms', PerformanceMetricsInput(
            ops=RMS_norm_ops,
            unit='vector',
            load_act=c_q,
            store_act=c_q)
                                                          )

        analysis['mla_q_uq'] = self._analyze_to_results('mla_q_uq', PerformanceMetricsInput(
            ops=2 * b * s_act * d_q * d_nope * n_heads / self.tp,
            load_weight=w_uq,
            load_act=c_q,
            store_act=q_c)
                                                        )

        analysis['mla_q_uk'] = self._analyze_to_results('mla_q_uk', PerformanceMetricsInput(
            ops=2 * b * s_act * d_c * d_nope * n_heads / self.tp,
            load_weight=w_uk,
            load_act=q_c,
            store_act=q_n)
                                                        )

        analysis['mla_q_qr'] = self._analyze_to_results('mla_q_qr', PerformanceMetricsInput(
            ops=2 * b * s_act * d_q * d_R * n_heads / self.tp,
            load_weight=w_qr,
            load_act=c_q,
            store_act=q_R)
                                                        )

        analysis['mla_q_rope'] = self._analyze_to_results('mla_q_rope', PerformanceMetricsInput(
            ops=ROPE_ops,
            unit='vector',
            load_act=q_R,
            store_act=q_R)
                                                          )

        analysis['mla_k_nope'] = self._analyze_to_results('mla_k_nope', PerformanceMetricsInput(
            ops=2 * b * s_act * h * d_c,
            load_weight=w_dkv,
            load_act=x_sz,
            store_act=c_kv),
                                                          )

        analysis['mla_k_rope'] = self._analyze_to_results('mla_k_rope', PerformanceMetricsInput(
            ops=2 * b * s_act * h * d_R,
            load_weight=w_kr,
            load_act=x_sz,
            store_act=k_R)
                                                          )

        ## flashMLA fused
        qk_ops = 2 * b * (s_kv + s_act) * s_act * (d_c + d_R) * n_heads / self.tp
        sv_ops = 2 * b * (s_kv + s_act) * s_act * d_c * n_heads / self.tp
        softmax_ops = 5 * b * s_act * (s_kv + s_act) * n_heads / self.tp
        block_r = min(math.ceil(self.hw.hw_conf.onchip_buffer_size / (kv * (d_nope + d_R))), d_nope)  # blocks of size
        n_blocks_r = math.ceil((s_act) / block_r) if block_r > 0 else 0

        analysis['flash_attn'] = self._analyze_to_results('flash_attn', PerformanceMetricsInput(
            ops=qk_ops + sv_ops + softmax_ops,  # sofmax running on vector unit :S
            # unit = 'vector',
            load_act=q_n + q_R,
            store_act=max(1, 44 * b ** 2 - 378 * b + 1148) * 2 * o_hat,  # init and save
            load_kv_cache=kv * (6.5 * b ** 2 - 35.5 * b + 94) * n_blocks_r * (s_kv + s_act) * b * (d_c + d_R)))

        ########## part 2
        analysis['uv_proj'] = self._analyze_to_results('uv_proj', PerformanceMetricsInput(
            ops=2 * b * s_act * d_nope * d_c * n_heads / self.tp,  # 2 *b * s_act * d_c * h, #
            load_weight=w_uv,  # 
            load_act=o_hat,
            store_act=o_h
        ))

        analysis['dynamic_quant'] = self._analyze_to_results('dtnamic_quant', PerformanceMetricsInput(
            ops=2 * o_h,  # (finding min, max and round)
            unit='vector',
            load_act=o_h,
            store_act=(32 * (b - 1) ** 4) * o_h // 2,  # half of the size  ########## sefi fix max(1,1200*b - 1600)
        ))

        self.model.dtype.itemsize = 1
        self.model.dtype.itemsize = 1

        analysis['o_proj'] = self._analyze_to_results('o_proj', PerformanceMetricsInput(
            ops=2 * b * n_heads / self.tp * s_act * d_nope * h,  # 2 *b * s_act * d_c * h, #
            load_weight=w_uv + w_o,  # 
            load_act=o_h,
            store_act=y, tp_comm_time=tp_comm))

        self.model.dtype.itemsize = 2
        self.model.dtype.itemsize = 2

        analysis['routing'] = self._analyze_to_results('routing', PerformanceMetricsInput(
            ops=2 * b * s_act * h * self.model.n_routed_experts + out,  # including top_k
            load_weight=w_router,
            load_act=y,
            store_act=out))

        return analysis

    def _analyze_deepseek_attention_projections(self, chunk_ctx: ChunkContext, tp_comm: float) -> dict:
        """Analyzes Q, K, V, and Output projections for a DeepSeek layer."""
        # Unpack params
        (b, s_act) = (chunk_ctx.batch_size, chunk_ctx.chunk_size_act)
        (h, h_kv, d_kv) = (self.model.get_hidden_size(), self.model.kv_lora_rank, self.model.qk_rope_head_dim)
        (q_rank, n_heads, n_nope) = (self.model.q_lora_rank, self.model.get_num_attention_heads(self.vllm_config.parallel_config),
                                     self.model.qk_nope_head_dim)
        (w, a, kv) = (self.model.dtype.itemsize, self.model.dtype.itemsize, self.model.dtype.itemsize)

        analysis = {}
        analysis['q_proj'] = self._analyze_to_results('q_proj', PerformanceMetricsInput(
            ops=2 * b * s_act * h * q_rank, load_weight=w * h * q_rank,
            load_act=a * b * s_act * h, store_act=a * b * s_act * q_rank))

        analysis['kv_proj'] = self._analyze_to_results('kv_proj', PerformanceMetricsInput(
            ops=2 * b * s_act * h * h_kv + 2 * b * s_act * h * d_kv,
            load_weight=w * h * h_kv + w * h * d_kv,
            load_act=a * b * s_act * h, ## qkv_weights_concat =True
            store_act=kv * b * s_act * h_kv + a * b * s_act * d_kv))

        analysis['o_proj'] = self._analyze_to_results('o_proj', PerformanceMetricsInput(
            ops=2 * b * s_act * h * n_heads * n_nope / self.tp,
            load_weight=w * h * n_heads * n_nope / self.tp,
            load_act=a * b * s_act * n_heads * n_nope / self.tp,
            store_act=a * b * s_act * h, tp_comm_time=tp_comm))

        return analysis

    def _analyze_deepseek_attention_core(self, chunk_ctx: ChunkContext) -> dict:
        """Analyzes the core attention mechanism (Flash vs. Standard)."""
        # Unpack params
        (b, s_act, s_kv) = (chunk_ctx.batch_size, chunk_ctx.chunk_size_act, chunk_ctx.chunk_size_kv)
        (h_kv, d_kv, n_nope) = (self.model.kv_lora_rank, self.model.qk_rope_head_dim, self.model.qk_nope_head_dim)
        (q_rank, n_heads) = (self.model.q_lora_rank, self.model.get_num_attention_heads(self.vllm_config.parallel_config))
        (a, kv) = (self.model.dtype.itemsize, self.model.dtype.itemsize)

        qk_ops = 2 * b * s_kv * h_kv * n_heads * n_nope / self.tp + 2 * b * s_act * q_rank * n_heads * \
                 n_nope / self.tp + \
                 2 * b * s_act * q_rank * n_heads * d_kv / self.tp + 2 * b * s_act * n_heads * (
                         n_nope + d_kv) * s_kv / self.tp
        sv_ops = 2 * b * s_act * n_heads * n_nope * s_kv / self.tp + 2 * b * s_kv * h_kv * \
                 n_heads * n_nope / self.tp
        softmax_ops = 5 * b * n_heads / self.tp * s_act * s_kv

        analysis = {}
        if True : #exec_config.use_flash_attn:
            block_r = min(math.ceil(self.hw.hw_conf.onchip_buffer_size / (kv * n_nope)), n_nope)
            n_blocks_r = math.ceil(s_act / block_r) if block_r > 0 else 0
            analysis['flash_attn'] = self._analyze_to_results('flash_attn', PerformanceMetricsInput(
                ops=qk_ops + sv_ops + softmax_ops, load_act=a * b * s_act * q_rank,
                store_act=a * b * s_act * n_nope * n_heads / self.tp * 2,
                load_kv_cache=kv * n_blocks_r * s_kv * b * h_kv * 2 + kv * n_blocks_r * s_kv * b * d_kv))
        else:
            analysis['qk_matmul'] = self._analyze_to_results('qk_matmul', PerformanceMetricsInput(
                ops=qk_ops, load_act=a * b * s_act * q_rank, load_kv_cache=kv * b * s_kv * (h_kv + d_kv),
                store_act=a * b * s_act * s_kv * n_heads / self.tp))
            analysis['sv_matmul'] = self._analyze_to_results('sv_matmul', PerformanceMetricsInput(
                ops=sv_ops, load_act=a * b * s_act * s_kv * n_heads / self.tp, load_kv_cache=kv * b * s_kv * h_kv,
                store_act=a * b * s_act * n_heads * n_nope / self.tp))
            analysis['softmax'] = self._analyze_to_results('softmax', PerformanceMetricsInput(
                ops=softmax_ops, load_act=a * b * n_heads / self.tp * s_act * s_kv,
                store_act=a * b * n_heads / self.tp * s_act * s_kv))

        return analysis

    def _analyze_deepseek_ffn(self, chunk_ctx: ChunkContext) -> dict:
        """Analyzes the standard (non-MoE) FFN layers."""
        (b, s_act) = (chunk_ctx.batch_size, chunk_ctx.chunk_size_act)
        (h, i) = (self.model.get_hidden_size(), self.vllm_config.model_config.hf_config.intermediate_size)
        (w, a) = (self.model.dtype.itemsize, self.model.dtype.itemsize)

        ops = 2 * b * s_act * h * i
        analysis = {}
        analysis['ffn_gate_proj'] = self._analyze_to_results('ffn_gate_proj', PerformanceMetricsInput(
            ops=ops, load_weight=w * h * i, load_act=a * b * s_act * h, store_act=a * b * s_act * i))
        analysis['ffn_up_proj'] = self._analyze_to_results('ffn_up_proj', PerformanceMetricsInput(
            ops=ops, load_weight=w * h * i, load_act=a * b * s_act * h, store_act=a * b * s_act * i))
        analysis['ffn_down_proj'] = self._analyze_to_results('ffn_down_proj', PerformanceMetricsInput(
            ops=ops, load_weight=w * h * i, load_act=a * b * s_act * i, store_act=a * b * s_act * h))
        return analysis

    def _analyze_deepseek_moe(self, chunk_ctx: ChunkContext, tp_comm: float) -> dict:
        """Analyzes the Mixture-of-Experts (MoE) layers."""
        # Unpack params
        (b, s_act) = (chunk_ctx.batch_size, chunk_ctx.chunk_size_act)
        (h, moe_i) = (self.model.get_hidden_size(), self.model.moe_intermediate_size)
        (w, a) = (self.model.dtype.itemsize, self.model.dtype.itemsize)
        (n_exp, n_tok) = (self.model.n_routed_experts, self.model.num_experts_per_tok)
        bw_inter = self.hw.hw_conf.inter_node_bandwidth

        is_prefill = 'prefill' in chunk_ctx.stage
        ep_comm_factor = (n_tok / n_exp) * max(n_exp / self.tp - n_tok,
                                               (self.dp - 1) * \
                                               (n_exp // self.ep)) if is_prefill else (n_tok + 1) / self.tp
        ep_comm = (a * b * h * s_act * ep_comm_factor)
        moe_op_factor = n_tok / n_exp if is_prefill else 1

        ops = 2 * b * s_act * moe_op_factor * h * moe_i
        analysis = {}
        analysis['moe_gate_proj'] = self._analyze_to_results('moe_gate_proj', PerformanceMetricsInput(
            ops=ops, load_weight=w * h * moe_i, load_act=a * b * s_act * moe_op_factor * h,
            store_act=a * b * s_act * moe_op_factor * moe_i, ep_comm_time=ep_comm / bw_inter if bw_inter > 0 else 0))
        analysis['moe_up_proj'] = self._analyze_to_results('moe_up_proj', PerformanceMetricsInput(
            ops=2 * b * (s_act * moe_op_factor) * h * moe_i, load_weight=w * h * moe_i,
            load_act=a * b * s_act * moe_op_factor * h * 0, # since we contact weights
            store_act=a * b * s_act * moe_op_factor * moe_i))
        analysis['moe_down_proj'] = self._analyze_to_results('moe_down_proj', PerformanceMetricsInput(
            ops=2 * b * (s_act * moe_op_factor) * h * moe_i, load_weight=w * h * moe_i,
            load_act=a * b * s_act * moe_op_factor * moe_i, store_act=a * b * s_act * moe_op_factor * h,
            ep_comm_time=ep_comm / bw_inter if bw_inter > 0 else 0, tp_comm_time=tp_comm))
        return analysis

    def _analyze_deepseek_moe_per_die_per_exp(self, chunk_ctx: ChunkContext, 
                                              tp_comm: float) -> dict:
        """Analyzes the Mixture-of-Experts (MoE) layers."""
        # Unpack params
        (b, s_act) = (chunk_ctx.batch_size, chunk_ctx.chunk_size_act)
        (h, moe_i) = (self.model.get_hidden_size(), self.model.moe_intermediate_size)
        (w, a) = (self.model.dtype.itemsize, self.model.dtype.itemsize)
        (n_exp, n_tok) = (self.model.n_routed_experts, self.model.num_experts_per_tok)
        bw_inter = self.hw.hw_conf.inter_node_bandwidth

        is_prefill = 'prefill' in chunk_ctx.stage
        ep_comm_factor = (n_tok / n_exp) * max(n_exp / self.tp - n_tok,
                                               (self.dp - 1) * \
                                               (n_exp // self.ep)) if is_prefill else (n_tok + 1) / self.tp
        ep_comm = (a * b * h * s_act * ep_comm_factor)

        experts_per_die = b  ################################ temporary - please fix191
        ops = 2 * b * s_act * h * moe_i
        activation_ops = 5 * b * s_act * moe_i + b * s_act
        analysis = {}
        analysis['moe_gate_proj'] = self._analyze_to_results('moe_gate_proj', PerformanceMetricsInput(
            ops=ops + activation_ops,
            load_weight=experts_per_die * w * h * moe_i,
            load_act=a * b * s_act * h,
            store_act=a * b * s_act * moe_i, ep_comm_time=ep_comm / bw_inter if bw_inter > 0 else 0))
        analysis['moe_up_proj'] = self._analyze_to_results('moe_up_proj', PerformanceMetricsInput(
            ops=2 * b * (s_act) * h * moe_i,
            load_weight=experts_per_die * w * h * moe_i,
            load_act=a * b * s_act * h * 0, # since we contact weights
            store_act=a * b * s_act * moe_i))
        analysis['moe_down_proj'] = self._analyze_to_results('moe_down_proj', PerformanceMetricsInput(
            ops=2 * b * (s_act) * h * moe_i, load_weight=experts_per_die * w * h * moe_i,
            load_act=a * b * s_act * moe_i, store_act=a * b * s_act * h,
            ep_comm_time=ep_comm / bw_inter if bw_inter > 0 else 0, tp_comm_time=tp_comm))
        return analysis
    

def test():
    from conf.hardware_config import HardwareTopology, DeviceType
    from conf.model_config import ModelConfig, ModelType

    try:
        from math import log
        model_type = ModelType["DEEPSEEK_V3"]  # ["QWEN2_5_7B"]
        sizes = [1, 1, 1]  # [1024, 4096, 8192, 16*1024, 32*1024] #], 24*1024, 32*1024, 64*1024, 128*1024]
        # kv_sizes = [64*1024, 64*1024, 64*1024, 64*1024, 64*1024] # [1024, 4096, 8192,  16*1024, 32*1024] 
        # kv_sizes = [1024, 4096, 8192, 16*1024, 32*1024]
        kv_sizes = [2048, 4096, 8192]
        result = []
        for s_act, s_kv in zip(sizes, kv_sizes):
            inpt_length = s_act
            kv_length = s_kv
            outpt_length = 1  # if s > 24*1024 else 16*1024*(s==16*1024) + 8*1024*(s==24*1024)
            batch_size = 2

            compute_util = 0.54  # 0.66
            mem_bw_util = 0.9
            model_config = ModelConfig.create_model_config(model_type)
            hw_topology = HardwareTopology.create(number_of_ranks=2,
                                                  npus_per_rank=8,
                                                  ascend_type=DeviceType.ASCEND910C,  # DeviceType.ASCEND910B4,
                                                  compute_util=compute_util, mem_bw_util=mem_bw_util)

            
            p_config = ParallelismConfig(pp=1, tp=1, chunk_size=s_act)
            cm = CostModel(model_config, hw_topology, p_config)

            # infer_config = InferenceConfig(batch_size=batch_size, input_length=inpt_length,
            #                                output_length=outpt_length, kv_length=kv_length)
            # exec_config = ExecutionConfig(chunk_size=s_act, use_flash_attn=True, qkv_weights_concat=True)
            infer_config = ChunkContext(
                stage='prefill',
                batch_size=batch_size,
                chunk_size_act=inpt_length,
                # chunk_idx=0,
                output_length=outpt_length,
                chunk_size_kv=1024
            )
            

            timings = cm.get_groups_timing_deepseek(infer_config)
            print(
                f"For decode - with batch {batch_size}, kv_len {s_kv}: attn_1 {timings['attn_1']:.2f}, attn_2 {timings['attn_2']:.2f}, total {timings['attn_1'] + timings['attn_2']:.2f} ")
            # result += [cm.estimate_cost_of(infer_config,ParallelismConfig(pp=1, tp=1), exec_config)]

            # p_config_prefill = ParallelismConfig(pp=8, tp=4, ep=32, dp=8)
            # p_config_decode = ParallelismConfig(pp=8, tp=4, ep=320, dp=80)
            # chunk_ctx = ChunkContext(batch_size=5, chunk_size_act=300,
            #                  chunk_idx=0, stage="prefill", pp_stage_idx=0)
            # cm.estimate_cost_per_chunk_per_pp_stage_deepseek(chunk_ctx, p_config_prefill)
            # result += [cm.estimate_cost_of_deepseek(infer_config, p_config_prefill, p_config_decode, exec_config)]

        # [print(f"TTFT {res['TTFT']}, TBT {res['TBT']}")  for res in result]

        sizes = [32, 128, 256, 384, 512,
                 640]  # [1024, 4096, 8192, 16*1024, 32*1024] #], 24*1024, 32*1024, 64*1024, 128*1024]
        # kv_sizes = [64*1024, 64*1024, 64*1024, 64*1024, 64*1024] # [1024, 4096, 8192,  16*1024, 32*1024] 
        # kv_sizes = [1024, 4096, 8192, 16*1024, 32*1024]
        kv_sizes = [0, 0, 0, 0, 0, 0]
        result = []
        for s_act, s_kv in zip(sizes, kv_sizes):
            inpt_length = s_act
            kv_length = s_kv
            outpt_length = 1  # if s > 24*1024 else 16*1024*(s==16*1024) + 8*1024*(s==24*1024)
            batch_size = 4

            compute_util = 0.54  # 0.66
            mem_bw_util = 0.9
            model_config = ModelConfig.create_model_config(model_type)
            hw_topology = HardwareTopology.create(number_of_ranks=2,
                                                  npus_per_rank=8,
                                                  ascend_type=DeviceType.ASCEND910C,  # DeviceType.ASCEND910B4,
                                                  compute_util=compute_util, mem_bw_util=mem_bw_util)

            p_config = ParallelismConfig(pp=1, tp=1, chunk_size=s_act)
            cm = CostModel(model_config, hw_topology, p_config)

            # infer_config = InferenceConfig(batch_size=batch_size, input_length=inpt_length,
            #                                output_length=outpt_length, kv_length=kv_length)
            # exec_config = ExecutionConfig(chunk_size=s_act, use_flash_attn=True, qkv_weights_concat=True)
            infer_config = ChunkContext(
                stage='prefill',
                batch_size=batch_size,
                chunk_size_act=inpt_length,
                output_length=outpt_length,
                chunk_size_kv=1024
            )           
            

            timings = cm.get_groups_timing_deepseek(infer_config)
            print(f"For {batch_size} experts per die, seq_len {s_act}: MoE {timings['moe']:.2f}")
        return result

    except Exception as e:
        print(f"Error analyzing {model_type}: {e}")


if __name__ == "__main__":
    import os

    os.environ["PYTHONPATH"] = "./"
    res = test()
    print(f"results {res}")

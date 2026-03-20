from typing import Tuple
import pandas as pd
import logging
import math
import os
from conf.config import Config
from conf.common import SEC_2_US, MIN_ROUTED_EXPERT_PER_DIE, MEMORY_THRESHOLD_RATIO, MS_2_US, BYTE_2_GB
from src.search.base import BaseSearch
from src.model.register import get_model, get_attention_family


class AfdSearch(BaseSearch):
    '''
    Description:
        The AFD search algorithm.
        It is used to search the optimal attention batch size, 
        attention die count, FFN die count for the model used AFD serving.
    Attributes:
        config: The configuration of the AFD search task.
        perf_afd_results: The performance results of the AFD search,
        it contains the following columns:
            attn_bs: Attention batch size for per micro batch, int.
            ffn_bs: FFN batch size for per micro batch, float.
            kv_len: KV cache length, int.
            attn_die: The number of Attention die, int.
            ffn_die: The number of FFN die, int.
            total_die: The number of Total die, int.
            attn_time: Attention time for per layer per micro batch (μs), float.
            moe_time: MoE time for per layer per micro batch (μs), float.
            dispatch_time: Dispatch time for per layer per micro batch (μs), float.
            combine_time: Combine time for per layer per micro batch (μs), float.
            commu_time: Communication time for per layer per micro batch (μs), float.
            e2e_time: End-to-end time (ms), float.
            e2e_time_per_dense_layer: End-to-end time for per dense layers (μs), float.
            e2e_time_per_moe_layer: End-to-end time for per MoE layers (μs), float.
            throughput: Throughput (tokens/second), float.
    '''
    ATTN_DIE_MULTIPLIER = 7

    def __init__(self, config: Config):
        super().__init__(config)
        self.perf_afd_results = []

    def search_attn_bs(self) -> Tuple[float, int]:
        """
        Description:
            Search the maximum attention batch size that satisfies the latency and memory constraints.
        Returns:
            attn_time: Attention time for per layer per micro batch (μs), float.
            attn_bs: Attention batch size for per micro batch, int.
        """
        attn_bs_min, attn_bs_max = self.config.min_attn_bs, self.config.max_attn_bs

        while attn_bs_max - attn_bs_min > 1:
            attn_bs = (attn_bs_min + attn_bs_max) // 2
            self.config.attn_bs = attn_bs
            model = get_model(self.config)
            attn = model["attn"]
            attn()
            attn_time = attn.e2e_time * SEC_2_US
            attn_latency_constraint = (
                self.config.tpot * MS_2_US / self.config.model_config.num_layers * 
                (1 + self.config.multi_token_ratio) / self.config.micro_batch_num
            )

            if get_attention_family(self.config.model_type) == "MLA":
                kv_size, attn_static_memory, _, _ = self.compute_MLA_memory_size(self.config.model_config, attn_bs)
            elif get_attention_family(self.config.model_type) == "GQA":
                kv_size, attn_static_memory, _, _ = self.compute_GQA_memory_size(self.config.model_config, attn_bs)
            attn_memory = kv_size * self.config.micro_batch_num + attn_static_memory

            if attn_time > attn_latency_constraint or attn_memory > self.config.aichip_config.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO:
                attn_bs_max = attn_bs
            else:
                attn_bs_min = attn_bs

        if attn_time > attn_latency_constraint or attn_memory > self.config.aichip_config.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO:
            attn_bs = attn_bs_min
            self.config.attn_bs = attn_bs
            attn = get_model(self.config)["attn"]
            attn()
            attn_time = attn.e2e_time * SEC_2_US
        return attn_time, attn_bs

    def search(self, attn_time: float, attn_bs: int):
        '''
        Description:
            Search the optimal attention batch size, 
            attention die count, FFN die count for the model used AFD serving.
        Args:
            attn_time: Attention time for per layer per micro batch (μs), float.
            attn_bs: Attention batch size for per micro batch, int.
        '''
        if get_attention_family(self.config.model_type) == "MLA":
            kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_MLA_memory_size(self.config.model_config, attn_bs)
        elif get_attention_family(self.config.model_type) == "GQA":
            kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_GQA_memory_size(self.config.model_config, attn_bs)

        # compute per dense layer time
        if self.config.model_config.num_layers > self.config.model_config.num_moe_layers:
            self.config.attn_bs = attn_bs * self.config.micro_batch_num
            self.config.ffn_bs = self.config.attn_bs
            model = get_model(self.config)
            attn = model["attn"]
            mlp = model["mlp"]
            attn()
            mlp()
            dense_attn_time = attn.e2e_time * SEC_2_US
            mlp_time = mlp.e2e_time * SEC_2_US
            e2e_time_per_dense_layer = dense_attn_time + mlp_time
        else:
            e2e_time_per_dense_layer = 0.0

        # search ffn_die, attn_die
        self.config.attn_bs = attn_bs
        min_ffn_die, max_ffn_die, ffn_die_step = self.config.min_die, self.config.max_die + 1, self.config.die_step
        for ffn_die in range(min_ffn_die, max_ffn_die, ffn_die_step):
            routed_expert_per_die = self.config.model_config.n_shared_experts + max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.config.model_config.n_routed_experts / ffn_die)
            )
            ffn_static_memory = per_router_expert_memory * routed_expert_per_die
            self.config.routed_expert_per_die = routed_expert_per_die
            if ffn_static_memory > self.config.aichip_config.aichip_memory * MEMORY_THRESHOLD_RATIO:
                continue

            for attn_die in range(ffn_die, self.ATTN_DIE_MULTIPLIER * ffn_die, ffn_die_step):
                total_die = ffn_die + attn_die
                if total_die % self.config.aichip_config.num_dies_per_node != 0:
                    continue
                self.config.ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok * attn_die / ffn_die
                self.config.attn_die = attn_die
                self.config.ffn_die = ffn_die
                model = get_model(self.config)
                moe = model["moe"]
                moe()
                # compute per moe layer time
                moe_time = moe.e2e_time * SEC_2_US
                dispatch_time = moe.dispatch_time * SEC_2_US
                combine_time = moe.combine_time * SEC_2_US
                commu_time = moe.commu_time * SEC_2_US
                e2e_time_per_moe_layer = max(
                    attn_time + moe_time + commu_time,
                    max(attn_time, moe_time) * self.config.micro_batch_num
                )

                e2e_time = (
                    e2e_time_per_dense_layer * self.config.model_config.first_k_dense_replace + 
                    e2e_time_per_moe_layer * self.config.model_config.num_moe_layers
                )
                throughput = (
                    attn_bs * self.config.micro_batch_num * attn_die / total_die / e2e_time * 
                    (1 + self.config.multi_token_ratio) * SEC_2_US
                )

                logging.info(f"-------AFD Search Result:-------")
                logging.info(
                    f"attn_bs: {attn_bs}, ffn_bs: {self.config.ffn_bs}, "
                    f"kv_len: {self.config.kv_len}, attn_die: {attn_die}, "
                    f"ffn_die: {ffn_die}, total_die: {total_die}, "
                    f"attn_time: {attn_time:.2f}us, moe_time: {moe_time:.2f}us, "
                    f"dispatch_time: {dispatch_time:.2f}us, combine_time: {combine_time:.2f}us, "
                    f"commu_time: {commu_time:.2f}us, e2e_time: {e2e_time:.2f}ms, "
                    f"e2e_time_per_dense_layer: {e2e_time_per_dense_layer:.2f}us, "
                    f"e2e_time_per_moe_layer: {e2e_time_per_moe_layer:.2f}us, throughput: {throughput:.2f} tokens/die/s, "
                    f"kv_size:{kv_size} GB, attn_static_memory:{attn_static_memory} GB, "
                    f"mlp_static_memory:{mlp_static_memory} GB, ffn_static_memory:{ffn_static_memory} GB"
                )

                self.perf_afd_results.append([
                    attn_bs, self.config.ffn_bs, self.config.kv_len, attn_die, ffn_die, total_die,
                    attn_time, moe_time, dispatch_time, combine_time, commu_time, e2e_time / MS_2_US,
                    e2e_time_per_dense_layer, e2e_time_per_moe_layer, throughput,
                    kv_size, attn_static_memory, mlp_static_memory, ffn_static_memory
                ])

        columns = [
            'attn_bs', 'ffn_bs', 'kv_len', 'attn_die', 'ffn_die', 'total_die',
            'attn_time(us)', 'moe_time(us)', 'dispatch_time(us)', 'combine_time(us)', 'commu_time(us)', 'e2e_time(ms)',
            'e2e_time_per_dense_layer(us)', 'e2e_time_per_moe_layer(us)', 'throughput(tokens/die/s)',
            'kv_size(GB)', 'attn_static_memory(GB)', 'mlp_static_memory(GB)', 'ffn_static_memory(GB)'
        ]
        df = pd.DataFrame(self.perf_afd_results, columns=columns)

        result_dir = f"data/afd/mbn{self.config.micro_batch_num}/"
        file_name = f"{self.config.device_type.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        os.makedirs(result_dir, exist_ok=True)
        result_path = result_dir + file_name
        df.to_csv(result_path, index=False)

        df_best = df.sort_values(by=['throughput(tokens/die/s)'], ascending=False).drop_duplicates(subset=['total_die'])
        df_best = df_best.sort_values(by=['total_die'], ascending=True)
        best_result_dir = result_dir + "best/"
        best_file_name = f"{self.config.device_type.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        os.makedirs(best_result_dir, exist_ok=True)
        best_result_path = best_result_dir + best_file_name
        df_best.to_csv(best_result_path, index=False)

    def deployment(self):
        attn_time, attn_bs= self.search_attn_bs()
        self.search(attn_time, attn_bs)

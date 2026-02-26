import pandas as pd
import logging
import math
import os
from conf.config import Config
from conf.common import SEC_2_US, MEMORY_THRESHOLD_RATIO, MS_2_US, BYTE_2_GB, US_2_MS
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

    def __init__(self, config: Config):
        super().__init__(config)

    def search_bs(self) -> list[list[float]]:
        perf_afd_results = []
        for attn_die in range(self.config.min_die1, self.config.max_die1, self.config.die_step1):
            for ffn_die in range(self.config.min_die2, self.config.max_die2, self.config.die_step2):
                total_die = attn_die + ffn_die
                if self.config.device_type1 == self.config.device_type2:
                    if total_die % self.config.aichip_config1.num_dies_per_node != 0:
                        continue
                if ffn_die < 64:
                    # router + shared expert
                    routed_expert_per_die = (
                        self.config.model_config.n_shared_experts +
                        math.ceil(self.config.model_config.n_routed_experts / ffn_die)
                    )
                elif ffn_die >= 64 and ffn_die < 128:
                    # router + shared expert + 1 redundant expert for EPLB
                    routed_expert_per_die = (
                        self.config.model_config.n_shared_experts +
                        math.ceil(self.config.model_config.n_routed_experts / ffn_die) + 1
                    )
                else:
                    # router + 1 redundant experts for EPLB
                    routed_expert_per_die = math.ceil(self.config.model_config.n_routed_experts / ffn_die) + 1
                self.config.routed_expert_per_die = routed_expert_per_die
                self.config.ffn_die = ffn_die
                self.config.attn_die = attn_die
                for attn_bs in range(self.config.min_attn_bs, self.config.max_attn_bs):
                    self.config.attn_bs = attn_bs
                    self.config.ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok * attn_die / ffn_die
                    model = get_model(self.config)
                    attn = model["attn"]
                    attn()
                    attn_time = attn.e2e_time * SEC_2_US
                    moe = model["moe"]
                    moe()
                    commu_time = moe.commu_time * SEC_2_US
                    dispatch_time = moe.dispatch_time * SEC_2_US
                    combine_time = moe.combine_time * SEC_2_US
                    moe_time = moe.e2e_time * SEC_2_US

                    # compute moe layer time
                    e2e_time_per_moe_layer = max(
                        attn_time + moe_time + commu_time,
                        max(commu_time, max(attn_time, moe_time)) * self.config.micro_batch_num
                    )

                    # compute memory
                    if get_attention_family(self.config.model_type) == "MLA":
                        kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_MLA_memory_size(self.config.model_config, attn_bs)
                    elif get_attention_family(self.config.model_type) == "GQA":
                        kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_GQA_memory_size(self.config.model_config, attn_bs)
                    ffn_dynamic_memory = (
                        self.config.ffn_bs * self.config.model_config.hidden_size *
                        self.config.seq_len * self.config.model_config.num_layers * BYTE_2_GB
                    )
                    ffn_static_memory = per_router_expert_memory * routed_expert_per_die
                    attn_memory = kv_size * self.config.micro_batch_num + attn_static_memory + mlp_static_memory
                    ffn_memory = ffn_dynamic_memory + ffn_static_memory
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
                        dense_mlp_time = mlp.e2e_time * SEC_2_US
                        e2e_time_per_dense_layer = dense_attn_time + dense_mlp_time
                    else:
                        dense_mlp_time = 0.0
                        e2e_time_per_dense_layer = 0.0
                    self.config.attn_bs = attn_bs
                    self.config.ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok * attn_die / ffn_die
                    # compute e2e time
                    e2e_time = (
                        e2e_time_per_moe_layer * (self.config.model_config.num_moe_layers + self.config.seq_len - 1) +
                        e2e_time_per_dense_layer * self.config.model_config.first_k_dense_replace
                    )
                    throughput = (
                        attn_bs * self.config.micro_batch_num * attn_die / total_die / e2e_time * 
                        (1 + self.config.multi_token_ratio) * SEC_2_US
                    )
                    # check latency and memory constraints
                    latency_constraint = (
                        self.config.tpot * MS_2_US * (1 + self.config.multi_token_ratio)
                    )
                    # check latency and memory constraints
                    attn_memory_constraint = self.config.aichip_config1.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO
                    ffn_memory_constraint = self.config.aichip_config2.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO
                    if e2e_time > latency_constraint or attn_memory > attn_memory_constraint or ffn_memory > ffn_memory_constraint:
                        break

                    e2e_time = e2e_time * US_2_MS

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
                        f"kv_size: {kv_size} GB, attn_static_memory: {attn_static_memory} GB, "
                        f"mlp_static_memory: {mlp_static_memory} GB, ffn_static_memory: {ffn_static_memory} GB"
                    )
                    perf_afd_results.append([
                        attn_bs,
                        round(self.config.ffn_bs, 1),
                        self.config.kv_len,
                        attn_die,
                        ffn_die,
                        total_die,
                        round(attn_time, 1),
                        round(moe_time, 1),
                        round(dispatch_time, 1),
                        round(combine_time, 1),
                        round(commu_time, 1),
                        round(e2e_time, 1),
                        round(e2e_time_per_dense_layer, 1),
                        round(e2e_time_per_moe_layer, 1),
                        round(throughput, 1),
                        round(kv_size, 1),
                        round(attn_static_memory, 1),
                        round(mlp_static_memory, 1),
                        round(ffn_static_memory, 1)
                    ])

        unique_results = {}
        for result in perf_afd_results:
            total_die = result[5]
            throughput = result[14]
            if total_die not in unique_results or throughput > unique_results[total_die][14]:
                unique_results[total_die] = result
        perf_afd_results = list(unique_results.values())
        return perf_afd_results


    def deployment(self):
        perf_afd_results = self.search_bs()

        columns = [
            'attn_bs', 'ffn_bs', 'kv_len', 'attn_die', 'ffn_die', 'total_die',
            'attn_time(us)', 'moe_time(us)', 'dispatch_time(us)', 'combine_time(us)', 'commu_time(us)', 'e2e_time(ms)',
            'e2e_time_per_dense_layer(us)', 'e2e_time_per_moe_layer(us)', 'throughput(tokens/die/s)',
            'kv_size(GB)', 'attn_static_memory(GB)', 'mlp_static_memory(GB)', 'ffn_static_memory(GB)'
        ]

        df = pd.DataFrame(perf_afd_results, columns=columns)

        result_dir = f"data/afd/mbn{self.config.micro_batch_num}/"
        file_name = f"{self.config.device_type1.name}-{self.config.device_type2.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        os.makedirs(result_dir, exist_ok=True)
        result_path = result_dir + file_name
        df.to_csv(result_path, index=False)

        df_best = df.sort_values(by=['throughput(tokens/die/s)'], ascending=False).drop_duplicates(subset=['total_die'])
        df_best = df_best.sort_values(by=['total_die'], ascending=True)
        best_result_dir = result_dir + "best/"
        best_file_name = f"{self.config.device_type1.name}-{self.config.device_type2.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        os.makedirs(best_result_dir, exist_ok=True)
        best_result_path = best_result_dir + best_file_name
        df_best.to_csv(best_result_path, index=False)

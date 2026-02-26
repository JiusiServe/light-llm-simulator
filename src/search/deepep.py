import logging
import math
import os
import pandas as pd
from conf.common import US_2_MS, SEC_2_US, BYTE_2_GB, MEMORY_THRESHOLD_RATIO, MS_2_SEC, MS_2_US
from src.search.base import BaseSearch
from src.model.register import get_model, get_attention_family
from conf.config import Config
from conf.hardware_config import HWConf


class DeepEpSearch(BaseSearch):
    '''
    Description:
        The DeepEP search algorithm.
        It is used to search the optimal attention batch size for the model used DeepEP serving.
    Attributes:
        config: The configuration of the DeepEP search task.
        perf_deepep_results: The performance results of the DeepEP search,
        it contains the following columns:
            attn_bs: Attention batch size, int.
            ffn_bs: FFN batch size(tokens per die), float.
            kv_len: KV cache length, int.
            total_die: The number of Total die, int.
            attn_time: Attention time for per layer (μs), float.
            mlp_time: MLP time for per dense layer (μs), float.
            moe_time: MoE time for per layer (μs), float.
            commu_time: Communication time for per layer (μs), float.
            dispatch_time: Dispatch time for per layer (μs), float.
            combine_time: Combine time for per layer (μs), float.
            e2e_time: End-to-end time (ms), float.
            throughput: Throughput (tokens/second), float.
    '''
    def __init__(self, config: Config):
        super().__init__(config)

    def search_bs(self, min_die, max_die, die_step, attn_bs_min, attn_bs_max) -> list[list[float]]:
        perf_deepep_results = []
        for total_die in range(min_die, max_die, die_step):
            if total_die < 64:
                # router + shared expert
                routed_expert_per_die = (
                    self.config.model_config.n_shared_experts +
                    math.ceil(self.config.model_config.n_routed_experts / total_die)
                )
            elif total_die >= 64 and total_die < 128:
                # router + shared expert + 1 redundant expert for EPLB
                routed_expert_per_die = (
                    self.config.model_config.n_shared_experts +
                    math.ceil(self.config.model_config.n_routed_experts / total_die) + 1
                )
            else:
                # router + 1 redundant experts for EPLB
                routed_expert_per_die = math.ceil(self.config.model_config.n_routed_experts / total_die) + 1
            for attn_bs in range(attn_bs_min, attn_bs_max):
            # for attn_bs in [24, 28, 30, 48, 72, 90, 144]:
                if get_attention_family(self.config.model_type) == "MLA":
                    kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_MLA_memory_size(self.config.model_config, attn_bs)
                elif get_attention_family(self.config.model_type) == "GQA":
                    kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_GQA_memory_size(self.config.model_config, attn_bs)
                latency_constraint = (
                    self.config.tpot * MS_2_US * (1 + self.config.multi_token_ratio) / self.config.micro_batch_num
                )
                ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok
                self.config.attn_bs = attn_bs
                self.config.ffn_bs = ffn_bs
                self.config.attn_die = total_die
                self.config.ffn_die = total_die
                self.config.routed_expert_per_die = routed_expert_per_die
                model = get_model(self.config)
                embedding = model["embedding"]
                embedding()
                embedding_time = embedding.e2e_time * SEC_2_US
                lm_head = model["lm_head"]
                lm_head()
                lm_head_time = lm_head.e2e_time * SEC_2_US
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
                    max(commu_time, max(attn_time, moe_time)) *
                    self.config.micro_batch_num
                )
                # moe layer time + embedding time + lm head time + MTP layer time
                e2e_time = e2e_time_per_moe_layer * (self.config.model_config.num_moe_layers + self.config.seq_len - 1) + embedding_time + lm_head_time

                # compute memory
                ffn_dynamic_memory = (
                    ffn_bs * self.config.model_config.hidden_size * self.config.seq_len *
                    self.config.model_config.num_layers * BYTE_2_GB
                )
                ffn_static_memory = per_router_expert_memory * routed_expert_per_die
                total_memory = (
                    kv_size * self.config.micro_batch_num + attn_static_memory +
                    mlp_static_memory + ffn_dynamic_memory + ffn_static_memory
                )

                if self.config.model_config.num_layers > self.config.model_config.num_moe_layers:
                    # compute dense layer time
                    mlp = model["mlp"]
                    mlp()
                    mlp_time = mlp.e2e_time * SEC_2_US
                    e2e_time_per_dense_layer = attn_time + mlp_time
                    e2e_time = e2e_time + e2e_time_per_dense_layer * self.config.model_config.first_k_dense_replace
                else:
                    mlp_time = 0.0
                    e2e_time_per_dense_layer = 0.0

                # check latency and memory constraints
                if e2e_time > latency_constraint or total_memory > self.config.aichip_config1.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO:
                    break

                e2e_time = e2e_time * US_2_MS
                throughput = attn_bs / e2e_time / MS_2_SEC * (1 + self.config.multi_token_ratio)

                logging.info(f"-------DeepEP Search Result:-------")
                logging.info(
                    f"attn_bs:{attn_bs}, ffn_bs:{ffn_bs}, "
                    f"kv_len:{self.config.kv_len}, total_die:{total_die}, "
                    f"attn_time:{attn_time} us, moe_time:{moe_time} us, "
                    f"commu_time:{commu_time} us, dispatch_time:{dispatch_time} us, combine_time:{combine_time} us, "
                    f"e2e_time:{e2e_time} ms, throughput:{throughput} tokens/die/s, "
                    f"e2e_time_per_dense_layer:{e2e_time_per_dense_layer} us, e2e_time_per_moe_layer:{e2e_time_per_moe_layer} us, "
                    f"kv_size:{kv_size} GB, attn_static_memory:{attn_static_memory} GB, "
                    f"mlp_static_memory:{mlp_static_memory} GB, ffn_static_memory:{ffn_static_memory} GB"
                )

                perf_deepep_results.append([
                    attn_bs,
                    ffn_bs,
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
        for result in perf_deepep_results:
            total_die = result[2]
            throughput = result[11]
            if total_die not in unique_results or throughput > unique_results[total_die][11]:
                unique_results[total_die] = result
        perf_deepep_results = list(unique_results.values())
        return perf_deepep_results

    def deployment(self):
        min_die1, max_die1, die_step1 = self.config.min_die1, self.config.max_die1 + 1, self.config.die_step1
        min_die2, max_die2, die_step2 = self.config.min_die2, self.config.max_die2 + 1, self.config.die_step2
        self.config.aichip_config1 = HWConf.create(self.config.device_type1)
        self.config.aichip_config2 = HWConf.create(self.config.device_type1)
        perf_deepep_results1 = self.search_bs(min_die1, max_die1, die_step1, self.config.min_attn_bs, self.config.max_attn_bs + 1)
        self.config.aichip_config1 = HWConf.create(self.config.device_type2)
        self.config.aichip_config2 = HWConf.create(self.config.device_type2)
        perf_deepep_results2 = self.search_bs(min_die2, max_die2, die_step2, self.config.min_attn_bs, self.config.max_attn_bs + 1)
        perf_deepep_results = []
        for result1 in perf_deepep_results1:
            for result2 in perf_deepep_results2:
                total_die = result1[2] + result2[2]
                result = result1 + result2
                avg_bs = (result1[2] * result1[0] + result2[2] * result2[0]) / total_die
                avg_e2e_time = (result1[2] * result1[8] + result2[2] * result2[8]) / total_die
                avg_throughput = (result1[2] * result1[11] + result2[2] * result2[11]) / total_die
                result.append(round(avg_bs, 1))
                result.append(self.config.kv_len)
                result.append(total_die)
                result.append(round(avg_e2e_time, 1))
                result.append(round(avg_throughput, 1))
                perf_deepep_results.append(result)

        columns = [
            'attn_bs1', 'ffn_bs1', 'total_die1', 'attn_time1(us)', 
            'moe_time1(us)', 'dispatch_time1(us)', 'combine_time1(us)', 'commu_time1(us)', 'e2e_time1(ms)',
            'e2e_time_per_dense_layer1(us)', 'e2e_time_per_moe_layer1(us)', 'throughput1(tokens/die/s)',
            'kv_size1(GB)', 'attn_static_memory1(GB)', 'mlp_static_memory1(GB)', 'ffn_static_memory1(GB)',
            'attn_bs2', 'ffn_bs2', 'total_die2', 'attn_time2(us)', 
            'moe_time2(us)', 'dispatch_time2(us)', 'combine_time2(us)', 'commu_time2(us)', 'e2e_time2(ms)',
            'e2e_time_per_dense_layer2(us)', 'e2e_time_per_moe_layer2(us)', 'throughput2(tokens/die/s)',
            'kv_size2(GB)', 'attn_static_memory2(GB)', 'mlp_static_memory2(GB)', 'ffn_static_memory2(GB)',
            'avg_bs', 'kv_len', 'total_die', 'avg_e2e_time', 'throughput(tokens/die/s)'
        ]

        df = pd.DataFrame(perf_deepep_results, columns=columns).sort_values(by=['total_die', 'total_die1', 'total_die2'], ascending=[True, True, True])
        result_dir = f"data/deepep/"
        file_name = f"{self.config.device_type1.name}-{self.config.device_type2.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        os.makedirs(result_dir, exist_ok=True)
        result_path = result_dir + file_name
        df.to_csv(result_path, index=False)

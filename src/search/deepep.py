import logging
import math
import pandas as pd
from conf.common import MIN_ROUTED_EXPERT_PER_DIE, US_2_MS, SEC_2_US, BYTE_2_GB, MEMORY_THRESHOLD_RATIO, MS_2_SEC, MS_2_US
from src.search.base import BaseSearch
from src.search.config import SearchConfig
from src.module.register import get_model


class DeepEpSearch(BaseSearch):
    MIN_TOTAL_DIE = 16
    MAX_TOTAL_DIE = 769
    DIE_STEP = 16
    MIN_ATTN_BS = 2
    MAX_ATTN_BS = 1000

    def __init__(
        self,
        model_type,
        model_config,
        aichip_config,
        kv_len: int,
        seq_len: int,
        multi_token_ratio: float,
        latency: float,
        micro_batch_num: int
    ):
        super().__init__(
            model_type,
            model_config,
            aichip_config,
            kv_len,
            seq_len,
            multi_token_ratio,
            latency * MS_2_US
        )
        self.micro_batch_num = micro_batch_num
        self.perf_deepep_results = []

    def search_bs(self):
        search_mode = "DeepEp"
        for total_die in range(self.MIN_TOTAL_DIE, self.MAX_TOTAL_DIE, self.DIE_STEP):
            routed_expert_per_die = max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.model_config.n_routed_experts / total_die)
            )
            attn_bs_min, attn_bs_max = self.MIN_ATTN_BS, self.MAX_ATTN_BS
            e2e_time = 0.0
            total_memory = 0.0

            # search max attention bs
            while attn_bs_max - attn_bs_min > 1:
                attn_bs = (attn_bs_min + attn_bs_max) // 2
                kv_size, attn_static_memory, per_router_expert_memory = self.compute_memory_size(
                    self.model_config, attn_bs
                )
                latency_constraint = (
                    self.latency / self.micro_batch_num * (1 + self.multi_token_ratio)
                )
                attn_die, ffn_die = total_die, total_die
                search_config = SearchConfig(
                    self.model_config,
                    search_mode,
                    attn_bs,
                    self.seq_len,
                    self.kv_len,
                    attn_die,
                    ffn_die
                )
                model = get_model(self.model_type, self.model_config, self.aichip_config, search_config)
                attn = model["attn"]
                attn()
                moe = model["moe"]
                moe()
                attn_time = attn.attention_e2e_time * SEC_2_US
                moe_time = moe.moe_e2e_time * SEC_2_US
                commu_time = moe.commu_time * SEC_2_US
                dispatch_time = moe.dispatch_time * SEC_2_US
                combine_time = moe.combine_time * SEC_2_US

                ffn_dynamic_memory = (
                    search_config.ffn_bs * self.model_config.hidden_size * 
                    self.model_config.num_layers * BYTE_2_GB
                )
                total_memory = (
                    kv_size * self.micro_batch_num + attn_static_memory + 
                    per_router_expert_memory * routed_expert_per_die + ffn_dynamic_memory
                )
                e2e_time_per_moe_layer = attn_time + moe_time + commu_time
                e2e_time = e2e_time_per_moe_layer * self.model_config.num_moe_layers

                if self.model_config.num_layers > self.model_config.num_moe_layers:
                    # compute per dense layer time
                    mlp = model["mlp"]
                    mlp()
                    mlp_time = mlp.mlp_e2e_time * SEC_2_US
                    dense_commu_time = mlp.commu_time * SEC_2_US
                    e2e_time_per_dense_layer = attn_time + mlp_time + dense_commu_time
                    e2e_time = e2e_time + e2e_time_per_dense_layer * self.model_config.first_k_dense_replace

                if (e2e_time > latency_constraint or 
                    total_memory > self.aichip_config.hw_conf.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO):
                    attn_bs_max = attn_bs
                else:
                    attn_bs_min = attn_bs

            e2e_time = e2e_time * US_2_MS
            throughput = attn_bs / e2e_time / MS_2_SEC * (1 + self.multi_token_ratio)

            logging.info(f"-------DeepEP Search Result:-------")
            logging.info(
                f"attn_bs:{attn_bs}, ffn_bs:{search_config.ffn_bs}, "
                f"kv_len:{self.kv_len}, total_die:{total_die}, "
                f"attn_time:{attn_time}, mlp_time:{mlp_time}, "
                f"moe_time:{moe_time}, commu_time:{commu_time}, "
                f"dispatch_time:{dispatch_time}, combine_time:{combine_time}, "
                f"e2e_time:{e2e_time}, throughput:{throughput}"
            )

            self.perf_deepep_results.append([
                attn_bs, search_config.ffn_bs, self.kv_len, total_die, attn_time, moe_time,
                commu_time, dispatch_time, combine_time, e2e_time, throughput
            ])

        columns = [
            'attn_bs', 'ffn_bs', 'kv_len', 'total_die', 'attn_time', 
            'moe_time', 'commu_time', 'dispatch_time', 'combine_time', 'e2e_time', 'throughput'
        ]
        df = pd.DataFrame(self.perf_deepep_results, columns=columns)
        result_path = f'data/deepep/perf_deepep_latency{int(self.latency * US_2_MS)}_results.csv'
        df.to_csv(result_path, index=False)

    def deployment(self):
        self.search_bs()

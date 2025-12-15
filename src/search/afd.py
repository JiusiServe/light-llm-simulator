from typing import Tuple
import pandas as pd
import logging
import math
from conf.common import SEC_2_US, MIN_ROUTED_EXPERT_PER_DIE, MEMORY_THRESHOLD_RATIO, US_2_MS, MS_2_US
from src.search.base import BaseSearch
from src.search.config import SearchConfig
from src.module.register import get_model


class AfdSearch(BaseSearch):
    MIN_ATTN_BS = 2
    MAX_ATTN_BS = 1000
    MIN_FFN_DIE = 8
    MAX_FFN_DIE = 289
    FFN_DIE_STEP = 8
    ATTN_DIE_MULTIPLIER = 7

    def __init__(self, model_type, model_config, aichip_config, kv_len: int, seq_len: int,
                 multi_token_ratio: float, latency: float, micro_batch_num: int):
        super().__init__(model_type, model_config, aichip_config, kv_len, seq_len, multi_token_ratio, latency * MS_2_US)
        self.micro_batch_num = micro_batch_num
        self.perf_afd_results = []

    def search_attn_bs(self) -> Tuple[float, int]:
        """
        search max attention bs

        Returns:
            (attn_time, attn_bs) tuple
        """
        attn_bs_min, attn_bs_max = self.MIN_ATTN_BS, self.MAX_ATTN_BS
        attn_time = 0.0
        search_mode = "AFD"

        while attn_bs_max - attn_bs_min > 1:
            attn_bs = (attn_bs_min + attn_bs_max) // 2
            search_config = SearchConfig(self.model_config, search_mode, attn_bs, self.seq_len, self.kv_len)
            model = get_model(self.model_type, self.model_config, self.aichip_config, search_config)
            attn = model["attn"]
            attn()
            attn_time = attn.attention_e2e_time * SEC_2_US
            attn_latency_constraint = (
                self.latency / self.model_config.num_layers * 
                (1 + self.multi_token_ratio) / self.micro_batch_num
            )
            kv_size, attn_static_memory, _ = self.compute_memory_size(self.model_config, attn_bs)
            attn_memory = kv_size * self.micro_batch_num + attn_static_memory
            
            if attn_time > attn_latency_constraint or attn_memory > self.aichip_config.hw_conf.aichip_memory:
                attn_bs_max = attn_bs
            else:
                attn_bs_min = attn_bs

        if attn_time > attn_latency_constraint or attn_memory > self.aichip_config.hw_conf.aichip_memory:
            attn_bs = attn_bs_min
            search_config = SearchConfig(self.model_config, search_mode, attn_bs_min, self.seq_len, self.kv_len)
            attn = get_model(self.model_type, self.model_config, self.aichip_config, search_config)["attn"]
            attn()
            attn_time = attn.attention_e2e_time * SEC_2_US

        return attn_time, attn_bs

    def search(self, attn_time: float, attn_bs: int):
        search_mode = "AFD"
        _, _, per_router_expert_memory = self.compute_memory_size(self.model_config, attn_bs)
        logging.info(f'search args - attn_time: {attn_time:.2f}us, attn_bs: {attn_bs}')

        # compute per dense layer time
        if self.model_config.num_layers > self.model_config.num_moe_layers:
            dense_attn_bs = attn_bs * self.micro_batch_num
            search_config = SearchConfig(self.model_config, search_mode, dense_attn_bs, self.seq_len, self.kv_len)
            model = get_model(self.model_type, self.model_config, self.aichip_config, search_config)
            attn = model["attn"]
            mlp = model["mlp"]
            attn()
            mlp()
            dense_attn_time = attn.attention_e2e_time * SEC_2_US
            mlp_time = mlp.mlp_e2e_time * SEC_2_US
            dense_commu_time = mlp.commu_time * SEC_2_US
            e2e_time_per_dense_layer = dense_attn_time + mlp_time + dense_commu_time
        else:
            e2e_time_per_dense_layer = 0.0

        # search ffn_die, attn_die
        for ffn_die in range(self.MIN_FFN_DIE, self.MAX_FFN_DIE, self.FFN_DIE_STEP):
            routed_expert_per_die = max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.model_config.n_routed_experts / ffn_die)
            )
            ffn_static_memory = per_router_expert_memory * routed_expert_per_die
            if ffn_static_memory > self.aichip_config.hw_conf.aichip_memory * MEMORY_THRESHOLD_RATIO:
                continue

            for attn_die in range(ffn_die, self.ATTN_DIE_MULTIPLIER * ffn_die, self.FFN_DIE_STEP):
                total_die = ffn_die + attn_die
                search_config = SearchConfig(self.model_config, search_mode, attn_bs, self.seq_len,
                                             self.kv_len, attn_die, ffn_die)
                model = get_model(self.model_type, self.model_config, self.aichip_config, search_config)
                moe = model["moe"]
                moe()
                # compute per moe layer time
                moe_time = moe.moe_e2e_time * SEC_2_US
                dispatch_time = moe.dispatch_time * SEC_2_US
                combine_time = moe.combine_time * SEC_2_US
                commu_time = moe.commu_time * SEC_2_US
                e2e_time_per_moe_layer = max(
                    attn_time + moe_time + commu_time,
                    max(attn_time, moe_time) * self.micro_batch_num
                )

                latency_constraint = (
                    self.latency * (1 + self.multi_token_ratio) / 
                    self.model_config.num_layers
                )

                if e2e_time_per_moe_layer > latency_constraint:
                    continue

                e2e_time = (
                    e2e_time_per_dense_layer * self.model_config.first_k_dense_replace + 
                    e2e_time_per_moe_layer * self.model_config.num_moe_layers
                )
                throughput = (
                    attn_bs * self.micro_batch_num * attn_die / total_die / e2e_time * 
                    (1 + self.multi_token_ratio) * SEC_2_US
                )

                logging.info(f"-------AFD Search Result:-------")
                logging.info(
                    f"attn_bs: {attn_bs}, ffn_bs: {search_config.ffn_bs}, "
                    f"kv_len: {self.kv_len}, attn_die: {attn_die}, "
                    f"ffn_die: {ffn_die}, total_die: {total_die}, "
                    f"attn_time: {attn_time:.2f}us, moe_time: {moe_time:.2f}us, "
                    f"dispatch_time: {dispatch_time:.2f}us, combine_time: {combine_time:.2f}us, "
                    f"commu_time: {commu_time:.2f}us, e2e_time: {e2e_time:.2f}us, "
                    f"latency_per_layer: {latency_constraint:.2f}us, latency:{self.latency * US_2_MS} ms"
                    f"e2e_time_per_dense_layer: {e2e_time_per_dense_layer:.2f}us, "
                    f"e2e_time_per_moe_layer: {e2e_time_per_moe_layer:.2f}us, throughput: {throughput:.2f}"
                )

                self.perf_afd_results.append([
                    attn_bs, search_config.ffn_bs, self.kv_len, attn_die, ffn_die, total_die,
                    attn_time, moe_time, dispatch_time, combine_time, commu_time, e2e_time / MS_2_US,
                    e2e_time_per_dense_layer, e2e_time_per_moe_layer, throughput
                ])

        columns = [
            'attn_bs', 'ffn_bs', 'kv_len', 'attn_die', 'ffn_die', 'total_die',
            'attn_time', 'moe_time', 'dispatch_time', 'combine_time', 'commu_time', 'e2e_time',
            'e2e_time_per_dense_layer', 'e2e_time_per_moe_layer', 'throughput'
        ]
        df = pd.DataFrame(self.perf_afd_results, columns=columns)

        result_path = (
            f'data/afd/mbn{self.micro_batch_num}/'
            f'perf_afd_mbn{self.micro_batch_num}_latency{int(self.latency * US_2_MS)}_results.csv'
        )
        df.to_csv(result_path, index=False)

        df_best = df.sort_values(by=['throughput'], ascending=False).drop_duplicates(subset=['total_die'])
        df_best = df_best.sort_values(by=['total_die'], ascending=True)
        best_path = (
            f'data/afd/mbn{self.micro_batch_num}/best/'
            f'perf_afd_mbn{self.micro_batch_num}_latency{int(self.latency * US_2_MS)}_best_results.csv'
        )
        df_best.to_csv(best_path, index=False)

    def deployment(self):
        attn_time, attn_bs= self.search_attn_bs()
        self.search(attn_time, attn_bs)

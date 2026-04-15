import pandas as pd
import logging
import os
from conf.config import Config
from conf.common import SEC_2_US, MEMORY_THRESHOLD_RATIO, BYTE_2_GB, US_2_MS, MS_2_SEC
from src.search.base import BaseSearch
from src.model.register import get_model, get_attention_family


class AfdSearch(BaseSearch):
    '''
    Description:
        The AFD search algorithm.
        It is used to search the optimal attention batch size,
        attention die count, FFN die count for the model used AFD serving.
        Supports both Homogeneous (single device type) and Heterogeneous
        (different device types for attention and FFN) deployment modes.
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
            deployment_mode: "Homogeneous" or "Heterogeneous".
            device_type_attn: Device type for attention.
            device_type_ffn: Device type for FFN.
    '''
    ATTN_DIE_MULTIPLIER = 7

    def __init__(self, config: Config):
        super().__init__(config)
        self.perf_afd_results = []

    def _evaluate_config(self, attn_bs, attn_die, ffn_die, routed_expert_per_die, e2e_latency_target):
        """Evaluate a (attn_bs, attn_die, ffn_die) configuration.
        Returns results dict if all constraints are satisfied, None otherwise.
        """
        # Memory computation
        if get_attention_family(self.config.model_type) == "MLA":
            kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_MLA_memory_size(
                self.config.model_config, attn_bs)
        elif get_attention_family(self.config.model_type) == "GQA":
            kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_GQA_memory_size(
                self.config.model_config, attn_bs)

        ffn_static_memory = per_router_expert_memory * routed_expert_per_die

        # Memory constraints
        attn_used_memory = kv_size * self.config.micro_batch_num + attn_static_memory
        attn_memory_threshold = self.config.aichip_config_attn.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO
        ffn_memory_threshold = self.config.aichip_config_ffn.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO

        if attn_used_memory > attn_memory_threshold or ffn_static_memory > ffn_memory_threshold:
            return None

        # Compute MoE layer timing
        self.config.attn_bs = attn_bs
        self.config.ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok * attn_die / ffn_die
        moe_ffn_bs = self.config.ffn_bs
        self.config.attn_die = attn_die
        self.config.ffn_die = ffn_die
        self.config.routed_expert_per_die = routed_expert_per_die

        model = get_model(self.config)
        attn = model["attn"]
        attn()
        moe = model["moe"]
        moe()

        attn_time = attn.e2e_time * SEC_2_US
        moe_time = moe.e2e_time * SEC_2_US
        commu_time = moe.commu_time * SEC_2_US

        e2e_time_per_moe_layer = max(
            attn_time + moe_time + commu_time,
            max(attn_time, max(moe_time, commu_time)) * self.config.micro_batch_num
        )

        e2e_time = e2e_time_per_moe_layer * (self.config.model_config.num_moe_layers + self.config.seq_len - 1)
        embedding = model["embedding"]
        embedding()
        lm_head = model["lm_head"]
        lm_head()
        e2e_time = e2e_time + embedding.e2e_time * SEC_2_US + lm_head.e2e_time * SEC_2_US

        # Dense layer computation
        if self.config.model_config.num_layers > self.config.model_config.num_moe_layers:
            self.config.attn_bs = attn_bs * self.config.micro_batch_num
            self.config.ffn_bs = self.config.attn_bs
            dense_ffn_bs = self.config.ffn_bs
            model_dense = get_model(self.config)
            attn_dense = model_dense["attn"]
            mlp = model_dense["mlp"]
            attn_dense()
            mlp()
            e2e_time_per_dense_layer = attn_dense.e2e_time * SEC_2_US + mlp.e2e_time * SEC_2_US
            e2e_time = e2e_time + e2e_time_per_dense_layer * self.config.model_config.first_k_dense_replace
        else:
            dense_ffn_bs = 0
            e2e_time_per_dense_layer = 0.0

        e2e_time = e2e_time / (1 + self.config.multi_token_ratio) * US_2_MS

        if e2e_time > e2e_latency_target:
            return None

        a2e_send = attn.a2e_send.e2e_time * SEC_2_US
        a2e_recv = attn.a2e_recv.e2e_time * SEC_2_US
        e2a_recv = moe.e2a_recv.e2e_time * SEC_2_US
        dispatch_time = moe.dispatch_time * SEC_2_US
        combine_time = moe.combine_time * SEC_2_US

        ffn_used_memory = ffn_static_memory
        attn_available_memory = attn_memory_threshold - attn_used_memory
        ffn_available_memory = ffn_memory_threshold - ffn_used_memory

        return {
            'moe_ffn_bs': moe_ffn_bs,
            'dense_ffn_bs': dense_ffn_bs,
            'attn_time': attn_time,
            'moe_time': moe_time,
            'dispatch_time': dispatch_time,
            'combine_time': combine_time,
            'commu_time': commu_time,
            'e2e_time': e2e_time,
            'e2e_time_per_dense_layer': e2e_time_per_dense_layer,
            'e2e_time_per_moe_layer': e2e_time_per_moe_layer,
            'a2e_send': a2e_send,
            'a2e_recv': a2e_recv,
            'e2a_recv': e2a_recv,
            'kv_size': kv_size,
            'attn_static_memory': attn_static_memory,
            'mlp_static_memory': mlp_static_memory,
            'ffn_static_memory': ffn_static_memory,
            'attn_used_memory': attn_used_memory,
            'ffn_used_memory': ffn_used_memory,
            'attn_available_memory': attn_available_memory,
            'ffn_available_memory': ffn_available_memory,
        }

    def search(self):
        '''
        Description:
            Search the optimal attention batch size,
            attention die count, FFN die count for the model used AFD serving.
            For each (ffn_die, attn_die) pair, binary search for the max attn_bs
            that satisfies latency and memory constraints.
        '''

        # Get die ranges based on deployment mode
        if self.config.deployment_mode == "Heterogeneous":
            min_ffn_die = self.config.min_die2
            max_ffn_die = self.config.max_die2 + 1
            ffn_die_step = self.config.die_step2
            min_attn_die = self.config.min_die
            max_attn_die = self.config.max_die + 1
            attn_die_step = self.config.die_step
        else:
            min_ffn_die = self.config.min_die
            max_ffn_die = self.config.max_die + 1
            ffn_die_step = self.config.die_step
            min_attn_die = None
            max_attn_die = None
            attn_die_step = self.config.die_step

        for ffn_die in range(min_ffn_die, max_ffn_die, ffn_die_step):
            routed_expert_per_die = Config.calc_routed_expert_per_die(
                self.config.model_config.n_routed_experts,
                self.config.model_config.n_shared_experts,
                ffn_die
            )

            # Determine attention die range
            if self.config.deployment_mode == "Heterogeneous":
                attn_die_start = min_attn_die
                attn_die_end = max_attn_die
            else:
                attn_die_start = ffn_die
                attn_die_end = self.ATTN_DIE_MULTIPLIER * ffn_die

            for attn_die in range(attn_die_start, attn_die_end, attn_die_step):
                total_die = ffn_die + attn_die

                # Node alignment check based on deployment mode
                if self.config.deployment_mode == "Heterogeneous":
                    if attn_die % self.config.aichip_config_attn.num_dies_per_node != 0:
                        continue
                    if ffn_die % self.config.aichip_config_ffn.num_dies_per_node != 0:
                        continue
                else:
                    if total_die % self.config.aichip_config.num_dies_per_node != 0:
                        continue

                # Binary search on attn_bs
                attn_bs_min, attn_bs_max = self.config.min_attn_bs, self.config.max_attn_bs
                while attn_bs_max - attn_bs_min > 1:
                    attn_bs = (attn_bs_min + attn_bs_max) // 2
                    result = self._evaluate_config(
                        attn_bs, attn_die, ffn_die, routed_expert_per_die, self.config.tpot
                    )
                    if result is not None:
                        attn_bs_min = attn_bs
                    else:
                        attn_bs_max = attn_bs

                # Use attn_bs_min as the optimal bs
                attn_bs = attn_bs_min

                # Recompute final results with the optimal attn_bs
                result = self._evaluate_config(
                    attn_bs, attn_die, ffn_die, routed_expert_per_die, self.config.tpot
                )
                if result is None:
                    continue

                throughput = (
                    attn_bs * self.config.micro_batch_num * attn_die / total_die / result['e2e_time'] / MS_2_SEC
                )

                logging.info(f"-------AFD Search Result:-------")
                logging.info(
                    f"deployment_mode: {self.config.deployment_mode}, "
                    f"attn_bs: {attn_bs}, dense_ffn_bs: {result['dense_ffn_bs']}, moe_ffn_bs: {result['moe_ffn_bs']}, "
                    f"kv_len: {self.config.kv_len}, attn_die: {attn_die}, "
                    f"ffn_die: {ffn_die}, total_die: {total_die}, "
                    f"attn_time: {result['attn_time']:.2f}us, moe_time: {result['moe_time']:.2f}us, "
                    f"a2e_send: {result['a2e_send']:.2f}us, a2e_recv: {result['a2e_recv']:.2f}us, "
                    f"dispatch_time: {result['dispatch_time']:.2f}us, combine_time: {result['combine_time']:.2f}us, "
                    f"e2a_recv: {result['e2a_recv']:.2f}us, commu_time: {result['commu_time']:.2f}us, e2e_time: {result['e2e_time']:.2f}ms, "
                    f"e2e_time_per_dense_layer: {result['e2e_time_per_dense_layer']:.2f}us, "
                    f"e2e_time_per_moe_layer: {result['e2e_time_per_moe_layer']:.2f}us, throughput: {throughput:.2f} tokens/die/s, "
                    f"kv_size: {result['kv_size']} GB, attn_static_memory: {result['attn_static_memory']} GB, "
                    f"mlp_static_memory: {result['mlp_static_memory']} GB, ffn_static_memory: {result['ffn_static_memory']} GB, "
                    f"attn_used_memory: {result['attn_used_memory']} GB, ffn_used_memory: {result['ffn_used_memory']} GB, "
                    f"attn_available_memory: {result['attn_available_memory']} GB, ffn_available_memory: {result['ffn_available_memory']} GB"
                )

                self.perf_afd_results.append([
                    attn_bs, result['dense_ffn_bs'], result['moe_ffn_bs'], self.config.kv_len, attn_die, ffn_die, total_die,
                    result['attn_time'], result['moe_time'], result['a2e_send'], result['a2e_recv'],
                    result['dispatch_time'], result['combine_time'], result['e2a_recv'], result['commu_time'],
                    result['e2e_time'], result['e2e_time_per_dense_layer'], result['e2e_time_per_moe_layer'], throughput,
                    result['kv_size'], result['attn_static_memory'], result['mlp_static_memory'], result['ffn_static_memory'],
                    result['attn_used_memory'], result['ffn_used_memory'],
                    result['attn_available_memory'], result['ffn_available_memory'],
                    self.config.deployment_mode, self.config.device_type.name, self.config.device_type2.name
                ])

        columns = [
            'attn_bs(per_micro_batch)', 'dense_ffn_bs', 'moe_ffn_bs(per_micro_batch)',
            'kv_len', 'attn_die', 'ffn_die', 'total_die',
            'attn_time(us)', 'moe_time(us)', 'a2e_send(us)', 'a2e_recv(us)', 'dispatch_time(us)', 'combine_time(us)', 'e2a_recv(us)', 'commu_time(us)',
            'e2e_time(ms)', 'e2e_time_per_dense_layer(us)', 'e2e_time_per_moe_layer(us)', 'throughput(tokens/die/s)',
            'kv_size(GB)', 'attn_static_memory(GB)', 'mlp_static_memory(GB)', 'ffn_static_memory(GB)',
            'attn_used_memory(GB)', 'ffn_used_memory(GB)', 'attn_available_memory(GB)', 'ffn_available_memory(GB)',
            'deployment_mode', 'device_type_attn', 'device_type_ffn'
        ]
        df = pd.DataFrame(self.perf_afd_results, columns=columns)

        # Generate file name and directory based on deployment mode
        if self.config.deployment_mode == "Heterogeneous":
            result_dir = f"data/afd/mbn{self.config.micro_batch_num}/heterogeneous/"
            file_name = f"{self.config.device_type.name}_{self.config.device_type2.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        else:
            result_dir = f"data/afd/mbn{self.config.micro_batch_num}/homogeneous/"
            file_name = f"{self.config.device_type.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        os.makedirs(result_dir, exist_ok=True)
        result_path = result_dir + file_name
        df.to_csv(result_path, index=False, float_format='%.2f')

        if len(df) > 0:
            df_best = df.sort_values(by=['throughput(tokens/die/s)'], ascending=False).drop_duplicates(subset=['total_die'])
            df_best = df_best.sort_values(by=['total_die'], ascending=True)
            if self.config.deployment_mode == "Heterogeneous":
                best_result_dir = f"data/afd/mbn{self.config.micro_batch_num}/best/heterogeneous/"
            else:
                best_result_dir = f"data/afd/mbn{self.config.micro_batch_num}/best/homogeneous/"
            os.makedirs(best_result_dir, exist_ok=True)
            best_result_path = best_result_dir + file_name
            df_best.to_csv(best_result_path, index=False, float_format='%.2f')

    def deployment(self):
        self.search()

import logging
import math
import os
import pandas as pd
from conf.common import MIN_ROUTED_EXPERT_PER_DIE, US_2_MS, SEC_2_US, BYTE_2_GB, MEMORY_THRESHOLD_RATIO, MS_2_SEC, MS_2_US
from src.search.base import BaseSearch
from src.model.register import get_model, get_attention_family
from conf.config import Config


class DeepEpSearch(BaseSearch):
    '''
    Description:
        The DeepEP search algorithm.
        It is used to search the optimal attention batch size for the model used DeepEP serving.

        Supports two deployment modes:
        - Homogeneous: Run DeepEP on a single device type
        - Heterogeneous: Run homogeneous DeepEP on two device types separately and compute weighted average

        NOTE: DeepEP Heterogeneous mode runs homogeneous DeepEP on device_type1 and device_type2
        separately. It does NOT run a truly heterogeneous deployment like AFD does.
        The results are combined using weighted average for comparison purposes.

    Attributes:
        config: The configuration of the DeepEP search task.
        perf_deepep_results: The performance results of the DeepEP search.
    '''
    def __init__(self, config: Config):
        super().__init__(config)
        self.perf_deepep_results = []

    def _run_homogeneous_deepep(self, device_type_str: str, min_die: int, max_die: int, die_step: int) -> dict:
        '''
        Run homogeneous DeepEP on a single device type.

        Args:
            device_type_str: Device type string (e.g., "Ascend_A3Pod")
            min_die, max_die, die_step: Die search range

        Returns:
            Dictionary mapping total_die -> throughput
        '''
        from conf.hardware_config import DeviceType, HWConf

        device_type = DeviceType(device_type_str)
        aichip_config = HWConf.create(device_type)

        results = {}

        for total_die in range(min_die, max_die + 1, die_step):
            routed_expert_per_die = self.config.model_config.n_shared_experts + max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.config.model_config.n_routed_experts / total_die)
            )
            attn_bs_min, attn_bs_max = self.config.min_attn_bs, self.config.max_attn_bs

            # search max attention bs
            while attn_bs_max - attn_bs_min > 1:
                attn_bs = (attn_bs_min + attn_bs_max) // 2
                if get_attention_family(self.config.model_type) == "MLA":
                    kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_MLA_memory_size(
                        self.config.model_config, attn_bs
                    )
                elif get_attention_family(self.config.model_type) == "GQA":
                    kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_GQA_memory_size(
                        self.config.model_config, attn_bs
                    )
                latency_constraint = (
                    self.config.tpot * MS_2_US / self.config.micro_batch_num * (1 + self.config.multi_token_ratio)
                )
                attn_die, ffn_die = total_die, total_die
                ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok

                # Create a temporary config with the device type
                temp_config = Config(
                    serving_mode="DeepEP",
                    model_type=self.config.model_type.value,
                    device_type=device_type_str,
                    min_attn_bs=self.config.min_attn_bs,
                    max_attn_bs=self.config.max_attn_bs,
                    min_die=min_die,
                    max_die=max_die,
                    die_step=die_step,
                    tpot=self.config.tpot,
                    kv_len=self.config.kv_len,
                    micro_batch_num=1,
                    next_n=self.config.seq_len - 1,
                    multi_token_ratio=self.config.multi_token_ratio,
                    attn_tensor_parallel=self.config.attn_tensor_parallel,
                    ffn_tensor_parallel=self.config.ffn_tensor_parallel,
                    deployment_mode="Homogeneous"
                )
                temp_config.attn_bs = attn_bs
                temp_config.ffn_bs = ffn_bs
                temp_config.attn_die = attn_die
                temp_config.ffn_die = ffn_die
                temp_config.routed_expert_per_die = routed_expert_per_die

                model = get_model(temp_config)
                attn = model["attn"]
                attn()
                moe = model["moe"]
                moe()
                attn_time = attn.e2e_time * SEC_2_US
                moe_time = moe.e2e_time * SEC_2_US
                commu_time = moe.commu_time * SEC_2_US

                ffn_dynamic_memory = (
                    ffn_bs * self.config.model_config.hidden_size *
                    self.config.model_config.num_layers * BYTE_2_GB
                )
                ffn_static_memory = per_router_expert_memory * routed_expert_per_die
                total_memory = (
                    kv_size * self.config.micro_batch_num + attn_static_memory +
                    mlp_static_memory + ffn_dynamic_memory + ffn_static_memory
                )
                e2e_time_per_moe_layer = attn_time + moe_time + commu_time
                e2e_time = e2e_time_per_moe_layer * self.config.model_config.num_moe_layers

                if self.config.model_config.num_layers > self.config.model_config.num_moe_layers:
                    mlp = model["mlp"]
                    mlp()
                    mlp_time = mlp.e2e_time * SEC_2_US
                    e2e_time_per_dense_layer = attn_time + mlp_time
                    e2e_time = e2e_time + e2e_time_per_dense_layer * self.config.model_config.first_k_dense_replace
                else:
                    e2e_time_per_dense_layer = 0.0

                if (e2e_time > latency_constraint or
                    total_memory > aichip_config.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO):
                    attn_bs_max = attn_bs
                else:
                    attn_bs_min = attn_bs

            e2e_time = e2e_time * US_2_MS
            throughput = attn_bs / e2e_time / MS_2_SEC * (1 + self.config.multi_token_ratio)
            results[total_die] = {
                'attn_bs': attn_bs,
                'ffn_bs': ffn_bs,
                'throughput': throughput,
                'e2e_time': e2e_time,
                'attn_time': attn_time,
                'moe_time': moe_time,
                'commu_time': commu_time,
                'kv_size': kv_size,
                'attn_static_memory': attn_static_memory,
                'mlp_static_memory': mlp_static_memory,
                'ffn_static_memory': ffn_static_memory,
                'e2e_time_per_dense_layer': e2e_time_per_dense_layer,
                'e2e_time_per_moe_layer': e2e_time_per_moe_layer
            }

        return results

    def search_bs_heterogeneous(self):
        '''
        Run heterogeneous DeepEP search.

        Heterogeneous DeepEP runs homogeneous DeepEP on device_type1 and device_type2
        separately, then computes weighted average throughput for comparison.

        Weighted average: (die1 * throughput1 + die2 * throughput2) / total_die
        '''
        logging.info("=" * 60)
        logging.info("NOTE: DeepEP Heterogeneous mode runs homogeneous DeepEP on two")
        logging.info("device types separately and computes weighted average throughput.")
        logging.info("This is for comparison purposes only, NOT a truly heterogeneous deployment.")
        logging.info("=" * 60)

        # Run DeepEP on device_type1 (attention device)
        logging.info(f"Running DeepEP on {self.config.device_type.value} (device_type1)...")
        results_device1 = self._run_homogeneous_deepep(
            self.config.device_type.value,
            self.config.min_die,
            self.config.max_die,
            self.config.die_step
        )

        # Run DeepEP on device_type2 (FFN device)
        logging.info(f"Running DeepEP on {self.config.device_type2.value} (device_type2)...")
        results_device2 = self._run_homogeneous_deepep(
            self.config.device_type2.value,
            self.config.min_die2,
            self.config.max_die2,
            self.config.die_step2
        )

        # Combine results with weighted average throughput
        for die1, r1 in results_device1.items():
            for die2, r2 in results_device2.items():
                total_die = die1 + die2
                # Weighted average throughput
                weighted_throughput = (die1 * r1['throughput'] + die2 * r2['throughput']) / total_die

                logging.info(f"-------DeepEP Heterogeneous Search Result:-------")
                logging.info(
                    f"device1_die:{die1}, device2_die:{die2}, total_die:{total_die}, "
                    f"throughput_device1:{r1['throughput']:.2f}, throughput_device2:{r2['throughput']:.2f}, "
                    f"weighted_throughput:{weighted_throughput:.2f} tokens/die/s"
                )

                self.perf_deepep_results.append([
                    r1['attn_bs'], r1['ffn_bs'], self.config.kv_len, die1, die2, total_die,
                    r1['attn_time'], r1['moe_time'], r1['commu_time'], r1['e2e_time'],
                    r1['e2e_time_per_dense_layer'], r1['e2e_time_per_moe_layer'],
                    r1['throughput'], r2['throughput'], weighted_throughput,
                    r1['kv_size'], r1['attn_static_memory'], r1['mlp_static_memory'], r1['ffn_static_memory'],
                    "Heterogeneous", self.config.device_type.name, self.config.device_type2.name
                ])

        columns = [
            'attn_bs', 'ffn_bs', 'kv_len', 'device1_die', 'device2_die', 'total_die',
            'attn_time(us)', 'moe_time(us)', 'commu_time(us)', 'e2e_time(ms)',
            'e2e_time_per_dense_layer(us)', 'e2e_time_per_moe_layer(us)',
            'throughput_device1(tokens/die/s)', 'throughput_device2(tokens/die/s)', 'weighted_throughput(tokens/die/s)',
            'kv_size(GB)', 'attn_static_memory(GB)', 'mlp_static_memory(GB)', 'ffn_static_memory(GB)',
            'deployment_mode', 'device_type1', 'device_type2'
        ]
        df = pd.DataFrame(self.perf_deepep_results, columns=columns)
        result_dir = f"data/deepep/heterogeneous/"
        file_name = f"{self.config.device_type.name}_{self.config.device_type2.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        os.makedirs(result_dir, exist_ok=True)
        result_path = result_dir + file_name
        df.to_csv(result_path, index=False)

    def search_bs(self):
        '''
        Description:
            Search the optimal attention batch size for the model used DeepEP serving.
        '''
        min_total_die, max_total_die, die_step = self.config.min_die, self.config.max_die, self.config.die_step
        for total_die in range(min_total_die, max_total_die + 1, die_step):
            routed_expert_per_die = self.config.model_config.n_shared_experts + max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.config.model_config.n_routed_experts / total_die)
            )
            attn_bs_min, attn_bs_max = self.config.min_attn_bs, self.config.max_attn_bs

            # search max attention bs
            while attn_bs_max - attn_bs_min > 1:
                attn_bs = (attn_bs_min + attn_bs_max) // 2
                if get_attention_family(self.config.model_type) == "MLA":
                    kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_MLA_memory_size(
                        self.config.model_config, attn_bs
                    )
                elif get_attention_family(self.config.model_type) == "GQA":
                    kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_GQA_memory_size(
                        self.config.model_config, attn_bs
                    )
                latency_constraint = (
                    self.config.tpot * MS_2_US / self.config.micro_batch_num * (1 + self.config.multi_token_ratio)
                )
                attn_die, ffn_die = total_die, total_die
                ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok
                self.config.attn_bs = attn_bs
                self.config.ffn_bs = ffn_bs
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
                dispatch_time = moe.dispatch_time * SEC_2_US
                combine_time = moe.combine_time * SEC_2_US

                ffn_dynamic_memory = (
                    ffn_bs * self.config.model_config.hidden_size *
                    self.config.model_config.num_layers * BYTE_2_GB
                )
                ffn_static_memory = per_router_expert_memory * routed_expert_per_die
                total_memory = (
                    kv_size * self.config.micro_batch_num + attn_static_memory +
                    mlp_static_memory + ffn_dynamic_memory + ffn_static_memory
                )
                e2e_time_per_moe_layer = attn_time + moe_time + commu_time
                e2e_time = e2e_time_per_moe_layer * self.config.model_config.num_moe_layers

                if self.config.model_config.num_layers > self.config.model_config.num_moe_layers:
                    mlp = model["mlp"]
                    mlp()
                    mlp_time = mlp.e2e_time * SEC_2_US
                    e2e_time_per_dense_layer = attn_time + mlp_time
                    e2e_time = e2e_time + e2e_time_per_dense_layer * self.config.model_config.first_k_dense_replace
                else:
                    mlp_time = 0.0
                    e2e_time_per_dense_layer = 0.0
                if (e2e_time > latency_constraint or
                    total_memory > self.config.aichip_config.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO):
                    attn_bs_max = attn_bs
                else:
                    attn_bs_min = attn_bs

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

            self.perf_deepep_results.append([
                attn_bs, ffn_bs, self.config.kv_len, total_die,
                attn_time, moe_time, dispatch_time, combine_time, commu_time, e2e_time,
                e2e_time_per_dense_layer, e2e_time_per_moe_layer, throughput,
                kv_size, attn_static_memory, mlp_static_memory, ffn_static_memory,
                "Homogeneous", self.config.device_type.name, self.config.device_type.name
            ])

        columns = [
            'attn_bs', 'ffn_bs', 'kv_len', 'total_die',
            'attn_time(us)', 'moe_time(us)', 'dispatch_time(us)', 'combine_time(us)', 'commu_time(us)', 'e2e_time(ms)',
            'e2e_time_per_dense_layer(us)', 'e2e_time_per_moe_layer(us)', 'throughput(tokens/die/s)',
            'kv_size(GB)', 'attn_static_memory(GB)', 'mlp_static_memory(GB)', 'ffn_static_memory(GB)',
            'deployment_mode', 'device_type1', 'device_type2'
        ]
        df = pd.DataFrame(self.perf_deepep_results, columns=columns)
        result_dir = f"data/deepep/homogeneous/"
        file_name = f"{self.config.device_type.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        os.makedirs(result_dir, exist_ok=True)
        result_path = result_dir + file_name
        df.to_csv(result_path, index=False)

    def deployment(self):
        '''
        Run DeepEP deployment based on deployment_mode.
        - Homogeneous: Run standard DeepEP on single device type
        - Heterogeneous: Run homogeneous DeepEP on two device types separately
        '''
        if self.config.deployment_mode == "Heterogeneous":
            self.search_bs_heterogeneous()
        else:
            self.search_bs()

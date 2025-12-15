"""
DeepSeek example: run AFD and DeepEP search once.
"""
from conf.model_config import ModelType, ModelConfig
from conf.hardware_config import HardwareTopology, DeviceType
from src.search.afd import AfdSearch
from src.search.deepep import DeepEpSearch
import logging


def main() -> None:
    # ----- 1. Model & hardware configuration -----
    model_type = ModelType.DEEPSEEK_V3
    model_config = ModelConfig.create_model_config(model_type)

    hw_topology = HardwareTopology.create(
        number_of_ranks=8,
        npus_per_rank=8,
        device_type=DeviceType.ASCEND910B2_376T_64G,
    )

    # ----- 2. Search parameters (keep it small for a quick demo) -----
    kv_len = 4096
    next_n = 1
    seq_len = 1 + next_n
    multi_token_ratio = 0.7
    latency = list(range(50, 201, 10))
    micro_batch_num = 3     # single micro-batch setting

    for latency in latency:
        # ----- 3. Run AFD search -----
        logging.info("------- AFD Search -------")
        afd_search = AfdSearch(
            model_type=model_type,
            model_config=model_config,
            aichip_config=hw_topology,
            kv_len=kv_len,
            seq_len=seq_len,
            multi_token_ratio=multi_token_ratio,
            latency=latency,
            micro_batch_num=micro_batch_num,
        )
        afd_search.deployment()

        # ----- 4. Run DeepEP as baseline search -----
        logging.info("------- DeepEP Search -------")
        deepep_search = DeepEpSearch(
            model_type=model_type,
            model_config=model_config,
            aichip_config=hw_topology,
            kv_len=kv_len,
            seq_len=seq_len,
            multi_token_ratio=multi_token_ratio,
            latency=latency,
            micro_batch_num=1,   # DeepEP uses micro_batch_num = 1
        )
        deepep_search.deployment()

if __name__ == "__main__":
    main()

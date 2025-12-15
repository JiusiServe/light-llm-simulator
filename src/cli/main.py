import argparse
import logging
from conf.model_config import ModelType, ModelConfig
from conf.hardware_config import HardwareTopology, DeviceType
from src.search.afd import AfdSearch
from src.search.deepep import DeepEpSearch


def add_default_mode_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--model_name',
        type=str,
        default="deepseek-ai/DeepSeek-V3"
    )
    parser.add_argument(
        '--device_name',
        type=str,
        default="Ascend_A3Pod"
    )
    parser.add_argument(
        '--num_chips',
        type=int,
        default=64
    )
    parser.add_argument(
        '--latency',
        nargs='+',
        type=int,
        default=list[int](range(50, 201, 10))
    )
    parser.add_argument(
        '--kv_len',
        type=int,
        default=4096
    )
    parser.add_argument(
        '--micro_batch_num',
        nargs='+',
        type=int,
        default=[2, 3]
    )
    parser.add_argument(
        '--next_n',
        type=int,
        default=1
    )
    parser.add_argument(
        '--multi_token_ratio',
        type=float,
        default=0.7
    )


def run_search(
    model_type: ModelType,
    model_config: ModelConfig,
    hw_topology: HardwareTopology,
    kv_len: int,
    seq_len: int,
    multi_token_ratio: float,
    latency: int,
    micro_batch_num: list[int]
):
    logging.info(f'- latency: {latency}ms, micro_batch_num: {micro_batch_num}')

    logging.info('-------AFD Search-------')
    for mbn in micro_batch_num:
        afd_search = AfdSearch(
            model_type, model_config, hw_topology,
            kv_len=kv_len,
            seq_len=seq_len,
            multi_token_ratio=multi_token_ratio,
            latency=latency,
            micro_batch_num=mbn
        )
        afd_search.deployment()

    logging.info('-------DeepEP Search-------')
    baseline_search = DeepEpSearch(
        model_type, model_config, hw_topology,
        kv_len=kv_len,
        seq_len=seq_len,
        multi_token_ratio=multi_token_ratio,
        latency=latency,
        micro_batch_num=1
    )
    baseline_search.deployment()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_default_mode_arguments(parser)
    args = parser.parse_args()

    model_type = ModelType(args.model_name)
    model_config = ModelConfig.create_model_config(model_type)

    device_type = DeviceType(args.device_name)
    hw_topology = HardwareTopology.create(
        number_of_ranks=8,
        npus_per_rank=8,
        device_type=device_type
    )

    seq_len = 1 + args.next_n

    for latency in args.latency:
        run_search(
            model_type, model_config, hw_topology,
            kv_len=args.kv_len,
            seq_len=seq_len,
            multi_token_ratio=args.multi_token_ratio,
            latency=latency,
            micro_batch_num=args.micro_batch_num
        )


if __name__ == "__main__":
    main()

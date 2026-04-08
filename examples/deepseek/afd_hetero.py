"""
DeepSeek V3 Heterogeneous AFD example: run AFD search with different device types for attention and FFN.

In Heterogeneous mode:
- Attention module runs on device_type1 (e.g., Ascend_A3Pod)
- FFN module runs on device_type2 (e.g., Ascend_David121)
- Total die = attn_die + ffn_die

Usage:
    python examples/deepseek/afd_hetero.py
    python examples/deepseek/afd_hetero.py --device_type1 Ascend_A3Pod --device_type2 Ascend_David121
    python examples/deepseek/afd_hetero.py --tpot 50 --kv_len 4096
"""
import argparse
from src.search.afd import AfdSearch
from conf.config import Config


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--serving_mode', type=str, default="AFD")
    parser.add_argument('--model_type', type=str, default="deepseek-ai/DeepSeek-V3")
    parser.add_argument('--device_type1', type=str, default="Ascend_A3Pod",
                        help="Device type for attention module")
    parser.add_argument('--device_type2', type=str, default="Ascend_David121",
                        help="Device type for FFN module")
    parser.add_argument('--min_attn_bs', type=int, default=2)
    parser.add_argument('--max_attn_bs', type=int, default=1000)
    parser.add_argument('--min_attn_die', type=int, default=16,
                        help="Min attention dies")
    parser.add_argument('--max_attn_die', type=int, default=256,
                        help="Max attention dies")
    parser.add_argument('--attn_die_step', type=int, default=16,
                        help="Step for attention die search")
    parser.add_argument('--min_ffn_die', type=int, default=16,
                        help="Min FFN dies")
    parser.add_argument('--max_ffn_die', type=int, default=256,
                        help="Max FFN dies")
    parser.add_argument('--ffn_die_step', type=int, default=16,
                        help="Step for FFN die search")
    parser.add_argument('--tpot', nargs='+', type=int, default=[50])
    parser.add_argument('--kv_len', nargs='+', type=int, default=[4096])
    parser.add_argument('--micro_batch_num', nargs='+', type=int, default=[3])
    parser.add_argument('--next_n', type=int, default=1)
    parser.add_argument('--multi_token_ratio', type=float, default=0.7)
    parser.add_argument('--attn_tensor_parallel', type=int, default=1)
    parser.add_argument('--ffn_tensor_parallel', type=int, default=1)


def run_search(args):
    for mbn in args.micro_batch_num:
        for tpot in args.tpot:
            for kv_len in args.kv_len:
                config = Config(
                    serving_mode=args.serving_mode,
                    model_type=args.model_type,
                    device_type=args.device_type1,
                    min_attn_bs=args.min_attn_bs,
                    max_attn_bs=args.max_attn_bs,
                    min_die=args.min_attn_die,
                    max_die=args.max_attn_die,
                    die_step=args.attn_die_step,
                    tpot=tpot,
                    kv_len=kv_len,
                    micro_batch_num=mbn,
                    next_n=args.next_n,
                    multi_token_ratio=args.multi_token_ratio,
                    attn_tensor_parallel=args.attn_tensor_parallel,
                    ffn_tensor_parallel=args.ffn_tensor_parallel,
                    deployment_mode="Heterogeneous",
                    device_type2=args.device_type2,
                    min_die2=args.min_ffn_die,
                    max_die2=args.max_ffn_die,
                    die_step2=args.ffn_die_step
                )
                afd_search = AfdSearch(config)
                afd_search.deployment()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run Heterogeneous AFD search for DeepSeek V3"
    )
    add_arguments(parser)
    args = parser.parse_args()

    print(f"Running Heterogeneous AFD search:")
    print(f"  Attention device: {args.device_type1}")
    print(f"  FFN device: {args.device_type2}")
    print(f"  Attention die range: {args.min_attn_die} - {args.max_attn_die}")
    print(f"  FFN die range: {args.min_ffn_die} - {args.max_ffn_die}")

    run_search(args)


if __name__ == "__main__":
    main()

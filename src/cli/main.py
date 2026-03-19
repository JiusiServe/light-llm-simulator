import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from conf.config import Config
from src.search.afd import AfdSearch
from src.search.deepep import DeepEpSearch


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments to the parser.
    Parameters:
        serving_mode: The serving mode of the task.
        model_type: The type of the model.
        device_type1: The type of the device1.
                      For AFD, it is the device type to explore attention module.
        device_type2: The type of the device2.
                      For AFD, it is the device type to explore ffn module.
        min_attn_bs: The min number of attention batch size to explore.
        max_attn_bs: The max number of attention batch size to explore.
        min_die1: The min number of die to explore for device_type1.
        max_die1: The max number of die to explore for device_type1.
        min_die2: The min number of die to explore for device_type2.
        max_die2: The max number of die to explore for device_type2.
        die_step1: The step size of the die to explore for device_type1.
        die_step2: The step size of the die to explore for device_type2.
        max_die: The max number of die to explore.
        die_step: The step size of the die to explore.
        tpot: The target TPOT.
        kv_len: The input sequence length.
        micro_batch_num: The micro batch number.
        next_n: Predict the next n tokens through the MTP(Multi-Token Prediction) technique.
        multi_token_ratio: The acceptance rate of the additionally predicted token.
        attn_tensor_parallel: Number of dies used for tensor model parallelism.
        ffn_tensor_parallel: Number of dies used for tensor model parallelism.
    """
    parser.add_argument('--serving_mode', type=str, default="AFD")
    parser.add_argument('--model_type', type=str, default="deepseek-ai/DeepSeek-V3")
    parser.add_argument(
        '--device_type1',
        type=str,
        default="Ascend_David120",
        help="For AFD, it is the device type to explore attention module."
    )
    parser.add_argument(
        '--device_type2',
        type=str,
        default="Ascend_David100",
        help="For AFD, it is the device type to explore ffn module."
    )
    parser.add_argument('--min_attn_bs', type=int, default=2)
    parser.add_argument('--max_attn_bs', type=int, default=1000)
    parser.add_argument(
        '--min_die1',
        type=int,
        default=8,
        help="The min number of die to explore for device_type1."
    )
    parser.add_argument(
        '--max_die1',
        type=int,
        default=128,
        help="The max number of die to explore for device_type1."
    )
    parser.add_argument(
        '--min_die2',
        type=int,
        default=8,
        help="The min number of die to explore for device_type2."
    )
    parser.add_argument(
        '--max_die2',
        type=int,
        default=128,
        help="The max number of die to explore for device_type2."
    )
    parser.add_argument(
        '--die_step1',
        type=int,
        default=8,
        help="The step size of the die to explore for device_type1."
    )
    parser.add_argument(
        '--die_step2',
        type=int,
        default=8,
        help="The step size of the die to explore for device_type2."
    )
    parser.add_argument('--tpot', nargs='+', type=int, default=[20, 50, 70, 100, 150])
    parser.add_argument('--kv_len', nargs='+', type=int, default=[2048, 4096, 8192, 16384, 131072])
    parser.add_argument('--micro_batch_num', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--next_n', type=int, default=1)
    parser.add_argument('--multi_token_ratio', type=float, default=0.7)
    parser.add_argument('--attn_tensor_parallel', type=int, default=1)
    parser.add_argument('--ffn_tensor_parallel', type=int, default=1)

def run_search(args):
    """
    Run the search to find the best configuration and the best throughput.
    Parameters:
        args: The arguments from the parser.
    """
    if args.serving_mode == "AFD":
        for mbn in args.micro_batch_num:
            for tpot in args.tpot:
                for kv_len in args.kv_len:
                    config = Config(
                        serving_mode=args.serving_mode,
                        model_type=args.model_type,
                        device_type1=args.device_type1,
                        device_type2=args.device_type2,
                        min_attn_bs=args.min_attn_bs,
                        max_attn_bs=args.max_attn_bs,
                        min_die1=args.min_die1,
                        max_die1=args.max_die1,
                        min_die2=args.min_die2,
                        max_die2=args.max_die2,
                        die_step1=args.die_step1,
                        die_step2=args.die_step2,
                        tpot=tpot,
                        kv_len=kv_len,
                        micro_batch_num=mbn,
                        next_n=args.next_n,
                        multi_token_ratio=args.multi_token_ratio,
                        attn_tensor_parallel=args.attn_tensor_parallel,
                        ffn_tensor_parallel=args.ffn_tensor_parallel
                    )
                    afd_search = AfdSearch(config)
                    afd_search.deployment()
    elif args.serving_mode == "DeepEP":
        for tpot in args.tpot:
            for kv_len in args.kv_len:
                config = Config(
                    serving_mode=args.serving_mode,
                    model_type=args.model_type,
                    device_type1=args.device_type1,
                    device_type2=args.device_type2,
                    min_attn_bs=args.min_attn_bs,
                    max_attn_bs=args.max_attn_bs,
                    min_die1=args.min_die1,
                    max_die1=args.max_die1,
                    min_die2=args.min_die2,
                    max_die2=args.max_die2,
                    die_step1=args.die_step1,
                    die_step2=args.die_step2,
                    tpot=tpot,
                    kv_len=kv_len,
                    micro_batch_num=1,
                    next_n=args.next_n,
                    multi_token_ratio=args.multi_token_ratio,
                    attn_tensor_parallel=args.attn_tensor_parallel,
                    ffn_tensor_parallel=args.ffn_tensor_parallel
                )
                deepep_search = DeepEpSearch(config)
                deepep_search.deployment()
    else:
        raise ValueError(f"Invalid serving mode: {args.serving_mode}")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
    import subprocess
    subprocess.run(["python", "src/visualization/throughput.py"])
    subprocess.run(["python", "src/visualization/pipeline.py"])

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from conf.config import Config
from src.search.afd import AfdSearch
from src.search.deepep import DeepEpSearch
from conf.model_config import ModelType
from conf.hardware_config import DeviceType


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments to the parser.
    Parameters:
        serving_mode: The serving mode of the task.
        model_type: The type of the model.
        device_type: The type of the device.
        min_attn_bs: The min number of attention batch size to explore.
        max_attn_bs: The max number of attention batch size to explore.
        min_die: The min number of die to explore.
        max_die: The max number of die to explore.
        die_step: The step size of the die to explore.
        tpot: The target TPOT.
        kv_len: The input sequence length.
        micro_batch_num: The micro batch number.
        next_n: Predict the next n tokens through the MTP(Multi-Token Prediction) technique.
        multi_token_ratio: The acceptance rate of the additionally predicted token.
        attn_tensor_parallel: Number of dies used for tensor model parallelism.
        ffn_tensor_parallel: Number of dies used for tensor model parallelism.
        deployment_mode: "Homogeneous" or "Heterogeneous" (AFD and DeepEP).
        device_type2: The second device type for FFN (Heterogeneous mode only).
        min_die2: The min number of FFN dies to explore (Heterogeneous mode only).
        max_die2: The max number of FFN dies to explore (Heterogeneous mode only).
        die_step2: The step size for FFN dies (Heterogeneous mode only).
    """
    parser.add_argument('--serving_mode', type=str, default="AFD")
    parser.add_argument('--model_type', type=str, default="deepseek-ai/DeepSeek-V3")
    parser.add_argument('--device_type', type=str, default="Ascend_A3Pod")
    parser.add_argument('--min_attn_bs', type=int, default=2)
    parser.add_argument('--max_attn_bs', type=int, default=1000)
    parser.add_argument('--min_die', type=int, default=16)
    parser.add_argument('--max_die', type=int, default=768)
    parser.add_argument('--die_step', type=int, default=16)
    parser.add_argument('--tpot', nargs='+', type=int, default=[20, 50, 70, 100, 150])
    parser.add_argument('--kv_len', nargs='+', type=int, default=[2048, 4096, 8192, 16384, 131072])
    parser.add_argument('--micro_batch_num', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--next_n', type=int, default=1)
    parser.add_argument('--multi_token_ratio', type=float, default=0.7)
    parser.add_argument('--attn_tensor_parallel', type=int, default=1)
    parser.add_argument('--ffn_tensor_parallel', type=int, default=1)
    # Heterogeneous mode arguments
    parser.add_argument('--deployment_mode', type=str, default="Homogeneous",
                        choices=["Homogeneous", "Heterogeneous"],
                        help="Deployment mode. Supported for both AFD and DeepEP. "
                             "Note: DeepEP Heterogeneous runs homogeneous DeepEP on two device types separately "
                             "and computes weighted average throughput for comparison only, NOT a truly heterogeneous deployment like AFD.")
    parser.add_argument('--device_type2', type=str, default=None,
                        help="Second device type for FFN (required for Heterogeneous mode)")
    parser.add_argument('--min_die2', type=int, default=None,
                        help="Min FFN dies for Heterogeneous mode")
    parser.add_argument('--max_die2', type=int, default=None,
                        help="Max FFN dies for Heterogeneous mode")
    parser.add_argument('--die_step2', type=int, default=None,
                        help="Die step for FFN in Heterogeneous mode")


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
                        device_type=args.device_type,
                        min_attn_bs=args.min_attn_bs,
                        max_attn_bs=args.max_attn_bs,
                        min_die=args.min_die,
                        max_die=args.max_die,
                        die_step=args.die_step,
                        tpot=tpot,
                        kv_len=kv_len,
                        micro_batch_num=mbn,
                        next_n=args.next_n,
                        multi_token_ratio=args.multi_token_ratio,
                        attn_tensor_parallel=args.attn_tensor_parallel,
                        ffn_tensor_parallel=args.ffn_tensor_parallel,
                        deployment_mode=args.deployment_mode,
                        device_type2=args.device_type2,
                        min_die2=args.min_die2,
                        max_die2=args.max_die2,
                        die_step2=args.die_step2
                    )
                    afd_search = AfdSearch(config)
                    afd_search.deployment()
    elif args.serving_mode == "DeepEP":
        for tpot in args.tpot:
            for kv_len in args.kv_len:
                config = Config(
                    serving_mode=args.serving_mode,
                    model_type=args.model_type,
                    device_type=args.device_type,
                    min_attn_bs=args.min_attn_bs,
                    max_attn_bs=args.max_attn_bs,
                    min_die=args.min_die,
                    max_die=args.max_die,
                    die_step=args.die_step,
                    tpot=tpot,
                    kv_len=kv_len,
                    micro_batch_num=1,
                    next_n=args.next_n,
                    multi_token_ratio=args.multi_token_ratio,
                    attn_tensor_parallel=args.attn_tensor_parallel,
                    ffn_tensor_parallel=args.ffn_tensor_parallel,
                    deployment_mode=args.deployment_mode,
                    device_type2=args.device_type2,
                    min_die2=args.min_die2,
                    max_die2=args.max_die2,
                    die_step2=args.die_step2
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

    if os.environ.get("LIGHT_LLM_SKIP_POST_PLOTS", "").lower() in ("1", "true", "yes"):
        return

    import subprocess
    throughput_cmd = [
        "python", "src/visualization/throughput.py",
        "--model_type", args.model_type,
        "--device_type", args.device_type,
        "--deployment_mode", args.deployment_mode,
        "--min_die", str(args.min_die),
        "--max_die", str(args.max_die),
    ]
    # Add heterogeneous parameters if applicable
    if args.deployment_mode == "Heterogeneous":
        if args.device_type2 is None:
                raise ValueError("device_type2 is required for Heterogeneous deployment mode")
        throughput_cmd.extend(["--device_type2", args.device_type2])
        _min_die2 = args.min_die2 if args.min_die2 is not None else args.min_die
        _max_die2 = args.max_die2 if args.max_die2 is not None else args.max_die
        _die_step2 = args.die_step2 if args.die_step2 is not None else args.die_step
        total_dies = list(range(args.min_die + _min_die2, args.max_die + _max_die2 + 1, min(args.die_step, _die_step2)))
    else:
        total_dies = list(range(args.min_die, args.max_die + 1, args.die_step))
    throughput_cmd.extend(["--micro_batch_num", "2", "3"])
    throughput_cmd.extend(["--tpot_list"] + [str(t) for t in args.tpot])
    throughput_cmd.extend(["--kv_len_list"] + [str(k) for k in args.kv_len])
    throughput_cmd.extend(["--total_die"] + [str(t) for t in total_dies])
    subprocess.run(throughput_cmd)

    for tpot in args.tpot:
        for kv_len in args.kv_len:
            # Construct file name based on deployment mode
            if args.deployment_mode == "Heterogeneous":
                device_type2_enum = DeviceType(args.device_type2)
                file_name = f"{DeviceType(args.device_type).name}_{device_type2_enum.name}-{ModelType(args.model_type).name}-tpot{tpot}-kv_len{kv_len}.csv"
            else:
                file_name = f"{DeviceType(args.device_type).name}-{ModelType(args.model_type).name}-tpot{tpot}-kv_len{kv_len}.csv"
            pipeline_cmd = [
                "python", "src/visualization/pipeline.py",
                "--file_name", file_name,
                "--deployment_mode", args.deployment_mode,
            ]
            if args.deployment_mode == "Heterogeneous":
                pipeline_cmd.extend(["--device_type2", args.device_type2])
            subprocess.run(pipeline_cmd)


if __name__ == "__main__":
    main()

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
print("sys.path", sys.path)
from conf.model_config import ModelType
from conf.hardware_config import DeviceType


COLOR_MAP = {20: '#1f77b4', 50: '#ff7f52', 70: '#2ca02c',
             100: '#9467bd', 150: '#d62728'}


def throughput_vs_dies(
        file_name,
        min_die,
        max_die,
        deployment_mode="Homogeneous",
        device_type2=None
    ):
    """Generate throughput vs dies plot."""
    if deployment_mode == "Heterogeneous" and device_type2:
        # Heterogeneous mode - read combined DeepEP file
        deepep_dir = "data/deepep/heterogeneous/"
        afd_mbn2_dir = "data/afd/mbn2/best/heterogeneous/"
        afd_mbn3_dir = "data/afd/mbn3/best/heterogeneous/"
        deepep_path = deepep_dir + file_name  # Combined file: DEVICE1_DEVICE2-MODEL-...csv
        afd_mbn2_path = afd_mbn2_dir + file_name
        afd_mbn3_path = afd_mbn3_dir + file_name
        output_subdir = "heterogeneous/"
    else:
        deepep_dir = f"data/deepep/homogeneous/"
        afd_mbn2_dir = f"data/afd/mbn2/best/homogeneous/"
        afd_mbn3_dir = f"data/afd/mbn3/best/homogeneous/"
        deepep_path = deepep_dir + file_name
        afd_mbn2_path = afd_mbn2_dir + file_name
        afd_mbn3_path = afd_mbn3_dir + file_name
        output_subdir = "homogeneous/"

    # Check which files exist
    files_exist = {
        'deepep': os.path.exists(deepep_path),
        'afd_mbn2': os.path.exists(afd_mbn2_path),
        'afd_mbn3': os.path.exists(afd_mbn3_path)
    }

    if not any(files_exist.values()):
        print(f"No files found for {file_name}")
        return

    # Set up the plot
    plt.figure(figsize=(12, 8), dpi=300)
    plt.rcParams.update({'font.size': 14})

    # Load and plot data for existing files
    if deployment_mode == "Heterogeneous" and device_type2 and files_exist['deepep']:
        # For DeepEP heterogeneous, read the combined file with device1_die, device2_die, weighted_throughput
        deepep_df = pd.read_csv(deepep_path)

        # Read AFD files to get attn_die and ffn_die for each configuration
        if files_exist['afd_mbn2']:
            afd_df = pd.read_csv(afd_mbn2_path)
        elif files_exist['afd_mbn3']:
            afd_df = pd.read_csv(afd_mbn3_path)
        else:
            afd_df = None

        if afd_df is not None:
            weighted_throughputs = []
            total_dies = []
            for _, row in afd_df.iterrows():
                total_d = row['total_die']
                if total_d < min_die or total_d > max_die:
                    continue
                attn_die = int(row['attn_die'])
                ffn_die = int(row['ffn_die'])

                # Find DeepEP row where device1_die == attn_die and device2_die == ffn_die
                deepep_row = deepep_df[
                    (deepep_df['device1_die'] == attn_die) &
                    (deepep_df['device2_die'] == ffn_die)
                ]

                if len(deepep_row) > 0:
                    weighted_t = deepep_row['weighted_throughput(tokens/die/s)'].values[0]
                    weighted_throughputs.append(weighted_t)
                    total_dies.append(total_d)

            if total_dies:
                plt.plot(total_dies, weighted_throughputs, label='DeepEP (weighted)', color='#1f77b4', marker='o', linestyle='-')
    elif files_exist['deepep']:
        deepep_df = pd.read_csv(deepep_path)
        deepep_df = deepep_df[(deepep_df['total_die'] >= min_die) & (deepep_df['total_die'] <= max_die)]
        plt.plot(deepep_df['total_die'], deepep_df['throughput(tokens/die/s)'], label='DeepEP', color='#1f77b4', marker='o', linestyle='-')

    if files_exist['afd_mbn2']:
        afd_mbn2_df = pd.read_csv(afd_mbn2_path)
        afd_mbn2_df = afd_mbn2_df[(afd_mbn2_df['total_die'] >= min_die) & (afd_mbn2_df['total_die'] <= max_die)]
        plt.plot(afd_mbn2_df['total_die'], afd_mbn2_df['throughput(tokens/die/s)'], label='AFD MBN2', color='#ff7f0e', marker='s', linestyle='--')

    if files_exist['afd_mbn3']:
        afd_mbn3_df = pd.read_csv(afd_mbn3_path)
        afd_mbn3_df = afd_mbn3_df[(afd_mbn3_df['total_die'] >= min_die) & (afd_mbn3_df['total_die'] <= max_die)]
        plt.plot(afd_mbn3_df['total_die'], afd_mbn3_df['throughput(tokens/die/s)'], label='AFD MBN3', color='#2ca02c', marker='^', linestyle=':')

    # Add legend, labels, and title
    plt.legend(fontsize=12)
    plt.xlabel('Number of dies', fontsize=14, fontweight='bold')
    plt.ylabel('Throughput (tokens/s/die)', fontsize=14, fontweight='bold')
    title = file_name.split('.')[0]
    plt.title(title, fontsize=16, fontweight='bold')

    # Add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot
    result_dir = f'data/images/throughput/{output_subdir}'
    result_file_name = file_name
    result_path = result_dir + result_file_name.replace('.csv', '.png')
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {result_path}")


def throughput_vs_tpot_kvlen(device_type: DeviceType,
                             model_type: ModelType,
                             total_die: int,
                             tpot_list: list[int],
                             kv_len_list: list[int],
                             micro_batch_num: int,
                             deployment_mode: str = "Homogeneous",
                             device_type2: str = None):
    """Generate throughput comparison chart for different TPOT and KV lengths."""
    if deployment_mode == "Heterogeneous" and device_type2:
        device_type2_enum = DeviceType(device_type2)
        deepep_dir = "data/deepep/heterogeneous/"
        afd_dir = f"data/afd/mbn{micro_batch_num}/best/heterogeneous/"
        output_subdir = "heterogeneous/"
    else:
        deepep_dir = "data/deepep/homogeneous/"
        afd_dir = f"data/afd/mbn{micro_batch_num}/best/homogeneous/"
        output_subdir = "homogeneous/"

    fig, ax = plt.subplots(figsize=(8, 4))

    width = 0.15
    x_base = np.arange(len(kv_len_list))

    for idx, tpot in enumerate(tpot_list):
        improvement = []
        for kv_len in kv_len_list:
            if deployment_mode == "Heterogeneous" and device_type2:
                # AFD heterogeneous file
                afd_file_name = f"{device_type.name}_{device_type2_enum.name}-{model_type.name}-tpot{tpot}-kv_len{kv_len}.csv"
                afd_path = os.path.join(afd_dir, afd_file_name)

                # DeepEP homogeneous files for each device type separately
                deepep_file = f"{device_type.name}_{device_type2_enum.name}-{model_type.name}-tpot{tpot}-kv_len{kv_len}.csv"
                deepep_path = os.path.join(deepep_dir, deepep_file)

                if not (os.path.exists(afd_path) and os.path.exists(deepep_path)):
                    improvement.append(np.nan)
                    continue

                afd_df = pd.read_csv(afd_path)
                deepep_df = pd.read_csv(deepep_path)

                # Get AFD heterogeneous row with matching total_die
                afd_row = afd_df.loc[afd_df['total_die'] == total_die]
                if len(afd_row) == 0:
                    improvement.append(np.nan)
                    continue

                # Get attn_die and ffn_die from AFD configuration
                attn_die = int(afd_row['attn_die'].values[0])
                ffn_die = int(afd_row['ffn_die'].values[0])

                # Get DeepEP throughput on device1 with attn_die and device2 with ffn_die
                d = deepep_df.loc[
                    (deepep_df['device1_die'] == attn_die) &
                    (deepep_df['device2_die'] == ffn_die), 
                    'weighted_throughput(tokens/die/s)'].values

                if len(d) == 0:
                    improvement.append(np.nan)
                    continue

                a = afd_row['throughput(tokens/die/s)'].values
                improvement.append((a[0] - d[0]) / d[0] * 100)
            else:
                file_name = f"{device_type.name}-{model_type.name}-tpot{tpot}-kv_len{kv_len}.csv"
                deepep_path = os.path.join(deepep_dir, file_name)
                afd_path = os.path.join(afd_dir, file_name)
                if not (os.path.exists(deepep_path) and os.path.exists(afd_path)):
                    improvement.append(np.nan)
                    continue
                deepep_df = pd.read_csv(deepep_path)
                afd_df = pd.read_csv(afd_path)

                d = deepep_df.loc[deepep_df['total_die'] == total_die, 'throughput(tokens/die/s)'].values
                a = afd_df.loc[afd_df['total_die'] == total_die, 'throughput(tokens/die/s)'].values
                if len(d) and len(a):
                    improvement.append((a[0] - d[0]) / d[0] * 100)
                else:
                    improvement.append(np.nan)

        mask = ~pd.isna(improvement)
        ax.bar(x_base[mask] + idx * width,
               np.array(improvement)[mask],
               width,
               color=COLOR_MAP[tpot],
               label=f'TPOT={tpot}ms')

        miss = ~mask
        ax.scatter(x_base[miss] + idx * width,
                    [0]*miss.sum(),
                    marker='x', color='red', s=60, zorder=10)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('kv_len')
    ax.set_ylabel('improvement ratio / %')
    if deployment_mode == "Heterogeneous" and device_type2:
        ax.set_title(f'{device_type.name}_{device_type2_enum.name}-{model_type.name}-mbn{micro_batch_num}-total_die{total_die}')
    else:
        ax.set_title(f'{device_type.name}-{model_type.name}-mbn{micro_batch_num}-total_die{total_die}')
    ax.set_xticks(x_base + width * (len(tpot_list) - 1) / 2)
    ax.set_xticklabels(kv_len_list)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    os.makedirs(f'data/images/throughput/{output_subdir}', exist_ok=True)
    if deployment_mode == "Heterogeneous" and device_type2:
        out_path = f'data/images/throughput/{output_subdir}{device_type.name}_{device_type2_enum.name}-{model_type.name}-mbn{micro_batch_num}-total_die{total_die}.png'
    else:
        out_path = f'data/images/throughput/{output_subdir}{device_type.name}-{model_type.name}-mbn{micro_batch_num}-total_die{total_die}.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# -------------------- CLI --------------------
def add_args(p):
    p.add_argument('--model_type', type=str, default='deepseek-ai/DeepSeek-V3')
    p.add_argument('--device_type', type=str, default='Ascend_A3Pod')
    p.add_argument('--deployment_mode', type=str, default='Homogeneous',
                   choices=['Homogeneous', 'Heterogeneous'],
                   help='Deployment mode for visualization')
    p.add_argument('--device_type2', type=str, default=None,
                   help='Second device type for Heterogeneous mode')
    p.add_argument('--tpot_list', nargs='+', type=int, default=[20, 50, 70, 100, 150])
    p.add_argument('--kv_len_list', nargs='+', type=int,
                   default=[2048, 4096, 8192, 16384, 131072])
    p.add_argument('--total_die', nargs='+', type=int, default=[64])
    p.add_argument('--micro_batch_num', nargs='+', type=int, default=[2, 3])
    p.add_argument('--min_die', type=int, default=0)
    p.add_argument('--max_die', type=int, default=784)


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    device_type = DeviceType(args.device_type)
    model_type = ModelType(args.model_type)

    # Validate heterogeneous mode parameters
    if args.deployment_mode == "Heterogeneous" and not args.device_type2:
        print("Warning: device_type2 is required for Heterogeneous mode, skipping...")
        return

    for total_die in args.total_die:
        for micro_batch_num in args.micro_batch_num:
            throughput_vs_tpot_kvlen(device_type,
                                    model_type,
                                    total_die,
                                    args.tpot_list,
                                    args.kv_len_list,
                                    micro_batch_num,
                                    args.deployment_mode,
                                    args.device_type2)
    for tpot in args.tpot_list:
        for kv_len in args.kv_len_list:
            if args.deployment_mode == "Heterogeneous" and args.device_type2:
                device_type2_enum = DeviceType(args.device_type2)
                # For heterogeneous, pass the combined file name format
                # The function will extract device names and look up separate DeepEP files
                file_name = f"{device_type.name}_{device_type2_enum.name}-{model_type.name}-tpot{int(tpot)}-kv_len{kv_len}.csv"
            else:
                file_name = f"{device_type.name}-{model_type.name}-tpot{int(tpot)}-kv_len{kv_len}.csv"
            throughput_vs_dies(file_name, args.min_die, args.max_die, args.deployment_mode, args.device_type2)


if __name__ == '__main__':
    main()
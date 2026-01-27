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


def throughput_vs_dies(file_name, min_die, max_die):
    deepep_dir = f"data/deepep/"
    afd_mbn2_dir = f"data/afd/mbn2/best/"
    afd_mbn3_dir = f"data/afd/mbn3/best/"
    deepep_path = deepep_dir + file_name
    afd_mbn2_path = afd_mbn2_dir + file_name
    afd_mbn3_path = afd_mbn3_dir + file_name

    # Check if files exist
    if not os.path.exists(deepep_path):
        raise FileNotFoundError(f"File not found: {deepep_path}")
    if not os.path.exists(afd_mbn2_path):
        raise FileNotFoundError(f"File not found: {afd_mbn2_path}")
    if not os.path.exists(afd_mbn3_path):
        raise FileNotFoundError(f"File not found: {afd_mbn3_path}")

    # Load data
    deepep_df = pd.read_csv(deepep_path)
    afd_mbn2_df = pd.read_csv(afd_mbn2_path)
    afd_mbn3_df = pd.read_csv(afd_mbn3_path)

    # Filter data to include only total_die values in the range min_die to max_die
    deepep_df = deepep_df[(deepep_df['total_die'] >= min_die) & (deepep_df['total_die'] <= max_die)]
    afd_mbn2_df = afd_mbn2_df[(afd_mbn2_df['total_die'] >= min_die) & (afd_mbn2_df['total_die'] <= max_die)]
    afd_mbn3_df = afd_mbn3_df[(afd_mbn3_df['total_die'] >= min_die) & (afd_mbn3_df['total_die'] <= max_die)]

    # Set up the plot
    plt.figure(figsize=(12, 8), dpi=300)
    plt.rcParams.update({'font.size': 14})

    # Plot data
    plt.plot(deepep_df['total_die'], deepep_df['throughput(tokens/die/s)'], label='DeepEP', color='#1f77b4', marker='o', linestyle='-')
    plt.plot(afd_mbn2_df['total_die'], afd_mbn2_df['throughput(tokens/die/s)'], label='AFD MBN2', color='#ff7f0e', marker='s', linestyle='--')
    plt.plot(afd_mbn3_df['total_die'], afd_mbn3_df['throughput(tokens/die/s)'], label='AFD MBN3', color='#2ca02c', marker='^', linestyle=':')

    # Add legend, labels, and title
    plt.legend(fontsize=12)
    plt.xlabel('Number of dies', fontsize=14, fontweight='bold')
    plt.ylabel('Throughput (tokens/s/die)', fontsize=14, fontweight='bold')
    plt.title(file_name.split('.')[0], fontsize=16, fontweight='bold')

    # Add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot
    result_dir = 'data/images/throughput/'
    result_file_name = file_name.split('.')[0] + '.png'
    result_path = result_dir + result_file_name
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()

def throughput_vs_tpot_kvlen(device_type: DeviceType,
                             model_type: ModelType,
                             total_die: int,
                             tpot_list: list[int],
                             kv_len_list: list[int],
                             micro_batch_num: int):
    deepep_dir = "data/deepep/"
    afd_dir = f"data/afd/mbn{micro_batch_num}/best/"
    fig, ax = plt.subplots(figsize=(8, 4))

    width = 0.15
    x_base = np.arange(len(kv_len_list))

    for idx, tpot in enumerate(tpot_list):
        improvement = []
        for kv_len in kv_len_list:
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
    ax.set_title(f'{device_type.name}-{model_type.name}-mbn{micro_batch_num}-total_die{total_die}')
    ax.set_xticks(x_base + width * (len(tpot_list) - 1) / 2)
    ax.set_xticklabels(kv_len_list)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    os.makedirs('data/images/throughput', exist_ok=True)
    out_path = f'data/images/throughput/{device_type.name}-{model_type.name}-mbn{micro_batch_num}-total_die{total_die}.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -------------------- CLI --------------------
def add_args(p):
    p.add_argument('--model_type', type=str, default='deepseek-ai/DeepSeek-V3')
    p.add_argument('--device_type', type=str, default='Ascend_A3Pod')
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
    for total_die in args.total_die:
        for micro_batch_num in args.micro_batch_num:
            throughput_vs_tpot_kvlen(device_type,
                                    model_type,
                                    total_die,
                                    args.tpot_list,
                                    args.kv_len_list,
                                    micro_batch_num)
    for tpot in args.tpot_list:
        for kv_len in args.kv_len_list:
            file_name = f"{device_type.name}-{model_type.name}-tpot{int(tpot)}-kv_len{kv_len}.csv"
            throughput_vs_dies(file_name, args.min_die, args.max_die)

if __name__ == '__main__':
    main()
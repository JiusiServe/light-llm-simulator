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


def throughput_vs_tpot_kvlen(device_type1: DeviceType,
                             device_type2: DeviceType,
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
            file_name = f"{device_type1.name}-{device_type2.name}-{model_type.name}-tpot{tpot}-kv_len{kv_len}.csv"
            deepep_path = os.path.join(deepep_dir, file_name)
            afd_path = os.path.join(afd_dir, file_name)
            if not (os.path.exists(deepep_path) and os.path.exists(afd_path)):
                improvement.append(np.nan)
                continue
            deepep_df = pd.read_csv(deepep_path)
            afd_df = pd.read_csv(afd_path)
            a = afd_df.loc[afd_df['total_die'] == total_die, 'throughput(tokens/die/s)'].values
            total_die1 = afd_df.loc[afd_df['total_die'] == total_die, 'attn_die'].values
            total_die2 = afd_df.loc[afd_df['total_die'] == total_die, 'ffn_die'].values
            if len(total_die1) != 0 or len(total_die2) != 0:
                d = deepep_df.loc[
                    (deepep_df['total_die'] == total_die) &
                    (deepep_df['total_die1'] == total_die1[0]) &
                    (deepep_df['total_die2'] == total_die2[0]),
                    'throughput(tokens/die/s)'
                ].values
            else:
                d = []
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
    ax.set_title(f'{device_type1.name}-{device_type2.name}-{model_type.name}-mbn{micro_batch_num}-total_die{total_die}')
    ax.set_xticks(x_base + width * (len(tpot_list) - 1) / 2)
    ax.set_xticklabels(kv_len_list)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    os.makedirs('data/images/throughput', exist_ok=True)
    out_path = f'data/images/throughput/{device_type1.name}-{device_type2.name}-{model_type.name}-mbn{micro_batch_num}-total_die{total_die}.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -------------------- CLI --------------------
def add_args(p):
    p.add_argument('--model_type', type=str, default='deepseek-ai/DeepSeek-V3')
    p.add_argument('--device_type1', type=str, default='Ascend_David121')
    p.add_argument('--device_type2', type=str, default='Ascend_David120')
    p.add_argument('--tpot_list', nargs='+', type=int, default=[20, 50, 70, 100, 150])
    p.add_argument('--kv_len_list', nargs='+', type=int,
                   default=[2048, 4096, 8192, 16384, 131072])
    p.add_argument('--total_die', nargs='+', type=int, default=[64])
    p.add_argument('--micro_batch_num', nargs='+', type=int, default=[2, 3])

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    device_type1 = DeviceType(args.device_type1)
    device_type2 = DeviceType(args.device_type2)
    model_type = ModelType(args.model_type)
    for total_die in args.total_die:
        for micro_batch_num in args.micro_batch_num:
            throughput_vs_tpot_kvlen(device_type1,
                                    device_type2,
                                    model_type,
                                    total_die,
                                    args.tpot_list,
                                    args.kv_len_list,
                                    micro_batch_num)

if __name__ == '__main__':
    main()
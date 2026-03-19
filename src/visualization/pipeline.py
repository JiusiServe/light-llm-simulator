import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from conf.hardware_config import DeviceType
from conf.model_config import ModelType


color_map = {
    'attn': '#1f77b4',
    'dispatch': '#ff7f0e',
    'moe': '#2ca02c',
    'combine': '#d62728'
}

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model_type', type=str, default='deepseek-ai/DeepSeek-V3')
    parser.add_argument('--device_type1', type=str, default='Ascend_David120')
    parser.add_argument('--device_type2', type=str, default='Ascend_David100')
    parser.add_argument('--tpot', nargs='+', type=int, default=[20, 50, 70, 100, 150])
    parser.add_argument('--kv_len', nargs='+', type=int, default=[2048, 4096, 8192, 16384, 131072])


def create_gantt_chart(mbn, attn_time, dispatch_time, moe_time, combine_time, total_die, file_name, attn_die=None, ffn_die=None, device_type=None):
    start_time, attn_tmp, dispatch_tmp, ffn_tmp = 0.0, 0.0, 0.0, 0.0
    fig, ax = plt.subplots(figsize=(10, 4))
    for micro_id in range(mbn):
        attn_end = start_time + attn_time
        dispatch_end = max(attn_end, attn_tmp) + dispatch_time
        ffn_end = max(dispatch_end, dispatch_tmp) + moe_time
        combine_end = max(ffn_end, ffn_tmp) + combine_time

        tasks = [
            ("attn", start_time, attn_end),
            ("dispatch", max(attn_end, attn_tmp), dispatch_end),
            ("moe", max(dispatch_end, dispatch_tmp), ffn_end),
            ("combine", max(ffn_end, ffn_tmp), combine_end)
        ]
        attn_tmp, dispatch_tmp, ffn_tmp = dispatch_end, ffn_end, combine_end
        start_time = attn_end  

        for i, (label, start, end) in enumerate(tasks):
            ax.barh(len(tasks) - i - 1, end - start, left=start, height=1, align='center', edgecolor='black', color=color_map[label])
            ax.text(start + (end - start) / 2, len(tasks) - i - 1, str(int(end-start)) + 'us', ha='center', va='center')

    ax.set_yticks([len(tasks) - i - 1 for i in range(len(tasks))])
    ax.set_yticklabels([task[0] for task in tasks])
    if mbn == 1:
        title = 'deepep-total_die' + str(int(total_die))
    elif mbn == 2:
        title = 'AFD-mbn2-total_die' + str(int(total_die)) + '-attn' + str(int(attn_die)) + '-ffn' + str(int(ffn_die))
    elif mbn == 3:
        title = 'AFD-mbn3-total_die' + str(int(total_die)) + '-attn' + str(int(attn_die)) + '-ffn' + str(int(ffn_die))
    else:
        raise ValueError('Invalid mbn')

    pipeline_name = 'total_die' + str(int(total_die))

    ax.set_title(title)
    if mbn == 1:
        save_dir = 'data/images/pipeline/deepep/'
        os.makedirs(save_dir, exist_ok=True)
        if device_type is not None:
            file_name = device_type + '-' + file_name.split('-')[2] + '-' + file_name.split('-')[3] + '-' + file_name.split('.')[0].split('-')[4]
        save_path = save_dir + file_name + '-' + pipeline_name + '.png'
        ax.set_title('deepep-' + title)
    elif mbn == 2:
        save_dir = 'data/images/pipeline/mbn2/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir + file_name.split('.')[0] + '-' + pipeline_name + '.png'
        ax.set_title('AFD-mbn2-' + title)
    elif mbn == 3:
        save_dir = 'data/images/pipeline/mbn3/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir + file_name.split('.')[0] + '-' + pipeline_name + '.png'
        ax.set_title('AFD-mbn3-' + title)
    fig.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()
    device_type1 = DeviceType(args.device_type1)
    device_type2 = DeviceType(args.device_type2)
    model_type = ModelType(args.model_type)
    for tpot in args.tpot:
        for kv_len in args.kv_len:
            file_name = f"{device_type1.name}-{device_type2.name}-{model_type.name}-tpot{tpot}-kv_len{kv_len}.csv"
    for mbn in [1, 2, 3]:
        if mbn == 1:
            mbn_path = 'data/deepep/' + file_name
        elif mbn == 2:
            mbn_path = 'data/afd/mbn2/best/' + file_name
        elif mbn == 3:
            mbn_path = 'data/afd/mbn3/best/' + file_name
        df = pd.read_csv(mbn_path)
        for index, row in df.iterrows():
            if mbn == 1:
                attn_time, dispatch_time, moe_time, combine_time = row['attn_time1(us)'], row['dispatch_time1(us)'], row['moe_time1(us)'], row['combine_time1(us)']
                total_die = row['total_die1']
                device_type = file_name.split('-')[0]
                create_gantt_chart(mbn, attn_time, dispatch_time, moe_time, combine_time, total_die, file_name, device_type=device_type)
                attn_time, dispatch_time, moe_time, combine_time = row['attn_time2(us)'], row['dispatch_time2(us)'], row['moe_time2(us)'], row['combine_time2(us)']
                total_die = row['total_die2']
                device_type = file_name.split('-')[1]
                create_gantt_chart(mbn, attn_time, dispatch_time, moe_time, combine_time, total_die, file_name, device_type=device_type)
            else:
                attn_die = row['attn_die']
                ffn_die = row['ffn_die']
                attn_time, dispatch_time, moe_time, combine_time = row['attn_time(us)'], row['dispatch_time(us)'], row['moe_time(us)'], row['combine_time(us)']
                total_die= row['total_die']
                create_gantt_chart(mbn, attn_time, dispatch_time, moe_time, combine_time, total_die, file_name, attn_die=attn_die, ffn_die=ffn_die)

if __name__ == "__main__":
    main()

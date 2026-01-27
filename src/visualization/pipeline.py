import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


color_map = {
    'attn': '#1f77b4',
    'dispatch': '#ff7f0e',
    'moe': '#2ca02c',
    'combine': '#d62728'
}

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--file_name', type=str, default="ASCENDA3_Pod-DEEPSEEK_V3-tpot50-kv_len4096.csv")


def create_gantt_chart(mbn, attn_time, dispatch_time, moe_time, combine_time, total_die, file_name):
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
    title = 'total_die' + str(int(total_die))

    ax.set_title(title)
    if mbn == 1:
        save_dir = 'data/images/pipeline/deepep/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir + file_name.split('.')[0] + '-' + title + '.png'
        ax.set_title('deepep-' + title)
    elif mbn == 2:
        save_dir = 'data/images/pipeline/mbn2/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir + file_name.split('.')[0] + '-' + title + '.png'
        ax.set_title('AFD-mbn2-' + title)
    elif mbn == 3:
        save_dir = 'data/images/pipeline/mbn3/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir + file_name.split('.')[0] + '-' + title + '.png'
        ax.set_title('AFD-mbn3-' + title)
    fig.savefig(save_path)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()
    for mbn in [1, 2, 3]:
        if mbn == 1:
            mbn_path = 'data/deepep/' + args.file_name
        elif mbn == 2:
            mbn_path = 'data/afd/mbn2/best/' + args.file_name
        elif mbn == 3:
            mbn_path = 'data/afd/mbn3/best/' + args.file_name
        df = pd.read_csv(mbn_path)
        for index, row in df.iterrows():
            attn_time, dispatch_time, moe_time, combine_time = row['attn_time(us)'], row['dispatch_time(us)'], row['moe_time(us)'], row['combine_time(us)']
            attn_bs, ffn_bs, total_die, throughput = row['attn_bs'], row['ffn_bs'], row['total_die'], row['throughput(tokens/die/s)']
            create_gantt_chart(mbn, attn_time, dispatch_time, moe_time, combine_time, total_die, args.file_name)

if __name__ == "__main__":
    main()

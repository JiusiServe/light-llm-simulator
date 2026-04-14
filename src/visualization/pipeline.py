import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


color_map = {
    'attn': '#1f77b4',
    'attn+a2e': '#1f77b4',
    'dispatch': '#ff7f0e',
    'moe': '#2ca02c',
    'combine': '#d62728',
    'combine+e2a-recv': '#d62728',
    'commu': '#9467bd'
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--file_name', type=str, default="ASCENDA3_Pod-DEEPSEEK_V3-tpot50-kv_len4096.csv")
    parser.add_argument('--deployment_mode', type=str, default="Homogeneous",
                        choices=['Homogeneous', 'Heterogeneous'],
                        help='Deployment mode for visualization')
    parser.add_argument('--device_type2', type=str, default=None,
                        help='Second device type for Heterogeneous mode')


def create_gantt_chart_afd(mbn, attn_time, a2e_time, dispatch_time, moe_time, combine_time, e2a_recv_time, total_die, file_name, deployment_mode="Homogeneous"):
    """Create Gantt chart for AFD serving mode (has dispatch/combine phases)."""
    start_time, attn_tmp, dispatch_tmp, ffn_tmp = 0.0, 0.0, 0.0, 0.0
    fig, ax = plt.subplots(figsize=(12, 4))
    for micro_id in range(mbn):
        attn_end = start_time + attn_time + a2e_time
        dispatch_end = max(attn_end, attn_tmp) + dispatch_time
        ffn_end = max(dispatch_end, dispatch_tmp) + moe_time
        combine_end = max(ffn_end, ffn_tmp) + combine_time + e2a_recv_time

        tasks = [
            ("attn+a2e", start_time, attn_end),
            ("dispatch", max(attn_end, attn_tmp), dispatch_end),
            ("moe", max(dispatch_end, dispatch_tmp), ffn_end),
            ("combine+e2a-recv", max(ffn_end, ffn_tmp), combine_end)
        ]
        attn_tmp, dispatch_tmp, ffn_tmp = dispatch_end, ffn_end, combine_end
        start_time = attn_end

        for i, (label, start, end) in enumerate(tasks):
            ax.barh(len(tasks) - i - 1, end - start, left=start, height=1, align='center', edgecolor='black', color=color_map[label])
            ax.text(start + (end - start) / 2, len(tasks) - i - 1, str(int(end-start)) + 'us', ha='center', va='center')

    ax.set_yticks([len(tasks) - i - 1 for i in range(len(tasks))])
    ax.set_yticklabels([task[0] for task in tasks])

    # Build title and save path
    base_name = file_name.split('.')[0]
    die_str = f"-total_die{int(total_die)}"

    if deployment_mode == "Heterogeneous":
        save_dir = 'data/images/pipeline/afd/heterogeneous/'
        os.makedirs(save_dir, exist_ok=True)
        title = f"afd-mbn{mbn}{die_str}"
        save_path = save_dir + base_name + f"-afd-mbn{mbn}{die_str}.png"
    else:
        save_dir = 'data/images/pipeline/afd/homogeneous/'
        os.makedirs(save_dir, exist_ok=True)
        title = f"afd-mbn{mbn}{die_str}"
        save_path = save_dir + base_name + f"-afd-mbn{mbn}{die_str}.png"

    ax.set_title(title)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_gantt_chart_deepep(attn_time, dispatch_time, moe_time, combine_time, total_die, file_name, deployment_mode="Homogeneous"):
    """Create Gantt chart for DeepEP serving mode."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # DeepEP pipeline: attn -> dispatch -> moe -> combine
    attn_end = attn_time
    dispatch_end = attn_end + dispatch_time
    moe_end = dispatch_end + moe_time
    combine_end = moe_end + combine_time

    tasks = [
        ("attn", 0, attn_end),
        ("dispatch", attn_end, dispatch_end),
        ("moe", dispatch_end, moe_end),
        ("combine", moe_end, combine_end)
    ]

    for i, (label, start, end) in enumerate(tasks):
        ax.barh(len(tasks) - i - 1, end - start, left=start, height=1, align='center', edgecolor='black', color=color_map[label])
        ax.text(start + (end - start) / 2, len(tasks) - i - 1, str(int(end-start)) + 'us', ha='center', va='center')

    ax.set_yticks([len(tasks) - i - 1 for i in range(len(tasks))])
    ax.set_yticklabels([task[0] for task in tasks])

    # Build title and save path
    base_name = file_name.split('.')[0]
    die_str = f"-total_die{int(total_die)}"

    if deployment_mode == "Heterogeneous":
        save_dir = 'data/images/pipeline/deepep/heterogeneous/'
        os.makedirs(save_dir, exist_ok=True)
        title = f"deepep-mbn1{die_str}"
        save_path = save_dir + base_name + f"-deepep-mbn1{die_str}.png"
    else:
        save_dir = 'data/images/pipeline/deepep/homogeneous/'
        os.makedirs(save_dir, exist_ok=True)
        title = f"deepep-mbn1{die_str}"
        save_path = save_dir + base_name + f"-deepep-mbn1{die_str}.png"

    ax.set_title(title)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()

    # Determine file paths based on deployment mode
    if args.deployment_mode == "Heterogeneous" and args.device_type2:
        paths = {
            1: f'data/deepep/heterogeneous/{args.file_name}',
            2: f'data/afd/mbn2/best/heterogeneous/{args.file_name}',
            3: f'data/afd/mbn3/best/heterogeneous/{args.file_name}'
        }
    else:
        paths = {
            1: f'data/deepep/homogeneous/{args.file_name}',
            2: f'data/afd/mbn2/best/homogeneous/{args.file_name}',
            3: f'data/afd/mbn3/best/homogeneous/{args.file_name}'
        }

    for mbn in [1, 2, 3]:
        mbn_path = paths[mbn]
        if not os.path.exists(mbn_path):
            print(f"Skipping mbn={mbn}: file not found: {mbn_path}")
            continue
        df = pd.read_csv(mbn_path)

        if mbn == 1:
            # DeepEP mode - uses attn, dispatch, moe, combine columns
            required_cols = ['attn_time(us)', 'dispatch_time(us)', 'moe_time(us)', 'combine_time(us)']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping mbn={mbn}: missing required columns for DeepEP")
                continue
            for index, row in df.iterrows():
                attn_time = row['attn_time(us)']
                dispatch_time = row['dispatch_time(us)']
                moe_time = row['moe_time(us)']
                combine_time = row['combine_time(us)']
                total_die = row['total_die']
                create_gantt_chart_deepep(attn_time, dispatch_time, moe_time, combine_time, total_die, args.file_name, args.deployment_mode)
        else:
            # AFD mode - uses attn, a2e, dispatch, moe, combine, e2a_recv columns
            required_cols = ['attn_time(us)', 'dispatch_time(us)', 'moe_time(us)', 'combine_time(us)']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping mbn={mbn}: missing required columns for AFD")
                continue
            a2e_col1 = 'a2e_send(us)'
            a2e_col2 = 'a2e_recv(us)'
            e2a_recv_col = 'e2a_recv(us)'
            for index, row in df.iterrows():
                attn_time = row['attn_time(us)']
                a2e_time = (row['a2e_send(us)'] + row['a2e_recv(us)']) if (a2e_col1 in row and a2e_col2 in row) else 0
                dispatch_time = row['dispatch_time(us)']
                moe_time = row['moe_time(us)']
                combine_time = row['combine_time(us)']
                e2a_recv_time = row[e2a_recv_col] if e2a_recv_col in row else 0
                total_die = row['total_die']
                create_gantt_chart_afd(mbn, attn_time, a2e_time, dispatch_time, moe_time, combine_time, e2a_recv_time, total_die, args.file_name, args.deployment_mode)


if __name__ == "__main__":
    main()

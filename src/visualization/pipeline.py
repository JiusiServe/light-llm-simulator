import pandas as pd
import matplotlib.pyplot as plt

def create_gantt_chart(mbn, attn_time, dispatch_time, moe_time, combine_time, total_die):
    start_time, attn_tmp, dispatch_tmp, ffn_tmp = 0.0, 0.0, 0.0, 0.0
    fig, ax = plt.subplots(figsize=(10, 4))
    for micro_id in range(mbn):
        attn_end = start_time + attn_time
        dispatch_end = max(attn_end, attn_tmp) + dispatch_time
        ffn_end = max(dispatch_end, dispatch_tmp) + moe_time
        combine_end = max(ffn_end, ffn_tmp) + combine_time

        tasks = [
            ("attn_time", start_time, attn_end),
            ("dispatch_time", max(attn_end, attn_tmp), dispatch_end),
            ("moe_time", max(dispatch_end, dispatch_tmp), ffn_end),
            ("combine_time", max(ffn_end, ffn_tmp), combine_end)
        ]
        attn_tmp, dispatch_tmp, ffn_tmp = dispatch_end, ffn_end, combine_end
        start_time = attn_end  

        for i, (label, start, end) in enumerate(tasks):
            ax.barh(len(tasks) - i - 1, end - start, left=start, height=1, align='center', edgecolor='black')
            ax.text(start + (end - start) / 2, len(tasks) - i - 1, label, ha='center', va='center')

    ax.set_yticks([len(tasks) - i - 1 for i in range(len(tasks))])
    ax.set_yticklabels([task[0] for task in tasks])
    title = 'total_die' + str(total_die)
    ax.set_title(title)
    if mbn == 2:
        save_path = 'data/image/pipeline/mbn2/' + title + '.png'
    else:
        save_path = 'data/image/pipeline/mbn3/' + title + '.png'
    fig.savefig(save_path)

for mbn in [2, 3]:
    if mbn == 2:
        mbn_path = 'data/afd/mbn2/best/perf_afd_mbn2_latency50_best_results.csv'
    else:
        mbn_path = 'data/afd/mbn3/best/perf_afd_mbn3_latency50_best_results.csv'
    df = pd.read_csv(mbn_path)
    for index, row in df.iterrows():
        attn_time, dispatch_time, moe_time, combine_time = row['attn_time'], row['dispatch_time'], row['moe_time'], row['combine_time']
        attn_bs, ffn_bs, total_die, throughput = row['attn_bs'], row['ffn_bs'], row['total_die'], row['throughput']
        create_gantt_chart(mbn, attn_time, dispatch_time, moe_time, combine_time, total_die)

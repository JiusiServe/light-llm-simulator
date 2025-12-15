import matplotlib.pyplot as plt
import pandas as pd
from conf.common import US_2_MS


tpx = [1/(latency*US_2_MS) for latency in range(50, 201, 10)]

for total_die in range(16, 769, 16):
    throughput_afd_mbn2_best, throughput_afd_mbn3_best, throughput_deepep = [], [], []
    for latency in range(50, 201, 10):
        value_afd_mbn2_best, value_afd_mbn3_best, value_deepep = 0, 0, 0
        mbn2_path = 'data/afd/mbn2/best/perf_afd_mbn2_latency'+str(latency)+'_best_results.csv'
        mbn3_path = 'data/afd/mbn3/best/perf_afd_mbn3_latency'+str(latency)+'_best_results.csv'
        deepep_path = 'data/deepep/perf_deepep_latency'+str(latency)+'_results.csv'
        df_afd_mbn2_best = pd.read_csv(mbn2_path)
        df_afd_mbn3_best = pd.read_csv(mbn3_path)
        df_deepep = pd.read_csv(deepep_path)
        for index, row in df_afd_mbn2_best.iterrows():
            if row['total_die'] == total_die:
                value_afd_mbn2_best = row['throughput']
                break
        throughput_afd_mbn2_best.append(value_afd_mbn2_best)
        for index, row in df_afd_mbn3_best.iterrows():
            if row['total_die'] == total_die:
                value_afd_mbn3_best = row['throughput']
                break
        throughput_afd_mbn3_best.append(value_afd_mbn3_best)
        for index, row in df_deepep.iterrows():
            if row['total_die'] == total_die:
                value_deepep = row['throughput']
                break
        throughput_deepep.append(value_deepep)
    plt.scatter(tpx, throughput_afd_mbn2_best, marker='2', label='mbn2', color='#1f77b4')
    plt.scatter(tpx, throughput_afd_mbn3_best, marker='3', label='mbn3',color='#ff7f0e')
    plt.scatter(tpx, throughput_deepep, marker='d', label='deepep',color='#2ca02c')
    plt.legend()
    plt.xlabel('tpx(tokens/s/user)')
    plt.ylabel('throughput')
    title = 'total_die'+ str(int(total_die))
    plt.title(title)
    save_path = 'data/image/pareto/' + title + '.png'
    plt.savefig(save_path)
    plt.clf()

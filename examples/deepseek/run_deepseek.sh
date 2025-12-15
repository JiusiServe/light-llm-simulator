# run in INFO level to get output.log
rm -rf ./output.log
cur_date=$(LC_ALL=en_US.utf8 date + "%b%d-%H%M") && echo $cur_date

export LOG_LEVEL=INFO
python examples/deepseek/deepseek.py info > data/output-${cur_date}.log 2>&1
ln -s data/output-${cur_date}.log ./output.log
echo "saved perf datas to data/output-${cur_date}.log"

# visualization
# Generates Pareto frontier images showing the trade-off between latency and throughput for different configurations.
python ./src/visualization/pareto.py
# Visualizes the AFD (Attention-FFN disaggregated) pipeline to identify bubble
python ./src/visualization/pipeline.py
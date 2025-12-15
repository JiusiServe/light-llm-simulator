## Examples

This directory contains runnable examples for **Light LLM Simulator**.

### Layout

- `deepseek/`
  - `deepseek.py`: Python example that runs AFD and DeepEP once with a small search space.
  - `run_deepseek.sh`: Convenience shell script to run the example and generate visualizations.

### DeepSeek Example

#### Run with the helper shell script

```bash
bash examples/deepseek/run_deepseek.sh
```
This will:
This script will:
- Set `LOG_LEVEL=INFO`.
- Run the `deepseek.py` example
    - Construct a **DeepSeek-V3** model config.
    - Use **Ascend 910B2 (8 ranks × 8 NPUs)** as the hardware topology.
    - Run **AFD** search once with:
        - `latency = [50, 60, 70, ..., 190, 200] ms`
        - `micro_batch_num = 3`
        - `kv_len = 4096`
        - `seq_len = 1 + next_n` (with `next_n = 1`)
    - Run **DeepEP** search once as a baseline.
- Save logs to:
  - `data/output-<timestamp>.log`
  - Also create a symlink `output.log` pointing to the latest run.
- Run visualization scripts:
  - `src/visualization/pareto.py` (Pareto frontier plots)
  - `src/visualization/pipeline.py` (AFD pipeline visualization)

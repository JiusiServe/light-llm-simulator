# Installation Guide

This guide covers installing and setting up Light LLM Simulator.

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/light-llm-simulator.git
cd light-llm-simulator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Simulator

- Run with default settings (DeepSeek-V3, latency 50-200ms, micro_batch_num 2/3):

    ```bash
    python src/cli/main.py
    ```

 - Run with Customize Parameters

    ```bash
    python src/cli/main.py \
        --model_name "deepseek-ai/DeepSeek-V3" \
        --latency 100 150 200 \
        --kv_len 4096 \
        --micro_batch_num 2 3 \
        --next_n 1 \
        --multi_token_ratio 0.7
    ```

### 4. Visualization

```bash
# Generates Pareto frontier images showing the trade-off between latency and throughput for different configurations.
python src/visualization/pareto.py
# Visualizes the AFD (Attention-FFN disaggregated) pipeline to identify bubble
python src/visualization/pipeline.py
```

## Parameter Explanation

#### `--model_name`
- Model identifier (see `ModelType` enum)
- Default: `"deepseek-ai/DeepSeek-V3"`

#### `--latency`
- Latency constraints in milliseconds
- Can specify multiple values: `--latency 50 100 150`
- Default: `50, 60, 70, ..., 200` (step 10)

#### `--kv_len`
- KV cache length
- Affects memory usage
- Default: `4096`

#### `--micro_batch_num`
- Micro-batch numbers to explore
- Default: `[2, 3]`

#### `--next_n`
- Predict the next n tokens through the MTP(Multi-Token Prediction) technique.
- `seq_len = 1 + next_n`
- Default: `1`

#### `--multi_token_ratio`
- The acceptance rate of the additionally predicted token
- Affects throughput calculation
- Default: `0.7`

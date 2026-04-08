# AFD (Attention-FFN Disaggregated) Search

AFD is a deployment strategy that disaggregates Attention and FFN computations across different hardware resources to optimize throughput while meeting latency targets.

## Overview

In AFD serving, Attention and FFN workers are separated, allowing independent scaling and optimization. The search algorithm finds optimal configurations by:

1. Determining the maximum attention batch size that satisfies latency and memory targets
2. Exploring different combinations of Attention and FFN die allocations
3. Selecting configurations with highest throughput, deduplicating by total die and sorting by throughput

## Deployment Modes

AFD supports two deployment modes:

### Homogeneous (Default)
- Both Attention and FFN run on the same device type
- `device_type` applies to all modules
- Original AFD behavior

### Heterogeneous
- Attention runs on `device_type` (device1)
- FFN runs on `device_type2` (device2)
- Total die = attn_die + ffn_die
- Allows mixing different accelerator types for optimal cost-performance

See [Heterogeneous AFD Design](../plans/heterogeneous-afd-design.md) for detailed documentation.

## Output Columns

The search outputs the following columns:

- `attn_bs`: Attention batch size
- `ffn_bs`: FFN batch size
- `kv_len`: KV cache length
- `attn_die`: Number of attention dies
- `ffn_die`: Number of FFN dies
- `total_die`: Number of total dies
- `attn_time`: Attention time for per layer (μs)
- `ffn_time`: FFN time for per layer (μs)
- `commu_time`: communication time per layer(μs)
- `e2e_time`: End-to-end time (ms)
- `e2e_time_per_dense_layer`: End-to-end time for per dense layers (μs)
- `e2e_time_per_moe_layer`: End-to-end time for per MoE layers (μs)
- `throughput`: Throughput (tokens/second)
- `deployment_mode`: Deployment mode ("Homogeneous" or "Heterogeneous")
- `device_type_attn`: Device type for attention module
- `device_type_ffn`: Device type for FFN module

The above output results are cached in CSV files for analysis

## Usage

### Homogeneous Mode

```python
from src.search.afd import AfdSearch

config = Config(
   serving_mode,
   model_type,
   device_type,
   min_attn_bs,
   max_attn_bs,
   min_die,
   max_die,
   tpot,
   kv_len,
   micro_batch_num,
   next_n,
   multi_token_ratio,
   attn_tensor_parallel,
   ffn_tensor_parallel
)
afd_search = AfdSearch(config)
afd_search.deployment()
```

### Heterogeneous Mode

```python
from src.search.afd import AfdSearch

config = Config(
   serving_mode="AFD",
   model_type,
   device_type,              # Attention device
   device_type2,             # FFN device (new)
   min_attn_bs,
   max_attn_bs,
   min_die,                  # Attention die range
   max_die,
   min_die2,                 # FFN die range (new)
   max_die2,                 # (new)
   die_step2,                # (new)
   tpot,
   kv_len,
   micro_batch_num,
   next_n,
   multi_token_ratio,
   attn_tensor_parallel,
   ffn_tensor_parallel,
   deployment_mode="Heterogeneous"  # new
)
afd_search = AfdSearch(config)
afd_search.deployment()
```

### CLI Usage

```bash
# Homogeneous (default)
python src/cli/main.py --serving_mode AFD --device_type Ascend_A3Pod

# Heterogeneous
python src/cli/main.py \
  --serving_mode AFD \
  --deployment_mode Heterogeneous \
  --device_type Ascend_A3Pod \
  --device_type2 Ascend_David121 \
  --min_die 16 --max_die 256 \
  --min_die2 16 --max_die2 128
```

## Targets

### Latency Targets

1. **Attention Module Latency**:
   ```
   micro_batch_num * attn_time < latency / num_layers * (1 + multi_token_ratio)
   ```

2. **MoE Module Latency**:
    ```
    micro_batch_num * moe_time < latency / num_layers * (1 + multi_token_ratio)
    ```
3. **MoE Layer Latency**:
   ```
   e2e_time_per_moe_layer = max(attn_time + moe_time + commu_time,max(attn_time, moe_time) * self.micro_batch_num)

   e2e_time_per_moe_layer * num_layers < latency * (1 + multi_token_ratio)
   ```

### Memory Targets

1. **Attention Memory**:
   ```
   kv_size * micro_batch_num + attn_static_memory < npu_memory * MEMORY_THRESHOLD_RATIO
   ```

2. **FFN Static Memory**:
   ```
   ffn_static_memory + ffn dynamic memory < npu_memory * MEMORY_THRESHOLD_RATIO
   ```

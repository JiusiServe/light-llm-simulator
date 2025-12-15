# AFD (Attention-FFN Disaggregated) Search

AFD is a deployment strategy that disaggregates Attention and FFN computations across different hardware resources to optimize throughput while meeting latency targets.

## Overview

In AFD serving, Attention and FFN workers are separated, allowing independent scaling and optimization. The search algorithm finds optimal configurations by:

1. Determining the maximum attention batch size that satisfies latency and memory targets
2. Exploring different combinations of Attention and FFN die allocations
3. Selecting configurations with highest throughput, deduplicating by total die and sorting by throughput

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

The above output results are cached in CSV files for analysis

## Usage

```python
from conf.model_config import ModelType, ModelConfig
from conf.hardware_config import HardwareTopology, DeviceType
from src.search.afd import AfdSearch

# Create configurations
model_type = ModelType.DEEPSEEK_V3
model_config = ModelConfig.create_model_config(model_type)
hw_topology = HardwareTopology.create(
    number_of_ranks=8,
    npus_per_rank=8,
    device_type=DeviceType.ASCEND910B2_376T_64G
)

# Run AFD search
afd_search = AfdSearch(
    model_type=model_type,
    model_config=model_config,
    aichip_config=hw_topology,
    kv_len=4096,
    seq_len=2,
    multi_token_ratio=0.7,
    latency=100,  # ms
    micro_batch_num=2
)

afd_search.deployment()
```

## Parameters

- `model_type`: Model type enum
- `model_config`: Model configuration object
- `aichip_config`: Hardware topology configuration
- `kv_len`: KV cache length
- `seq_len`: Sequence length (1 + next_x)
- `multi_token_ratio`: Multi-token generation ratio
- `latency`: Latency targets in milliseconds
- `micro_batch_num`: Micro-batch number for pipelining

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

# DeepEP Search

DeepEP is a baseline deployment strategy that uses traditional expert parallelism for MoE models, serving as a comparison baseline for AFD search.

## Overview

DeepEP distributes experts across available hardware resources using standard expert parallelism. Unlike AFD, it doesn't disaggregate Attention and FFN workers, instead using a unified die allocation.The search algorithm finds optimal configurations by:

1. Explore total die counts from 16 to 769 (step 16)
2. Calculate Routed Experts Per Die
    ```
    routed_expert_per_die = max(2, ceil(n_routed_experts / total_die))
    ```
3. Binary search to find the maximum `attn_bs` that satisfies latency and memory targets

## Usage

```python
from conf.model_config import ModelType, ModelConfig
from conf.hardware_config import HardwareTopology, DeviceType
from src.search.deepep import DeepEpSearch

# Create configurations
model_type = ModelType.DEEPSEEK_V3
model_config = ModelConfig.create_model_config(model_type)
hw_topology = HardwareTopology.create(
    number_of_ranks=8,
    npus_per_rank=8,
    device_type=DeviceType.ASCEND910B2_376T_64G
)

# Run DeepEP search
deepep_search = DeepEpSearch(
    model_type=model_type,
    model_config=model_config,
    aichip_config=hw_topology,
    kv_len=4096,
    seq_len=2,
    multi_token_ratio=0.7,
    latency=100,  # ms
    micro_batch_num=1
)

deepep_search.deployment()
```

## Targets

### Latency Targets
```
e2e_time < latency / micro_batch_num * (1 + multi_token_ratio)
```

Where `e2e_time` is:
```
attn_time * num_layers + 
mlp_time * first_k_dense_replace + 
moe_time * num_moe_layers + 
commu_time * num_layers
```

### Memory Targets

```
total_memory < npu_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO
```

Where `total_memory` includes:
- KV cache (dynamic)
- Attention static memory
- MoE expert memory (static + dynamic)

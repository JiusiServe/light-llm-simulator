# DeepEP Search

DeepEP is a deployment strategy that uses traditional expert parallelism for MoE models.

## Overview

DeepEP distributes experts across available hardware resources using standard expert parallelism. The search algorithm finds optimal configurations by:

1. Explore total die counts from 16 to 769 (step 16)
2. Calculate Routed Experts Per Die (via `Config.calc_routed_expert_per_die`)
3. Binary search to find the maximum `attn_bs` that satisfies latency and memory targets

## Usage

```python
from src.search.deepep import DeepEpSearch

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
deepep_search = DeepEpSearch(config)
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
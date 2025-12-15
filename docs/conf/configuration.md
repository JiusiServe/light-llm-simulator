# Configuration Guide

This guide explains how to configure models, hardware in Light LLM Simulator.

## Model Configuration

### Supported Models

Models are configured in `conf/model_config.py`. Currently supported:

- **DeepSeek V3**: `deepseek-ai/DeepSeek-V3` (fully supported)
- **Qwen3-235B-A22B**: Configuration available (partial support)

### Using a Model

```python
from conf.model_config import ModelType, ModelConfig

# Select model
model_type = ModelType.DEEPSEEK_V3

# Create configuration
model_config = ModelConfig.create_model_config(model_type)
```

### Model Parameters

Key parameters in `ModelConfig`:

- `hidden_size`: Hidden dimension (e.g., 7168 for DeepSeek V3)
- `num_layers`: Number of transformer layers (e.g., 61)
- `num_attention_heads`: Attention heads (e.g., 128)
- `intermediate_size`: MLP intermediate size
- `moe_intermediate_size`: MoE intermediate size
- `n_routed_experts`: Number of routed experts
- `num_experts_per_tok`: Experts per token

## Hardware Configuration

### Supported Devices

Hardware is configured in `conf/hardware_config.py`. Supported devices:

- **Ascend**: 910B2, 910B3, 910B4, 910C, A3Pod, David

### Using Hardware

```python
from conf.hardware_config import HardwareTopology, DeviceType

# Create hardware topology
hw_topology = HardwareTopology.create(
    number_of_ranks=8,      # Number of nodes
    npus_per_rank=8,        # NPUs per node
    device_type=DeviceType.ASCEND910B2_376T_64G
)
```

### Hardware Parameters

- `npu_memory`: NPU memory in bytes
- `npu_flops_fp16`: FP16 FLOPS
- `npu_flops_int8`: INT8 FLOPS
- `intra_node_bandwidth`: Intra-node bandwidth (GB/s)
- `inter_node_bandwidth`: Inter-node bandwidth (GB/s)
- `local_memory_bandwidth`: Local memory bandwidth (GB/s)

### Custom Hardware

To add custom hardware:

```python
# In conf/hardware_config.py
DeviceType.CUSTOM = "Custom_Device"

configs[DeviceType.CUSTOM] = cfg(
    npu_memory=64 * 1e9,
    npu_flops_fp16=400 * 1e12,
    npu_flops_int8=800 * 1e12,
    intra_node_bandwidth=200 * 1e9,
    inter_node_bandwidth=100 * 1e9,
    local_memory_bandwidth=2000 * 1e9,
    onchip_buffer_size=256 * 1e6
)
```

## Environment Variables

Defined in `conf/common.py`:

### Logging Level

```bash
export LOG_LEVEL=INFO  # DEBUG, INFO, WARNING
python src/cli/main.py
```

### Common Constants

- Data type sizes: `DTYPE_FP16`, `DTYPE_INT8`, etc.
- Time conversions: `MS_2_US`, `SEC_2_US`, etc.
- Size conversions: `BYTE_2_GB`, `GB_2_BYTE`, etc.

# Supported Operators

This document lists all operators supported by Light LLM Simulator and their cost modeling characteristics.

## Operator Architecture

All operators inherit from `BaseOp` which provides:
- Unified execution flow: `op_disc_factor()` → `compute_cost()` → `memory_cost()` → `e2e_cost()`
- Hardware configuration integration
- Automatic FLOPS and bandwidth calculations

## Matrix Operations

### OpGeMatmul
**Description**: General GEMM (General Matrix Multiply) operator supporting FP16/INT8 precision.

**Parameters**:
- `m, n, k`: Matrix dimensions
- `aichip_config`: Hardware configuration
- `elem_size`: Element size (1 for INT8, 2 for FP16)

**Cost Model**:
- Compute: `2 * m * n * k` FLOPS
- Memory: `elem_size * n * k` bytes
- Discount factor: Variable based on `bs`

**Use Cases**: General matrix multiplications in attention layers.

### OpQuantMatmul
**Description**: Quantized GEMM operator optimized for INT8 precision.

**Parameters**:
- `m, n, k`: Matrix dimensions
- `aichip_config`: Hardware configuration
- `elem_size`: Element size (default: 1 for INT8)

**Cost Model**:
- Compute: `2 * m * n * k` FLOPS
- Memory: `elem_size * n * k` bytes
- Discount factor: Variable based on `bs`

**Use Cases**: Quantized matrix multiplications in attention layers.

### OpGroupedMatmul
**Description**: Grouped matrix multiplication for MoE (Mixture of Experts) operations.

**Parameters**:
- `num_experts`: Number of expert groups
- `bs`: Batch size
- `m, n`: Matrix dimensions
- `aichip_config`: Hardware configuration
- `elem_size`: Element size (default: 1)

**Cost Model**:
- Compute: `2 * bs * m * n` FLOPS
- Memory: `elem_size * (bs * m + m * n * num_experts)` bytes
- Discount factor: Variable based on `num_experts` and `bs` (see implementation)

**Use Cases**: MoE expert computations with grouped matrix operations.

## Attention Operations

### OpMlaProlog
**Description**: MLA (Multi-head Latent Attention) prolog stage for query/key/value computation.

**Components**:
- `mla_q_a_proj`: Query LoRA projection
- `mla_q_nope`: Query non-positional encoding
- `mla_q_rope`: Query RoPE (Rotary Position Embedding)
- `mla_k_nope`: Key non-positional encoding
- `mla_k_rope`: Key RoPE
- `mla_q_absorb`: Query absorption operation

**Cost Model**: Aggregates costs from all sub-operations.

**Use Cases**: DeepSeek V3 attention preprocessing.

### DeepSeekV3PageAttentionInt8
**Description**: DeepSeek V3 page attention with INT8 quantization.

**Operations**:
- QK RoPE computation (FP16, cube core)
- QK matrix multiplication (FP16, cube core)
- Softmax (FP16, vector core)
- SV matrix multiplication (FP16, cube core)

**Cost Model**:
- Cube operations: QK RoPE + QK matmul + SV matmul
- Vector operations: Softmax
- Discount factor: Variable based on `bs` 

**Use Cases**: DeepSeek V3 attention computation with page attention optimization.

### DeepSeekV3PageAttentionFP16
**Description**: DeepSeek V3 page attention with FP16 precision.

**Operations**:
- QK FLOPS computation
- Softmax FLOPS
- UV absorption FLOPS

**Cost Model**:
- Compute: QK + Softmax + UV absorption
- Memory: KV cache + attention weights

**Use Cases**: FP16 precision attention computation.

### DeepSeekV3FlashAttentionInt8
**Description**: Flash Attention implementation with INT8 quantization.

**Operations**:
- Quantization (FP16 → INT8)
- Softmax (FP16)
- Matrix multiplication (INT8)

**Cost Model**:
- Vector time: Quantization + Softmax
- Cube time: Matrix multiplication

**Use Cases**: Memory-efficient attention with quantization.

## Activation Operations

### OpSwiglu
**Description**: SWiGLU (Swish-Gated Linear Unit) activation function.

**Operations**:
- Dequantization (INT8 → FP16)
- SWiGLU activation
- Quantization (FP16 → INT8)

**Cost Model**:
- Compute: `2 * m * n + 6 * m * n / 2 + m * n / 2 + 2 * m * n / 2` FLOPS
- Memory: 0 (in-memory operation, no explicit memory transfer)

**Use Cases**: Activation function in MLP and MoE layers.

## Communication Operations

### MoeDispatch
**Description**: MoE dispatch operation for routing tokens to experts.

**Parameters**:
- `model_config`: Model configuration
- `aichip_config`: Hardware configuration
- `search_config`: Search configuration

**Cost Model**:
- Packet size: `attn_bs * seq_len * hidden_size * num_experts_per_tok * elem_size`
- Bandwidth: `intra_node_bandwidth * comm_intra_ratio`
- Time: `packet_size / bandwidth`

**Use Cases**: Routing tokens to MoE experts in distributed settings.

### MoeCombine
**Description**: MoE combine operation for aggregating expert outputs.

**Parameters**:
- `model_config`: Model configuration
- `aichip_config`: Hardware configuration
- `search_config`: Search configuration

**Cost Model**:
- Packet size: `ffn_bs * seq_len * hidden_size * elem_size`
- Bandwidth: `intra_node_bandwidth * comm_intra_ratio`
- Time: `packet_size / bandwidth`

**Use Cases**: Aggregating MoE expert outputs in distributed settings.

## Adding New Operators

To add a new operator:

1. Inherit from `BaseOp`
2. Implement `compute_cost()` and `memory_cost()` methods
3. Optionally override `op_disc_factor()` for custom discount factors
4. Register in `src/ops/__init__.py`

Example:
```python
from src.ops.base_ops import BaseOp

class MyCustomOp(BaseOp):
    def compute_cost(self):
        self.compute_flops = ...  # Calculate FLOPS
        self.compute_time = self.compute_flops / self.cube_flops
        return self.compute_time
    
    def memory_cost(self):
        self.bytes = ...  # Calculate memory bytes
        self.memory_time = self.bytes / self.mem_bw_local
        return self.memory_time
```

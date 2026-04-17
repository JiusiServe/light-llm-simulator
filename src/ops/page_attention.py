from src.ops.base import BaseOp
from conf.common import BLOCK_SIZE, SPARSE_COUNT
from conf.common import US_2_SEC
import math


class MLAFlashAttentionFP16(BaseOp):
    '''
    Description:
        The Flash Attention operation for the model used MLA attention mechanism in FP16 precision.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=2):
        self.op_disc_factor()
        super().__init__("MLAFlashAttentionFP16", config.aichip_config, elem_size)
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len

    def compute_cost(self):
        self.qk_flops = (
            2 * self.attn_bs * self.model_config.num_attention_heads * 
            self.kv_len * (2 * self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
        )
        self.softmax_flops =(
            5 * self.attn_bs * self.model_config.num_attention_heads * self.kv_len
        )
        # matrix absorption
        self.uv_absorb_flops =(
            2 * self.attn_bs * self.model_config.kv_lora_rank * 
            self.model_config.num_attention_heads * self.model_config.v_head_dim
        )
        self.compute_time = (
            self.qk_flops / self.cube_flops +
            self.softmax_flops / self.vec_flops +
            self.uv_absorb_flops / self.cube_flops
        )
        return self.compute_time

    def memory_cost(self):
        # q_block: [B, n/tp, S, (512+64)] fp16
        q_block = 2 * self.attn_bs * self.model_config.num_attention_heads * self.seq_len * (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
        # kv_nope_block: [B, 1, KV, 512] fp16
        kv_nope_block = 2 * self.attn_bs * self.kv_len * self.model_config.kv_lora_rank
        # kv_rope_block: [B, 1, KV, 64] fp16
        kv_rope_block = 2 * self.attn_bs * self.kv_len * self.model_config.qk_rope_head_dim
        # o_block: [B, n/tp, S, 512] fp16
        o_block = 2 * self.attn_bs * self.model_config.num_attention_heads * self.seq_len * self.model_config.kv_lora_rank

        self.bytes = q_block + kv_nope_block + kv_rope_block + o_block
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time


class MLAFlashAttentionInt8(BaseOp):
    '''
    Description:
        The Flash Attention operation for the model used MLA attention mechanism in INT8 precision.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=1):
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len
        self.seq_len = config.seq_len
        self.static_cost = 30 * US_2_SEC
        super().__init__("MLAFlashAttentionInt8", config.aichip_config, elem_size, static_cost=self.static_cost)
        self.memory_ratio = 0.8

    def compute_cost(self):
        # qk_position = 2*B*128/TP*S*64*KV
        # BF16, cube core
        # - q_rope = [B, 128/TP, S, 64]
        # - k_rope = [B, 128/TP, KV, 64] (trans) [B, 128/TP, 64, KV]
        # - output_qk_rope = [B, 128/TP, S, KV]
        qk_rope = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.qk_rope_head_dim * self.kv_len
        )
        # qk_matmul = 2*B*128/TP*S*512*KV
        # INT8, cube core
        # - q_nope = [B, 128/TP, S, 512]
        # - k_nope = [B, 128/TP, KV, 512] (trans) [B, 128/TP, 512, KV]
        # - output_qk_nope = [B, 128/TP, S, KV]
        qk_matmul =(
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.kv_lora_rank * self.kv_len
        )
        # (output_qk + output_qkv) -> softmax = 5*[B, 128/TP, S, KV]
        # BF16, vector core
        # - output_qk = [B, 128/TP, S, KV]
        # - output_qk = [B, 128/TP, S, KV]
        # - output_softmax = [B, 128/TP, S, KV]
        softmax =(
            5 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.kv_len
        )
        # qkv_matmul = 2*B*128/TP*S*KV*512
        # INT8, cube core
        # - output_softmax = [B, 128/TP, S, KV]
        # - v_nope = [B, 128/TP, KV, 512]
        # - output_matmul = [B, 128/TP, S, 512]
        sv_matmul =(
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.kv_len * self.model_config.kv_lora_rank
        )

        self.total_computation = qk_rope + qk_matmul + softmax + sv_matmul
        qk_rope_time = qk_rope / self.cube_flops_fp16
        qk_matmul_time = qk_matmul / self.cube_flops_int8
        softmax_time = softmax / self.vec_flops_fp16
        sv_matmul_time = sv_matmul / self.cube_flops_int8
        self.compute_time = qk_rope_time + qk_matmul_time + softmax_time + sv_matmul_time
        return self.compute_time

    def memory_cost(self):
        # q_nope_block: [B, S, n/tp, 512] INT8
        q_nope_block = self.attn_bs * self.seq_len * self.model_config.num_attention_heads * self.model_config.kv_lora_rank
        # q_rope_block: [B, S, n/tp, 64] BF16
        q_rope_block = 2 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads * self.model_config.qk_rope_head_dim
        # kv_nope_block: [block_num, 1, block_size, 512] INT8
        # key_nope and value_nope are loaded separately
        block_num = math.ceil(self.attn_bs * self.kv_len / BLOCK_SIZE)
        # int8
        kv_nope_block = 2 * block_num * BLOCK_SIZE * self.model_config.kv_lora_rank
        # bf16 [vllm-ascend only support bf16]
        # kv_nope_block = 4 * block_num * BLOCK_SIZE * self.model_config.kv_lora_rank
        # kv_rope_block: [block_num, 1, n/tp, 64] BF16
        # key_rope and value_rope are loaded separately
        kv_rope_block = 4 * block_num * BLOCK_SIZE * self.model_config.qk_rope_head_dim
        # o_block: [B, S, n/tp, 512] BF16
        o_block = 2 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads * self.model_config.kv_lora_rank

        self.total_data_movement = q_nope_block + q_rope_block + kv_nope_block + kv_rope_block + o_block
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth / self.memory_ratio
        return self.memory_time


class MLASparseFlashAttentionInt8(BaseOp):
    '''
    Description:
        Sparse Flash Attention for MLA models in INT8 precision.
        Uses sparse block selection (top-K) to attend only to selected KV blocks,
        reducing both compute and memory compared to full attention.

        Based on CANN ops-transformer sparse_flash_attention operator:
        - query: [B, S, N1, D+rope_head_dim] (nope + rope parts)
        - key/value: stored in paged KV cache, accessed via sparse_indices + block_table
        - sparse_indices: [B, q_blocks, N2, K] specifies which KV blocks to attend to
        - Only selected KV blocks are loaded and computed, skipping irrelevant tokens

    Attributes:
        config: The configuration of the search task.
        sparse_block_size: The sparse block size for block-sparse selection.
        sparse_k: Number of selected KV blocks per query block (top-K).
    '''
    def __init__(self, config, sparse_block_size=128, sparse_k=16, elem_size=1):
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len
        self.seq_len = config.seq_len
        self.sparse_block_size = sparse_block_size
        self.sparse_k = sparse_k
        self.static_cost = 30 * US_2_SEC
        super().__init__("MLASparseFlashAttentionInt8", config.aichip_config, elem_size, static_cost=self.static_cost)
        self.memory_ratio = 0.8

    def compute_cost(self):
        # Effective KV length: only selected sparse blocks are attended to
        selected_kv_len = self.sparse_k * self.sparse_block_size

        # qk_rope: 2 * B * N1 * S * rope_head_dim * selected_kv_len
        # BF16, cube core
        # - q_rope = [B, N1, S, rope_head_dim]
        # - k_rope = [B, N1, selected_kv, rope_head_dim]
        # - output_qk_rope = [B, N1, S, selected_kv]
        qk_rope = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.qk_rope_head_dim * selected_kv_len
        )
        # qk_nope: 2 * B * N1 * S * D * selected_kv_len
        # INT8, cube core
        # - q_nope = [B, N1, S, D]
        # - k_nope = [B, N1, selected_kv, D]
        # - output_qk_nope = [B, N1, S, selected_kv]
        qk_nope = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.kv_lora_rank * selected_kv_len
        )
        # softmax: 5 * B * N1 * S * selected_kv_len
        # BF16, vector core
        softmax = (
            5 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * selected_kv_len
        )
        # sv_matmul: 2 * B * N1 * S * selected_kv_len * D
        # INT8, cube core
        # - softmax_weights = [B, N1, S, selected_kv]
        # - v_nope = [B, N1, selected_kv, D]
        # - output = [B, N1, S, D]
        sv_matmul = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * selected_kv_len * self.model_config.kv_lora_rank
        )

        self.total_computation = qk_rope + qk_nope + softmax + sv_matmul
        qk_rope_time = qk_rope / self.cube_flops_fp16
        qk_nope_time = qk_nope / self.cube_flops_int8
        softmax_time = softmax / self.vec_flops_fp16
        sv_matmul_time = sv_matmul / self.cube_flops_int8
        self.compute_time = qk_rope_time + qk_nope_time + softmax_time + sv_matmul_time
        return self.compute_time

    def memory_cost(self):
        # Number of selected KV blocks per batch
        selected_kv_blocks = self.sparse_k * math.ceil(self.seq_len / self.sparse_block_size)

        # q_nope: [B, S, N1, D] INT8
        q_nope_block = self.attn_bs * self.seq_len * self.model_config.num_attention_heads * self.model_config.kv_lora_rank
        # q_rope: [B, S, N1, rope_head_dim] BF16
        q_rope_block = 2 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads * self.model_config.qk_rope_head_dim

        # KV cache: only load selected blocks (not full KV cache)
        # kv_nope: [selected_blocks, block_size, N2, D] INT8
        # key_nope and value_nope are loaded separately
        kv_nope_block = 2 * selected_kv_blocks * self.sparse_block_size * self.model_config.kv_lora_rank
        # kv_rope: [selected_blocks, block_size, N2, rope_head_dim] BF16
        # key_rope and value_rope are loaded separately
        kv_rope_block = 4 * selected_kv_blocks * self.sparse_block_size * self.model_config.qk_rope_head_dim

        # Overhead: sparse_indices and block_table
        q_blocks = math.ceil(self.seq_len / self.sparse_block_size)
        sparse_indices_bytes = self.attn_bs * q_blocks * self.sparse_k * 4  # int32
        block_table_bytes = self.attn_bs * math.ceil(self.kv_len / BLOCK_SIZE) * 4  # int32

        # o_block: [B, S, N1, D] BF16
        o_block = 2 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads * self.model_config.kv_lora_rank

        self.total_data_movement = (q_nope_block + q_rope_block + kv_nope_block +
                                    kv_rope_block + o_block + sparse_indices_bytes + block_table_bytes)
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth / self.memory_ratio
        return self.memory_time


class MLASparseFlashAttentionFP16(BaseOp):
    '''
    Description:
        Sparse Flash Attention for MLA models in FP16 precision.
        Uses sparse block selection (top-K) to attend only to selected KV blocks.

        - query: [B, S, N1, D+rope_head_dim] (nope + rope parts)
        - key/value: stored in paged KV cache, accessed via sparse_indices + block_table
        - sparse_indices: [B, q_blocks, N2, K] specifies which KV blocks to attend to
        - Only selected KV blocks are loaded and computed

    Attributes:
        config: The configuration of the search task.
        sparse_block_size: The sparse block size for block-sparse selection.
        sparse_k: Number of selected KV blocks per query block (top-K).
    '''
    def __init__(self, config, elem_size=2):
        super().__init__("MLASparseFlashAttentionFP16", config.aichip_config, elem_size)
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len
        self.seq_len = config.seq_len

    def compute_cost(self):
        # qk_position = 2 * b * s * n * sparse_count * (512+64)
        # BF16, cube core
        # q = [b, s, n, (512+64)]
        # k = [b, sparse_count, n, (512+64)]
        qk_flops = (
            2 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads *
            SPARSE_COUNT * (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
        )
        # softmax: 5 * B * N1 * S * sparse_count
        softmax_flops = (
            5 * self.attn_bs * self.model_config.num_attention_heads * self.seq_len * SPARSE_COUNT
        )
        # qkv_matmul (sv): 2 * B * N1 * S * D * sparse_count
        # BF16, cube core
        # - output_softmax = [B, 128/TP, S, sparse_count]
        # - v_nope = [B, 128/TP, sparse_count, 512]
        # - output_matmul = [B, 128/TP, S, 512]
        sv_flops = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.kv_lora_rank * SPARSE_COUNT
        )
        self.total_computation = qk_flops + softmax_flops + sv_flops
        self.compute_time = (
            qk_flops / self.cube_flops_fp16 +
            softmax_flops / self.vec_flops_fp16 +
            sv_flops / self.cube_flops_fp16
        )
        return self.compute_time

    def memory_cost(self):
        # q_block: [B, N, S, D+R] bf16
        q_bytes = 2 * self.attn_bs * self.model_config.num_attention_heads * self.seq_len * (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
        # kv_block: only selected blocks, key+value bf16, N2=1 for MLA
        # key(D+R) + value(D+R) = 2 * sel * (D+R) * elem_size
        kv_bytes = 4 * self.attn_bs * SPARSE_COUNT * (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
        # sparse_indices: [B, S, SPARSE_COUNT] int32
        sparse_idx_bytes = self.attn_bs * self.seq_len * SPARSE_COUNT * 4
        # block_table: [b, kv_len/block_size] int32
        block_table_bytes = self.attn_bs * self.kv_len / BLOCK_SIZE * 4
        # o_block: [B, N, S, D] bf16
        o_bytes = 2 * self.attn_bs * self.model_config.num_attention_heads * self.seq_len * self.model_config.kv_lora_rank

        self.total_data_movement = q_bytes + kv_bytes + o_bytes + sparse_idx_bytes + block_table_bytes
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time


class GQAFlashAttentionFP16(BaseOp):
    '''
    Description:
        The Flash Attention operation for the model used GQA attention mechanism in FP16 precision.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=2):
        self.op_compute_disc()
        super().__init__("FlashAttentionFP16", config.aichip_config, elem_size)
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len
        self.seq_len = config.seq_len

    def op_compute_disc(self):
        return 0.651

    def compute_cost(self):
        # qk_matmul: 2*B*n*s*D*kv
        # query_states: [B, n, s, D]
        # key_states: [B, n_kv, kv, D]
        # qk: [B, n, s, kv]
        qk_matmul = (
            2 * self.attn_bs *
            self.model_config.num_heads *
            self.seq_len *
            self.model_config.head_size *
            self.kv_len
        )
        # softmax
        softmax = (
            5 * self.attn_bs *
            self.model_config.num_heads *
            self.seq_len *
            self.kv_len
        )
        # qkv_matmul: 2*B*n*s*kv*D
        # qk: [B, n, s, kv]
        # value_statue: [B, n_kv, kv, D]
        qkv_matmul = (
            2 * self.attn_bs *
            self.model_config.num_heads *
            self.seq_len *
            self.model_config.head_size *
            self.kv_len
        )
        cube_time = (qk_matmul + qkv_matmul) / self.cube_flops_fp16
        vec_time = softmax / self.vec_flops_fp16
        self.compute_time = cube_time + vec_time
        return self.compute_time

    def memory_cost(self):
        # q_block: [B, n, s, D]
        # kv_cache: [B, n_kv, kv, D]
        # o_block: [B, n, s, D]
        self.bytes = (
            2 * self.elem_size *
            self.attn_bs *
            self.model_config.kv_heads *
            self.kv_len *
            self.model_config.head_size
        )
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time

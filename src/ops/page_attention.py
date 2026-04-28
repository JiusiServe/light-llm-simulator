from src.ops.base import BaseOp
from conf.common import BLOCK_SIZE, SPARSE_COUNT
from conf.common import US_2_SEC
import math


class MLAFlashAttention(BaseOp):
    '''
    Description:
        The Flash Attention operation for the model used MLA attention mechanism.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=1):
        self.config = config
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len
        self.seq_len = config.seq_len
        self.static_cost = 30 * US_2_SEC
        super().__init__("MLAFlashAttention", config.aichip_config, elem_size, static_cost=self.static_cost)
        self.memory_ratio = 0.8

    def compute_cost(self):
        # qk_position = 2*B*N/TP*S*Dr*KV
        # BF16, cube core
        # - q_rope = [B, N/TP, S, Dr]
        # - k_rope = [B, N/TP, KV, Dr] (trans) [B, N/TP, Dr, KV]
        # - output_qk_rope = [B, N/TP, S, KV]
        qk_rope = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.qk_rope_head_dim * self.kv_len
        )
        # qk_matmul = 2*B*N/TP*S*D*KV
        # INT8 for kv_cache_quant is int8; BF16 for kv_cache_quant is bf16, cube core
        # - q_nope = [B, N/TP, S, D]
        # - k_nope = [B, N/TP, KV, D] (trans) [B, N/TP, D, KV]
        # - output_qk_nope = [B, N/TP, S, KV]
        qk_matmul =(
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.kv_lora_rank * self.kv_len
        )
        # (output_qk + output_qkv) -> softmax = 5*[B, N/TP, S, KV]
        # BF16, vector core
        # - output_qk = [B, N/TP, S, KV]
        # - output_qk = [B, N/TP, S, KV]
        # - output_softmax = [B, N/TP, S, KV]
        softmax =(
            5 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.kv_len
        )
        # qkv_matmul = 2*B*N/TP*S*KV*D
        # INT8 for kv_cache_quant is int8; BF16 for kv_cache_quant is bf16, cube core
        # - output_softmax = [B, N/TP, S, KV]
        # - v_nope = [B, N/TP, KV, D]
        # - output_matmul = [B, N/TP, S, D]
        sv_matmul =(
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.kv_len * self.model_config.kv_lora_rank
        )

        self.total_computation = qk_rope + qk_matmul + softmax + sv_matmul
        qk_rope_time = qk_rope / self.cube_flops_fp16
        if self.config.kv_cache_quant == "bf16":
            qk_matmul_time = qk_matmul / self.cube_flops_fp16
            sv_matmul_time = sv_matmul / self.cube_flops_fp16
        else:
            qk_matmul_time = qk_matmul / self.cube_flops_int8
            sv_matmul_time = sv_matmul / self.cube_flops_int8
        softmax_time = softmax / self.vec_flops_fp16
        self.compute_time = qk_rope_time + qk_matmul_time + softmax_time + sv_matmul_time
        return self.compute_time

    def memory_cost(self):
        # q_nope_block: [B, S, n/tp, D]
        # INT8 for kv_cache_quant is int8; BF16 for kv_cache_quant is bf16
        q_nope_block = self.attn_bs * self.seq_len * self.model_config.num_attention_heads * self.model_config.kv_lora_rank
        # q_rope_block: [B, S, n/tp, Dr] BF16
        q_rope_block = 2 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads * self.model_config.qk_rope_head_dim
        # kv_nope_block: [block_num, 1, block_size, D] INT8
        # INT8 for kv_cache_quant is int8; BF16 for kv_cache_quant is bf16
        # key_nope and value_nope are loaded separately
        block_num = math.ceil(self.attn_bs * self.kv_len / BLOCK_SIZE)
        kv_nope_block = 2 * block_num * BLOCK_SIZE * self.model_config.kv_lora_rank
        # k_rope_block: [block_num, 1, n/tp, Dr] BF16
        k_rope_block = 2 * block_num * BLOCK_SIZE * self.model_config.qk_rope_head_dim
        # o_block: [B, S, n/tp, D] BF16
        o_block = 2 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads * self.model_config.kv_lora_rank
        if self.config.kv_cache_quant == "bf16":
            q_nope_block = q_nope_block * 2
            kv_nope_block = kv_nope_block * 2

        self.total_data_movement = q_nope_block + q_rope_block + kv_nope_block + k_rope_block + o_block
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth / self.memory_ratio
        return self.memory_time


class MLASparseFlashAttention(BaseOp):
    '''
    Description:
        Sparse Flash Attention for MLA models.

        - query: [B, S, N1, D+rope_head_dim] (nope + rope parts)
        - key/value: stored in paged KV cache, accessed via sparse_indices + block_table
        - sparse_indices: [B, q_blocks, N2, K] specifies which KV blocks to attend to
        - Only selected KV blocks are loaded and computed

    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=2):
        self.config = config
        super().__init__("MLASparseFlashAttention", config.aichip_config, elem_size, static_cost=70*US_2_SEC)
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len
        self.seq_len = config.seq_len

    def compute_cost(self):
        # qk_position = 2 * B * S * N1 * sparse_count * Dr
        # q = [B, S, N1, Dr]
        # k = [B, sparse_count, N1, Dr]
        qk_rope = (
            2 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads *
            SPARSE_COUNT * self.model_config.qk_rope_head_dim
        )
        # qk_matmul = 2 * B * S * N1 * sparse_count * D
        # q = [B, S, N1, D]
        # k = [B, sparse_count, N1, D]
        qk_matmul = (
            2 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads *
            SPARSE_COUNT * self.model_config.kv_lora_rank
        )
        # softmax: 5 * B * N1 * S * sparse_count
        softmax = (
            5 * self.attn_bs * self.model_config.num_attention_heads * self.seq_len * SPARSE_COUNT
        )
        # qkv_matmul (sv): 2 * B * N1 * S * D * sparse_count
        # - output_softmax = [B, N1/TP, S, sparse_count]
        # - v_nope = [B, N1/TP, sparse_count, 512]
        # - output_matmul = [B, N1/TP, S, 512]
        sv_matmul = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.kv_lora_rank * SPARSE_COUNT
        )
        self.total_computation = qk_rope + qk_matmul + softmax + sv_matmul
        qk_rope_time = qk_rope / self.cube_flops_fp16
        if self.config.kv_cache_quant == "bf16":
            qk_matmul_time = qk_matmul / self.cube_flops_fp16
            sv_matmul_time = sv_matmul / self.cube_flops_fp16
        else:
            qk_matmul_time = qk_matmul / self.cube_flops_int8
            sv_matmul_time = sv_matmul / self.cube_flops_int8
        softmax_time = softmax / self.vec_flops_fp16
        self.compute_time = qk_rope_time + qk_matmul_time + softmax_time + sv_matmul_time
        return self.compute_time

    def memory_cost(self):
        # q_nope_block: [B, S, N1/tp, D]
        # INT8 for kv_cache_quant is int8; BF16 for kv_cache_quant is bf16
        q_nope_block = self.attn_bs * self.model_config.num_attention_heads * self.seq_len * self.model_config.kv_lora_rank
        # q_rope_block: [B, S, N1/tp, Dr] BF16
        q_rope_block = 2 * self.attn_bs * self.model_config.num_attention_heads * self.seq_len * self.model_config.qk_rope_head_dim
        # kv_block: only selected blocks, key+value bf16, N2=1 for MLA
        kv_nope_block = 2 * self.attn_bs * SPARSE_COUNT * self.model_config.kv_lora_rank
        k_rope_block = 2 * self.attn_bs * SPARSE_COUNT * self.model_config.qk_rope_head_dim
        # sparse_indices: [B, S, SPARSE_COUNT] int32
        sparse_idx_bytes = self.attn_bs * self.seq_len * SPARSE_COUNT * 4
        # block_table: [b, kv_len/block_size] int32
        block_table_bytes = self.attn_bs * self.kv_len / BLOCK_SIZE * 4
        # o_block: [B, N1, S, D] bf16
        o_block = 2 * self.attn_bs * self.model_config.num_attention_heads * self.seq_len * self.model_config.kv_lora_rank
        # softmaxMaxOut+softmaxSumOut: [B, N2, S1, N1/N2] FLOAT
        softmax_out_bytes = 2 * 4 * self.attn_bs * self.seq_len * self.model_config.num_attention_heads
        if self.config.kv_cache_quant == "bf16":
            q_nope_block = q_nope_block * 2
            kv_nope_block = kv_nope_block * 2

        self.total_data_movement = q_nope_block + q_rope_block + kv_nope_block + k_rope_block + o_block + sparse_idx_bytes + block_table_bytes + softmax_out_bytes
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time


class GQAFlashAttention(BaseOp):
    '''
    Description:
        The Flash Attention operation for the model used GQA attention mechanism.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=2):
        self.config = config
        super().__init__("GQAFlashAttention", config.aichip_config, elem_size)
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len
        self.seq_len = config.seq_len

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
        self.total_computation = qk_matmul + softmax + qkv_matmul
        if self.config.kv_cache_quant == "bf16":
            cube_time = (qk_matmul + qkv_matmul) / self.cube_flops_fp16
        else:
            cube_time = (qk_matmul + qkv_matmul) / self.cube_flops_int8
        vec_time = softmax / self.vec_flops_fp16
        self.compute_time = cube_time + vec_time
        return self.compute_time

    def memory_cost(self):
        # q_block: [B, n, s, D]
        q_block = self.attn_bs * self.model_config.num_heads * self.seq_len * self.model_config.head_size
        # kv_cache: [B, n_kv, kv, D](key+value loaded separately)
        kv_block = 2 * self.attn_bs * self.model_config.kv_heads * self.kv_len * self.model_config.head_size
        # o_block: [B, n, s, D]
        o_block = self.attn_bs * self.model_config.num_heads * self.seq_len * self.model_config.head_size

        if self.config.kv_cache_quant == "bf16":
            kv_block = kv_block * 2
        self.total_data_movement = q_block + kv_block + o_block
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time

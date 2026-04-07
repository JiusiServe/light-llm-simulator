from src.ops.base import BaseOp
from conf.common import BLOCK_SIZE
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

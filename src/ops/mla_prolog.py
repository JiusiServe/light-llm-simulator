from src.ops.base import BaseOp
from conf.common import BLOCK_SIZE
import math

class OpMlaProlog(BaseOp):
    '''
    Description:
        It is used to compute the query, key and value for the MLA attention.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config):
        self.model_config = config.model_config
        self.aichip_config = config.aichip_config
        self.config = config
        self.elem_size = 1
        super().__init__("mla_prolog", self.aichip_config, self.elem_size)

    def __call__(self):
        self.compute_cost()
        self.memory_cost()
        self.e2e_cost()

    def compute_cost(self):
        # mla_q_nope:
        # [b, s, h] , [dq, h] -> [b, s, dq]
        # [b, s, dq] , [n*d_h^N, dq] -> [b, s, n*d_h^N]
        # [b, s, n*d_h^N], [dc, n*d_h^N] -> [b, s, dc]
        q_a_proj = 2 * (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            self.model_config.q_lora_rank
        )
        q_nope = 2 * (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.q_lora_rank *
            self.model_config.num_attention_heads *
            self.model_config.qk_nope_head_dim
        )
        q_absorb = 2 * (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.num_attention_heads *
            self.model_config.qk_nope_head_dim *
            self.model_config.kv_lora_rank
        )
        mla_q_nope = q_a_proj + q_nope + q_absorb
        # mla_q_pe: [b, s, dq], [n*d_h^R, dq] -> [b, s, n*d_h^R]
        mla_q_pe = 2 * (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.q_lora_rank *
            self.model_config.num_attention_heads *
            self.model_config.qk_rope_head_dim
        )
        # mla_k_nope: [b, s, h] , [dc, h] -> [b, s, dc]
        mla_k_nope = 2 * (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            self.model_config.kv_lora_rank
        )
        # mla_k_rope: [b, s, h], [d_h^R, h] -> [b, s, d_h^R] fp16
        mla_k_rope = 2 * (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            self.model_config.qk_rope_head_dim
        )
        self.total_computation = mla_q_nope + mla_q_pe + mla_k_nope + mla_k_rope
        self.compute_time = self.total_computation / self.aichip_config.cube_flops_int8

        return self.compute_time

    def memory_cost(self):
        # INPUT
        # tensor x:[B, S, H] int8
        x = self.config.attn_bs * self.config.seq_len * self.model_config.hidden_size
        # tensor W_dq:[dq, H] int8
        weight_dq = self.model_config.q_lora_rank * self.model_config.hidden_size
        # tensor W_uq:[n*d_h^N, dq] int8
        weight_uq = self.model_config.num_attention_heads * self.model_config.qk_nope_head_dim * self.model_config.q_lora_rank
        # tensor W_uk:[n*d_h^N, dc] bf16
        weight_uk = 2 *self.model_config.num_attention_heads * self.model_config.qk_nope_head_dim * self.model_config.kv_lora_rank
        # tensor W_qr:[n*d_h^R, dq] int8
        weight_qr = self.model_config.num_attention_heads * self.model_config.qk_rope_head_dim * self.model_config.q_lora_rank
        # tensor W_dkv:[dc, h] int8
        weight_dkv = self.model_config.kv_lora_rank * self.model_config.hidden_size
        # tensor W_kr:[d_h^R, h] int8
        weight_kr = self.model_config.qk_rope_head_dim * self.model_config.hidden_size * 2
        # tensor kv_cache_in:[b, s, dc] int8 [vllm-Ascend bf16]
        cache_capacity_per_seq = math.ceil(self.model_config.max_kv_length / BLOCK_SIZE)
        block_num = math.ceil(self.config.attn_bs * self.config.seq_len * cache_capacity_per_seq / BLOCK_SIZE)
        kv_cache_in = BLOCK_SIZE * block_num * self.model_config.kv_lora_rank
        # tensor kr_cache_in:[b, s, d_h^R] bf16
        kr_cache_in = 2 * BLOCK_SIZE * block_num * self.model_config.qk_rope_head_dim
        # tensor rmsnorm_gamma_cq: [dq] bf16
        rmsnorm_gamma_cq = 2 * self.model_config.q_lora_rank
        # tensor rmsnorm_gamma_ckv: [dc] bf16
        rmsnorm_gamma_ckv = 2 * self.model_config.kv_lora_rank
        # tensor rope_sin: [b, s, d_h^R] bf16
        rope_sin = 2 * self.config.attn_bs * self.config.seq_len * self.model_config.qk_rope_head_dim
        # tensor rope_cos: [b, s, d_h^R] bf16
        rope_cos = 2 * self.config.attn_bs * self.config.seq_len * self.model_config.qk_rope_head_dim
        # tensor cache_index: [b, s] int64
        cache_index = 8 * self.config.attn_bs * self.config.seq_len
        # dequant_scale_x: [b*s, 1] float
        dequant_scale_x = 4 * self.config.attn_bs * self.config.seq_len
        # dequant_scale_w_dq: [dq] float
        dequant_scale_w_dq = 4 * self.model_config.q_lora_rank
        # dequant_scale_w_uq_qr: [n*(d_h^N+d_h^R)] float
        dequant_scale_w_uq_qr = 4 * self.model_config.num_attention_heads * (self.model_config.qk_nope_head_dim + self.model_config.qk_rope_head_dim)
        # dequant_scale_w_dkv_kr: [dc + d_h^R] float
        dequant_scale_w_dkv_kr = 4 * (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
        # quant_scale_ckv: [d_h^R] float
        quant_scale_ckv = 4 * self.model_config.kv_lora_rank
        # OUTPUT
        # tensor q_nope:[b, s, n, dc] int8 [vllm-Ascend bf16]
        q_nope = self.config.attn_bs * self.config.seq_len * self.model_config.num_attention_heads * self.model_config.kv_lora_rank
        # tensor q_rope:[b, s, n, d_h^R] bf16
        q_rope = 2 * self.config.attn_bs * self.config.seq_len * self.model_config.num_attention_heads * self.model_config.qk_rope_head_dim
        # tensor dequant_scale_q_nope: [b*s, n, 1] float
        dequant_scale_q_nope = 4 * self.config.attn_bs * self.config.seq_len * self.model_config.num_attention_heads
        self.total_data_movement = (
            x + weight_dq + weight_uq + weight_uk + weight_qr + weight_dkv + weight_kr + kv_cache_in + kr_cache_in +
            rmsnorm_gamma_cq + rmsnorm_gamma_ckv + rope_sin + rope_cos + cache_index + dequant_scale_x + dequant_scale_w_dq +
            dequant_scale_w_uq_qr + dequant_scale_w_dkv_kr + quant_scale_ckv + q_nope + q_rope + dequant_scale_q_nope
        )
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time

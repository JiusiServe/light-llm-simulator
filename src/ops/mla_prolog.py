from src.ops.base import BaseOp
from conf.common import BLOCK_SIZE


class OpMlaProlog(BaseOp):
    '''
    Description:
        It is used to compute the query, key and value for the MLA attention.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config):
        self.model_config = config.model_config
        self.aichip_config = config.aichip_config1
        self.config = config
        self.attn_bs = config.attn_bs * config.seq_len
        self.elem_size = 1
        super().__init__("mla_prolog", self.aichip_config, self.elem_size)
        self.memory_ratio = 0.7

    def __call__(self):
        self.compute_cost()
        self.memory_cost()
        self.e2e_cost()

    def compute_cost(self):
        # mla_q_nope:
        # [b, s, h] , [dq, h] -> [b, s, dq]
        # [b, s, dq] , [n*d_h^N, dq] -> [b, s, n*d_h^N]
        # [b, s, n*d_h^N], [dc, n*d_h^N] -> [b, s, dc]
        q_a_proj = (
            self.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            self.model_config.q_lora_rank
        )
        q_nope = (
            self.attn_bs *
            self.config.seq_len *
            self.model_config.q_lora_rank *
            self.model_config.num_attention_heads *
            self.model_config.qk_nope_head_dim
        )
        q_absorb = 2 * (
            self.attn_bs *
            self.config.seq_len *
            self.model_config.num_attention_heads *
            self.model_config.qk_nope_head_dim *
            self.model_config.kv_lora_rank
        )
        mla_q_nope = q_a_proj + q_nope + q_absorb
        # mla_q_pe: [b, s, dq], [n*d_h^R, dq] -> [b, s, n*d_h^R]
        mla_q_pe = 2 * (
            self.attn_bs *
            self.config.seq_len *
            self.model_config.q_lora_rank *
            self.model_config.num_attention_heads *
            self.model_config.qk_rope_head_dim
        )
        # mla_k_nope: [b, s, h] , [dc, h] -> [b, s, dc]
        mla_k_nope = (
            self.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            self.model_config.kv_lora_rank
        )
        # mla_k_rope: [b, s, h], [d_h^R, h] -> [b, s, d_h^R] fp16
        mla_k_rope = 2 * (
            self.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            self.model_config.qk_rope_head_dim
        )
        self.total_computation = mla_q_nope + mla_q_pe + mla_k_nope + mla_k_rope
        self.compute_time = self.total_computation / self.aichip_config.cube_flops_int8

        return self.compute_time

    def memory_cost(self):
        self.attn_bs = self.attn_bs * 2
        # tensor x:[B, S, H] int8
        x = self.attn_bs * self.config.seq_len * self.model_config.hidden_size
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
        weight_kr = self.model_config.qk_rope_head_dim * self.model_config.hidden_size
        # tensor q_nope:[b, s, n, dc] int8
        q_nope = self.attn_bs * self.config.seq_len * self.model_config.num_attention_heads * self.model_config.kv_lora_rank
        # tensor q_rope:[b, s, n, d_h^R] fp16
        q_rope = 2 * self.attn_bs * self.config.seq_len * self.model_config.num_attention_heads * self.model_config.qk_rope_head_dim
        # tensor kv_cache_out:[b, s, dc] int8
        kv_cache_out = BLOCK_SIZE * self.attn_bs * self.config.seq_len * self.model_config.kv_lora_rank
        # tensor kr_cache_out:[b, s, d_h^R] fp16
        kr_cache_out = 2 * BLOCK_SIZE * self.attn_bs * self.config.seq_len * self.model_config.qk_rope_head_dim
        self.total_data_movement = (
            x + weight_dq + weight_uq + weight_uk + weight_qr + weight_dkv + weight_kr + 
            q_nope + q_rope + kv_cache_out + kr_cache_out
        )
        self.memory_time = self.total_data_movement / self.aichip_config.local_memory_bandwidth / self.memory_ratio

        return self.memory_time

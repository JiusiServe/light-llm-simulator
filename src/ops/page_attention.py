from src.ops.base import BaseOp
import logging


class DeepSeekV3PageAttentionFP16(BaseOp):
    def __init__(self, model_config, aichip_config, search_config, elem_size=2):
        self.op_disc_factor()
        super().__init__(aichip_config, elem_size)
        logging.debug(f"-------Dsv3PageAttentionFP16-------:")
        self.model_config = model_config
        self.search_config = search_config
        self.attn_bs = search_config.attn_bs
        self.kv_len = search_config.kv_len
        self.mem_bw_inter = self.mem_bw_inter * 0.7
        self.mem_bw_intra = self.mem_bw_inter * 0.7

    def __call__(self):
        self.op_disc_factor()
        self.compute_cost()
        self.memory_cost()
        self.e2e_cost()

    def compute_cost(self):
        self.qk_flops = (
            2 * self.attn_bs * self.model_config.num_attention_heads * 
            self.kv_len * (2 * self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
        )
        self.softmax_flops =(
            4 * self.attn_bs * self.model_config.num_attention_heads * self.kv_len
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
        self.bytes = (
            self.elem_size * self.attn_bs * self.kv_len * 
            (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim) + 
            self.elem_size * self.model_config.kv_lora_rank *
            self.model_config.num_attention_heads * self.model_config.v_head_dim
        )
        self.memory_time = self.bytes / self.mem_bw_inter
        return self.memory_time


class DeepSeekV3PageAttentionInt8(BaseOp):
    def __init__(self, model_config, aichip_config, search_config, elem_size=1):
        self.op_disc_factor()
        super().__init__(aichip_config, elem_size)
        logging.debug(f"-------Dsv3PageAttentionInt8-------:")
        self.model_config = model_config
        self.search_config = search_config
        self.attn_bs = search_config.attn_bs
        self.kv_len = search_config.kv_len
        self.seq_len = search_config.seq_len
        self.mem_bw_inter = self.mem_bw_inter * 0.7
        self.mem_bw_intra = self.mem_bw_inter * 0.7

    def __call__(self):
        self.op_disc_factor()
        self.compute_cost()
        self.memory_cost()
        self.e2e_cost()

    def op_disc_factor(self):
        return 0.77

    def compute_cost(self):
        # qk_position = 2*B*128/TP*S*64*KV
        # FP16, cube core
        # - q_rope = [B, 128/TP, S, 64]
        # - k_rope = [B, 1, KV, 64] (trans) [B, 1, 64, KV]
        # - output_qk = [B, 128/TP, S, KV]
        qk_rope = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.qk_rope_head_dim * self.kv_len
        )
        # qk_matmul = 2*B*128/TP*S*512*KV
        # FP16, cube core
        # - qk = [B, 128/TP, S, 512]
        # - kv_nope = [B, 1, KV, 512] (trans) [B, 1, 512, KV]
        # - output_qkv = [B, 128/TP, S, KV]
        qk_matmul =(
            2 * self.attn_bs * self.model_config.num_attention_heads * 
            self.seq_len * self.model_config.kv_lora_rank * self.kv_len
        )
        # (output_qk + output_qkv) -> safe_softmax = 5*[B, 128/TP, S, KV]
        # FP16, vector core
        # - output_qk = [B, 128/TP, S, KV]
        # - output_qkv = [B, 128/TP, S, KV]
        # - output_softmax = [B, 128/TP, S, KV]
        softmax =(
            5 * self.attn_bs * self.model_config.num_attention_heads * 
            self.seq_len * self.kv_len
        )
        # qkv_matmul = 2*B*128/TP*S*KV*512
        # FP16, cube core
        # - output_softmax = [B, 128/TP, S, KV]
        # - kv_nope = [B, 1, KV, 512]
        # - output_matmul = [B, 128/TP, S, 512]
        sv_matmul =(
            2 * self.attn_bs * self.model_config.num_attention_heads * 
            self.seq_len * self.kv_len * self.model_config.kv_lora_rank
        )
        cube_time = (qk_rope + qk_matmul + sv_matmul) / self.cube_flops_fp16
        vec_time = softmax / self.vec_flops_fp16
        self.compute_time = cube_time + vec_time
        return self.compute_time

    def memory_cost(self):
        self.bytes = (
            self.attn_bs * self.kv_len * 
            (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim) + 
            self.model_config.kv_lora_rank*self.model_config.num_attention_heads * 
            self.model_config.v_head_dim
        )
        self.memory_time = self.bytes / self.mem_bw_local
        return self.memory_time


class DeepSeekV3FlashAttentionInt8(BaseOp):
    def __init__(self, model_config, aichip_config, search_config, elem_size):
        self.op_disc_factor()
        super().__init__(aichip_config, elem_size)
        logging.debug(f"-------Dsv3FlashAttentionInt8-------:")
        self.model_config = model_config
        self.attn_bs = search_config.attn_bs
        self.kv_len = search_config.kv_len
        self.seq_len = search_config.seq_len
        self.mem_bw_inter = self.mem_bw_inter * 0.7
        self.mem_bw_intra = self.mem_bw_intra * 0.7

    def __call__(self):
        self.op_disc_factor()
        self.compute_cost()
        self.memory_cost()
        self.e2e_cost()

    def compute_cost(self):
        # Quant: query, key, value
        # FP16-->int8
        # mean + scale
        quant_flops = (
            2 * self.attn_bs * self.model_config.num_attention_heads * self.seq_len * 
            (self.model_config.qk_nope_head_dim + self.model_config.qk_rope_head_dim) + 
            2 * self.attn_bs * self.model_config.num_attention_heads * self.kv_len * 
            (self.model_config.qk_nope_head_dim + self.model_config.qk_rope_head_dim) + 
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.kv_len * self.model_config.v_head_dim
        )
        # softmax FP16
        softmax_flops =(
            4 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.kv_len
        )
        # matmul int8
        matmul_flops = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.kv_len * 
            (self.model_config.qk_nope_head_dim + self.model_config.qk_rope_head_dim + self.model_config.v_head_dim)
        )
        vec_time1 = quant_flops / self.vec_flops_fp16
        vec_time2 = softmax_flops / self.vec_flops_fp16
        cube_time = matmul_flops / self.cube_flops_int8
        self.compute_time = vec_time1 + vec_time2 + cube_time
        return self.compute_time

    def memory_cost(self):
        self.bytes = (
            self.attn_bs * self.kv_len * 
            (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim) +
            self.model_config.kv_lora_rank*self.model_config.num_attention_heads *
            self.model_config.v_head_dim
        )
        self.memory_time = self.bytes / self.mem_bw_inter
        return self.memory_time

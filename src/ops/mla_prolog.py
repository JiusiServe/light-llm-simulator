from src.ops.matmul import OpGeMatmul
import logging


class OpMlaProlog:
    def __init__(self, model_config, aichip_config, search_config):
        logging.debug(f"-------OpMlaProlog-------:")
        self.model_config = model_config
        self.aichip_config = aichip_config
        self.search_config = search_config
        self.attn_bs = search_config.attn_bs * search_config.seq_len
        # compute query, key, value
        self.mla_q_a_proj = OpGeMatmul(
            self.attn_bs,
            model_config.hidden_size,
            model_config.q_lora_rank,
            aichip_config
        )
        self.mla_q_nope = OpGeMatmul(
            self.attn_bs,
            model_config.q_lora_rank,
            model_config.num_attention_heads*model_config.qk_nope_head_dim / search_config.attn_tensor_parallel,
            aichip_config
        )
        self.mla_q_rope = OpGeMatmul(
            self.attn_bs,
            model_config.q_lora_rank,
            model_config.num_attention_heads * model_config.qk_rope_head_dim / search_config.attn_tensor_parallel,
            aichip_config
        )
        self.mla_k_nope = OpGeMatmul(
            self.attn_bs,
            model_config.hidden_size,
            model_config.kv_lora_rank,
            aichip_config
        )
        self.mla_k_rope = OpGeMatmul(
            self.attn_bs,
            model_config.hidden_size,
            model_config.qk_rope_head_dim,
            aichip_config
        )
        self.mla_q_absorb = OpGeMatmul(
            self.attn_bs,
            model_config.num_attention_heads*model_config.qk_nope_head_dim / search_config.attn_tensor_parallel,
            model_config.kv_lora_rank,
            aichip_config,
            elem_size=1
        )

    def __call__(self):
        self.mla_q_a_proj()
        self.mla_q_nope()
        self.mla_q_rope()
        self.mla_k_nope()
        self.mla_k_rope()
        self.mla_q_absorb()
        self.compute_cost()
        self.memory_cost()
        self.e2e_cost()

    def compute_cost(self):
        self.compute_flops = (
            self.mla_q_a_proj.compute_flops +
            self.mla_q_nope.compute_flops +
            self.mla_q_rope.compute_flops +
            self.mla_k_nope.compute_flops +
            self.mla_k_rope.compute_flops +
            self.mla_q_absorb.compute_flops
        )
        self.compute_time = self.mla_q_a_proj.compute_time + self.mla_q_nope.compute_time + self.mla_q_rope.compute_time + self.mla_k_nope.compute_time + self.mla_k_rope.compute_time + self.mla_q_absorb.compute_time
        return self.compute_time

    def memory_cost(self):
        self.bytes = self.mla_q_a_proj.bytes + self.mla_q_nope.bytes + self.mla_q_rope.bytes + self.mla_k_nope.bytes + self.mla_k_rope.bytes + self.mla_q_absorb.bytes
        self.memory_time = self.mla_q_a_proj.memory_time + self.mla_q_nope.memory_time + self.mla_q_rope.memory_time + self.mla_k_nope.memory_time + self.mla_k_rope.memory_time + self.mla_q_absorb.memory_time
        return self.memory_time

    def e2e_cost(self):
        self.e2e_time = self.mla_q_a_proj.e2e_time + self.mla_q_nope.e2e_time + self.mla_q_rope.e2e_time + self.mla_k_nope.e2e_time + self.mla_k_rope.e2e_time + self.mla_q_absorb.e2e_time
        return self.e2e_time

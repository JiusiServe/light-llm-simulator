from src.ops.matmul import OpGeMatmul


class OpMlaProlog:
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
        # compute query, key, value
        if self.model_config.q_lora_rank == 0:
            self.mla_q_a_proj = OpGeMatmul(
                "mla_q_a_proj",
                0, 0, 0,
                self.aichip_config
            )
            self.mla_q_nope = OpGeMatmul(
                "mla_q_nope",
                self.attn_bs,
                self.model_config.hidden_size,
                self.model_config.num_attention_heads*self.model_config.qk_nope_head_dim / config.attn_tensor_parallel,
                self.aichip_config
            )
            self.mla_q_rope = OpGeMatmul(
                "mla_q_rope",
                self.attn_bs,
                self.model_config.hidden_size,
                self.model_config.num_attention_heads * self.model_config.qk_rope_head_dim / config.attn_tensor_parallel,
                self.aichip_config
            )
        else:
            self.mla_q_a_proj = OpGeMatmul(
                "mla_q_a_proj",
                self.attn_bs,
                self.model_config.hidden_size,
                self.model_config.q_lora_rank,
                self.aichip_config
            )
            self.mla_q_nope = OpGeMatmul(
                "mla_q_nope",
                self.attn_bs,
                self.model_config.q_lora_rank,
                self.model_config.num_attention_heads*self.model_config.qk_nope_head_dim / config.attn_tensor_parallel,
                self.aichip_config
            )
            self.mla_q_rope = OpGeMatmul(
                "mla_q_rope",
                self.attn_bs,
                self.model_config.q_lora_rank,
                self.model_config.num_attention_heads * self.model_config.qk_rope_head_dim / config.attn_tensor_parallel,
                self.aichip_config
            )
        self.mla_k_nope = OpGeMatmul(
            "mla_k_nope",
            self.attn_bs,
            self.model_config.hidden_size,
            self.model_config.kv_lora_rank,
            self.aichip_config
        )
        self.mla_k_rope = OpGeMatmul(
            "mla_k_rope",
            self.attn_bs,
            self.model_config.hidden_size,
            self.model_config.qk_rope_head_dim,
            self.aichip_config
        )
        self.mla_q_absorb = OpGeMatmul( 
            "mla_q_absorb",
            self.attn_bs,
            self.model_config.num_attention_heads*self.model_config.qk_nope_head_dim / config.attn_tensor_parallel,
            self.model_config.kv_lora_rank,
            self.aichip_config,
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
        self.compute_time = (
            self.mla_q_a_proj.compute_time +
            self.mla_q_nope.compute_time +
            self.mla_q_rope.compute_time +
            self.mla_k_nope.compute_time +
            self.mla_k_rope.compute_time +
            self.mla_q_absorb.compute_time
        )
        return self.compute_time

    def memory_cost(self):
        self.bytes = (
            self.mla_q_a_proj.bytes +
            self.mla_q_nope.bytes +
            self.mla_q_rope.bytes +
            self.mla_k_nope.bytes +
            self.mla_k_rope.bytes +
            self.mla_q_absorb.bytes
        )
        self.memory_time = (
            self.mla_q_a_proj.memory_time +
            self.mla_q_nope.memory_time +
            self.mla_q_rope.memory_time +
            self.mla_k_nope.memory_time +
            self.mla_k_rope.memory_time +
            self.mla_q_absorb.memory_time
        )
        return self.memory_time

    def e2e_cost(self):
        self.e2e_time = (
            self.mla_q_a_proj.e2e_time +
            self.mla_q_nope.e2e_time +
            self.mla_q_rope.e2e_time +
            self.mla_k_nope.e2e_time +
            self.mla_k_rope.e2e_time +
            self.mla_q_absorb.e2e_time
        )
        return self.e2e_time

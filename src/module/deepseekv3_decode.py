import logging
from src.module.base import BaseModule
from src.ops import (
    OpMlaProlog,
    DeepSeekV3PageAttentionInt8,
    OpGeMatmul,
    OpQuantMatmul,
    OpSwiglu,
    OpGroupedMatmul,
    MoeDispatch,
    MoeCombine,
)


class DeepSeekV3DecodeAttn(BaseModule):
    def __init__(self, model_config, aichip_config, search_config):
        super().__init__(model_config, aichip_config, search_config)
        self.aichip_config = aichip_config.hw_conf
        self.attn_bs = search_config.attn_bs

        self.attention_e2e_time: float = 0.0
        self.attention_compute_time: float = 0.0
        self.attention_memory_time: float = 0.0

        self._build_ops()

    def _build_ops(self):
        # mla prolog
        self.mla_prolog = OpMlaProlog(self.model_config, self.aichip_config, self.search_config)
        # page attention
        self.page_attention = DeepSeekV3PageAttentionInt8(self.model_config, self.aichip_config, self.search_config)
        # matrix absorption
        self.bmm_uv_absorb = OpGeMatmul(
            self.attn_bs * self.search_config.seq_len,
            self.model_config.kv_lora_rank,
            self.model_config.num_attention_heads * self.model_config.v_head_dim,
            self.aichip_config
        )
        # compute o_proj
        self.bmm_o_proj = OpQuantMatmul(
            self.attn_bs * self.search_config.seq_len,
            self.model_config.num_attention_heads * self.model_config.v_head_dim,
            self.model_config.hidden_size,
            self.aichip_config
        )

        self.ops = [
            self.mla_prolog,
            self.page_attention,
            self.bmm_uv_absorb,
            self.bmm_o_proj
        ]

    def _aggregate_times(self):
        self.attention_e2e_time = (
            self.mla_prolog.e2e_time +
            self.page_attention.e2e_time +
            self.bmm_uv_absorb.e2e_time +
            self.bmm_o_proj.e2e_time
        )
        logging.debug(
            f"Attention Module: - mla_prolog: {self.mla_prolog.e2e_time * 1e6:.2f}us, "
            f"page_attention: {self.page_attention.e2e_time * 1e6:.2f}us, "
            f"bmm_uv_absorb: {self.bmm_uv_absorb.e2e_time * 1e6:.2f}us, "
            f"bmm_o_proj: {self.bmm_o_proj.e2e_time * 1e6:.2f}us"
        )
        self.attention_compute_time = (
            self.mla_prolog.compute_time +
            self.page_attention.compute_time +
            self.bmm_uv_absorb.compute_time +
            self.bmm_o_proj.compute_time
        )
        self.attention_memory_time = (
            self.mla_prolog.memory_time +
            self.page_attention.memory_time +
            self.bmm_uv_absorb.memory_time +
            self.bmm_o_proj.memory_time
        )

        logging.debug(
            f"Attention Module - mla_prolog: {self.mla_prolog.e2e_time * 1e6:.2f}us, "
            f"page_attention: {self.page_attention.e2e_time * 1e6:.2f}us, "
            f"bmm_uv_absorb: {self.bmm_uv_absorb.e2e_time * 1e6:.2f}us, "
            f"bmm_o_proj: {self.bmm_o_proj.e2e_time * 1e6:.2f}us"
        )


class DeepSeekV3DecodeMLP(BaseModule):
    def __init__(self, model_config, aichip_config, search_config):
        super().__init__(model_config, aichip_config, search_config)
        self.mlp_e2e_time: float = 0.0
        self.mlp_compute_time: float = 0.0
        self.mlp_memory_time: float = 0.0
        self.commu_time: float = 0.0
        self.dispatch_time: float = 0.0
        self.combine_time: float = 0.0
        self._build_ops()

    def _build_ops(self):
        bs = self.search_config.attn_bs * self.model_config.num_experts_per_tok * self.search_config.seq_len
        self.dispatch_time = MoeDispatch().dispatch_latency(
            self.model_config, self.aichip_config.hw_conf, self.search_config
        )
        self.mlp_up = OpQuantMatmul(
            bs, self.model_config.hidden_size, 2 * self.model_config.intermediate_size, 
            self.aichip_config.hw_conf
        )
        self.mlp_swiglu = OpSwiglu(
            bs, 2 * self.model_config.moe_intermediate_size, 
            self.aichip_config.hw_conf, elem_size=1
        )
        self.mlp_down = OpQuantMatmul(
            bs, self.model_config.intermediate_size, self.model_config.hidden_size, 
            self.aichip_config.hw_conf
        )
        self.combine_time = MoeCombine().combine_latency(
            self.model_config, self.aichip_config.hw_conf, self.search_config
        )
        
        self.ops = [self.mlp_up, self.mlp_swiglu, self.mlp_down]

    def _aggregate_times(self):
        self.mlp_e2e_time = (
            self.mlp_up.e2e_time + self.mlp_swiglu.e2e_time + self.mlp_down.e2e_time
        )
        self.mlp_compute_time = (
            self.mlp_up.compute_time + self.mlp_swiglu.compute_time + self.mlp_down.compute_time
        )
        self.mlp_memory_time = (
            self.mlp_up.memory_time + self.mlp_swiglu.memory_time + self.mlp_down.memory_time
        )
        self.commu_time = self.dispatch_time + self.combine_time


class DeepSeekV3DecodeMoe(BaseModule):
    def __init__(self, model_config, aichip_config, search_config):
        super().__init__(model_config, aichip_config, search_config)
        self.tokens_per_ffn_die = search_config.ffn_bs * search_config.seq_len
        self.routed_expert_per_die = search_config.routed_expert_per_die
        self.moe_e2e_time: float = 0.0
        self.moe_compute_time: float = 0.0
        self.moe_memory_time: float = 0.0
        self.commu_time: float = 0.0
        self.dispatch_time: float = 0.0
        self.combine_time: float = 0.0
        self._build_ops()

    def _build_ops(self):
        self.dispatch_time = MoeDispatch().dispatch_latency(
            self.model_config, self.aichip_config.hw_conf, self.search_config
        )
        self.moe_up = OpGroupedMatmul(
            self.routed_expert_per_die,
            self.tokens_per_ffn_die,
            self.model_config.hidden_size,
            2 * self.model_config.moe_intermediate_size / self.search_config.ffn_tensor_parallel,
            self.aichip_config.hw_conf,
            elem_size=1
        )
        self.moe_swiglu = OpSwiglu(
            self.tokens_per_ffn_die,
            2 * self.model_config.moe_intermediate_size,
            self.aichip_config.hw_conf,
            elem_size=1
        )
        self.moe_down = OpGroupedMatmul(
            self.routed_expert_per_die,
            self.tokens_per_ffn_die,
            self.model_config.moe_intermediate_size / self.search_config.ffn_tensor_parallel,
            self.model_config.hidden_size,
            self.aichip_config.hw_conf,
            elem_size=1
        )
        self.combine_time = MoeCombine().combine_latency(
            self.model_config, self.aichip_config.hw_conf, self.search_config
        )

        self.ops = [self.moe_up, self.moe_swiglu, self.moe_down]

    def _aggregate_times(self):
        self.moe_e2e_time = (
            self.moe_up.e2e_time + self.moe_swiglu.e2e_time + self.moe_down.e2e_time
        )
        self.moe_compute_time = (
            self.moe_up.compute_time + self.moe_swiglu.compute_time + self.moe_down.compute_time
        )
        self.moe_memory_time = (
            self.moe_up.memory_time + self.moe_swiglu.memory_time + self.moe_down.memory_time
        )
        self.commu_time = self.dispatch_time + self.combine_time

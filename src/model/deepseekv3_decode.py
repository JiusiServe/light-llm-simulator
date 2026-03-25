import logging
from conf.common import MAX_AVG_RATIO
from conf.config import Config
from src.model.base import BaseModule
from src.ops import (
    OpMlaProlog,
    MLAFlashAttentionInt8,
    OpGeMatmul,
    OpQuantBatchMatmul,
    OpSwiglu,
    OpGroupedMatmul,
    Dispatch,
    Combine,
    OpNorm
)


class DeepSeekV3DecodeAttn(BaseModule):
    """
    Description:
        The attention module of the DeepSeekV3-671B model.
        It is composed of a MLA prolog, a page attention, a matrix absorption and a output projection.
        It uses Multi-Latent Attention (MLA) to calculate the attention.
        It can calculate the end-to-end time, compute time and memory time of the attention.
    Attributes:
        attn_bs: The batch size of the attention.
        mla_prolog: The MLA preprocessing to get the query, key and value.
        page_attention: The page attention operator.
        bmm_uv_absorb: The Value matrix absorption operator.
        bmm_o_proj: The output projection operator.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_bs = self.config.attn_bs

        self._build_ops()

    def _build_ops(self):
        # mla prolog
        self.mla_prolog = OpMlaProlog(self.config)
        # page attention
        self.page_attention = MLAFlashAttentionInt8(self.config)
        # matrix absorption
        self.bmm_uv_absorb = OpGeMatmul(
            "bmm_uv_absorb",
            self.attn_bs * self.config.seq_len,
            self.model_config.kv_lora_rank,
            self.model_config.num_attention_heads * self.model_config.v_head_dim,
            self.aichip_config
        )
        # compute o_proj
        self.bmm_o_proj = OpQuantBatchMatmul(
            "bmm_o_proj",
            self.attn_bs * self.config.seq_len,
            self.model_config.num_attention_heads * self.model_config.v_head_dim,
            self.model_config.hidden_size,
            self.aichip_config
        )
        # compute norm
        self.norm = OpNorm(self.attn_bs, self.aichip_config)

        self.ops = [
            self.mla_prolog,
            self.page_attention,
            self.bmm_uv_absorb,
            self.bmm_o_proj,
            self.norm
        ]

    def _aggregate_times(self):
        self.e2e_time = (
            self.mla_prolog.e2e_time +
            self.page_attention.e2e_time +
            self.bmm_uv_absorb.e2e_time +
            self.bmm_o_proj.e2e_time +
            self.norm.e2e_time
        )
        self.compute_time = (
            self.mla_prolog.compute_time +
            self.page_attention.compute_time +
            self.bmm_uv_absorb.compute_time +
            self.bmm_o_proj.compute_time +
            self.norm.compute_time
        )
        self.memory_time = (
            self.mla_prolog.memory_time +
            self.page_attention.memory_time +
            self.bmm_uv_absorb.memory_time +
            self.bmm_o_proj.memory_time + 
            self.norm.memory_time
        )
        logging.info(
            f"Attention Module - attn_bs: {self.config.attn_bs}, "
            f"mla_prolog: {self.mla_prolog.e2e_time * 1e6:.2f}us, "
            f"page_attention: {self.page_attention.e2e_time * 1e6:.2f}us, "
            f"bmm_uv_absorb: {self.bmm_uv_absorb.e2e_time * 1e6:.2f}us, "
            f"bmm_o_proj: {self.bmm_o_proj.e2e_time * 1e6:.2f}us, "
            f"norm: {self.norm.e2e_time * 1e6:.2f}us"
        )


class DeepSeekV3DecodeMLP(BaseModule):
    """
    Description:
        The MLP module of the DeepSeekV3-671B model.
        It is composed of a dispatch, a MLP up, a MLP swiglu, a MLP down and a combine.
        It can calculate the end-to-end time, compute time and memory time of the MLP.
    Attributes:
        bs: The batch size of the MLP.
    """
    def __init__(self, config: Config):
        super().__init__(config, hw_type="ffn")
        self.commu_time: float = 0.0
        self.dispatch_time: float = 0.0
        self.combine_time: float = 0.0
        self._build_ops()

    def _build_ops(self):
        bs = self.config.attn_bs * self.config.seq_len
        self.mlp_up = OpQuantBatchMatmul(
            "mlp_up",
            bs,
            self.model_config.hidden_size,
            2 * self.model_config.intermediate_size,
            self.aichip_config
        )
        self.mlp_swiglu = OpSwiglu(
            bs,
            2 * self.model_config.moe_intermediate_size,
            self.aichip_config
        )
        self.mlp_down = OpQuantBatchMatmul(
            "mlp_down",
            bs,
            self.model_config.intermediate_size,
            self.model_config.hidden_size,
            self.aichip_config
        )

        self.ops = [
            self.mlp_up,
            self.mlp_swiglu,
            self.mlp_down
        ]

    def _aggregate_times(self):
        self.e2e_time = (
            self.mlp_up.e2e_time + self.mlp_swiglu.e2e_time + self.mlp_down.e2e_time
        )
        self.compute_time = (
            self.mlp_up.compute_time + self.mlp_swiglu.compute_time + self.mlp_down.compute_time
        )
        self.memory_time = (
            self.mlp_up.memory_time + self.mlp_swiglu.memory_time + self.mlp_down.memory_time
        )

        logging.info(
            f"MLP Module - mlp_up: {self.mlp_up.e2e_time * 1e6:.2f}us, "
            f"mlp_swiglu: {self.mlp_swiglu.e2e_time * 1e6:.2f}us, "
            f"mlp_down: {self.mlp_down.e2e_time * 1e6:.2f}us, "
        )


class DeepSeekV3DecodeMoe(BaseModule):
    """
    Description:
        The MoE module of the DeepSeekV3-671B model.
        It is composed of a dispatch, a MoE up, a MoE swiglu, a MoE down and a combine.
        It can calculate the end-to-end time, compute time and memory time of the MoE.
    Attributes:
        tokens_per_ffn_die: The number of tokens per FFN die.
        routed_expert_per_die: The number of routed experts per die.
        commu_time: The communication time of the MoE.
        dispatch_time: The dispatch time of the MoE.
        combine_time: The combine time of the MoE.
    """
    def __init__(self, config: Config):
        super().__init__(config, hw_type="ffn")
        self.tokens_per_ffn_die = config.ffn_bs * config.seq_len
        self.routed_expert_per_die = config.routed_expert_per_die
        self.commu_time: float = 0.0
        self.dispatch_time: float = 0.0
        self.combine_time: float = 0.0
        self._build_ops()

    def _build_ops(self):
        self.dispatch = Dispatch(self.config)
        self.moe_up = OpGroupedMatmul(
            "moe_up",
            self.routed_expert_per_die,
            self.tokens_per_ffn_die,
            self.model_config.hidden_size,
            2 * self.model_config.moe_intermediate_size / self.config.ffn_tensor_parallel,
            self.aichip_config,
            elem_size=1
        )
        self.moe_swiglu = OpSwiglu(
            self.tokens_per_ffn_die,
            2 * self.model_config.moe_intermediate_size,
            self.aichip_config
        )
        self.moe_down = OpGroupedMatmul(
            "moe_down",
            self.routed_expert_per_die,
            self.tokens_per_ffn_die,
            self.model_config.moe_intermediate_size / self.config.ffn_tensor_parallel,
            self.model_config.hidden_size,
            self.aichip_config,
            elem_size=1
        )
        self.combine = Combine(self.config)

        self.ops = [
            self.dispatch,
            self.moe_up,
            self.moe_swiglu,
            self.moe_down,
            self.combine
        ]

    def _aggregate_times(self):
        self.dispatch_time = self.dispatch.e2e_time
        self.e2e_time = (
            self.moe_up.e2e_time * MAX_AVG_RATIO +
            self.moe_swiglu.e2e_time +
            self.moe_down.e2e_time * MAX_AVG_RATIO
        )
        self.compute_time = (
            self.moe_up.compute_time * MAX_AVG_RATIO +
            self.moe_swiglu.compute_time +
            self.moe_down.compute_time * MAX_AVG_RATIO
        )
        self.memory_time = (
            self.moe_up.memory_time * MAX_AVG_RATIO +
            self.moe_swiglu.memory_time +
            self.moe_down.memory_time * MAX_AVG_RATIO
        )
        self.combine_time = self.combine.e2e_time
        self.commu_time = self.dispatch_time + self.combine_time

        logging.info(
            f"MoE Module - ffn_bs: {self.config.ffn_bs}, "
            f"moe_up: {self.moe_up.e2e_time * 1e6:.2f}us, "
            f"moe_swiglu: {self.moe_swiglu.e2e_time * 1e6:.2f}us, "
            f"moe_down: {self.moe_down.e2e_time * 1e6:.2f}us, "
            f"dispatch_time: {self.dispatch_time * 1e6:.2f}us, "
            f"combine_time: {self.combine_time * 1e6:.2f}us, "
            f"commu_time: {self.commu_time * 1e6:.2f}us"
        )

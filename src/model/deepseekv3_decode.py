import logging
from conf.common import MAX_AVG_RATIO
from conf.config import Config
from src.model.base import BaseModule
from src.ops import (
    OpMlaProlog,
    MLAFlashAttentionInt8,
    OpMatmul,
    OpBatchMatmul,
    OpTransposeBatchMatmul,
    OpQuantBatchMatmul,
    OpSwiglu,
    OpGroupedMatmul,
    Dispatch,
    Combine,
    OpAddRmsNorm,
    OpDynamicQuant,
    OpMoeGating
)


class DeepSeekV3DecodeEmbedding(BaseModule):
    """
    Description:
        The embedding module of the DeepSeekV3-671B model.
        It is composed of a embedding and a embedding dropout.
        It can calculate the end-to-end time, compute time and memory time of the embedding.
    Attributes:
        attn_bs: The batch size of the embedding.
        aichip_config: The AI chip configuration.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_bs = config.attn_bs
        self.aichip_config = config.aichip_config
        self._build_ops()

    def _build_ops(self):
        self.embedding = OpBatchMatmul(
            "embedding",
            self.attn_bs * self.config.seq_len,
            self.model_config.vocab_size,
            self.model_config.hidden_size,
            self.aichip_config
        )
        self.dropout = OpAddRmsNorm(
            "embedding_dropout",
            self.attn_bs,
            self.config.seq_len,
            self.model_config.hidden_size,
            self.aichip_config
        )
        self.ops = [
            self.embedding,
            self.dropout
        ]

    def _aggregate_times(self):
        self.e2e_time = (
            self.embedding.e2e_time +
            self.dropout.e2e_time
        )
        self.compute_time = (
            self.embedding.compute_time +
            self.dropout.compute_time
        )
        self.memory_time = (
            self.embedding.memory_time +
            self.dropout.memory_time
        )
        logging.info(
            f"Embedding Module - embedding: {self.embedding.e2e_time * 1e6:.2f}us, "
            f"dropout: {self.dropout.e2e_time * 1e6:.2f}us, "
            f"e2e_time: {self.e2e_time * 1e6:.2f}us"
        )


class DeepSeekV3DecodeAttn(BaseModule):
    """
    Description:
        The attention module of the DeepSeekV3-671B model.
        It is composed of a MLA prolog, a page attention, a matrix absorption and a output projection.
        It uses Multi-Latent Attention (MLA) to calculate the attention.
        It can calculate the end-to-end time, compute time and memory time of the attention.
    Attributes:
        input_norm: The input layernorm operator.
        attn_bs: The batch size of the attention.
        mla_prolog: The MLA preprocessing to get the query, key and value.
        page_attention: The page attention operator.
        bmm_uv_absorb: The Value matrix absorption operator.
        dynamic_quant: The dynamic quant operator.
        bmm_o_proj: The output projection operator.
        post_attention_norm: The post attention layernorm operator.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_bs = self.config.attn_bs

        self._build_ops()

    def _build_ops(self):
        # inputlayernorm
        self.input_norm = OpAddRmsNorm(
            "inputlayernorm",
            self.attn_bs,
            self.config.seq_len,
            self.model_config.hidden_size,
            self.aichip_config
        )
        # mla prolog
        self.mla_prolog = OpMlaProlog(self.config)
        # page attention
        self.page_attention = MLAFlashAttentionInt8(self.config)
        # matrix absorption
        self.bmm_uv_absorb = OpTransposeBatchMatmul(
            "bmm_uv_absorb",
            self.model_config.num_attention_heads,
            self.attn_bs * self.config.seq_len,
            self.model_config.kv_lora_rank,
            self.model_config.v_head_dim,
            self.aichip_config
        )
        # dynamic quant
        self.dynamic_quant = OpDynamicQuant(
            "attn_dynamic_quant",
            self.attn_bs * self.config.seq_len,
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
        # post attention layernorm
        self.post_attention_norm = OpAddRmsNorm(
            "post attention layernorm",
            self.attn_bs,
            self.config.seq_len,
            self.model_config.hidden_size,
            self.aichip_config
        )
        # compute gating score
        self.matmul = OpMatmul(
            "matmul",
            self.attn_bs * self.config.seq_len,
            self.model_config.hidden_size,
            self.model_config.n_routed_experts,
            self.aichip_config
        )
        self.gating = OpMoeGating(
            "gating",
            self.attn_bs,
            self.config.seq_len,
            self.model_config.n_routed_experts,
            self.model_config.num_experts_per_tok,
            self.aichip_config
        )
        self.ops = [
            self.input_norm,
            self.mla_prolog,
            self.page_attention,
            self.bmm_uv_absorb,
            self.dynamic_quant,
            self.bmm_o_proj,
            self.post_attention_norm,
            self.matmul,
            self.gating
        ]

    def _aggregate_times(self):
        self.e2e_time = (
            self.input_norm.e2e_time +
            self.mla_prolog.e2e_time +
            self.page_attention.e2e_time +
            self.bmm_uv_absorb.e2e_time +
            self.dynamic_quant.e2e_time +
            self.bmm_o_proj.e2e_time +
            self.post_attention_norm.e2e_time +
            self.matmul.e2e_time +
            self.gating.e2e_time
        )
        self.compute_time = (
            self.input_norm.compute_time +
            self.mla_prolog.compute_time +
            self.page_attention.compute_time +
            self.bmm_uv_absorb.compute_time +
            self.dynamic_quant.compute_time +
            self.bmm_o_proj.compute_time +
            self.post_attention_norm.compute_time +
            self.matmul.compute_time +
            self.gating.compute_time
        )
        self.memory_time = (
            self.input_norm.memory_time +
            self.mla_prolog.memory_time +
            self.page_attention.memory_time +
            self.bmm_uv_absorb.memory_time +
            self.dynamic_quant.memory_time +
            self.bmm_o_proj.memory_time + 
            self.post_attention_norm.memory_time +
            self.matmul.memory_time +
            self.gating.memory_time
        )
        logging.info(
            f"Attention Module - attn_bs: {self.config.attn_bs}, "
            f"input_norm: {self.input_norm.e2e_time * 1e6:.2f}us, "
            f"mla_prolog: {self.mla_prolog.e2e_time * 1e6:.2f}us, "
            f"page_attention: {self.page_attention.e2e_time * 1e6:.2f}us, "
            f"bmm_uv_absorb: {self.bmm_uv_absorb.e2e_time * 1e6:.2f}us, "
            f"dynamic_quant: {self.dynamic_quant.e2e_time * 1e6:.2f}us, "
            f"bmm_o_proj: {self.bmm_o_proj.e2e_time * 1e6:.2f}us, "
            f"post_attention_norm: {self.post_attention_norm.e2e_time * 1e6:.2f}us, "
            f"matmul: {self.matmul.e2e_time * 1e6:.2f}us, "
            f"gating: {self.gating.e2e_time * 1e6:.2f}us, "
            f"attn_time: {self.e2e_time * 1e6:.2f}us"
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
        self.dispatch_time = self.dispatch.e2e_time * MAX_AVG_RATIO
        self.e2e_time = (
            self.moe_up.e2e_time +
            self.moe_swiglu.e2e_time +
            self.moe_down.e2e_time
        ) * MAX_AVG_RATIO
        self.compute_time = (
            self.moe_up.compute_time +
            self.moe_swiglu.compute_time +
            self.moe_down.compute_time
        ) * MAX_AVG_RATIO
        self.memory_time = (
            self.moe_up.memory_time +
            self.moe_swiglu.memory_time +
            self.moe_down.memory_time
        ) * MAX_AVG_RATIO
        self.combine_time = self.combine.e2e_time * MAX_AVG_RATIO
        self.commu_time = self.dispatch_time + self.combine_time

        logging.info(
            f"MoE Module - attn_bs: {self.config.attn_bs}, ffn_bs: {self.config.ffn_bs}, "
            f"moe_up: {self.moe_up.e2e_time * 1e6:.2f}us, "
            f"moe_swiglu: {self.moe_swiglu.e2e_time * 1e6:.2f}us, "
            f"moe_down: {self.moe_down.e2e_time * 1e6:.2f}us, "
            f"dispatch_time: {self.dispatch_time * 1e6:.2f}us, "
            f"combine_time: {self.combine_time * 1e6:.2f}us, "
            f"commu_time: {self.commu_time * 1e6:.2f}us"
        )


class DeepSeekV3DecodeLMHead(BaseModule):
    """
    Description:
        The LMHead module of the DeepSeekV3-671B model.
        It is composed of a norm and a LMHead projection.
        It can calculate the end-to-end time, compute time and memory time of the LMHead.
    Attributes:
        attn_bs: The batch size of the LMHead.
        aichip_config: The AI chip configuration.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_bs = config.attn_bs
        self.aichip_config = config.aichip_config
        self._build_ops()

    def _build_ops(self):
        self.norm = OpAddRmsNorm(
            "norm",
            self.attn_bs,
            self.config.seq_len,
            self.model_config.hidden_size,
            self.aichip_config
        )
        self.lm_head_linear = OpBatchMatmul(
            "lm_head_linear",
            self.attn_bs * self.config.seq_len,
            self.model_config.hidden_size,
            self.model_config.vocab_size,
            self.aichip_config
        )
        self.ops = [
            self.norm,
            self.lm_head_linear
        ]

    def _aggregate_times(self):
        self.e2e_time = (
            self.norm.e2e_time +
            self.lm_head_linear.e2e_time
        )
        self.compute_time = (
            self.norm.compute_time +
            self.lm_head_linear.compute_time
        )
        self.memory_time = (
            self.norm.memory_time +
            self.lm_head_linear.memory_time
        )
        logging.info(
            f"Embedding Module - norm: {self.norm.e2e_time * 1e6:.2f}us, "
            f"lm_head_linear: {self.lm_head_linear.e2e_time * 1e6:.2f}us, "
            f"e2e_time: {self.e2e_time * 1e6:.2f}us"
        )

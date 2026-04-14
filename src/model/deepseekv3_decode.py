import logging
from conf.common import MAX_AVG_RATIO
from conf.config import Config
from src.model.base import BaseModule
from src.ops import (
    OpMlaProlog, MLAFlashAttentionInt8, OpMatmul, OpBatchMatmul,
    OpTransposeBatchMatmul, OpQuantBatchMatmul, OpSwiglu, OpGroupedMatmul,
    Dispatch, Combine, OpAddRmsNorm, OpDynamicQuant, OpMoeGating,
    OpA2ESend, OpA2ERecv, OpE2ARecv
)


class DeepSeekV3DecodeEmbedding(BaseModule):
    """The embedding module of DeepSeekV3-671B."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_bs = config.attn_bs
        self._build_ops()

    def _build_ops(self):
        bs = self.attn_bs * self.config.seq_len
        hidden = self.model_config.hidden_size
        self.embedding = OpBatchMatmul("embedding", bs, self.model_config.vocab_size, hidden, self.aichip_config)
        self.dropout = OpAddRmsNorm("embedding_dropout", self.attn_bs, self.config.seq_len, hidden, self.aichip_config)
        self.ops = [self.embedding, self.dropout]


class DeepSeekV3DecodeAttn(BaseModule):
    """The attention module of DeepSeekV3-671B using Multi-Latent Attention (MLA)."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_bs = config.attn_bs
        self._build_ops()

    def _build_ops(self):
        bs = self.attn_bs * self.config.seq_len
        hidden = self.model_config.hidden_size
        heads = self.model_config.num_attention_heads
        v_dim = heads * self.model_config.v_head_dim

        self.input_norm = OpAddRmsNorm("input_norm", self.attn_bs, self.config.seq_len, hidden, self.aichip_config)
        self.mla_prolog = OpMlaProlog(self.config)
        self.page_attention = MLAFlashAttentionInt8(self.config)
        self.bmm_uv_absorb = OpTransposeBatchMatmul("bmm_uv_absorb", heads, bs, self.model_config.kv_lora_rank, self.model_config.v_head_dim, self.aichip_config)
        self.dynamic_quant = OpDynamicQuant("dynamic_quant", bs, v_dim, self.aichip_config)
        self.bmm_o_proj = OpQuantBatchMatmul("bmm_o_proj", bs, v_dim, hidden, self.aichip_config)
        self.post_attention_norm = OpAddRmsNorm("post_norm", self.attn_bs, self.config.seq_len, hidden, self.aichip_config)
        self.matmul = OpMatmul("gating_matmul", bs, hidden, self.model_config.n_routed_experts, self.aichip_config)
        self.gating = OpMoeGating("gating", self.attn_bs, self.config.seq_len, self.model_config.n_routed_experts, self.model_config.num_experts_per_tok, self.aichip_config)

        self.ops = [
            self.input_norm, self.mla_prolog, self.page_attention, self.bmm_uv_absorb,
            self.dynamic_quant, self.bmm_o_proj, self.post_attention_norm, self.matmul, self.gating
        ]

        if self.config.serving_mode == "AFD":
            self.a2e_send = OpA2ESend(self.config)
            self.a2e_recv = OpA2ERecv(self.config)
            self.ops.extend([self.a2e_send, self.a2e_recv])


class DeepSeekV3DecodeMLP(BaseModule):
    """The MLP module of DeepSeekV3-671B."""
    def __init__(self, config: Config):
        super().__init__(config, hw_type="ffn")
        self._build_ops()

    def _build_ops(self):
        bs = self.config.attn_bs * self.config.seq_len
        hidden = self.model_config.hidden_size
        inter = self.model_config.intermediate_size

        self.mlp_up = OpQuantBatchMatmul("mlp_up", bs, hidden, 2 * inter, self.aichip_config)
        self.mlp_swiglu = OpSwiglu(bs, 2 * inter, self.aichip_config)
        self.mlp_down = OpQuantBatchMatmul("mlp_down", bs, inter, hidden, self.aichip_config)
        self.ops = [self.mlp_up, self.mlp_swiglu, self.mlp_down]


class DeepSeekV3DecodeMoe(BaseModule):
    """The MoE module of DeepSeekV3-671B."""
    def __init__(self, config: Config):
        super().__init__(config, hw_type="ffn")
        self.tokens_per_ffn_die = config.ffn_bs * config.seq_len
        self.routed_expert_per_die = config.routed_expert_per_die
        self.commu_time = 0.0
        self._build_ops()

    def _build_ops(self):
        hidden = self.model_config.hidden_size
        moe_inter = self.model_config.moe_intermediate_size
        tp = self.config.ffn_tensor_parallel

        self.dispatch = Dispatch(self.config)
        self.moe_up = OpGroupedMatmul("moe_up", self.routed_expert_per_die, self.tokens_per_ffn_die, hidden, 2 * moe_inter / tp, self.aichip_config, elem_size=1)
        self.moe_swiglu = OpSwiglu(self.tokens_per_ffn_die, 2 * moe_inter, self.aichip_config)
        self.moe_down = OpGroupedMatmul("moe_down", self.routed_expert_per_die, self.tokens_per_ffn_die, moe_inter / tp, hidden, self.aichip_config, elem_size=1)
        self.combine = Combine(self.config)
        self.dispatch()
        self.combine()
        self.ops = [self.moe_up, self.moe_swiglu, self.moe_down]
        if self.config.serving_mode == "AFD":
            self.e2a_recv = OpE2ARecv(self.config)
            self.e2a_recv()

    def _aggregate_times(self):
        super()._aggregate_times()
        self.dispatch_time = self.dispatch.e2e_time * MAX_AVG_RATIO
        self.combine_time = self.combine.e2e_time * MAX_AVG_RATIO
        self.commu_time = self.dispatch_time + self.combine_time
        if self.config.serving_mode == "AFD":
            self.commu_time = self.commu_time + self.e2a_recv.e2e_time


class DeepSeekV3DecodeLMHead(BaseModule):
    """The LMHead module of DeepSeekV3-671B."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_bs = config.attn_bs
        self._build_ops()

    def _build_ops(self):
        bs = self.attn_bs * self.config.seq_len
        hidden = self.model_config.hidden_size
        self.norm = OpAddRmsNorm("norm", self.attn_bs, self.config.seq_len, hidden, self.aichip_config)
        self.lm_head = OpBatchMatmul("lm_head", bs, hidden, self.model_config.vocab_size, self.aichip_config)
        self.ops = [self.norm, self.lm_head]

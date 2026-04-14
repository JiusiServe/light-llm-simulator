from conf.common import MAX_AVG_RATIO
from conf.config import Config
from src.model.base import BaseModule
from src.ops import (
    OpQuantBatchMatmul, OpBatchMatmul, OpRotary, GQAFlashAttentionFP16,
    Dispatch, Combine, OpGroupedMatmul, OpSwiglu, OpAddRmsNorm,
    OpDynamicQuant, OpA2ESend, OpA2ERecv, OpE2ARecv
)


class Qwen235DecodeEmbedding(BaseModule):
    """The embedding module of Qwen3-235B."""
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


class Qwen235DecodeAttn(BaseModule):
    """The attention module of Qwen3-235B using Grouped Query Attention (GQA)."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_bs = config.attn_bs
        self._build_ops()

    def _build_ops(self):
        bs = self.attn_bs * self.config.seq_len
        hidden = self.model_config.hidden_size
        num_heads = self.model_config.num_heads
        kv_heads = self.model_config.kv_heads
        head_size = self.model_config.head_size

        self.input_norm = OpAddRmsNorm("input_norm", self.attn_bs, self.config.seq_len, hidden, self.aichip_config)
        self.query_states = OpBatchMatmul("query_states", bs, hidden, num_heads * head_size, self.aichip_config)
        self.key_states = OpBatchMatmul("key_states", bs, hidden, kv_heads * head_size, self.aichip_config)
        self.value_states = OpBatchMatmul("value_states", bs, hidden, kv_heads * head_size, self.aichip_config)
        self.query_rope = OpRotary("query_rope", bs, num_heads, self.config.seq_len, hidden, self.aichip_config)
        self.key_rope = OpRotary("key_rope", bs, kv_heads, self.config.seq_len, hidden, self.aichip_config)
        self.page_attention = GQAFlashAttentionFP16(self.config)
        self.dynamic_quant = OpDynamicQuant("dynamic_quant", bs, num_heads * head_size, self.aichip_config)
        self.bmm_o_proj = OpQuantBatchMatmul("bmm_o_proj", bs, num_heads * head_size, hidden, self.aichip_config)
        self.post_attention_norm = OpAddRmsNorm("post_norm", self.attn_bs, self.config.seq_len, hidden, self.aichip_config)

        self.ops = [
            self.input_norm, self.query_states, self.key_states, self.value_states,
            self.query_rope, self.key_rope, self.page_attention,
            self.dynamic_quant, self.bmm_o_proj, self.post_attention_norm
        ]

        if self.config.serving_mode == "AFD":
            self.a2e_send = OpA2ESend(self.config)
            self.a2e_recv = OpA2ERecv(self.config)
            self.ops.extend([self.a2e_send, self.a2e_recv])


class Qwen235DecodeMoe(BaseModule):
    """The MoE module of Qwen3-235B."""
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
        self.moe_up = OpGroupedMatmul("Qwen235BMoEUP", self.routed_expert_per_die, self.tokens_per_ffn_die, hidden, 2 * moe_inter / tp, self.aichip_config, elem_size=1)
        self.moe_swiglu = OpSwiglu(self.tokens_per_ffn_die, 2 * moe_inter, self.aichip_config)
        self.moe_down = OpGroupedMatmul("Qwen235BMoEDown", self.routed_expert_per_die, self.tokens_per_ffn_die, moe_inter / tp, hidden, self.aichip_config, elem_size=1)
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


class Qwen235DecodeLMHead(BaseModule):
    """The LMHead module of Qwen3-235B."""
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

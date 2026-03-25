import logging
from conf.common import MAX_AVG_RATIO
from conf.config import Config
from src.model.base import BaseModule
from src.ops import (
    OpGeMatmul,
    OpQuantBatchMatmul,
    OpRotary,
    GQAFlashAttentionFP16,
    Dispatch,
    Combine,
    OpGroupedMatmul,
    OpSwiglu,
    OpNorm
)

class Qwen235DecodeAttn(BaseModule):
    '''
    Description:
        The attention module of the Qwen3-235B-A22B model.
        It is composed of a query projection, a key projection, a value projection, 
        a rotary position embedding, a page attention and a output projection.
        It can calculate the end-to-end time, compute time and memory time of the attention.
    Attributes:
        attn_bs: The batch size of the attention.
        query_states: The query projection.
        key_states: The key projection.
        value_states: The value projection
        query_rope: The query rotary position embedding.
        key_rope: The key rotary position embedding.
        page_attention: The page attention operator.
        bmm_o_proj: The output projection operator.
    '''
    def __init__(self, config: Config):
        super().__init__(config)
        self.model_config = config.model_config
        self.aichip_config = config.aichip_config
        self.attn_bs = config.attn_bs
        self._build_ops()

    def _build_ops(self):
        # q_proj
        self.bs = self.attn_bs * self.config.seq_len
        self.query_states = OpGeMatmul(
            "query_states",
            self.bs,
            self.model_config.hidden_size,
            self.model_config.num_heads * self.model_config.head_size,
            self.aichip_config
        )
        # k_proj
        self.key_states = OpGeMatmul(
            "key_states",
            self.bs,
            self.model_config.hidden_size,
            self.model_config.kv_heads * self.model_config.head_size,
            self.aichip_config
        )
        # v_proj
        self.value_states = OpGeMatmul(
            "value_states",
            self.bs,
            self.model_config.hidden_size,
            self.model_config.kv_heads * self.model_config.head_size,
            self.aichip_config
        )
        # rotary position embedding
        self.query_rope = OpRotary(
            "query_rope",
            self.bs,
            self.model_config.num_heads,
            self.config.seq_len,
            self.model_config.hidden_size,
            self.aichip_config
        )
        self.key_rope = OpRotary(
            "key_rope",
            self.bs,
            self.model_config.kv_heads,
            self.config.seq_len,
            self.model_config.hidden_size,
            self.aichip_config
        )
        # page attention
        self.page_attention = GQAFlashAttentionFP16(self.config)
        # compute o_proj
        self.bmm_o_proj = OpQuantBatchMatmul(
            "bmm_o_proj",
            self.bs,
            self.model_config.num_heads * self.model_config.head_size,
            self.model_config.hidden_size,
            self.aichip_config
        )
        # compute norm
        self.norm = OpNorm(self.attn_bs, self.aichip_config)

        self.ops = [
            self.query_states,
            self.key_states,
            self.value_states,
            self.query_rope,
            self.key_rope,
            self.page_attention,
            self.bmm_o_proj,
            self.norm
        ]

    def _aggregate_times(self):
        self.e2e_time = (
            self.query_states.e2e_time +
            self.key_states.e2e_time +
            self.value_states.e2e_time +
            self.query_rope.e2e_time +
            self.key_rope.e2e_time +
            self.page_attention.e2e_time +
            self.bmm_o_proj.e2e_time +
            self.norm.e2e_time
        )
        self.compute_time = (
            self.query_states.compute_time +
            self.key_states.compute_time +
            self.value_states.compute_time +
            self.query_rope.compute_time +
            self.key_rope.compute_time +
            self.page_attention.compute_time +
            self.bmm_o_proj.compute_time +
            self.norm.compute_time
        )
        self.memory_time = (
            self.query_states.memory_time +
            self.key_states.memory_time +
            self.value_states.memory_time +
            self.query_rope.memory_time +
            self.key_rope.memory_time +
            self.page_attention.memory_time +
            self.bmm_o_proj.memory_time +
            self.norm.memory_time
        )

        logging.info(
            f"Attention Module - attn_bs: {self.config.attn_bs}, "
            f"query_states: {self.query_states.e2e_time * 1e6:.2f}us, "
            f"key_states: {self.key_states.e2e_time * 1e6:.2f}us, "
            f"value_states: {self.value_states.e2e_time * 1e6:.2f}us, "
            f"query_rope: {self.query_rope.e2e_time * 1e6:.2f}us, "
            f"key_rope: {self.key_rope.e2e_time * 1e6:.2f}us, "
            f"page_attention: {self.page_attention.e2e_time * 1e6:.2f}us, "
            f"bmm_o_proj: {self.bmm_o_proj.e2e_time * 1e6:.2f}us, "
            f"norm: {self.norm.e2e_time * 1e6: .2f}us"
        )


class Qwen235DecodeMoe(BaseModule):
    '''
    Description:
        The MoE module of the Qwen3-235B-A22B model.
        It is composed of a dispatch, a MoE up, a MoE swiglu, a MoE down and a combine.
        It can calculate the end-to-end time, compute time and memory time of the MoE.
    Attributes:
        tokens_per_ffn_die: The number of tokens per FFN die.
        routed_expert_per_die: The number of routed experts per FFN die.
        commu_time: The communication time of the MoE.
        dispatch_time: The dispatch time of the MoE.
        combine_time: The combine time of the MoE.
    '''
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
            "Qwen235BMoEUP",
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
            "Qwen235BMoEDown",
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

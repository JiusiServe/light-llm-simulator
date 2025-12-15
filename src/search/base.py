from typing import Tuple
from abc import ABC, abstractmethod
from conf.common import BYTE_2_GB, DTYPE_BF16
from conf.model_config import ModelConfig, ModelType
from conf.hardware_config import HardwareTopology


class BaseSearch(ABC):
    def __init__(
        self,
        model_type: ModelType,
        model_config: ModelConfig,
        aichip_config: HardwareTopology,
        kv_len: int,
        seq_len: int,
        multi_token_ratio: float,
        latency: float
    ):
        self.model_type = model_type
        self.model_config = model_config
        self.aichip_config = aichip_config
        self.kv_len = kv_len
        self.seq_len = seq_len
        self.multi_token_ratio = multi_token_ratio
        self.latency = latency

    def compute_memory_size(
        self,
        model_config: ModelConfig,
        attn_bs: int
    ) -> Tuple[float, float, float]:

        # KVCache Size
        kv_size = (
            attn_bs * self.kv_len * 
            (model_config.kv_lora_rank + model_config.qk_rope_head_dim) * 
            model_config.num_layers * BYTE_2_GB * DTYPE_BF16
        )

        # Attention Static Memory
        q_a_proj = model_config.q_lora_rank * model_config.hidden_size
        q_nope = (
            model_config.num_attention_heads * 
            model_config.qk_nope_head_dim * 
            model_config.q_lora_rank
        )
        q_rope = (
            model_config.num_attention_heads * 
            model_config.qk_rope_head_dim * 
            model_config.q_lora_rank
        )
        k_nope = model_config.kv_lora_rank * model_config.hidden_size
        k_rope = model_config.qk_rope_head_dim * model_config.hidden_size
        q_absorb = (
            model_config.num_attention_heads * 
            model_config.qk_nope_head_dim * 
            model_config.kv_lora_rank
        )
        uv_absorb = (
            model_config.kv_lora_rank * 
            model_config.num_attention_heads * 
            model_config.v_head_dim
        )
        o_proj = (
            model_config.num_attention_heads * 
            model_config.v_head_dim * 
            model_config.hidden_size
        )

        attn_static_memory = (
            (q_a_proj + q_nope + q_rope + k_nope + k_rope + q_absorb + uv_absorb + o_proj) * 
            model_config.num_layers * BYTE_2_GB
        )

        # per router expert momory
        per_router_expert_memory = (
            3 * model_config.moe_intermediate_size * 
            model_config.hidden_size * 
            model_config.num_layers * BYTE_2_GB
        )
        
        return kv_size, attn_static_memory, per_router_expert_memory

    @abstractmethod
    def deployment(self):
        pass

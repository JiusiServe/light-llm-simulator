from typing import Tuple
from abc import ABC, abstractmethod
from conf.common import BYTE_2_GB, DTYPE_INT8, DTYPE_BF16
from conf.model_config import ModelConfig
from conf.config import Config


class BaseSearch(ABC):
    '''
    Description:
        The base class for the search algorithm.
        Inherit from this class to create a new search algorithm, such as AFD Search, DeepEP Search, etc.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config: Config):
        self.config = config

    def compute_MLA_memory_size(
        self,
        model_config: ModelConfig,
        attn_bs: int
    ) -> Tuple[float, float, float, float]:
        '''
        Description:
            Compute the memory size of the model used MLA attention mechanism.
        Args:
            model_config: The configuration of the model.
            attn_bs: The attention batch size.
        Returns:
            kv_size: The memory size of the KV cache, GB.
            attn_static_memory: The memory size of the attention static memory, GB.
            mlp_static_memory: The memory size of the MLP static memory, GB.
            per_router_expert_memory: The memory size of the per router expert memory, GB.
        TODO:
            Add the memory size of the attention dynamic memory.
            Add the memory size of the MLP dynamic memory.
        '''
        # KVCache Size
        # KVCache int8
        # kv_cache = (
        #     attn_bs * self.config.kv_len * model_config.kv_lora_rank * 
        #     model_config.num_layers * BYTE_2_GB * DTYPE_INT8
        # )
        # KVCache bf16
        kv_cache = (
            attn_bs * self.config.kv_len * model_config.kv_lora_rank * 
            model_config.num_layers * BYTE_2_GB * DTYPE_BF16
        )
        kr_cache = (
            attn_bs * self.config.kv_len * model_config.qk_rope_head_dim * 
            model_config.num_layers * BYTE_2_GB * DTYPE_BF16
        )
        kv_size = kv_cache + kr_cache

        # Attention Static Memory
        if model_config.q_lora_rank == 0:
            q_a_proj = 0
            q_nope = (
                model_config.num_attention_heads *
                model_config.qk_nope_head_dim *
                model_config.hidden_size
            )
            q_rope = (
                model_config.num_attention_heads * 
                model_config.qk_rope_head_dim * 
                model_config.hidden_size
            )
        else:
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

        # mlp memory
        if model_config.num_layers > model_config.num_moe_layers:
            mlp_static_memory = (
                    3 * model_config.intermediate_size * 
                    model_config.hidden_size * BYTE_2_GB
                ) * (model_config.num_layers - model_config.num_moe_layers)
        else:
            mlp_static_memory = 0

        # per router expert momory
        per_router_expert_memory = (
            3 * model_config.moe_intermediate_size * 
            model_config.hidden_size * 
            model_config.num_layers * BYTE_2_GB
        )
        
        return kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory

    def compute_GQA_memory_size(
        self,
        model_config: ModelConfig,
        attn_bs: int
    ) -> Tuple[float, float, float, float]:
        '''
        Description:
            Compute the memory size of the model used GQA attention mechanism.
        Args:
            model_config: The configuration of the model.
            attn_bs: The attention batch size.
        Returns:
            kv_size: The memory size of the KV cache, GB.
            attn_static_memory: The memory size of the attention static memory, GB.
            mlp_static_memory: The memory size of the MLP static memory, GB.
            per_router_expert_memory: The memory size of the per router expert memory, GB.
        TODO:
            Add the memory size of the attention dynamic memory.
            Add the memory size of the MLP dynamic memory.
        '''
        # KVCache Size
        kv_size = (
            attn_bs * self.config.kv_len * 
            model_config.kv_heads * 
            model_config.head_size * 
            model_config.num_layers * BYTE_2_GB * DTYPE_INT8 * 2
        )

        # Attention Static Memory
        q_proj = model_config.num_heads * model_config.head_size * model_config.hidden_size
        k_proj = model_config.kv_heads * model_config.head_size * model_config.hidden_size
        v_proj = model_config.kv_heads * model_config.head_size * model_config.hidden_size
        o_rope = model_config.num_heads * model_config.head_size * model_config.hidden_size

        attn_static_memory = (
            (q_proj + k_proj + v_proj + o_rope) * 
            model_config.num_layers * BYTE_2_GB
        )

        # mlp memory
        if model_config.num_layers > model_config.num_moe_layers:
            mlp_static_memory = (
                    3 * model_config.intermediate_size * 
                    model_config.hidden_size * BYTE_2_GB
                ) * (model_config.num_layers - model_config.num_moe_layers)
        else:
            mlp_static_memory = 0

        # per router expert momory
        per_router_expert_memory = (
            3 * model_config.moe_intermediate_size * 
            model_config.hidden_size * 
            model_config.num_layers * BYTE_2_GB
        )
        return kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory

    @abstractmethod
    def deployment(self):
        pass

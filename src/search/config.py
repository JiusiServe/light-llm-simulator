from conf.common import MIN_ROUTED_EXPERT_PER_DIE
import math


class SearchConfig:
    def __init__(
        self,
        model_config,
        search_mode,
        attn_bs,
        seq_len,
        kv_len,
        attn_die=8,
        ffn_die=8,
        attn_tensor_parallel=1,
        ffn_tensor_parallel=1
    ):
        self.model_config = model_config
        self.search_mode = search_mode
        self.attn_bs = attn_bs
        self.seq_len = seq_len
        self.kv_len = kv_len
        self.ffn_bs = attn_bs * model_config.num_experts_per_tok * attn_die / ffn_die
        self.attn_die = attn_die
        self.ffn_die = ffn_die
        self.routed_expert_per_die = max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.model_config.n_routed_experts / ffn_die)
            )
        self.attn_tensor_parallel = attn_tensor_parallel
        self.ffn_tensor_parallel = ffn_tensor_parallel

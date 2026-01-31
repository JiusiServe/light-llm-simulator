from src.ops.base import BaseOp

class Dispatch(BaseOp):
    '''
    Description:
        The dispatch operation.
        It is used to route the tokens to the experts.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=1):
        self.config = config
        super().__init__("Dispatch", config.aichip_config2, elem_size)
        self.model_config = config.model_config

    def op_memory_disc(self):
        return 0.7

    def memory_cost(self):
        if self.config.serving_mode == "DeepEP":
            dispatch_packet = (
                self.config.attn_bs *
                self.config.seq_len *
                self.model_config.hidden_size *
                self.model_config.num_experts_per_tok *
                self.elem_size
            )
        elif self.config.serving_mode == "AFD":
            coeff = max(
                self.config.attn_die / self.config.ffn_die,
                self.config.ffn_die / self.config.attn_die
            )
            dispatch_packet = (
                self.config.attn_bs *
                self.config.seq_len *
                self.model_config.hidden_size *
                self.model_config.num_experts_per_tok *
                self.elem_size *
                coeff
            )
            self.inter_node_bandwidth = min(
                self.config.aichip_config1.inter_node_bandwidth ,
                self.config.aichip_config2.inter_node_bandwidth
            ) * self.op_memory_disc()
        self.memory_time = dispatch_packet / self.inter_node_bandwidth
        return self.memory_time

class Combine(BaseOp):
    '''
    Description:
        The combine operation.
        It is used to aggregate the outputs of the experts.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=2):
        self.config = config
        super().__init__("Combine", config.aichip_config2, elem_size)
        self.model_config = config.model_config

    def op_memory_disc(self):
        return 0.7

    def memory_cost(self):
        if self.config.serving_mode == "DeepEP":
            combine_packet = (
                self.config.attn_bs *
                self.config.seq_len *
                self.model_config.hidden_size *
                self.model_config.num_experts_per_tok *
                self.elem_size
            )
        elif self.config.serving_mode == "AFD":
            coeff = max(
                self.config.attn_die / self.config.ffn_die,
                self.config.ffn_die / self.config.attn_die
            )
            combine_packet = (
                self.config.attn_bs *
                self.config.seq_len *
                self.model_config.hidden_size *
                self.model_config.num_experts_per_tok *
                self.elem_size *
                coeff
            )
            self.inter_node_bandwidth = min(
                self.config.aichip_config1.inter_node_bandwidth,
                self.config.aichip_config2.inter_node_bandwidth
            ) * self.op_memory_disc()
        self.memory_time = combine_packet / self.inter_node_bandwidth
        return self.memory_time

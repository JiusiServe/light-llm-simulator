from src.ops.base import BaseOp
from conf.common import US_2_SEC


class BaseOpP2P(BaseOp):
    '''
    Description:
        The base class for the p2p operators.
        Inherit from this class to create a new p2p operators, 
        such as A2E send, A2E recv, E2A send, E2A recv
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, name, config, elem_size=2):
        self.config = config
        super().__init__(name, config.aichip_config, elem_size, static_cost=3 * US_2_SEC)

    def compute_cost(self):
        self.compute_time = 0.0
        return self.compute_time

    def memory_cost(self):
        attn_die = self.config.attn_die
        ffn_die = self.config.ffn_die
        batch_size = self.config.attn_bs
        seq_len = self.config.seq_len
        hidden_size = self.config.model_config.hidden_size
        self.total_data_movement = max(attn_die / ffn_die, ffn_die / attn_die) * batch_size * seq_len * hidden_size * self.elem_size
        bw = min(self.config.aichip_config2.inter_node_bandwidth, self.config.aichip_config.inter_node_bandwidth) * self.op_memory_disc()
        self.memory_time = self.total_data_movement / bw
        return self.memory_time


class OpA2ESend(BaseOpP2P):
    def __init__(self, config, elem_size=2):
        super().__init__("A2ESend", config, elem_size)


class OpA2ERecv(BaseOpP2P):
    def __init__(self, config, elem_size=2):
        super().__init__("A2ERecv", config, elem_size)


class OpE2ARecv(BaseOpP2P):
    def __init__(self, config, elem_size=2):
        super().__init__("E2ARecv", config, elem_size)

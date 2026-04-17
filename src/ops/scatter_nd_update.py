from src.ops.base import BaseOp


class OpScatterNdUpdate(BaseOp):
    '''
    Description:
        Updates values in input tensor at positions specified by indices with values from updates.

    Attributes:
        config: The configuration of the search task.
    '''

    def __init__(self, config, elem_size=2):
        self.aichip_config = config.aichip_config
        self.attn_bs = config.attn_bs
        self.seq_len = config.seq_len
        self.kv_len = config.kv_len
        self.qk_nope_head_dim = config.model_config.qk_nope_head_dim

        super().__init__("ScatterNdUpdate", self.aichip_config, elem_size)

    def compute_cost(self):
        # ~3 vector ops per update: address computation, gather source, scatter write
        self.total_computation = 3 * self.attn_bs * self.seq_len * self.qk_nope_head_dim
        self.compute_time = self.total_computation / self.vector_flops
        return self.compute_time

    def memory_cost(self):
        # Read input tensor, [b, s, d], bf16
        input_bytes = self.attn_bs * self.seq_len * self.qk_nope_head_dim * self.elem_size
        # Read indices tensor: [b, s, 1], INT32
        indices_bytes = self.attn_bs * self.seq_len * 4
        # Read updates tensor: [b, s, d], bf16
        updates_bytes = self.attn_bs * self.seq_len * self.qk_nope_head_dim * self.elem_size

        self.total_data_movement = input_bytes + indices_bytes + updates_bytes
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time

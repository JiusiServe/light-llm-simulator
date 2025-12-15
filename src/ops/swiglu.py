# dequant, swiglu, quant
# m: bs, n:2*moe_intermediate_size

from src.ops.base import BaseOp

class OpSwiglu(BaseOp):
    def __init__(self, m, n, aichip_config, elem_size=2):
        super().__init__(aichip_config, elem_size)
        self.m = m
        self.n = n

    def __call__(self):
        self.op_disc_factor()
        self.compute_cost()
        self.memory_cost()
        self.e2e_cost()

    def compute_cost(self):
        # dequant: int8-->fp16
        self.compute_flops = 2 * self.m * self.n + 6 * self.m * self.n / 2 + self.m * self.n / 2+ 2 * self.m * self.n / 2
        self.compute_time = self.compute_flops / self.vector_flops
        return self.compute_time

    def memory_cost(self):
        self.bytes = 0
        self.memory_time = 0
        return self.memory_time

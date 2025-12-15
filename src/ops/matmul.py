import logging
from src.ops.base import BaseOp


class OpGeMatmul(BaseOp):
    def __init__(self, m, n, k, aichip_config, elem_size=2):
        super().__init__(aichip_config, elem_size)
        self.m = m
        self.n = n
        self.k = k

    def op_disc_factor(self):
        return 0.16

    def compute_cost(self):
        self.compute_flops = 2 * self.m * self.n * self.k
        self.compute_time = self.compute_flops / self.cube_flops
        return self.compute_time

    def memory_cost(self):
        self.bytes = self.elem_size * self.n * self.k + self.elem_size * self.m * self.n
        self.memory_time = self.bytes / self.mem_bw_local
        return self.memory_time


class OpQuantMatmul(BaseOp):
    def __init__(self, m, n, k, aichip_config, elem_size=1):
        super().__init__(aichip_config, elem_size)
        self.m = m
        self.n = n
        self.k = k

    def op_disc_factor(self):
        return 0.5

    def compute_cost(self):
        self.compute_flops = 2 * self.m * self.n * self.k
        self.compute_time = self.compute_flops / self.cube_flops
        return self.compute_time

    def memory_cost(self):
        self.bytes = self.elem_size * self.n * self.k + self.elem_size * self.m * self.n
        self.memory_time = self.bytes / self.mem_bw_local
        return self.memory_time


class OpGroupedMatmul(BaseOp):
    def __init__(self, num_experts, bs, m, n, aichip_config, elem_size=1):
        logging.debug(f"-------OpGroupedMatmul-------:")
        self.num_experts = num_experts
        self.bs = bs
        self.m = m
        self.n = n
        super().__init__(aichip_config, elem_size)
        self.mem_bw_inter = self.mem_bw_inter * self.op_disc_factor()
        self.mem_bw_intra = self.mem_bw_intra * self.op_disc_factor()
        self.mem_bw_local = self.mem_bw_local * self.op_disc_factor()

    def op_disc_factor(self):
        if self.num_experts == 2 or self.num_experts == 3:
            if self.bs <= 256:
                return 0.42
            elif self.bs > 256 and self.bs <= 512:
                return 0.5
            elif self.bs > 512 and self.bs <= 768:
                return 0.4
            elif self.bs > 768 and self.bs <= 1024:
                return 0.3
            elif self.bs > 1024 and self.bs <= 1280:
                return 0.26
            elif self.bs > 1280 and self.bs <= 1536:
                return 0.23
            elif self.bs > 1536 and self.bs <= 1792:
                return 0.195
            elif self.bs > 1792 and self.bs <= 2048:
                return 0.18
            else:
                return 0.16
        if self.num_experts == 4 or self.num_experts == 5:
            if self.bs <= 576:
                return 0.55
            elif self.bs > 576 and self.bs <= 704:
                return 0.5
            elif self.bs > 704 and self.bs <= 1024:
                return 0.37
            elif self.bs > 1024 and self.bs <= 1536:
                return 0.275
            elif self.bs > 1536 and self.bs <= 2048:
                return 0.25
            else:
                return 0.21
        if self.num_experts == 6 and self.num_experts == 7:
            if self.bs <= 768:
                return 0.58
            elif self.bs > 768 and self.bs <= 1536:
                return 0.34
            elif self.bs > 1536 and self.bs <= 3072:
                return 0.26
            else:
                return 0.20
        if self.num_experts == 8 and self.num_experts == 9:
            if self.bs <= 1024:
                return 0.58
            elif self.bs > 1024 and self.bs <= 2048:
                return 0.36
            elif self.bs > 2048 and self.bs <= 4096:
                return 0.26
            else:
                return 0.20
        if self.num_experts > 9 and self.num_experts <= 12:
            if self.bs <= 1536:
                return 0.6
            else:
                return 0.35
        if self.num_experts > 12:
            if self.bs <= 2048:
                return 0.6
            elif self.bs > 2048 and self.bs <= 4096:
                return 0.35
            else:
                return 0.25
        return 0.55

    def compute_cost(self):
        self.compute_flops = 2 * self.bs * self.m * self.n
        self.compute_time = self.compute_flops / self.cube_flops * self.op_disc_factor()
        return self.compute_time

    def memory_cost(self):
        self.bytes = self.elem_size * self.bs * self.m + self.elem_size * self.m * self.n * self.num_experts
        self.memory_time = self.bytes / self.mem_bw_local
        return self.memory_time

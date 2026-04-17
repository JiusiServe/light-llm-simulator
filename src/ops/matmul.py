from src.ops.base import BaseOp
from conf.common import US_2_SEC


class OpMatmul(BaseOp):
    '''
    Description:
        The matrix multiplication operation.
        It is used to multiply two matrices of shape (m, n) and (n, k).
    Attributes:
        name: The name of the operation.
        m: The number of rows of the first matrix.
        n: The number of columns of the first matrix.
        k: The number of columns of the second matrix.
        aichip_config: The hardware configuration.
    '''
    def __init__(self, name, m, n, k, aichip_config, elem_size=4):
        self.m = m
        self.n = n
        self.k = k
        self.elem_size = elem_size
        super().__init__(name, aichip_config, self.elem_size, static_cost=10*US_2_SEC)

    def compute_cost(self):
        self.total_computation = 2 * self.m * self.n * self.k
        self.compute_time = self.total_computation / self.cube_flops_fp16 / 0.8
        return self.compute_time

    def memory_cost(self):
        # input type: float, output type: float
        self.total_data_movement = self.elem_size * (self.m * self.n + self.n * self.k + self.m * self.k)
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time


class OpBatchMatmul(BaseOp):
    '''
    Description:
        The batch matrix multiplication operation.
        It is used to multiply two matrices of shape (m, n) and (n, k).
    Attributes:
        name: The name of the operation.
        m: The number of rows of the first matrix.
        n: The number of columns of the first matrix.
        k: The number of columns of the second matrix.
        aichip_config: The hardware configuration.
    '''
    def __init__(self, name, m, n, k, aichip_config):
        self.m = m
        self.n = n
        self.k = k
        self.elem_size = 2
        super().__init__(name, aichip_config, self.elem_size)

    def compute_cost(self):
        self.total_computation = 2 * self.m * self.n * self.k
        self.compute_time = self.total_computation / self.cube_flops_fp16 / 0.8
        return self.compute_time

    def memory_cost(self):
        # input type: bf16, output type: bf16
        self.total_data_movement = 2 * (self.m * self.n + self.n * self.k + self.m * self.k)
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time


class OpQuantBatchMatmul(BaseOp):
    '''
    Description:
        The quantized batch matrix multiplication operation.
        It is used to multiply two matrices of shape (m, n) and (n, k).
    Attributes:
        name: The name of the operation.
        m: The number of rows of the first matrix.
        n: The number of columns of the first matrix.
        k: The number of columns of the second matrix.
        aichip_config: The hardware configuration.
    '''
    def __init__(self, name, m, n, k, aichip_config):
        self.m = m
        self.n = n
        self.k = k
        self.elem_size = 1
        super().__init__(name, aichip_config, self.elem_size)

    def compute_cost(self):
        self.total_computation = 2 * self.m * self.n * self.k
        self.compute_time = self.total_computation / self.cube_flops_int8 / 0.8
        return self.compute_time

    def memory_cost(self):
        # input type: int8, output type: bf16
        self.total_data_movement = self.m * self.n + self.n * self.k + 2 * self.m * self.k
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time


class OpTransposeBatchMatmul(BaseOp):
    '''
    Description:
        Perform the matrix multiplication of tensor x1 and tensor x2.
        Tensors support transpose operation.
        tensor x1, x2 must be three-dimensional tensors.
        tensor x1: [B, M, N]
        tensor x2: [B, N, K]
        tensor output: [M, B, K]
    Attributes:
        name: The name of the operation.
        m: The number of rows of the first matrix.
        n: The number of columns of the first matrix.
        k: The number of columns of the second matrix.
        aichip_config: The hardware configuration.
    '''
    def __init__(self, name, b, m, n, k, aichip_config):
        self.b = b
        self.m = m
        self.n = n
        self.k = k
        self.static_cost = 5 * US_2_SEC
        self.elem_size = 2
        super().__init__(name, aichip_config, self.elem_size, self.static_cost)

    def compute_cost(self):
        self.compute_flops = 2 * self.m * self.n * self.k
        self.compute_time = self.compute_flops / self.cube_flops
        return self.compute_time

    def memory_cost(self):
        # input tensor x1: [B, M, N], bf16
        # input tensor x2: [B, N, K], bf16
        # output tensor output: [M, B, K], bf16
        self.bytes = (self.b * self.m * self.n + self.b * self.n * self.k + self.m * self.b * self.k) * self.elem_size
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time


class OpGroupedMatmul(BaseOp):
    '''
    Description:
        The grouped matrix multiplication operation.
        It is used to multiply two matrices of shape (m, n) and (n, k).
    Attributes:
        name: The name of the operation.
        num_experts: The number of experts.
        bs: The FFN batch size.
        m: The number of columns of the first matrix.
        n: The number of columns of the second matrix.
        aichip_config: The hardware configuration.
        elem_size: The element size of the data.
    '''
    def __init__(self, name, num_experts, bs, m, n, aichip_config, elem_size=1):
        self.num_experts = num_experts
        self.bs = bs
        self.m = m
        self.n = n
        super().__init__(name, aichip_config, elem_size)

    def op_memory_disc(self):
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
            if self.bs <= 512:
                return 0.55
            elif self.bs > 512 and self.bs <= 768:
                return 0.5
            elif self.bs > 768 and self.bs <= 1024:
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
        self.compute_time = self.compute_flops / self.cube_flops
        return self.compute_time

    def memory_cost(self):
        self.bytes = (
            self.elem_size * self.bs * self.m +
            self.elem_size * self.m * self.n * self.num_experts
        )
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time

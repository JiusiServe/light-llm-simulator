# aclnnDynamicQuant
from src.ops.base import BaseOp
from conf.common import US_2_SEC

class OpDynamicQuant(BaseOp):
    '''
    Description:
        The DynamicQuant operation.
        It is used to compute the DynamicQuant function.
    Attributes:
        bs: batch size.
        seq_len: sequence length.
        hidden_size: hidden size.
    Compute formula:
        scale = rowmax(abs(x)) / 127
        y = round(x / scale)
        where:
            x: input tensor
            y: output tensor
    '''
    def __init__(self, name, shape1, shape2, aichip_config, elem_size=2):
        self.shape1 = shape1
        self.shape2 = shape2
        super().__init__(name, aichip_config, elem_size, static_cost=20 * US_2_SEC)

    def compute_cost(self):
        # 5 times vector operation
        self.compute_time = 5 * self.shape1 * self.shape2 / self.vec_flops_fp16
        return self.compute_time

    def memory_cost(self):
        # input tensor x: [shape1, shape2] fp16
        # output tensor y: [shape1, shape2] int8
        self.bytes = 3 * self.shape1 * self.shape2
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time

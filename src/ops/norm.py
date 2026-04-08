# aclnnAddRmsNorm

from src.ops.base import BaseOp
from conf.common import US_2_SEC

class OpAddRmsNorm(BaseOp):
    '''
    Description:
        The AddRmsNorm operation.
        It is used to compute the AddRmsNorm function.
    Attributes:
        bs: batch size.
        seq_len: sequence length.
        hidden_size: hidden size.
    Compute formula:
        x = x1 + x2
        Rms(x) = sqrt(mean(x^2) + eps)
        RmsNorm = gamma * x / Rms(x)
        where:
            x1, x2: input tensor
            gamma: scale tensor
            eps: epsilon
            mean: mean of the input tensor
            sqrt: square root of the input tensor
    '''
    def __init__(self, name, bs, seq_len, hidden_size, aichip_config, elem_size=2):
        self.bs = bs
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        super().__init__(name, aichip_config, elem_size, static_cost=20 * US_2_SEC)

    def compute_cost(self):
        # norm(x) = x / sqrt(mean(x^2) + eps)
        # 6 times vector operation
        self.compute_time = 6 * self.bs * self.seq_len * self.hidden_size / self.vec_flops_fp16
        return self.compute_time

    def memory_cost(self):
        # input tensor x1, x2: [bs, seq_len, hidden_size]
        # gamma: [hidden_size]
        # output tensor y: [bs, seq_len, hidden_size]
        self.bytes = self.elem_size * (self.bs * self.seq_len * self.hidden_size * 3 + self.hidden_size)
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time

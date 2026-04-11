from src.ops.base import BaseOp
from conf.common import US_2_SEC


class OpMoeGating(BaseOp):
    '''
    Description:
        Perform a sigmoid function on the input x, sort the results into groups, 
        and finally select the top k experts based on the sorted groupings.
    Attributes:
        name: The name of the operation.
        bs: The batch size.
        seq_len: The sequence length.
        num_experts: The number of experts.
        topk: Number of experts activated per token.
        aichip_config: The hardware configuration.
    '''
    def __init__(self, name, bs, seq_len, num_experts, topk, aichip_config):
        self.bs = bs
        self.seq_len = seq_len
        self.num_experts = num_experts
        self.topk = topk
        self.elem_size = 1
        self.memory_ratio = 0.5
        super().__init__(name, aichip_config, self.elem_size)

    def compute_cost(self):
        # sigmoid function
        # normout = sigmoid(x) + bias
        sigmoid = 5 * self.bs * self.seq_len * self.num_experts
        # group
        # groupout, groupId = TopK(ReduceSum(TopK(Spilt(normout, groupcount), k=2, dim=-1), dim=-1), k=kgroup)
        group = 3 * self.bs * self.seq_len * self.num_experts
        # topk
        # y, expertIdxOut = TopK(normout[groupId, :], k)
        gettopk = self.bs * self.seq_len * self.topk
        # getyout
        # yout = y / (ReduceSum(y, dim=-1) + eps) * routed_scaling_factor
        getyout = 4 * self.bs * self.seq_len * self.topk
        self.total_computation = sigmoid + group + gettopk + getyout
        self.compute_time = self.total_computation / self.vec_flops_fp16
        return self.compute_time


    def memory_cost(self):
        # input tensor x: [bs * seq_len, num_experts], float
        # input tensor bias: [num_experts], float
        # output tensor y: [bs * seq_len, topk], float
        # output tensor expertIdxOut: [bs * seq_len, topk], int32
        # output tensor norm: [bs * seq_len, num_experts], float
        x = 2 * self.bs * self.seq_len * self.num_experts
        bias = 2 * self.num_experts
        y = 2 * self.bs * self.seq_len * self.topk
        expertIdxOut = 4 * self.bs * self.seq_len * self.topk
        norm = 2 * self.bs * self.seq_len * self.num_experts
        self.total_data_movement = x + bias + y + expertIdxOut + norm
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time

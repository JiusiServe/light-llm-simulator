from src.ops.base import BaseOp
from conf.common import US_2_SEC
import math

def calculate_expected_nodes(x, y, k):
    '''
    Description:
        Calculate the expected number of nodes(or dies in a node) that the tokens are sent to,

    Parameters:
        x (int): The total number of nodes(or the total number of dies in a node)
        y (int): The number of experts per node(or the number of experts per die)
        k (int): top-k(or activated experts per nodes)
    '''

    total_experts = x * y

    if k >= total_experts:
        return x

    try:
        prob_not_selected = math.comb(total_experts - y, k) / math.comb(total_experts, k)
    except ValueError:
        # Handle the case where k is too large and n is less than k in math.comb(n, k).
        prob_not_selected = 0

    expected_m = x * (1 - prob_not_selected)
    return expected_m


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
        self.static_cost = 5 * US_2_SEC
        super().__init__("Dispatch", config.aichip_config2, elem_size, self.static_cost)
        self.model_config = config.model_config

    def memory_cost(self):
        coeff = max(
            self.config.attn_die / self.config.ffn_die,
            self.config.ffn_die / self.config.attn_die
        )
        self.inter_node_bandwidth = min(
            self.config.aichip_config.inter_node_bandwidth,
            self.config.aichip_config2.inter_node_bandwidth
        ) * self.op_memory_disc()
        self.intra_node_bandwidth = min(
            self.config.aichip_config.intra_node_bandwidth,
            self.config.aichip_config2.intra_node_bandwidth
        ) * self.op_memory_disc()

        total_node = self.config.ffn_die // self.config.aichip_config2.num_dies_per_node
        experts_per_node = math.ceil(
            self.model_config.n_routed_experts / self.config.ffn_die
        ) * self.config.aichip_config2.num_dies_per_node
        expected_nodes = math.ceil(
            calculate_expected_nodes(
            total_node,
            experts_per_node,
            self.model_config.num_experts_per_tok
        ))
        activated_experts_per_node = math.ceil(self.model_config.num_experts_per_tok / expected_nodes)
        expected_die_per_node = math.ceil(
            calculate_expected_nodes(
            self.config.aichip_config2.num_dies_per_node,
            math.ceil(self.model_config.n_routed_experts / self.config.ffn_die),
            activated_experts_per_node
        ))
        dispatch_packet_inter_node = (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            expected_nodes *
            self.elem_size *
            coeff
        )
        dispatch_packet_intra_node = (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            expected_die_per_node *
            self.elem_size
        )
        self.memory_time = (
            dispatch_packet_inter_node / self.inter_node_bandwidth +
            dispatch_packet_intra_node / self.intra_node_bandwidth
        )
        if self.config.ffn_die <= 48:
            a = 1.93
            b = 34.64
        elif self.config.ffn_die <= 104:
            a = 2.02
            b = 39.56
        elif self.config.ffn_die <= 216:
            a = 1.57
            b = 54.23
        else:
            a = 1.00
            b = 108.09
        self.memory_time = a * self.memory_time + b * US_2_SEC + self.static_cost
        self.memory_time = self.memory_time / 2
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
        self.static_cost = 5 * US_2_SEC
        super().__init__("Combine", config.aichip_config2, elem_size, self.static_cost)
        self.model_config = config.model_config

    def memory_cost(self):
        coeff = max(
            self.config.attn_die / self.config.ffn_die,
            self.config.ffn_die / self.config.attn_die
        )
        comm_ratio = 1
        self.inter_node_bandwidth = min(
            self.config.aichip_config.inter_node_bandwidth,
            self.config.aichip_config2.inter_node_bandwidth
        ) * self.op_memory_disc() * comm_ratio
        self.intra_node_bandwidth = min(
            self.config.aichip_config.intra_node_bandwidth,
            self.config.aichip_config2.intra_node_bandwidth
        ) * self.op_memory_disc() * comm_ratio

        total_node = self.config.ffn_die // self.config.aichip_config2.num_dies_per_node
        experts_per_node = math.ceil(
            self.model_config.n_routed_experts / self.config.ffn_die
        ) * self.config.aichip_config2.num_dies_per_node
        expected_nodes = math.ceil(
            calculate_expected_nodes(
            total_node,
            experts_per_node,
            self.model_config.num_experts_per_tok
        ))
        activated_experts_per_node = math.ceil(self.model_config.num_experts_per_tok / expected_nodes)
        expected_die_per_node = math.ceil(
            calculate_expected_nodes(
            self.config.aichip_config2.num_dies_per_node,
            math.ceil(self.model_config.n_routed_experts / self.config.ffn_die),
            activated_experts_per_node
        ))
        combine_packet_inter_node = (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            expected_nodes *
            self.elem_size *
            coeff
        )
        combine_packet_intra_node = (
            self.config.attn_bs *
            self.config.seq_len *
            self.model_config.hidden_size *
            expected_die_per_node *
            self.elem_size
        )
        self.memory_time = (
            combine_packet_inter_node / self.inter_node_bandwidth +
            combine_packet_intra_node / self.intra_node_bandwidth
        )
        if self.config.ffn_die <= 48:
            a = 1.74
            b = 66.35
        elif self.config.ffn_die <= 104:
            a = 1.82
            b = 75.2
        elif self.config.ffn_die <= 216:
            a = 1.41
            b = 101.62
        else:
            a = 0.9
            b = 198.56
        self.memory_time = a * self.memory_time + b * US_2_SEC + self.static_cost
        self.memory_time = self.memory_time / 2
        return self.memory_time

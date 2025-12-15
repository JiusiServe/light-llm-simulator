import logging
from conf.common import BYTE_2_KB, SEC_2_US


class MoeDispatch():
    def dispatch_latency(self, model_config, aichip_config, search_config):
        elem_size = 1 # type:int8
        comm_intra_ratio = 0.25
        # dispatch tensor:[bs * topk, seq_len, h]
        dispatch_packet = search_config.attn_bs * search_config.seq_len * model_config.hidden_size * model_config.num_experts_per_tok * elem_size
        if search_config.search_mode == "AFD":
            comm_intra_ratio = 0.6
            dispatch_packet = dispatch_packet * search_config.attn_die / search_config.ffn_die
            logging.debug(f'attn_die: {search_config.attn_die}, ffn_die: {search_config.ffn_die}')
        comm_intra_bw = aichip_config.intra_node_bandwidth * comm_intra_ratio
        dispatch_time = dispatch_packet / comm_intra_bw
        logging.debug(
            f"Dispatch - bs: {search_config.attn_bs}, seq_len: {search_config.seq_len}, "
            f"dispatch_packet: {dispatch_packet * BYTE_2_KB:.2f}KB, comm_intra_bw: {comm_intra_bw:.2f}GB/s, "
            f"dispatch_time: {dispatch_time * SEC_2_US:.2f}us"
        )
        return dispatch_time


class MoeCombine():
    def combine_latency(self, model_config, aichip_config, search_config):
        elem_size = 2 # type:fp16
        comm_intra_ratio = 0.3
        # combine tensor:[bs, seq_len, h]
        combine_packet = search_config.ffn_bs *search_config.seq_len * model_config.hidden_size * elem_size
        if search_config.search_mode == "AFD":
            comm_intra_ratio = 0.65
        comm_intra_bw = aichip_config.intra_node_bandwidth * comm_intra_ratio
        combine_time = combine_packet / comm_intra_bw
        logging.debug(
            f"Combine - bs: {search_config.ffn_bs}, seq_len: {search_config.seq_len}, "
            f"combine_packet: {combine_packet * BYTE_2_KB:.2f}KB, comm_intra_bw: {comm_intra_bw:.2f}GB/s, "
            f"combine_time: {combine_time * SEC_2_US:.2f}us"
        )
        return combine_time

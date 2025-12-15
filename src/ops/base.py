from abc import ABC, abstractmethod
from conf.hardware_config import HardwareTopology
from conf.common import SEC_2_US
import logging

class BaseOp(ABC):
    def __init__(
        self,
        aichip_config: HardwareTopology,
        elem_size: int
    ) -> None:
        self.aichip_config = aichip_config
        self.elem_size = elem_size
        self.cube_flops_int8 = self.aichip_config.cube_flops_int8 * self.op_disc_factor()
        self.cube_flops_fp16 = self.aichip_config.cube_flops_fp16 * self.op_disc_factor()
        self.vec_flops_int8 = self.aichip_config.vector_flops_int8 * self.op_disc_factor()
        self.vec_flops_fp16 = self.aichip_config.vector_flops_fp16 * self.op_disc_factor()
        self.cube_flops = self.cube_flops_fp16 if elem_size == 2 else self.cube_flops_int8
        self.vector_flops = self.vec_flops_fp16 if elem_size == 2 else self.vec_flops_int8
        self.mem_bw_inter = self.aichip_config.inter_node_bandwidth
        self.mem_bw_intra = self.aichip_config.intra_node_bandwidth
        self.mem_bw_local = self.aichip_config.local_memory_bandwidth
        logging.debug(
            f"cube_flops_int8: {self.cube_flops_int8}, cube_flops_fp16: {self.cube_flops_fp16}, "
            f"vec_flops_int8: {self.vec_flops_int8}, vec_flops_fp16: {self.vec_flops_fp16}, "
            f"cube_flops: {self.cube_flops}, vector_flops: {self.vector_flops}, "
            f"mem_bw_inter: {self.mem_bw_inter}, mem_bw_intra: {self.mem_bw_intra}, "
            f"mem_bw_local: {self.mem_bw_local}"
        )
        self.compute_time: float = 0.0
        self.memory_time: float = 0.0
        self.e2e_time: float = 0.0


    def __call__(self) -> float:
        self.compute_cost()
        self.memory_cost()
        self.e2e_cost()
        return self.e2e_time

    def op_disc_factor(self) -> float:
        return 0.651

    @abstractmethod
    def compute_cost(self) -> float:
        return self.compute_time

    @abstractmethod
    def memory_cost(self) -> float:
        return self.memory_time

    def e2e_cost(self) -> float:
        self.e2e_time = max(self.memory_time, self.compute_time)
        logging.debug(
            f"compute_time: {self.compute_time * SEC_2_US:.2f} us, "
            f"memory_time: {self.memory_time * SEC_2_US:.2f} us, "
            f"e2e_time: {self.e2e_time * SEC_2_US:.2f} us"
        )
        return self.e2e_time

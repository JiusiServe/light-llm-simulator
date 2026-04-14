from abc import ABC
from conf.hardware_config import HWConf
from conf.common import SEC_2_US, US_2_SEC
import logging

class BaseOp(ABC):
    """
    Description:
        Base class for all operations.
        Inherit from this class to create a new operation.
        Operations are the basic building blocks of the model.
    Attributes:
        name: The name of the operation.
        aichip_config: The hardware configuration.
        elem_size: Size of a single element in bytes, 1 for INT8, 2 for FP16.
        cube_flops_int8: actual INT8 matrix-multiply computing power, TFLOPS.
        cube_flops_fp16: actual FP16 matrix-multiply computing power, TFLOPS.
        vec_flops_int8: actual INT8 vector computing power, TFLOPS.
        vec_flops_fp16: actual FP16 vector computing power, TFLOPS.
        cube_flops: actual matrix-multiply computing power, TFLOPS.
        vector_flops: actual vector computing power, TFLOPS.
        inter_node_bandwidth: Die-to-die bandwidth between nodes, GB/s.
        intra_node_bandwidth: Inter-card die-to-die bandwidth within the node, GB/s.
        local_memory_bandwidth: Peak HBM read+write bandwidth, GB/s.
        compute_time: The operator's compute time, seconds.
        memory_time: The operator's memory time, seconds.
        e2e_time: The operator's end-to-end time(max of compute and memory time), seconds.
        total_computation: The total computation of the operator.
        total_data_movement: The total data movement of the operator.
        arithmetic_intensity: The arithmetic intensity of the operator.
    """
    def __init__(
        self,
        name: str,
        aichip_config: HWConf,
        elem_size: int,
        static_cost: float = 10 * US_2_SEC
    ) -> None:
        self.name = name
        self.aichip_config = aichip_config
        self.elem_size = elem_size
        self.static_cost = static_cost
        self.cube_flops_int8 = self.aichip_config.cube_flops_int8 * self.op_compute_disc()
        self.cube_flops_fp16 = self.aichip_config.cube_flops_fp16 * self.op_compute_disc()
        self.vec_flops_int8 = self.aichip_config.vector_flops_int8 * self.op_compute_disc()
        self.vec_flops_fp16 = self.aichip_config.vector_flops_fp16 * self.op_compute_disc()
        self.cube_flops = self.cube_flops_fp16 if elem_size == 2 else self.cube_flops_int8
        self.vector_flops = self.vec_flops_fp16 if elem_size == 2 else self.vec_flops_int8
        self.inter_node_bandwidth = self.aichip_config.inter_node_bandwidth * self.op_memory_disc()
        self.intra_node_bandwidth = self.aichip_config.intra_node_bandwidth * self.op_memory_disc()
        self.local_memory_bandwidth = self.aichip_config.local_memory_bandwidth * self.op_memory_disc()
        self.compute_time: float = 0.0
        self.memory_time: float = 0.0
        self.e2e_time: float = 0.0
        self.total_computation: float = 1
        self.total_data_movement: float = 1
        self.arithmetic_intensity: float = 1

    def __call__(self) -> float:
        self.compute_cost()
        self.memory_cost()
        self.e2e_cost()
        self.get_arithmetic_intensity()
        return self.e2e_time

    def op_compute_disc(self) -> float:
        """
        Description:
            The computing power utilization.
            It is the ratio of the actual computing power to the peak computing power of the hardware.
            It varies with the input shape of the operator.
        Returns:
            The computing power utilization.
            Usually, the computing power utilization is around 0.651.
        """
        return 0.651

    def op_memory_disc(self) -> float:
        """
        Description:
            The memory bandwidth utilization.
            It is the ratio of the transferred bytes per second to the peak memory bandwidth of the hardware.
            It varies with bytes needed to be transferred.
        Returns:
            The memory bandwidth utilization.
            Usually, the memory bandwidth utilization is around 0.85.
        """
        return 0.7

    def compute_cost(self) -> float:
        """
        Description:
            Calculate the compute time of the operator.
        Returns:
            The compute time of the operator.
        """
        return self.compute_time

    def memory_cost(self) -> float:
        """
        Description:
            Calculate the memory time of the operator.
        Returns:
            The memory time of the operator.
        """
        return self.memory_time

    def e2e_cost(self) -> float:
        """
        Description:
            Calculate the end-to-end time of the operator.
            End-to-end time is the max of compute and memory time.
        Returns:
            The end-to-end time of the operator.
        """
        self.e2e_time = max(self.memory_time, self.compute_time) + self.static_cost
        logging.info(
            f"name: {self.name}, "
            f"compute_time: {self.compute_time * SEC_2_US:.2f} us, "
            f"memory_time: {self.memory_time * SEC_2_US:.2f} us, "
            f"e2e_time: {self.e2e_time * SEC_2_US:.2f} us"
        )
        return self.e2e_time

    def get_arithmetic_intensity(self) -> float:
        """
        Description:
            Calculate the arithmetic intensity of the operator.
        Returns:
            The arithmetic intensity of the operator.
        """
        if self.total_data_movement > 0:
            self.arithmetic_intensity = round(self.total_computation / self.total_data_movement, 2)
        else:
            self.arithmetic_intensity = 0.0
        return self.arithmetic_intensity

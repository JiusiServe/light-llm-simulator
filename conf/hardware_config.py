from dataclasses import dataclass
from enum import Enum
from conf.common import GB_2_BYTE, TB_2_BYTE, MB_2_BYTE


class DeviceType(Enum):
    """
    Enumerates all supported Hardwares.
    """
    ASCEND910B2 = "Ascend_910b2"
    ASCEND910B3 = "Ascend_910b3"
    ASCEND910B4 = "Ascend_910b4"
    ASCENDA3_Pod = "Ascend_A3Pod"
    ASCENDDAVID100 = "Ascend_David100"
    ASCENDDAVID120 = "Ascend_David120"
    ASCENDDAVID121 = "Ascend_David121"
    NvidiaA100SXM = "Nvidia_A100_SXM"
    NvidiaH100SXM = "Nvidia_H100_SXM"


@dataclass
class HWConf:
    """
    Hardware capability descriptor for **a single accelerator die**.

    Attributes:
    num_dies_per_node : int
        How many dies are electrically co-located on one node (socket).
    aichip_memory : float
        Per-die HBM capacity, GB.
    intra_node_bandwidth : float
        Inter-card die-to-die bandwidth within the node, GB/s.
    inter_node_bandwidth : float
        Die-to-die bandwidth between nodes, GB/s.
    bwsio_memory_bandwidth : float
        Die-to-die bandwidth on the same card, GB/s.
        (only meaningful for multi-die Ascend NPUs), GB/s.
    local_memory_bandwidth : float
        Peak HBM read+write bandwidth, GB/s.
    onchip_buffer_size : float
        Per-die L2 cache , MB.
    cube_flops_fp16 : float
        FP16 matrix-multiply peak computing power (Cube units), TFLOPS.
    cube_flops_int8 : float
        INT8 matrix-multiply peak computing power, TFLOPS (2× FP16 number).
    vector_flops_fp16 : float
        FP16 vector/element-wise peak computing power (Vector units), TFLOPS.
    vector_flops_int8 : float
        INT8 vector peak computing power, TFLOPS.
    """

    num_dies_per_node: int
    aichip_memory: float
    intra_node_bandwidth: float # within node bandwidth
    inter_node_bandwidth: float # between nodes bandwidth
    local_memory_bandwidth: float # HBM
    bwsio_memory_bandwidth: float # SIO bandwidth btw two dies
    onchip_buffer_size: float
    cube_flops_fp16: float
    cube_flops_int8: float
    vector_flops_fp16: float
    vector_flops_int8: float

    @classmethod
    def create(cls, device_type: DeviceType) -> 'HWConf':
        """
        Factory method: return a HWConf instance pre-filled with
        vendor-published hardware specifications for the requested device type.

        Parameters
        ----------
        device_type : DeviceType
            Enum member identifying the target accelerator.

        Returns
        -------
        HWConf
            Dataclass instance containing numeric hardware constants.

        Raises
        ------
        ValueError
            If the device type is not yet catalogued.
        """
        def cfg(**kwargs):
            return kwargs

        configs = {
            DeviceType.ASCEND910B2: cfg(
                num_dies_per_node=8, aichip_memory=64 * GB_2_BYTE, cube_flops_fp16=353.8 * TB_2_BYTE,
                cube_flops_int8=707.9 * TB_2_BYTE, vector_flops_fp16=22 * TB_2_BYTE,
                vector_flops_int8=44 * TB_2_BYTE, intra_node_bandwidth=196 * GB_2_BYTE,
                inter_node_bandwidth=50 * GB_2_BYTE, local_memory_bandwidth=1.6 * TB_2_BYTE,
                bwsio_memory_bandwidth = 196 * GB_2_BYTE, onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.ASCEND910B3: cfg(
                num_dies_per_node=8, aichip_memory=64 * GB_2_BYTE, cube_flops_fp16=294.9 * TB_2_BYTE,
                cube_flops_int8=589.8 * TB_2_BYTE, vector_flops_fp16=18.4 * TB_2_BYTE,
                vector_flops_int8=36.8 * TB_2_BYTE, intra_node_bandwidth=196 * GB_2_BYTE,
                inter_node_bandwidth=50 * GB_2_BYTE, local_memory_bandwidth=1.6 * TB_2_BYTE,
                bwsio_memory_bandwidth = 196 * GB_2_BYTE, onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.ASCEND910B4: cfg(
                num_dies_per_node=8, aichip_memory=32 * GB_2_BYTE, cube_flops_fp16=245.7 * TB_2_BYTE,
                cube_flops_int8=491.5 * TB_2_BYTE, vector_flops_fp16=15.4 * TB_2_BYTE,
                vector_flops_int8=30.7 * TB_2_BYTE, intra_node_bandwidth=32 * GB_2_BYTE,
                inter_node_bandwidth=32 * GB_2_BYTE, local_memory_bandwidth=0.8 * TB_2_BYTE,
                bwsio_memory_bandwidth = 32 * GB_2_BYTE, onchip_buffer_size=96 * MB_2_BYTE),
            DeviceType.ASCENDA3_Pod: cfg(
                num_dies_per_node=16, aichip_memory=64 * GB_2_BYTE, cube_flops_fp16=353.8 * TB_2_BYTE,
                cube_flops_int8=707.9 * TB_2_BYTE, vector_flops_fp16=22 * TB_2_BYTE,
                vector_flops_int8=44 * TB_2_BYTE, intra_node_bandwidth=196 * GB_2_BYTE,
                inter_node_bandwidth=196 * GB_2_BYTE, local_memory_bandwidth=1.6 * TB_2_BYTE,
                bwsio_memory_bandwidth = 224 * GB_2_BYTE, onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.ASCENDDAVID100: cfg(
                num_dies_per_node=8, aichip_memory=128 * GB_2_BYTE, cube_flops_fp16=486.6 * TB_2_BYTE,
                cube_flops_int8=973.2 * TB_2_BYTE, vector_flops_fp16=60.8 * TB_2_BYTE,
                vector_flops_int8=121.6 * TB_2_BYTE, intra_node_bandwidth=224 * GB_2_BYTE,
                inter_node_bandwidth=224 * GB_2_BYTE, local_memory_bandwidth=1.6 * TB_2_BYTE,
                bwsio_memory_bandwidth = 224 * GB_2_BYTE, onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.ASCENDDAVID120: cfg(
                num_dies_per_node=8, aichip_memory=144 * GB_2_BYTE, cube_flops_fp16=486.6 * TB_2_BYTE,
                cube_flops_int8=973.2 * TB_2_BYTE, vector_flops_fp16=60.8 * TB_2_BYTE,
                vector_flops_int8=121.6 * TB_2_BYTE, intra_node_bandwidth=224 * GB_2_BYTE,
                inter_node_bandwidth=224 * GB_2_BYTE, local_memory_bandwidth=4.2 * TB_2_BYTE,
                bwsio_memory_bandwidth = 224 * GB_2_BYTE, onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.ASCENDDAVID121: cfg(
                num_dies_per_node=8, aichip_memory=192 * GB_2_BYTE, cube_flops_fp16=917.5 * TB_2_BYTE,
                cube_flops_int8=1835 * TB_2_BYTE, vector_flops_fp16=57.3 * TB_2_BYTE,
                vector_flops_int8=114.7 * TB_2_BYTE, intra_node_bandwidth=1008 * GB_2_BYTE,
                inter_node_bandwidth=1008 * GB_2_BYTE, local_memory_bandwidth=8.4 * TB_2_BYTE,
                bwsio_memory_bandwidth = 1008 * GB_2_BYTE, onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.NvidiaA100SXM: cfg(
                num_dies_per_node=8, aichip_memory=80 * GB_2_BYTE, cube_flops_fp16=312 * TB_2_BYTE,
                cube_flops_int8=624 * TB_2_BYTE, vector_flops_fp16=312 * TB_2_BYTE,
                vector_flops_int8=624 * TB_2_BYTE, intra_node_bandwidth=300 * GB_2_BYTE,
                inter_node_bandwidth=25 * GB_2_BYTE, local_memory_bandwidth=4.0 * TB_2_BYTE,
                bwsio_memory_bandwidth = 300 * GB_2_BYTE, onchip_buffer_size=40 * MB_2_BYTE),
            DeviceType.NvidiaH100SXM: cfg(
                num_dies_per_node=8, aichip_memory=80 * GB_2_BYTE, cube_flops_fp16=989 * TB_2_BYTE,
                cube_flops_int8=1978 * TB_2_BYTE, vector_flops_fp16=989 * TB_2_BYTE,
                vector_flops_int8 = 1978 * TB_2_BYTE, intra_node_bandwidth=450 * GB_2_BYTE,
                inter_node_bandwidth=25 * GB_2_BYTE, local_memory_bandwidth=3.35 * TB_2_BYTE,
                bwsio_memory_bandwidth = 450 * GB_2_BYTE, onchip_buffer_size=50 * MB_2_BYTE),
        }

        if device_type not in configs:
            raise ValueError(f"Unsupported AscendType: {device_type}")

        return cls(**configs[device_type])

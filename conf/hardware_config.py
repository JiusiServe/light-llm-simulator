from dataclasses import dataclass
from enum import Enum
from conf.common import GB_2_BYTE, TB_2_BYTE, MB_2_BYTE


class DeviceType(Enum):
    ASCEND910B2_376T_64G = "Ascend_910b2"
    ASCEND910B3_313T_64G = "Ascend_910b3"
    ASCEND910B4 = "Ascend_910b4"
    ASCENDA3_Pod = "Ascend_A3Pod"
    ASCENDDAVID121 = "Ascend_David121"
    ASCENDDAVID120 = "Ascend_David120"


@dataclass
class HWConf:
    aichip_memory: float
    cube_core_cnt: int
    vector_core_cnt: int
    cube_freq: float # GHz
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
        def cfg(**kwargs):
            return kwargs

        configs = {
            DeviceType.ASCEND910B2_376T_64G: cfg(
                aichip_memory=64 * GB_2_BYTE, cube_core_cnt=24, vector_core_cnt=48, cube_freq=1.8,
                intra_node_bandwidth=196 * GB_2_BYTE, inter_node_bandwidth=50 * GB_2_BYTE,
                local_memory_bandwidth=1.6 * TB_2_BYTE, bwsio_memory_bandwidth = 196 * GB_2_BYTE,
                onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.ASCEND910B3_313T_64G: cfg(
                aichip_memory=64 * GB_2_BYTE, cube_core_cnt=20, vector_core_cnt=40, cube_freq=1.8,
                intra_node_bandwidth=196 * GB_2_BYTE, inter_node_bandwidth=50 * GB_2_BYTE,
                local_memory_bandwidth=1.6 * TB_2_BYTE, bwsio_memory_bandwidth = 196 * GB_2_BYTE,
                onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.ASCEND910B4: cfg(
                aichip_memory=32 * GB_2_BYTE, cube_core_cnt=20, vector_core_cnt=40, cube_freq=1.5,
                intra_node_bandwidth=32 * GB_2_BYTE, inter_node_bandwidth=32 * GB_2_BYTE,
                local_memory_bandwidth=0.8 * TB_2_BYTE, bwsio_memory_bandwidth = 32 * GB_2_BYTE,
                onchip_buffer_size=96 * MB_2_BYTE),
            DeviceType.ASCENDA3_Pod: cfg(
                aichip_memory=64 * GB_2_BYTE, cube_core_cnt=24, vector_core_cnt=48, cube_freq=1.8,
                intra_node_bandwidth=196 * GB_2_BYTE, inter_node_bandwidth=50 * GB_2_BYTE,
                local_memory_bandwidth=1.6 * TB_2_BYTE, bwsio_memory_bandwidth = 224 * GB_2_BYTE,
                onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.ASCENDDAVID121: cfg(
                aichip_memory=192 * GB_2_BYTE, cube_core_cnt=56, vector_core_cnt=112, cube_freq=2.0,
                intra_node_bandwidth=1008 * GB_2_BYTE, inter_node_bandwidth=50 * GB_2_BYTE,
                local_memory_bandwidth=8.4 * TB_2_BYTE, bwsio_memory_bandwidth = 1008 * GB_2_BYTE,
                onchip_buffer_size=192 * MB_2_BYTE),
            DeviceType.ASCENDDAVID120: cfg(
                aichip_memory=96 * GB_2_BYTE, cube_core_cnt=32, vector_core_cnt=128, cube_freq=1.7,
                intra_node_bandwidth=350 * GB_2_BYTE, inter_node_bandwidth=50 * GB_2_BYTE,
                local_memory_bandwidth=4.0 * TB_2_BYTE, bwsio_memory_bandwidth = 350 * GB_2_BYTE,
                onchip_buffer_size=192 * MB_2_BYTE),
        }

        if device_type not in configs:
            raise ValueError(f"Unsupported AscendType: {device_type}")

        param = configs[device_type]

        cube_flops_fp16 = param['cube_core_cnt'] * param['cube_freq'] * (16 ** 3) * 2 * GB_2_BYTE
        cube_flops_int8 = 2 * cube_flops_fp16
        vector_flops_fp16 = param['vector_core_cnt'] * param['cube_freq'] * (16 ** 2) * GB_2_BYTE
        vector_flops_int8 = 2 * vector_flops_fp16

        param.update(
            cube_flops_fp16=cube_flops_fp16,
            cube_flops_int8=cube_flops_int8,
            vector_flops_fp16=vector_flops_fp16,
            vector_flops_int8=vector_flops_int8
        )
        return cls(**configs[device_type])


@dataclass
class HardwareTopology:
    number_of_ranks: int
    npus_per_rank: int
    hw_conf: HWConf
    compute_util: float = 1.0
    mem_bw_util: float = 1.0

    @classmethod
    def create(cls, number_of_ranks: int, npus_per_rank: int, device_type: DeviceType) -> 'HardwareTopology':
        hw_conf = HWConf.create(device_type)
        return cls(npus_per_rank=npus_per_rank, number_of_ranks=number_of_ranks, hw_conf=hw_conf)

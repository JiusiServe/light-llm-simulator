# Copyright (c) 2025 Huawei. All rights reserved.

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class DeviceType(Enum):
    ASCEND910B2_376T_64G = "Ascend_910b2"
    ASCEND910B4 = "Ascend_910b4"
    ASCEND910B3_313T_64G = "Ascend_910b3"
    ASCEND910C = "Ascend_910c"
    ASCENDDAVID = "Ascend_David"
    NVidiaA100 = "Nvidia_A100"
    NVidiaH800 = "Nvidia_H800"
    NVidiaH20 = "Nvidia_H20"


@dataclass
class HWConf:
    npu_memory: float
    npu_flops_fp16: float
    npu_flops_int8: float
    vec_flops: float
    intra_node_bandwidth: float # GB/s
    inter_node_bandwidth: float # GB/s
    local_memory_bandwidth: float # GB/s
    onchip_buffer_size: float

    @classmethod
    def create(cls, device_type: DeviceType) -> 'HWConf':
        def cfg(**kwargs):
            return kwargs

        # e9, e12, e3, e6 = 1e9, 1e12, 1e3, 1e6
        #Batya checked. clock speed is 1.65 GHz (in spec. it is 1.8GHz)
        #computing capacity (TF/sec) = clock (cycles/sec) * Ops per cycle * #cores  ==> for exapmle, in 910B4, clock speed is 1.65GHz and cube can perform 16x16x16x2 ops * 20 (#cores) ==> ~ 270TF/sec (13.5 TF per core)
        # the vector unit handles element-wise or reduction operations, such as ReLU, sigmoid, softmax, or vector add.
        # It processes 128 FP16 data elements per cycle, performing operations like addition or multiplication.
        # vector unit capability: 1.65GHz * 128  (ops) * 2 (#units per core) * 20 (#cores) = 8 TF/sec (160 GF per core)
        configs = {
            DeviceType.ASCEND910B4: cfg(
                npu_memory=32e9, npu_flops_fp16=280e12, npu_flops_int8=560e12, vec_flops = 8e12,
                intra_node_bandwidth=30e9, inter_node_bandwidth=12.5e9,
                local_memory_bandwidth=800e9, onchip_buffer_size=96 * 1024*1024),
            DeviceType.ASCEND910B3_313T_64G: cfg(
                npu_memory=64e9, npu_flops_fp16=313e12, npu_flops_int8=626e12,  vec_flops = 8e12,
                intra_node_bandwidth=400e9, inter_node_bandwidth=90e9,
                local_memory_bandwidth=1200e9, onchip_buffer_size=192e6),
            DeviceType.ASCEND910B2_376T_64G: cfg(
                npu_memory=64e9, npu_flops_fp16=376e12, npu_flops_int8=752e12,  vec_flops = 8e12,
                intra_node_bandwidth=400e9, inter_node_bandwidth=90e9,
                local_memory_bandwidth=1200e9, onchip_buffer_size=192e6),
            DeviceType.ASCEND910C: cfg( # each die is 375TF/s
                npu_memory=128e9, npu_flops_fp16=750e12, npu_flops_int8=1500e12,  vec_flops = 23e12,
                intra_node_bandwidth=196e9, inter_node_bandwidth=50e9,
                local_memory_bandwidth=1600e9, onchip_buffer_size=192*1024*1024), #
            DeviceType.ASCENDDAVID: cfg(
                npu_memory=128e9, npu_flops_fp16=486e12, npu_flops_int8=972e12,  vec_flops = 8e12,
                intra_node_bandwidth=1008e9, inter_node_bandwidth=1600e9,
                local_memory_bandwidth=800e9, onchip_buffer_size=65536e3),
            DeviceType.NVidiaA100: cfg(
                npu_memory=40e9, npu_flops_fp16=312e12, npu_flops_int8=624e12,  vec_flops = 312e12,
                intra_node_bandwidth=600e9, inter_node_bandwidth=64e9,
                local_memory_bandwidth=1555e9, onchip_buffer_size=40960e3),
            DeviceType.NVidiaH800: cfg(
                npu_memory=80e9, npu_flops_fp16=1513e12, npu_flops_int8=3026e12,  vec_flops = 1513e12,
                intra_node_bandwidth=600e9, inter_node_bandwidth=128e9,
                local_memory_bandwidth=2000e9, onchip_buffer_size=51200e3),
            DeviceType.NVidiaH20: cfg(
                npu_memory=96e9, npu_flops_fp16=900e12, npu_flops_int8=900e12, vec_flops = 900e12,
                intra_node_bandwidth=900e9, inter_node_bandwidth=128e9,
                local_memory_bandwidth=4000e9, onchip_buffer_size=51200e3),
        }

        if device_type not in configs:
            raise ValueError(f"Unsupported AscendType: {device_type}")
        return cls(**configs[device_type])


@dataclass
class HardwareTopology:
    number_of_ranks: int
    npus_per_rank: int
    hw_conf: HWConf
    compute_util: float = 1.0
    mem_bw_util: float = 1.0

    @classmethod
    def create(cls, number_of_ranks: int, npus_per_rank: int, ascend_type: DeviceType, compute_util: float =1.0, mem_bw_util: float = 1.0) -> 'HardwareTopology':
        hw_conf = HWConf.create(ascend_type)
        hw_conf.npu_flops_fp16 *= compute_util
        hw_conf.npu_flops_int8 *= compute_util
        hw_conf.local_memory_bandwidth *= mem_bw_util
        return cls(npus_per_rank=npus_per_rank, number_of_ranks=number_of_ranks, hw_conf=hw_conf)


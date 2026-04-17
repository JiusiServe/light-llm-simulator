"""Lightweight operator cost smoke tests."""

import pytest

from conf.hardware_config import DeviceType, HWConf
from src.ops.matmul import OpBatchMatmul
from src.ops.scatter_nd_update import OpScatterNdUpdate


@pytest.mark.unit
def test_op_gematmul_e2e_time_positive() -> None:
    hw = HWConf.create(DeviceType.NvidiaH100SXM)
    op = OpBatchMatmul("test_gemm", 128, 256, 512, hw)
    op()
    assert op.compute_time > 0
    assert op.memory_time > 0
    assert op.e2e_time > 0


@pytest.mark.unit
def test_op_scatter_nd_update_positive_times() -> None:
    hw = HWConf.create(DeviceType.NvidiaH100SXM)
    op = OpScatterNdUpdate("scatter_nd_update", input_size=4096, num_updates=128, indices_last_dim=2, aichip_config=hw)
    op()
    assert op.compute_time > 0
    assert op.memory_time > 0
    assert op.e2e_time > 0


@pytest.mark.unit
def test_op_scatter_nd_update_memory_bound() -> None:
    """Scatter ND Update should be memory-bound (memory_time > compute_time)."""
    hw = HWConf.create(DeviceType.NvidiaH100SXM)
    op = OpScatterNdUpdate("scatter_nd_update", input_size=4096, num_updates=128, indices_last_dim=2, aichip_config=hw)
    op()
    assert op.memory_time > op.compute_time

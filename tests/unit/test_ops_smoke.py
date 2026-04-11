"""Lightweight operator cost smoke tests."""

import pytest

from conf.hardware_config import DeviceType, HWConf
from src.ops.matmul import OpBatchMatmul


@pytest.mark.unit
def test_op_gematmul_e2e_time_positive() -> None:
    hw = HWConf.create(DeviceType.NvidiaH100SXM)
    op = OpBatchMatmul("test_gemm", 128, 256, 512, hw)
    op()
    assert op.compute_time > 0
    assert op.memory_time > 0
    assert op.e2e_time > 0

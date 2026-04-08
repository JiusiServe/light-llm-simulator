"""Unit tests for hardware enums and Config construction."""

import pytest

from conf.config import Config
from conf.hardware_config import DeviceType, HWConf
from conf.model_config import ModelType


@pytest.mark.unit
@pytest.mark.parametrize("device_type", list(DeviceType))
def test_hwconf_create_positive(device_type: DeviceType) -> None:
    hw = HWConf.create(device_type)
    assert hw.aichip_memory > 0
    assert hw.local_memory_bandwidth > 0
    assert hw.num_dies_per_node >= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type",
    [
        ModelType.DEEPSEEK_V3,
        ModelType.QWEN3_235B,
        ModelType.DEEPSEEK_V2_LITE,
    ],
)
def test_config_builds_for_each_model(model_type: ModelType) -> None:
    cfg = Config(
        serving_mode="AFD",
        model_type=model_type.value,
        device_type=DeviceType.NvidiaH100SXM.value,
        min_attn_bs=2,
        max_attn_bs=32,
        min_die=16,
        max_die=16,
        die_step=16,
        tpot=200,
        kv_len=2048,
        micro_batch_num=2,
        next_n=1,
        multi_token_ratio=0.7,
        attn_tensor_parallel=1,
        ffn_tensor_parallel=1,
    )
    assert cfg.model_type == model_type
    assert cfg.model_config.hidden_size > 0


@pytest.mark.unit
def test_invalid_device_type_raises() -> None:
    with pytest.raises(ValueError):
        DeviceType("NotARealDevice")


@pytest.mark.unit
def test_invalid_model_type_raises() -> None:
    with pytest.raises(ValueError):
        ModelType("unknown/model-id")


@pytest.mark.unit
@pytest.mark.parametrize(
    "device_type1,device_type2",
    [
        (DeviceType.NvidiaH100SXM, DeviceType.NvidiaA100SXM),
        (DeviceType.ASCENDA3_Pod, DeviceType.ASCEND910B4),
        (DeviceType.ASCENDDAVID121, DeviceType.NvidiaH100SXM),
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        ModelType.DEEPSEEK_V3,
        ModelType.QWEN3_235B,
    ],
)
def test_config_heterogeneous_builds(
    device_type1: DeviceType, device_type2: DeviceType, model_type: ModelType
) -> None:
    """Test heterogeneous deployment mode with different device types for attention and FFN."""
    cfg = Config(
        serving_mode="AFD",
        model_type=model_type.value,
        device_type=device_type1.value,
        device_type2=device_type2.value,
        min_attn_bs=2,
        max_attn_bs=32,
        min_die=16,
        max_die=16,
        die_step=16,
        min_die2=8,
        max_die2=8,
        die_step2=8,
        tpot=200,
        kv_len=2048,
        micro_batch_num=2,
        next_n=1,
        multi_token_ratio=0.7,
        attn_tensor_parallel=1,
        ffn_tensor_parallel=1,
        deployment_mode="Heterogeneous",
    )
    assert cfg.model_type == model_type
    assert cfg.deployment_mode == "Heterogeneous"
    assert cfg.device_type == device_type1
    assert cfg.device_type2 == device_type2
    assert cfg.aichip_config_attn == cfg.aichip_config
    assert cfg.aichip_config_ffn == cfg.aichip_config2
    assert cfg.min_die == 16
    assert cfg.max_die == 16
    assert cfg.min_die2 == 8
    assert cfg.max_die2 == 8
    assert cfg.die_step2 == 8


@pytest.mark.unit
def test_heterogeneous_requires_device_type2() -> None:
    """Test that heterogeneous mode requires device_type2."""
    with pytest.raises(ValueError, match="device_type2 is required"):
        Config(
            serving_mode="AFD",
            model_type=ModelType.DEEPSEEK_V3.value,
            device_type=DeviceType.NvidiaH100SXM.value,
            min_attn_bs=2,
            max_attn_bs=32,
            min_die=16,
            max_die=16,
            die_step=16,
            tpot=200,
            kv_len=2048,
            micro_batch_num=2,
            next_n=1,
            multi_token_ratio=0.7,
            attn_tensor_parallel=1,
            ffn_tensor_parallel=1,
            deployment_mode="Heterogeneous",
        )

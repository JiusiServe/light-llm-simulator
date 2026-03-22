"""Model registration smoke tests."""

import pytest

from conf.config import Config
from conf.hardware_config import DeviceType
from src.model.register import get_model


def _base_config(model_type: str) -> Config:
    return Config(
        serving_mode="AFD",
        model_type=model_type,
        device_type=DeviceType.NvidiaH100SXM.value,
        min_attn_bs=2,
        max_attn_bs=8,
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


@pytest.mark.unit
def test_get_model_deepseek_v3_keys() -> None:
    m = get_model(_base_config("deepseek-ai/DeepSeek-V3"))
    assert set(m.keys()) == {"attn", "mlp", "moe"}


@pytest.mark.unit
def test_get_model_qwen_keys() -> None:
    m = get_model(_base_config("Qwen/Qwen3-235B-A22B"))
    assert set(m.keys()) == {"attn", "moe"}


@pytest.mark.unit
def test_get_model_deepseek_v2_lite_keys() -> None:
    m = get_model(_base_config("deepseek-ai/DeepSeek-V2-Lite"))
    assert set(m.keys()) == {"attn", "mlp", "moe"}

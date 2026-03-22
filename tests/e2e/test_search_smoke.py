"""End-to-end smoke: search writes CSVs under cwd (use tmp_path + chdir)."""

from pathlib import Path

import pandas as pd
import pytest

from conf.config import Config
from conf.hardware_config import DeviceType
from src.search.afd import AfdSearch
from src.search.deepep import DeepEpSearch


def _afd_config() -> Config:
    # Scalar tpot/kv_len/micro_batch_num match CLI loop behavior (see src/cli/main.py).
    return Config(
        serving_mode="AFD",
        model_type="Qwen/Qwen3-235B-A22B",
        device_type=DeviceType.NvidiaH100SXM.value,
        min_attn_bs=2,
        max_attn_bs=64,
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


def _deepep_config() -> Config:
    return Config(
        serving_mode="DeepEP",
        model_type="Qwen/Qwen3-235B-A22B",
        device_type=DeviceType.NvidiaH100SXM.value,
        min_attn_bs=2,
        max_attn_bs=64,
        min_die=16,
        max_die=24,
        die_step=8,
        tpot=200,
        kv_len=2048,
        micro_batch_num=1,
        next_n=1,
        multi_token_ratio=0.7,
        attn_tensor_parallel=1,
        ffn_tensor_parallel=1,
    )


@pytest.mark.e2e
@pytest.mark.build
def test_afd_search_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    AfdSearch(_afd_config()).deployment()
    out_dir = tmp_path / "data" / "afd" / "mbn2"
    assert out_dir.is_dir()
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) >= 1
    df = pd.read_csv(csvs[0])
    assert "throughput(tokens/die/s)" in df.columns
    assert not df["throughput(tokens/die/s)"].isna().all()


@pytest.mark.e2e
@pytest.mark.build
def test_deepep_search_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    DeepEpSearch(_deepep_config()).deployment()
    out_dir = tmp_path / "data" / "deepep"
    assert out_dir.is_dir()
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) == 1
    df = pd.read_csv(csvs[0])
    assert "throughput(tokens/die/s)" in df.columns
    assert len(df) >= 1

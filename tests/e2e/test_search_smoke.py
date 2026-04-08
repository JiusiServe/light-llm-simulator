"""End-to-end smoke: search writes CSVs under cwd (use tmp_path + chdir)."""

from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from conf.config import Config
from conf.hardware_config import DeviceType
from src.search.afd import AfdSearch
from src.search.deepep import DeepEpSearch


def _afd_config(deployment_mode: Literal["Homogeneous", "Heterogeneous"] = "Homogeneous") -> Config:
    """Create AFD config for the specified deployment mode."""
    common_kwargs = dict(
        serving_mode="AFD",
        model_type="Qwen/Qwen3-235B-A22B",
        device_type=DeviceType.NvidiaH100SXM.value,
        min_attn_bs=2,
        max_attn_bs=64,
        tpot=200,
        kv_len=2048,
        micro_batch_num=2,
        next_n=1,
        multi_token_ratio=0.7,
        attn_tensor_parallel=1,
        ffn_tensor_parallel=1,
        deployment_mode=deployment_mode,
    )

    if deployment_mode == "Heterogeneous":
        common_kwargs.update(
            device_type2=DeviceType.NvidiaA100SXM.value,
            min_die=8,
            max_die=16,
            die_step=8,
            min_die2=8,
            max_die2=16,
            die_step2=8,
        )
    else:
        common_kwargs.update(
            min_die=16,
            max_die=16,
            die_step=16,
        )

    return Config(**common_kwargs)


def _deepep_config(deployment_mode: Literal["Homogeneous", "Heterogeneous"] = "Homogeneous") -> Config:
    """Create DeepEP config for the specified deployment mode."""
    common_kwargs = dict(
        serving_mode="DeepEP",
        model_type="Qwen/Qwen3-235B-A22B",
        device_type=DeviceType.NvidiaH100SXM.value,
        min_attn_bs=2,
        max_attn_bs=64,
        tpot=200,
        kv_len=2048,
        micro_batch_num=1,
        next_n=1,
        multi_token_ratio=0.7,
        attn_tensor_parallel=1,
        ffn_tensor_parallel=1,
        deployment_mode=deployment_mode,
    )

    if deployment_mode == "Heterogeneous":
        common_kwargs.update(
            device_type2=DeviceType.NvidiaA100SXM.value,
            min_die=16,
            max_die=24,
            die_step=8,
            min_die2=8,
            max_die2=16,
            die_step2=8,
        )
    else:
        common_kwargs.update(
            min_die=16,
            max_die=24,
            die_step=8,
        )

    return Config(**common_kwargs)


@pytest.mark.e2e
@pytest.mark.build
@pytest.mark.parametrize("deployment_mode", ["Homogeneous", "Heterogeneous"])
def test_afd_search_writes_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deployment_mode: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _afd_config(deployment_mode)  # type: ignore
    AfdSearch(config).deployment()
    out_dir = tmp_path / "data" / "afd" / "mbn2" / deployment_mode.lower()
    assert out_dir.is_dir()
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) >= 1
    df = pd.read_csv(csvs[0])
    assert "throughput(tokens/die/s)" in df.columns
    assert not df["throughput(tokens/die/s)"].isna().all()
    # Verify deployment mode column
    assert "deployment_mode" in df.columns
    assert (df["deployment_mode"] == deployment_mode).all()


@pytest.mark.e2e
@pytest.mark.build
@pytest.mark.parametrize("deployment_mode", ["Homogeneous", "Heterogeneous"])
def test_deepep_search_writes_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deployment_mode: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _deepep_config(deployment_mode)  # type: ignore
    DeepEpSearch(config).deployment()
    out_dir = tmp_path / "data" / "deepep" / deployment_mode.lower()
    assert out_dir.is_dir()
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) == 1
    df = pd.read_csv(csvs[0])

    # Check throughput column (different for heterogeneous)
    if deployment_mode == "Heterogeneous":
        assert "weighted_throughput(tokens/die/s)" in df.columns
    else:
        assert "throughput(tokens/die/s)" in df.columns

    assert len(df) >= 1
    # Verify deployment mode column
    assert "deployment_mode" in df.columns
    assert (df["deployment_mode"] == deployment_mode).all()

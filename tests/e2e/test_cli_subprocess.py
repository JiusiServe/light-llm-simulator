"""CLI subprocess smoke (minimal grid, no post-search plots)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.build
def test_cli_deepep_minimal_grid(repo_root: Path, tmp_path: Path) -> None:
    env = {
        **os.environ,
        "LIGHT_LLM_SKIP_POST_PLOTS": "1",
        "PYTHONPATH": str(repo_root),
    }
    cmd = [
        sys.executable,
        str(repo_root / "src" / "cli" / "main.py"),
        "--serving_mode",
        "DeepEP",
        "--model_type",
        "Qwen/Qwen3-235B-A22B",
        "--device_type",
        "Nvidia_H100_SXM",
        "--min_die",
        "16",
        "--max_die",
        "24",
        "--die_step",
        "8",
        "--tpot",
        "200",
        "--kv_len",
        "2048",
        "--min_attn_bs",
        "2",
        "--max_attn_bs",
        "64",
    ]
    subprocess.run(cmd, cwd=tmp_path, env=env, check=True)
    deepep = tmp_path / "data" / "deepep"
    assert deepep.is_dir()
    assert list(deepep.glob("*.csv"))


@pytest.mark.unit
def test_cli_help_exits_zero(repo_root: Path) -> None:
    subprocess.run(
        [sys.executable, str(repo_root / "src" / "cli" / "main.py"), "-h"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

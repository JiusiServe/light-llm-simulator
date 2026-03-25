"""Smoke tests for the FastAPI webapp backend."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import webapp.backend.main as web_main

pytestmark = [pytest.mark.e2e, pytest.mark.build]


def api_request(method: str, path: str, **kwargs: object) -> httpx.Response:
    """Issue a request against the ASGI app without a live server."""

    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=web_main.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(_run())


@pytest.fixture
def fake_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect repo-root-relative data lookups into a temp tree."""

    fake_main = tmp_path / "webapp" / "backend" / "main.py"
    fake_main.parent.mkdir(parents=True, exist_ok=True)
    fake_main.write_text("# test path anchor\n")
    monkeypatch.setattr(web_main, "__file__", str(fake_main))
    return tmp_path


@pytest.fixture
def isolated_run_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect run output into a temp directory."""

    run_root = tmp_path / "runs"
    monkeypatch.setattr(web_main, "RUN_ROOT", run_root)
    return run_root


def test_root_serves_index_html() -> None:
    response = api_request("GET", "/")

    assert response.status_code == 200
    assert 'id="app"' in response.text
    assert "/static/app.js" in response.text


def test_constants_endpoint_returns_expected_values() -> None:
    response = api_request("GET", "/api/constants")

    assert response.status_code == 200
    assert response.json() == {
        "GB_2_BYTE": 1073741824,
        "TB_2_BYTE": 1099511627776,
        "MB_2_BYTE": 1048576,
    }


def test_model_config_endpoint_success_and_invalid() -> None:
    success = api_request(
        "GET",
        "/api/model_config",
        params={"model_type": "deepseek-ai/DeepSeek-V3"},
    )

    assert success.status_code == 200
    body = success.json()
    assert "hidden_size" in body
    assert body["hidden_size"] > 0

    failure = api_request(
        "GET",
        "/api/model_config",
        params={"model_type": "not-real"},
    )

    assert failure.status_code == 400
    assert "error" in failure.json()


def test_hardware_config_endpoint_success_and_invalid() -> None:
    success = api_request(
        "GET",
        "/api/hardware_config",
        params={"device_type": "Ascend_910b2"},
    )

    assert success.status_code == 200
    body = success.json()
    assert "aichip_memory" in body
    assert body["aichip_memory"] > 0

    failure = api_request(
        "GET",
        "/api/hardware_config",
        params={"device_type": "not-real"},
    )

    assert failure.status_code == 400
    assert "error" in failure.json()


def test_run_status_and_logs_lifecycle_with_mocked_background_task(
    isolated_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, list[str]]] = []

    def fake_run_process(run_dir: Path, args: list[str]) -> None:
        calls.append((run_dir, args))
        (run_dir / "output.log").write_text("simulated log\n", encoding="utf-8")
        (run_dir / ".done").write_text("done", encoding="utf-8")

    monkeypatch.setattr(web_main, "_run_process", fake_run_process)
    monkeypatch.setattr(
        web_main,
        "uuid4",
        lambda: SimpleNamespace(hex="deadbeefcafebabe"),
    )

    payload = {
        "serving_mode": "AFD",
        "model_type": "deepseek-ai/DeepSeek-V3",
        "device_type": "Ascend_910b2",
        "min_attn_bs": 4,
        "max_attn_bs": 64,
        "min_die": 16,
        "max_die": 128,
        "die_step": None,
        "tpot": [50, 100],
        "kv_len": [4096, 8192],
        "micro_batch_num": [2, 3],
        "next_n": 2,
        "multi_token_ratio": 0.8,
        "attn_tensor_parallel": 2,
        "ffn_tensor_parallel": 4,
    }
    response = api_request("POST", "/api/run", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "deadbeef"
    assert body["status"] == "started"
    assert body["log"] == str(isolated_run_root / "deadbeef" / "output.log")

    assert len(calls) == 1
    run_dir, args = calls[0]
    assert run_dir == isolated_run_root / "deadbeef"
    assert args[:2] == ["python", web_main.SRC_CLI]
    assert "--die_step" not in args
    assert args[args.index("--serving_mode") + 1] == "AFD"
    assert args[args.index("--model_type") + 1] == "deepseek-ai/DeepSeek-V3"
    assert args[args.index("--device_type") + 1] == "Ascend_910b2"
    assert args[args.index("--attn_tensor_parallel") + 1] == "2"
    assert args[args.index("--ffn_tensor_parallel") + 1] == "4"
    assert args[args.index("--tpot") + 1:args.index("--kv_len")] == ["50", "100"]
    assert args[args.index("--kv_len") + 1:args.index("--micro_batch_num")] == [
        "4096",
        "8192",
    ]
    assert args[args.index("--micro_batch_num") + 1:] == ["2", "3"]

    status_response = api_request("GET", f"/api/status/{body['run_id']}")
    assert status_response.status_code == 200
    assert status_response.json()["done"] is True

    logs_response = api_request("GET", f"/api/logs/{body['run_id']}")
    assert logs_response.status_code == 200
    assert logs_response.json() == {"log": "simulated log\n"}


def test_status_and_logs_404_for_missing_run(
    isolated_run_root: Path,
) -> None:
    assert isolated_run_root.exists() is False

    status_response = api_request("GET", "/api/status/missing123")
    assert status_response.status_code == 404
    assert status_response.json()["detail"] == "run_id not found"

    logs_response = api_request("GET", "/api/logs/missing123")
    assert logs_response.status_code == 404
    assert logs_response.json()["detail"] == "log not found"


@pytest.mark.parametrize("deployment_mode", ["Homogeneous", "Heterogeneous"])
def test_results_endpoint_returns_only_existing_image_urls(
    fake_repo_root: Path,
    deployment_mode: str,
) -> None:
    throughput_dir = fake_repo_root / "data" / "images" / "throughput" / deployment_mode.lower()
    pipeline_dir = fake_repo_root / "data" / "images" / "pipeline" / "afd" / deployment_mode.lower()
    throughput_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Build file names based on deployment mode
    if deployment_mode == "Heterogeneous":
        throughput_file = throughput_dir / (
            "Ascend_910b2_Nvidia_A100_SXM-deepseek-ai/DeepSeek-V3-mbn2-total_die128.png"
        )
        pipeline_file = pipeline_dir / (
            "Ascend_910b2_Nvidia_A100_SXM-deepseek-ai/DeepSeek-V3-tpot50-kv_len4096-afd-mbn3-total_die128.png"
        )
        expected_throughput = [
            "/data/images/throughput/heterogeneous/Ascend_910b2_Nvidia_A100_SXM-deepseek-ai/DeepSeek-V3-mbn2-total_die128.png"
        ]
        expected_pipeline = [
            "/data/images/pipeline/afd/heterogeneous/Ascend_910b2_Nvidia_A100_SXM-deepseek-ai/DeepSeek-V3-tpot50-kv_len4096-afd-mbn3-total_die128.png"
        ]
    else:
        throughput_file = throughput_dir / (
            "Ascend_910b2-deepseek-ai/DeepSeek-V3-mbn2-total_die128.png"
        )
        pipeline_file = pipeline_dir / (
            "Ascend_910b2-deepseek-ai/DeepSeek-V3-tpot50-kv_len4096-afd-mbn3-total_die128.png"
        )
        expected_throughput = [
            "/data/images/throughput/homogeneous/Ascend_910b2-deepseek-ai/DeepSeek-V3-mbn2-total_die128.png"
        ]
        expected_pipeline = [
            "/data/images/pipeline/afd/homogeneous/Ascend_910b2-deepseek-ai/DeepSeek-V3-tpot50-kv_len4096-afd-mbn3-total_die128.png"
        ]

    throughput_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)
    throughput_file.write_text("png", encoding="utf-8")
    pipeline_file.write_text("png", encoding="utf-8")

    params = {
        "model_type": "deepseek-ai/DeepSeek-V3",
        "device_type": "Ascend_910b2",
        "total_die": 128,
        "tpot": 50,
        "kv_len": 4096,
        "deployment_mode": deployment_mode,
    }
    if deployment_mode == "Heterogeneous":
        params["device_type2"] = "Nvidia_A100_SXM"

    response = api_request("GET", "/api/results", params=params)

    assert response.status_code == 200
    assert response.json() == {
        "throughput_images": expected_throughput,
        "pipeline_images": expected_pipeline,
    }


@pytest.mark.parametrize("deployment_mode", ["Homogeneous", "Heterogeneous"])
def test_fetch_csv_results_returns_filtered_rows_for_afd(
    fake_repo_root: Path,
    deployment_mode: str,
) -> None:
    # Build path based on deployment mode
    if deployment_mode == "Heterogeneous":
        csv_path = (
            fake_repo_root
            / "data"
            / "afd"
            / "mbn3"
            / "best"
            / "heterogeneous"
            / "Ascend_910b2_Nvidia_A100_SXM-deepseek-ai/DeepSeek-V3-tpot50-kv_len4096.csv"
        )
        device_type2 = "Nvidia_A100_SXM"
    else:
        csv_path = (
            fake_repo_root
            / "data"
            / "afd"
            / "mbn3"
            / "best"
            / "homogeneous"
            / "Ascend_910b2-deepseek-ai/DeepSeek-V3-tpot50-kv_len4096.csv"
        )
        device_type2 = None

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "\n".join(
            [
                "total_die,throughput(tokens/die/s),note",
                "64,12.5,small",
                "128,18.75,target",
            ]
        ),
        encoding="utf-8",
    )

    params = {
        "device_type": "Ascend_910b2",
        "model_type": "deepseek-ai/DeepSeek-V3",
        "tpot": 50,
        "kv_len": 4096,
        "serving_mode": "AFD",
        "micro_batch_num": 3,
        "total_die": 128,
        "deployment_mode": deployment_mode,
    }
    if device_type2:
        params["device_type2"] = device_type2

    response = api_request("GET", "/api/fetch_csv_results", params=params)

    assert response.status_code == 200
    assert response.json() == [
        {
            "total_die": 128,
            "throughput(tokens/die/s)": 18.75,
            "note": "target",
        }
    ]


@pytest.mark.parametrize("deployment_mode", ["Homogeneous", "Heterogeneous"])
def test_fetch_csv_results_returns_rows_for_deepep(
    fake_repo_root: Path,
    deployment_mode: str,
) -> None:
    # Build path based on deployment mode
    if deployment_mode == "Heterogeneous":
        csv_path = (
            fake_repo_root
            / "data"
            / "deepep"
            / "heterogeneous"
            / "Ascend_910b2_Nvidia_A100_SXM-deepseek-ai/DeepSeek-V3-tpot50-kv_len4096.csv"
        )
        device_type2 = "Nvidia_A100_SXM"
    else:
        csv_path = (
            fake_repo_root
            / "data"
            / "deepep"
            / "homogeneous"
            / "Ascend_910b2-deepseek-ai/DeepSeek-V3-tpot50-kv_len4096.csv"
        )
        device_type2 = None

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "\n".join(
            [
                "total_die,throughput(tokens/die/s)",
                "128,22.5",
                "256,19.0",
            ]
        ),
        encoding="utf-8",
    )

    params = {
        "device_type": "Ascend_910b2",
        "model_type": "deepseek-ai/DeepSeek-V3",
        "tpot": 50,
        "kv_len": 4096,
        "serving_mode": "DeepEP",
        "deployment_mode": deployment_mode,
    }
    if device_type2:
        params["device_type2"] = device_type2

    response = api_request("GET", "/api/fetch_csv_results", params=params)

    assert response.status_code == 200
    assert response.json() == [
        {"total_die": 128, "throughput(tokens/die/s)": 22.5},
        {"total_die": 256, "throughput(tokens/die/s)": 19.0},
    ]


def test_fetch_csv_results_errors_for_invalid_mode_and_missing_file(
    fake_repo_root: Path,
) -> None:
    invalid_mode = api_request(
        "GET",
        "/api/fetch_csv_results",
        params={
            "device_type": "Ascend_910b2",
            "model_type": "deepseek-ai/DeepSeek-V3",
            "tpot": 50,
            "kv_len": 4096,
            "serving_mode": "NotReal",
        },
    )
    assert invalid_mode.status_code == 400
    assert invalid_mode.json()["detail"] == "Unsupported serving_mode"

    missing_file = api_request(
        "GET",
        "/api/fetch_csv_results",
        params={
            "device_type": "Ascend_910b2",
            "model_type": "deepseek-ai/DeepSeek-V3",
            "tpot": 50,
            "kv_len": 4096,
            "serving_mode": "AFD",
            "micro_batch_num": 3,
        },
    )
    assert missing_file.status_code == 404
    assert "file not found:" in missing_file.json()["detail"]

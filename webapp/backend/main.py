from fastapi import FastAPI, BackgroundTasks, HTTPException, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
import pandas as pd
import subprocess
from pathlib import Path
from typing import Optional, List, Any, Dict
from conf.model_config import ModelConfig, ModelType
from conf.hardware_config import HWConf, DeviceType
from conf.common import GB_2_BYTE, TB_2_BYTE, MB_2_BYTE

RUN_ROOT = Path("webapp/runs")
SRC_CLI = Path("src/cli/main.py").as_posix()  # repository CLI entry

app = FastAPI(title="light-llm-simulator Web API")
api = APIRouter(prefix="/api", tags=["api"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount frontend and data directories
app.mount("/static", StaticFiles(directory="webapp/frontend"), name="static")
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory="data"), name="data")


class RunRequest(BaseModel):
    serving_mode: str = "AFD"
    model_type: str = "deepseek-ai/DeepSeek-V3"
    device_type: str = "Ascend_A3Pod"
    min_attn_bs: int = 2
    max_attn_bs: int = 1000
    min_die: int = 16
    max_die: int = 768
    die_step: Optional[int] = 16
    tpot: Optional[List[int]] = [50]
    kv_len: Optional[List[int]] = [4096]
    micro_batch_num: Optional[List[int]] = [2]
    next_n: int = 1
    multi_token_ratio: float = 0.7
    attn_tensor_parallel: int = 1
    ffn_tensor_parallel: int = 1


def _sanitize_args(req: RunRequest) -> List[str]:
    args = [
        "python", SRC_CLI,
        "--serving_mode", req.serving_mode,
        "--model_type", req.model_type,
        "--device_type", req.device_type,
        "--min_attn_bs", str(req.min_attn_bs),
        "--max_attn_bs", str(req.max_attn_bs),
        "--min_die", str(req.min_die),
        "--max_die", str(req.max_die),
        "--next_n", str(req.next_n),
        "--multi_token_ratio", str(req.multi_token_ratio),
        "--attn_tensor_parallel", str(req.attn_tensor_parallel),
        "--ffn_tensor_parallel", str(req.ffn_tensor_parallel),
    ]
    if req.die_step is not None:
        args += ["--die_step", str(req.die_step)]
    if req.tpot:
        args += ["--tpot"] + [str(x) for x in req.tpot]
    if req.kv_len:
        args += ["--kv_len"] + [str(x) for x in req.kv_len]
    if req.micro_batch_num:
        args += ["--micro_batch_num"] + [str(x) for x in req.micro_batch_num]
    return args


def _run_process(run_dir: Path, args: List[str]):
    run_log = run_dir / "output.log"
    with open(run_log, "wb") as fout:
        proc = subprocess.Popen(args, stdout=fout, stderr=subprocess.STDOUT, cwd=".")
        proc.wait()
    # mark complete
    (run_dir / ".done").write_text("done")


@api.post("/run")
def start_run(req: RunRequest, background_tasks: BackgroundTasks):
    run_id = uuid4().hex[:8]
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    args = _sanitize_args(req)
    background_tasks.add_task(_run_process, run_dir, args)
    return {"run_id": run_id, "status": "started", "log": str(run_dir / "output.log")}


@api.get("/status/{run_id}")
def status(run_id: str):
    run_dir = RUN_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run_id not found")
    done_file = run_dir / ".done"
    log_path = run_dir / "output.log"
    return {"run_id": run_id, "exists": True, "done": done_file.exists(), "log_path": str(log_path)}


@api.get("/logs/{run_id}")
def get_logs(run_id: str):
    run_dir = RUN_ROOT / run_id
    log = run_dir / "output.log"
    if not log.exists():
        raise HTTPException(status_code=404, detail="log not found")
    return {"log": log.read_text(errors="ignore")}


@api.get("/results")
def list_results(model_type: str, device_type: str, total_die: int, tpot: int, kv_len: int):
    image1 = f"/data/images/throughput/{device_type}-{model_type}-mbn2-total_die{total_die}.png"
    image2 = f"/data/images/throughput/{device_type}-{model_type}-mbn3-total_die{total_die}.png"
    image3 = f"/data/images/throughput/{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}.png"
    image4 = f"/data/images/pipeline/mbn2/{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}-total_die{total_die}.png"
    image5 = f"/data/images/pipeline/mbn3/{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}-total_die{total_die}.png"
    image6 = f"/data/images/pipeline/deepep/{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}-total_die{total_die}.png"
    throughput_images = [image1, image2, image3]
    pipeline_images = [image4, image5, image6]
    print(throughput_images)
    print(pipeline_images)
    return {"throughput_images": throughput_images, "pipeline_images": pipeline_images}

# model config endpoint: accepts model_type as string
@api.get("/model_config")
def get_model_config(model_type: str):
    # try to map provided string to ModelType enum
    mt = None
    for m in ModelType:
        if model_type == m.value or model_type == m.name or model_type.lower() in str(m.value).lower():
            mt = m
            break
    if mt is None:
        # try direct constructor (may raise)
        try:
            mt = ModelType(model_type)
        except Exception:
            return JSONResponse({"error": f"unknown model_type {model_type}"}, status_code=400)
    mc = ModelConfig.create_model_config(mt)
    # select common fields (fallback to inspect)
    keys = [
        "hidden_size", "num_layers", "max_kv_length", "num_heads",
        "kv_heads", "vocab_size", "head_size", "model_size_b",
        "intermediate_size", "kv_lora_rank", "max_position_embeddings",
        "moe_intermediate_size", "n_routed_experts", "n_shared_experts",
        "num_attention_heads", "num_experts_per_tok", "num_key_value_heads",
        "num_moe_layers", "first_k_dense_replace", "q_lora_rank",
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = getattr(mc, k, None)
    return out


@api.get("/hardware_config")
def get_hardware_config(device_type: str):
    # map device_type string to DeviceType enum
    dt = None
    for d in DeviceType:
        if device_type == d.value or device_type == d.name or device_type.lower() in d.value.lower():
            dt = d
            break
    if dt is None:
        try:
            dt = DeviceType(device_type)
        except Exception:
            return JSONResponse({"error": f"unknown device_type {device_type}"}, status_code=400)
    hw = HWConf.create(dt)
    return hw


@api.get("/constants")
def get_constants():
    return {"GB_2_BYTE": GB_2_BYTE, "TB_2_BYTE": TB_2_BYTE, "MB_2_BYTE": MB_2_BYTE}


@api.get("/fetch_csv_results")
def fetch_csv_results(
    device_type: str,
    model_type: str,
    tpot: int,
    kv_len: int,
    serving_mode: str = "AFD",
    micro_batch_num: Optional[int] = 1,
    total_die: Optional[int] = None,
):
    repo_root = Path(__file__).resolve().parents[2]
    if serving_mode == "AFD":
        dir_name = repo_root / "data" / "afd" / f"mbn{micro_batch_num}" / "best"
    elif serving_mode == "DeepEP":
        dir_name = repo_root / "data" / "deepep"
    else:
        raise HTTPException(status_code=400, detail="Unsupported serving_mode")
    file_name = f"{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}.csv"
    path = dir_name / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"read csv error: {e}")
    if total_die is not None:
        # allow both numeric and string comparisons
        df = df[df["total_die"].astype(str) == str(total_die)]
    return df.to_dict(orient="records")


# include router and root index
app.include_router(api)


@app.get("/")
def index():
    index_path = Path("webapp/frontend/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return {"msg": "Frontend not found. API docs at /api/docs"}


if __name__ == "__main__":
    import uvicorn
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    uvicorn.run("webapp.backend.main:app", host="127.0.0.1", port=8000, reload=True)
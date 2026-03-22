# Light LLM Simulator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

In large-model inference serving, choosing a good deployment is difficult. For Attention–FFN disaggregated (AFD) serving you must jointly pick attention and FFN worker counts, micro-batch sizes, and parallelism while meeting SLA targets on **TTFT** (time to first token) and **TPOT** (time per output token). **Light LLM Simulator** is an open-source, chip-agnostic performance explorer: it screens many deployment combinations and surfaces configurations that improve throughput within your latency budget.

Give it a **model**, **accelerator profile**, and **cluster scale** (searched over a die-count range). It runs **AFD** or **DeepEP** searches, writes CSV results under `data/`, and can drive throughput plots. A **Web UI** (Vue 3 + FastAPI) supports runs, config inspection, result tables, and charts.

This README follows the same onboarding shape as projects like [ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator): install → CLI (and web app) → how it works → support matrix → docs → limitations.

Let's get started.

## Build and install

### From source

```bash
git clone https://github.com/JiusiServe/light-llm-simulator.git
cd light-llm-simulator

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python 3.8+, `pandas`, `matplotlib`, `numpy`, `fastapi`, and `uvicorn` (see [`requirements.txt`](requirements.txt)). There is no PyPI package yet; install from a clone of this repository.

## Run

### CLI

From the repository root:

```bash
# Defaults: AFD, DeepSeek-V3, Ascend A3Pod; sweeps TPOT, KV length, micro-batch
python src/cli/main.py
```

**DeepEP, Qwen3-235B, NVIDIA H100**

```bash
python src/cli/main.py \
  --serving_mode DeepEP \
  --model_type "Qwen/Qwen3-235B-A22B" \
  --device_type "Nvidia_H100_SXM" \
  --min_die 16 --max_die 128 --die_step 16 \
  --tpot 50 100 \
  --kv_len 4096 8192
```

**AFD with a tighter sweep**

```bash
python src/cli/main.py \
  --serving_mode AFD \
  --model_type "deepseek-ai/DeepSeek-V3" \
  --device_type "Ascend_A3Pod" \
  --min_attn_bs 4 --max_attn_bs 256 \
  --min_die 32 --max_die 256 --die_step 32 \
  --tpot 70 \
  --kv_len 8192 \
  --micro_batch_num 2 3
```

CLI notes:

- **`--serving_mode`**: `AFD` or `DeepEP`.
- **`--model_type`**: values from `ModelType` in [`conf/model_config.py`](conf/model_config.py), e.g. `deepseek-ai/DeepSeek-V3`, `Qwen/Qwen3-235B-A22B`, `deepseek-ai/DeepSeek-V2-Lite`.
- **`--device_type`**: `DeviceType` string from [`conf/hardware_config.py`](conf/hardware_config.py) (see table below).
- **`--tpot`**, **`--kv_len`**, **`--micro_batch_num`**: list arguments; the search runs over the configured grid (DeepEP fixes micro-batch internally to `1` in code paths).
- **`--next_n`**, **`--multi_token_ratio`**: MTP-related knobs.
- **`--attn_tensor_parallel`**, **`--ffn_tensor_parallel`**: tensor-parallel widths.
- **`python src/cli/main.py -h`** for defaults and the full argument list.

The CLI entrypoint also invokes [`src/visualization/throughput.py`](src/visualization/throughput.py) after search; you can run that script alone for custom plots ([Visualization](docs/visualization/visualization.md)).

### Examples

Runnable scripts: [`examples/deepseek/`](examples/deepseek/), [`examples/qwen235B/`](examples/qwen235B/) (`afd.py`, `deepep.py`, and `run_*.sh` helpers).

## Webapp

```bash
pip install -r requirements.txt
LOG_LEVEL=INFO uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Backend: [`webapp/backend/main.py`](webapp/backend/main.py); frontend: [`webapp/frontend/`](webapp/frontend/).

- **Run experiment**: submit a simulation; live log / progress polling.
- **Configuration**: model and hardware for the selected run.
- **Results**: load a CSV from `data/afd/` or `data/deepep/`, filter and sort.
- **Visualizations**: throughput and pipeline images from `/data/images/`.

`LOG_LEVEL=INFO` helps the Run tab infer phases from logs. `/api/results` only returns image URLs that exist on disk to avoid unnecessary 404s.

## How it works

1. **Decompose** decode into operators (matmul, attention, communication, rotary, SwiGLU, etc.) with per-hardware cost.
2. **Parameterize** deployment (AFD or DeepEP): dies, batching, parallelism, sequence length, TPOT targets, optional MTP.
3. **Search** the grid and write metrics to CSV under `data/`.
4. **Visualize** throughput and pipeline behavior where implemented.

Details: [AFD](docs/search/AFD.md), [DeepEP](docs/search/DeepEP.md).

## Supported features

- **Serving modes**: **AFD** and **DeepEP** supported; **PD** (prefill/decode disaggregated) not yet implemented.
- **Models**: **DeepSeek V3** (MLA, MoE), **Qwen3-235B-A22B** (GQA, MoE); **DeepSeek-V2-Lite** is also registered in code.
- **Operators**: see [Supported operators](docs/ops/supported_ops.md).
- **Hardware**: eight accelerator profiles (table below).

### Hardware support matrix

| `--device_type` | Vendor | Status |
|-----------------|--------|--------|
| `Ascend_910b2` | Ascend | Supported |
| `Ascend_910b3` | Ascend | Supported |
| `Ascend_910b4` | Ascend | Supported |
| `Ascend_A3Pod` | Ascend | Supported |
| `Ascend_David121` | Ascend | Supported |
| `Ascend_David120` | Ascend | Supported |
| `Nvidia_A100_SXM` | NVIDIA | Supported |
| `Nvidia_H100_SXM` | NVIDIA | Supported |

## Documentation

| Topic | Link |
|--------|------|
| Quick start | [docs/quickstart.md](docs/quickstart.md) |
| Configuration | [docs/conf/configuration.md](docs/conf/configuration.md) |
| AFD | [docs/search/AFD.md](docs/search/AFD.md) |
| DeepEP | [docs/search/DeepEP.md](docs/search/DeepEP.md) |
| Operators | [docs/ops/supported_ops.md](docs/ops/supported_ops.md) |
| Models | [docs/model/supported_models.md](docs/model/supported_models.md) |
| Visualization | [docs/visualization/visualization.md](docs/visualization/visualization.md) |

## Repository layout

```
light-llm-simulator/
├── conf/           # Model, hardware, run configuration
├── data/           # CSVs, logs, images from runs
├── docs/
├── examples/       # deepseek/, qwen235B/
├── src/
│   ├── cli/main.py
│   ├── model/      # Decode modules
│   ├── ops/        # Operator cost models
│   ├── search/     # AFD, DeepEP
│   └── visualization/
└── webapp/         # FastAPI + Vue
```

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
MPLBACKEND=agg python -m pytest -m "unit or build"
```

See [`tests/README.md`](tests/README.md) for markers and commands.

## Contributing

Contributions are welcome via pull request.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Known limitations

- Outputs are **simulated** from analytic models; validate on real hardware and stacks before production use.
- **PD** serving is not modeled yet.
- Estimates can diverge from reality where memory, kernels, or networking are not fully captured; use results as strong starting points, not guarantees.

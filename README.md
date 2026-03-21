# Light LLM Simulator

> light-llm-simulator is an open-source, chip-agnostic performance explorer for large-model inference serving.It quickly screens thousands of deployment combinations to find the ones that maximize throughput while keeping TTFT and TPOT within your SLA.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## Overview

In large-model inference serving, finding an efficient deployment is far from trivial. For example, in AFD serving, you must jointly choose the number of Attention and FFN workers, the micro-batch size, and still meet strict SLA targets on TTFT and TPOT. **Light LLM Simulator** automates this search.

Tell it your model, chip type, and cluster size, and it returns a near-optimal configuration that maximizes throughput while respecting your SLA budget.

## Features

- 🎯 **AFD Search**: Attention-FFN Disaggregated deployment optimization
- 📊 **DeepEP Baseline**: DeepEP deployment optimization
- 📈 **Visualization**: Pareto frontier plots, pipeline analysis and throughput changes
- 🚀 **Multi-Token Prediction (MTP)**: Support for multi-token generation
- 🖥️ **Web UI**: Vue 3 + FastAPI browser UI for runs, configuration, results, and charts
- 🎨 **Extensible Architecture**: Easy to add new models, operators, or search strategies

## Supported Serving Mode
- ✅ **DeepEP**: Fully supported
- ✅ **AFD**: Fully supported
- ❌ **PD**: TODO

## Supported Models

- ✅ **DeepSeek V3**: Fully supported with MLA attention and MoE
- ✅ **Qwen3-235B-A22B**: Fully supported with GQA attention and MoE

## Supported Hardware

- **Ascend**: 910B2, 910B3, 910B4, A3Pod, David121, David120
- **Nvidia**: A100SXM, H100SXM

## Project Structure

```
light-llm-simulator/
├── data/              # Generated CSVs and visualization images
├── conf/              # Configuration files
│   ├── common.py            # Common constants
│   ├── config.py            # CLI configurations
│   ├── hardware_config.py   # Hardware specifications
│   └── model_config.py      # Model specifications
├── docs/              # Documentation
├── examples/    # runnable examples
│   ├── deepseek/    # DeepSeekV3-671B example
│   │   ├── afd.py    # Python example that runs AFD
│   │   ├── deepep.py    # Python example that runs DeepEP
│   │   ├── run_afd.sh    # Convenience shell script to run the AFD example
│   │   └──  run_deepep.sh    # Convenience shell script to run the DeepEP example
│   ├── qwen235B/      # Qwen3-235B-A22B example
│   │   ├── afd.py    # Python example that runs AFD
│   │   ├── deepep.py    # Python example that runs DeepEP
│   │   ├── run_afd.sh    # Convenience shell script to run the AFD example
│   │   └──  run_deepep.sh    # Convenience shell script to run the DeepEP example
├── src/               # Source code
│   ├── cli/        # Main entry point
│   │   └──  main.py
│   ├── model/             # Supported Models
│   │   ├── base.py         # Base model class
│   │   ├── deepseekv3_decode.py  # DeepSeekV3-671B decoder
│   │   ├── qwen235_decode.py     # Qwen3-235B-A22B decoder
│   │   └── register.py           # Model registration method
│   ├── ops/                # Operator cost models
│   │   ├── base.py       # Base operator class
│   │   ├── communication.py   # Communication ops
│   │   ├── matmul.py       # Matmul operations
│   │   ├── page_attention.py  # Attention operations
│   │   ├── rotary.py   # Rotary Position Embedding  ops
│   │   └── swiglu.py     # swiglu ops
│   ├── search/              # Search algorithms
│   │   ├── afd.py          # AFD search
│   │   ├── base.py         # Base search class
│   │   └── deepep.py       # DeepEP search
│   └── visualization/      # Visualization tools
│       └── throughput.py       # Visualize throughput changes
├── webapp/            # FastAPI backend + Vue 3 frontend
│   ├── backend/
│   │   └── main.py          # FastAPI app, API routes, static mounts
│   └── frontend/
│       ├── index.html       # SPA entrypoint
│       ├── app.js           # Shared frontend runtime + SFC loader bootstrap
│       ├── components/      # Vue tab components
│       └── styles/
│           └── main.css
└── README.md
```

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- [Installation Guide](docs/quickstart.md)
- [Configuration](docs/conf/configuration.md)
- [AFD Search Algorithms](docs/search/AFD.md)
- [DeepEP Search Algorithms](docs/search/DeepEP.md)
- [Supported Operators](docs/ops/supported_ops.md)
- [Supported Models](docs/model/supported_models.md)
- [Visualization](docs/visualization/visualization.md)

## Examples

See the [`examples/`](examples/) directory for runnable examples:

- [DeepSeekV3-671B Example](examples/deepseek/) - Complete example with AFD and DeepEP search
- [Qwen3-235B-A22B Example](examples/qwen235B/) - Complete example with AFD and DeepEP search

## Web UI

The repository includes a browser-based UI served directly by FastAPI. The frontend lives under [`webapp/frontend/`](webapp/frontend/) and uses Vue 3 with browser-side SFC loading. The backend entrypoint is [`webapp/backend/main.py`](webapp/backend/main.py).

### Start the web app

```bash
pip install -r requirements.txt
LOG_LEVEL=INFO uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open: <http://127.0.0.1:8000>

### UI workflow

- **Run Experiment**: Submit a simulation and watch live log/progress polling
- **Configuration**: Inspect model and hardware config for the latest run selection
- **Results**: Load one generated CSV and filter/sort rows within that file
- **Visualizations**: Show backend-generated throughput and pipeline images for the selected parameters

### Notes

- `LOG_LEVEL=INFO` is recommended so the Run tab can infer progress phases from logs.
- CSV results are loaded from generated files under `data/afd/` or `data/deepep/`.
- Visualization images are served from `/data/images/`.
- `/api/results` returns only image URLs that currently exist on disk, so the UI can avoid avoidable image 404s.

## Requirements

- Python 3.8+
- pandas
- matplotlib
- numpy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

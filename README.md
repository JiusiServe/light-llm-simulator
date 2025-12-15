# Light LLM Simulator

> A lightweight inference performance simulator for Attention-FFN Disaggregated (AFD) serving. Automatically finds near-optimal deployment configurations that maximize throughput while respecting latency budgets.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## Overview

In Attention-FFN disaggregated (AFD) serving, finding an efficient deployment is far from trivial: you must jointly choose the number of Attention and FFN workers, the micro-batch size, and still meet strict SLA targets on TTFT and TPOT. **Light LLM Simulator** automates this search.

Tell it your model, NPU type, and cluster size, and it returns a near-optimal configuration that maximizes throughput while respecting your latency budget.

## Features

- 🎯 **AFD Search**: Attention-FFN Disaggregated deployment optimization
- 📊 **DeepEP Baseline**: DeepEP for comparison
- 📈 **Visualization**: Pareto frontier plots and pipeline analysis
- 🚀 **Multi-Token Prediction (MTP)**: Support for multi-token generation
- 🎨 **Extensible Architecture**: Easy to add new models, operators, or search strategies

## Supported Models

- ✅ **DeepSeek V3**: Fully supported with MLA attention and MoE
- ❌ **Qwen3-235B-A22B**: TODO

## Supported Hardware

- **Ascend NPUs**: 910B2, 910B3, 910B4, A3Pod, David121, David120

## Project Structure

```
light-llm-simulator/
├── conf/              # Configuration files
│   ├── model_config.py      # Model configurations
│   ├── hardware_config.py   # Hardware specifications
│   └── common.py            # Common constants
├── src/               # Source code
│   ├── cli/        # Main entry point
│   │   ├── main.py
│   ├── search/              # Search algorithms
│   │   ├── afd.py          # AFD search
│   │   ├── deepep.py       # DeepEP search
│   │   └── base_search.py  # Base search class
│   ├── module/             # Model modules
│   │   ├── deepseekv3_decode.py  # DeepSeek V3 decoder
│   │   └── base_module.py  # Base module class
│   ├── ops/                # Operator cost models
│   │   ├── matmul.py       # Matmul operations
│   │   ├── page_attention.py  # FlashAttention operations
│   │   ├── communication.py   # Communication ops
│   │   └── base_ops.py     # Base operator class
│   └── visualization/      # Visualization tools
│       ├── pareto.py       # Pareto frontier
│       └── pipeline.py     # Pipeline visualization
├── examples/          # Example scripts
│   └── deepseek/      # DeepSeek example
├── docs/              # Documentation
├── data/              # Output directory
└── README.md
```

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- [Installation Guide](docs/installation.md)
- [Configuration](docs/conf/configuration.md)
- [Search Algorithms](docs/search/search.md)
- [Supported Operators](docs/ops/supported_ops.md)
- [Supported Models](docs/module/supported_models.md)
- [Visualization](docs/visualization/visualization.md)

## Examples

See the [`examples/`](examples/) directory for runnable examples:

- [DeepSeek Example](examples/deepseek/) - Complete example with AFD and DeepEP search

## Requirements

- Python 3.8+
- pandas
- matplotlib
- numpy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

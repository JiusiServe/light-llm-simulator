# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Light LLM Simulator is an experimental performance explorer for large-model inference serving. Currently in trial phase, the simulator screens deployment combinations to find configurations that maximize throughput while keeping TTFT and TPOT within SLA targets. AFD (Attention-FFN Disaggregated) serving mode is the primary demo feature. DeepEP is also supported for comparison. Models include DeepSeek V3 and Qwen3-235B on both Ascend and Nvidia hardware.

## Tech Stack

- Python 3.8+

## Architecture

**Core Hierarchy:** Config → Search → Model Modules → Operators → Compute/Memory Time

- [`conf/`](conf/): Configuration management (model specs, hardware specs, common constants)
- [`src/cli/main.py`](src/cli/main.py): Main entry point, argument parsing, orchestrates search and visualization
- [`src/search/`](src/search/): Search algorithms (AFD, DeepEP) that iterate over die/batch configurations
- [`src/model/`](src/model/): Model components (Attention, MLP, MoE) inheriting from `BaseModule`
- [`src/ops/`](src/ops/): Atomic operations (matmul, attention, communication) inheriting from `BaseOp`
- [`src/visualization/`](src/visualization/): Throughput plots and pipeline Gantt charts
- [`examples/`](examples/): Runnable Python scripts for specific model/config combinations
- Output: CSV results in `data/afd/` and `data/deepep/`; images in `data/images/`

Data flow: CLI args → Config → Search iterates die allocations → Model modules build operator lists → Operators compute e2e_time → Results saved to CSV → Visualizations generated.

## Coding Conventions

- Use dataclasses for configuration (see [`conf/model_config.py`](conf/model_config.py), [`conf/hardware_config.py`](conf/hardware_config.py))
- Use Enums for type-safe identifiers (ModelType, DeviceType)
- Factory pattern: `ModelConfig.create_model_config()` and `HWConf.create()`
- Abstract base classes: `BaseSearch`, `BaseModule`, `BaseOp` with abstractmethods
- Module pattern: Each model module implements `_build_ops()` and `_aggregate_times()`
- Logging: Use `logging` module (level controlled by LOG_LEVEL env var)
- Time conversions use constants from [`conf/common.py`](conf/common.py) (SEC_2_US, BYTE_2_GB, etc.)
- File naming: `{DeviceType}-{ModelType}-tpot{tpot}-kv_len{kv_len}.csv`

## Testing and Quality Bar

No automated tests currently. Validate changes by:
1. Running `python src/cli/main.py` with default args
2. Checking that CSV files are generated in `data/afd/` and `data/deepep/`
3. Verifying visualization scripts complete without errors
4. Comparing output values against known-good results when modifying models/ops

## File and Content Placement Rules

- Add new models: Create new file in [`src/model/`](src/model/) following `{model}_decode.py` pattern
- Add new operators: Create new file in [`src/ops/`](src/ops/) following `{op_type}.py` pattern
- Add new hardware: Update [`conf/hardware_config.py`](conf/hardware_config.py) only—do not create new config files
- Add new serving mode: Extend [`src/search/base.py`](src/search/base.py) and create new search file
- Results always go to `data/` subdirectory structure (not project root)

## Safe Change Rules

- **Do not modify** [`conf/common.py`](conf/common.py) constants (time conversions, dtype sizes) without strong reason
- **Do not change** the abstract base class interfaces (`BaseOp.__call__`, `BaseModule._aggregate_times`)
- **Do not modify** the output CSV column structure—downstream visualizations depend on it
- When adding models, preserve the registration pattern in [`src/model/register.py`](src/model/register.py)
- Hardware specs in [`conf/hardware_config.py`](conf/hardware_config.py) should be vendor-published values only

## Specific Commands

```bash
# Run simulator with defaults
python src/cli/main.py

# Run AFD example for DeepSeek V3
python examples/deepseek/afd.py --tpot 50 --kv_len 4096

# Run DeepEP example for Qwen
python examples/qwen235B/deepep.py

# Generate throughput visualization
python src/visualization/throughput.py --model_type deepseek-ai/DeepSeek-V3 --device_type Ascend_A3Pod

# Generate pipeline visualization
python src/visualization/pipeline.py --file_name ASCENDA3_Pod-DEEPSEEK_V3-tpot50-kv_len4096.csv

# Install dependencies
pip install -r requirements.txt
```

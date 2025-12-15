# Visualization Tools

Light LLM Simulator provides several visualization tools to analyze search results and deployment configurations.

## Available Tools

### 1. Pareto Frontier (`pareto.py`)

Generates Pareto frontier images showing the trade-off between latency and throughput for different configurations.

**Location**: `src/visualization/pareto.py`

**Output**: 
- Shows AFD vs DeepEP comparison
- Generate multiple images for different latency targets,  micro-batch numbers
- Focus on Pareto frontiers for showing the trade-off between latency and throughput

**Usage**:
```bash
python src/visualization/pareto.py
```

### 2. Pipeline Visualization (`pipeline.py`)

Visualizes the AFD (Attention-FFN disaggregated) pipeline to identify bubble

**Location**: `src/visualization/pipeline.py`

**Output**:
- Shows pipeline stages and timing
- Use pipeline visualizations to identify bubble

**Usage**:
```bash
python src/visualization/pipeline.py
```

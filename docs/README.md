# Light LLM Simulator Documentation

Welcome to the Light LLM Simulator documentation. This documentation provides comprehensive guides for using, understanding, and extending the simulator.

## Documentation Structure

### Getting Started
- [Installation](installation.md) - How to install and set up the simulator
- [Quick Start Guide](quickstart.md) - Get up and running in minutes

### Core Components

#### Operators
- [Supported Operators](ops/supported_ops.md) - List all supported operators and their cost models

#### Models
- [Supported Models](module/supported_models.md) - Describe available model architectures

#### Search Algorithms
- [AFD Search](search/search.md) - Attention-FFN Disaggregated search algorithm
- [DeepEP Search](search/deepep.md) - Deep Expert Parallelism search algorithm
- [Search Configuration](search/configuration.md) - Configuring search parameters

#### Visualization
- [Visualization](visualization/visualization.md) - Provides several visualization tools to analyze search results and deployment configurations
- [Pareto Frontier](../src/visualization/pareto.py) - Generates Pareto frontier images showing the trade-off between latency and throughput for different configurations.
- [Pipeline Visualization](../src/visualization/pipeline.py) - Visualizes the AFD (Attention-FFN disaggregated) pipeline to identify bubble

#### Configuration
- [Configuration](conf/configuration.md) - Understanding configuration files
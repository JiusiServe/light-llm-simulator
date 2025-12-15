# Supported Models

This document describes the model architectures supported by Light LLM Simulator.

## List of Supported Models for Ascend NPU

| Models | Example HF Models |
|--------|-------------------|
|deepseek-ai/DeepSeek-V3 | `deepseek-ai/DeepSeek-V3` |

## Model Module Structure

All model modules inherit from `BaseModule` which provides:

### BaseModule Interface

```python
class BaseModule(ABC):
    def __init__(self, model_config, aichip_config, search_config):
        # Initialize module with configurations
        
    def __call__(self):
        # Execute module: _execute_ops() → _aggregate_times()
        
    @abstractmethod
    def _build_ops(self):
        # Build operator list for this module
        
    @abstractmethod
    def _aggregate_times(self):
        # Aggregate times from all operators
```

### Module Components

Each model typically consists of:

1. **Attention Module**: Handles self-attention computation
   - Query/Key/Value projections
   - Attention computation
   - Output projection

2. **MLP Module**: Handles feed-forward network
   - Up projection
   - Activation (SWiGLU)
   - Down projection

3. **MoE Module**: Handles mixture of experts (if applicable)
   - Expert routing (dispatch)
   - Expert computation
   - Expert aggregation (combine)


## DeepSeek V3

**Status**: ✅ Fully Supported

**Model Identifier**: `deepseek-ai/DeepSeek-V3`

**Architecture**:
- Hidden size: 7168
- Number of layers: 61
- Number of attention heads: 128
- MoE layers: 58
- First K dense layers: 3
- Number of routed experts: 256
- Number of shared experts: 1
- Experts per token: 8

**Key Features**:
- MLA (Multi-head Latent Attention) with LoRA decomposition
- Page Attention with INT8 quantization
- MoE (Mixture of Experts) with grouped matrix operations
- Weight absorption optimization

**Module Components**:
- `DeepSeekV3DecodeAttn`: Attention module with MLA prolog and page attention
- `DeepSeekV3DecodeMLP`: MLP module for dense layers
- `DeepSeekV3DecodeMoe`: MoE module for expert layers

**Configuration**:
```python
from conf.model_config import ModelType, ModelConfig

model_type = ModelType.DEEPSEEK_V3
model_config = ModelConfig.create_model_config(model_type)
```

## Qwen3-235B-A22B

**Status**: ❌TODO

## Adding New Models

To add support for a new model:

1. **Define Model Configuration** in `conf/model_config.py`:
   ```python
   ModelType.NEW_MODEL = "model-identifier"
   
   configs[ModelType.NEW_MODEL] = cfg(
       hidden_size=...,
       num_layers=...,
       # ... other parameters
   )
   ```

2. **Create Model Modules** in `src/module/`:
   ```python
   class NewModelAttn(BaseModule):
       def _build_ops(self):
           # Build attention operators
           
       def _aggregate_times(self):
           # Aggregate attention times
   ```

3. **Update Model Factory** in `src/search/base_search.py`:
   ```python
   def get_model(model_name, ...):
       if model_name == ModelType.NEW_MODEL:
           # Return new model modules
   ```

4. **Test** with the new model configuration

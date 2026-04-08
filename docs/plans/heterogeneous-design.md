# Design Document: Heterogeneous Deployment Support

## 1. Overview

### 1.1 Background

The Light LLM Simulator previously only supported **Homogeneous** deployment mode, where both Attention and FFN modules run on the same device type. This document describes the implementation of **Heterogeneous** deployment mode for both AFD and DeepEP serving.

### 1.2 Motivation

In real-world deployments, different hardware accelerators may have different characteristics:
- Some devices excel at attention operations (memory-bound, requiring high HBM bandwidth)
- Other devices excel at FFN/MoE operations (compute-bound, requiring high FLOPS)

Heterogeneous deployment allows users to:
1. Match workload characteristics to hardware strengths
2. Optimize cost-performance by mixing different accelerator types
3. Compare heterogeneous AFD deployments against DeepEP baselines

### 1.3 Goals

1. Support Heterogeneous mode for AFD: attention on device_type1, FFN on device_type2
2. Support Heterogeneous mode for DeepEP: run homogeneous DeepEP on two device types separately and compute weighted average
3. Maintain backward compatibility with existing Homogeneous mode
4. Update CLI and examples to support new parameters
5. Update visualizations to support heterogeneous deployments
6. Update webapp UI to support heterogeneous mode selection

## 2. Design

### 2.1 Core Concepts

#### AFD Deployment Modes

| Mode | Attention Device | FFN Device | Total Die Calculation |
|------|------------------|------------|----------------------|
| Homogeneous | device_type | device_type | attn_die + ffn_die (same device) |
| Heterogeneous | device_type1 | device_type2 | attn_die + ffn_die (different devices) |

#### DeepEP Deployment Modes

| Mode | Description |
|------|-------------|
| Homogeneous | Run DeepEP on a single device type |
| Heterogeneous | Run homogeneous DeepEP on device_type1 and device_type2 **separately**, then compute weighted average throughput. **NOT** a truly heterogeneous deployment like AFD. |

**IMPORTANT**: DeepEP Heterogeneous mode is **NOT** a truly heterogeneous deployment. It runs homogeneous DeepEP on two device types separately and computes the weighted average for comparison purposes only.

#### Key Principle

**AFD Heterogeneous mode:**
- `total_die = attn_die (device1) + ffn_die (device2)`
- Each device type has independent memory constraints
- Node alignment is checked separately for each device type

**DeepEP Heterogeneous mode:**
- Runs homogeneous DeepEP on device_type1
- Runs homogeneous DeepEP on device_type2
- Computes weighted average throughput: `(die1 * throughput1 + die2 * throughput2) / total_die`

### 2.2 Architecture Changes

#### 2.2.1 Configuration (`conf/config.py`)

New parameters:

```python
class Config:
    def __init__(
        self,
        ...
        deployment_mode: str = "Homogeneous",  # NEW
        device_type2: Optional[str] = None,    # NEW: required for Heterogeneous
        min_die2: Optional[int] = None,        # NEW: FFN die range
        max_die2: Optional[int] = None,        # NEW
        die_step2: Optional[int] = None        # NEW
    )
```

New computed attributes:

```python
# Hardware configs for each module type
self.aichip_config_attn = self.aichip_config   # device_type
self.aichip_config_ffn = self.aichip_config2   # device_type2 (or same as attn in Homogeneous)
```

#### 2.2.2 Base Module (`src/model/base.py`)

Updated constructor to select hardware config based on module type:

```python
class BaseModule(ABC):
    def __init__(
        self,
        config: Config,
        hw_type: Literal["attn", "ffn"] = "attn"  # NEW parameter
    ) -> None:
        if hw_type == "attn":
            self.aichip_config = config.aichip_config_attn
        else:  # ffn
            self.aichip_config = config.aichip_config_ffn
```

#### 2.2.3 Model Modules

All model files updated to pass appropriate `hw_type`:

| Module Type | hw_type | Files |
|-------------|---------|-------|
| Attention | "attn" | `*_decode.py` (DecodeAttn classes) |
| MLP | "ffn" | `*_decode.py` (DecodeMLP classes) |
| MoE | "ffn" | `*_decode.py` (DecodeMoe classes) |

#### 2.2.4 AFD Search (`src/search/afd.py`)

Key changes:

1. **Memory constraints**: Use device-specific memory limits
   ```python
   # Attention memory check
   attn_memory > self.config.aichip_config_attn.aichip_memory * THRESHOLD

   # FFN memory check
   ffn_static_memory > self.config.aichip_config_ffn.aichip_memory * THRESHOLD
   ```

2. **Die iteration**: Separate ranges for heterogeneous mode
   ```python
   if self.config.deployment_mode == "Heterogeneous":
       attn_die_start = min_attn_die  # from min_die
       attn_die_end = max_attn_die    # from max_die
   else:
       attn_die_start = ffn_die       # original behavior
       attn_die_end = ATTN_DIE_MULTIPLIER * ffn_die
   ```

3. **Node alignment**: Independent checks per device type
   ```python
   if self.config.deployment_mode == "Heterogeneous":
       if attn_die % aichip_config_attn.num_dies_per_node != 0: continue
       if ffn_die % aichip_config_ffn.num_dies_per_node != 0: continue
   else:
       if total_die % aichip_config.num_dies_per_node != 0: continue
   ```

4. **Output file naming**: Include both device types
   ```python
   if deployment_mode == "Heterogeneous":
       file_name = f"{device_type1}_{device_type2}-{model}-tpot{tpot}-kv_len{kv_len}-heterogeneous.csv"
   else:
       file_name = f"{device_type}-{model}-tpot{tpot}-kv_len{kv_len}.csv"
   ```

#### 2.2.5 DeepEP Search (`src/search/deepep.py`)

Key changes for Heterogeneous mode:

1. **Homogeneous mode**: Original behavior, output includes new columns:
   - `deployment_mode`: "Homogeneous"
   - `device_type1`: Device type
   - `device_type2`: Device type (same as device_type1)

2. **Heterogeneous mode**: Run homogeneous DeepEP on two devices separately
   ```python
   # Run DeepEP on device_type1
   results_device1 = self._run_homogeneous_deepep(device_type1, ...)
   # Run DeepEP on device_type2
   results_device2 = self._run_homogeneous_deepep(device_type2, ...)
   # Compute weighted average
   weighted_throughput = (die1 * throughput1 + die2 * throughput2) / total_die
   ```

   **NOTE**: DeepEP Heterogeneous mode logs a reminder message explaining that it runs homogeneous DeepEP separately:
   ```
   NOTE: DeepEP Heterogeneous mode runs homogeneous DeepEP on two
   device types separately and computes weighted average throughput.
   This is for comparison purposes only, NOT a truly heterogeneous deployment.
   ```

#### 2.2.6 Visualization (`src/visualization/`)

##### Throughput Visualization (`throughput.py`)

Added support for heterogeneous CSV files:

1. **New CLI arguments**:
   ```python
   parser.add_argument('--deployment_mode', type=str, default="Homogeneous")
   parser.add_argument('--device_type2', type=str, default=None)
   ```

2. **Heterogeneous file naming**: Uses `{device_type1}_{device_type2}-{model}-tpot{tpot}-kv_len{kv_len}-heterogeneous.csv`

3. **Output path**: `data/images/throughput/heterogeneous/{device_type1}_{device_type2}-{model}-...`

##### Pipeline Visualization (`pipeline.py`)

Added support for heterogeneous pipeline Gantt charts:

1. **New CLI arguments**:
   ```python
   parser.add_argument('--deployment_mode', type=str, default="Homogeneous")
   parser.add_argument('--device_type2', type=str, default=None)
   ```

2. **Heterogeneous file lookup**: Checks for heterogeneous CSV files when `deployment_mode="Heterogeneous"`

3. **Output path**: `data/images/pipeline/heterogeneous/{device_type1}_{device_type2}-{model}-...`

#### 2.2.7 Webapp Updates (`webapp/`)

##### Backend (`webapp/backend/main.py`)

1. **RunRequest model**: Added heterogeneous parameters
   ```python
   deployment_mode: str = "Homogeneous"
   device_type2: Optional[str] = None
   min_die2: Optional[int] = None
   max_die2: Optional[int] = None
   die_step2: Optional[int] = None
   ```

2. **_sanitize_args**: Passes heterogeneous parameters to CLI

3. **fetch_csv_results**: Supports heterogeneous file naming
   ```python
   if deployment_mode == "Heterogeneous" and device_type2:
       file_name = f"{device_type}_{device_type2}-{model_type}-tpot{tpot}-kv_len{kv_len}-heterogeneous.csv"
   ```

##### Frontend Components

1. **RunForm.vue**:
   - Added Deployment Mode dropdown (Homogeneous/Heterogeneous)
   - Added Device Type 2 dropdown (conditional, shown when Heterogeneous)
   - Added FFN die range fields (conditional)

2. **CsvSelector.vue**:
   - Added Deployment Mode dropdown
   - Added Device Type 2 dropdown (conditional)

3. **ConfigurationTab/index.vue**:
   - Shows both device configurations when heterogeneous mode is selected
   - Updated summary text to show both device types

4. **HardwareConfig.vue**:
   - Added `label` prop to support different section titles ("Attention Device" vs "FFN Device")

### 2.3 CLI Changes (`src/cli/main.py`)

New arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--deployment_mode` | "Homogeneous" | Deployment mode (Homogeneous/Heterogeneous) |
| `--device_type2` | None | Second device type for Heterogeneous mode |
| `--min_die2` | None | Min FFN dies for Heterogeneous |
| `--max_die2` | None | Max FFN dies for Heterogeneous |
| `--die_step2` | None | Die step for FFN in Heterogeneous |

### 2.4 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Homogeneous Mode                            │
├─────────────────────────────────────────────────────────────────┤
│  device_type ──► aichip_config_attn                             │
│              ──► aichip_config_ffn (same)                       │
│                                                                 │
│  min_die/max_die ──► both attn_die and ffn_die search ranges    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Heterogeneous Mode                          │
├─────────────────────────────────────────────────────────────────┤
│  device_type  ──► aichip_config_attn (for Attention modules)    │
│  device_type2 ──► aichip_config_ffn (for FFN/MoE modules)       │
│                                                                 │
│  min_die/max_die   ──► attn_die search range                    │
│  min_die2/max_die2 ──► ffn_die search range                     │
│                                                                 │
│  total_die = attn_die + ffn_die                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 Memory Constraints

#### Homogeneous Mode
```
attn_memory < aichip_memory * MEMORY_THRESHOLD_RATIO
ffn_memory  < aichip_memory * MEMORY_THRESHOLD_RATIO
```

#### Heterogeneous Mode
```
attn_memory < aichip_config_attn.aichip_memory * MEMORY_THRESHOLD_RATIO
ffn_memory  < aichip_config_ffn.aichip_memory  * MEMORY_THRESHOLD_RATIO
```

## 3. Usage Examples

### 3.1 CLI - Heterogeneous AFD

```bash
python src/cli/main.py \
  --serving_mode AFD \
  --deployment_mode Heterogeneous \
  --device_type Ascend_A3Pod \
  --device_type2 Ascend_David121 \
  --min_die 16 --max_die 256 --die_step 16 \
  --min_die2 16 --max_die2 128 --die_step2 16 \
  --tpot 50 --kv_len 4096 \
  --micro_batch_num 3
```

### 3.2 CLI - Heterogeneous DeepEP

```bash
python src/cli/main.py \
  --serving_mode DeepEP \
  --deployment_mode Heterogeneous \
  --device_type Ascend_A3Pod \
  --device_type2 Ascend_David121 \
  --min_die 16 --max_die 256 --die_step 16 \
  --min_die2 16 --max_die2 128 --die_step2 16 \
  --tpot 50 --kv_len 4096
```

**NOTE**: DeepEP Heterogeneous mode will log a warning message explaining that it runs homogeneous DeepEP on two device types separately.

### 3.3 Python API - Heterogeneous AFD

```python
from conf.config import Config
from src.search.afd import AfdSearch

config = Config(
    serving_mode="AFD",
    model_type="deepseek-ai/DeepSeek-V3",
    device_type="Ascend_A3Pod",
    min_attn_bs=2,
    max_attn_bs=1000,
    min_die=16,
    max_die=256,
    die_step=16,
    tpot=50,
    kv_len=4096,
    micro_batch_num=3,
    next_n=1,
    multi_token_ratio=0.7,
    attn_tensor_parallel=1,
    ffn_tensor_parallel=1,
    deployment_mode="Heterogeneous",
    device_type2="Ascend_David121",
    min_die2=16,
    max_die2=128,
    die_step2=16
)

afd_search = AfdSearch(config)
afd_search.deployment()
```

### 3.4 Python API - Heterogeneous DeepEP
```python
from conf.config import Config
from src.search.deepep import DeepEpSearch

config = Config(
    serving_mode="DeepEP",
    model_type="deepseek-ai/DeepSeek-V3",
    device_type="Ascend_A3Pod",
    min_attn_bs=2,
    max_attn_bs=1000,
    min_die=16,
    max_die=256,
    die_step=16,
    tpot=50,
    kv_len=4096,
    micro_batch_num=1,
    next_n=1,
    multi_token_ratio=0.7,
    attn_tensor_parallel=1,
    ffn_tensor_parallel=1,
    deployment_mode="Heterogeneous",
    device_type2="Ascend_David121",
    min_die2=16,
    max_die2=128,
    die_step2=16
)

deepep_search = DeepEpSearch(config)
deepep_search.deployment()
# Output: data/deepep/ASCENDA3_Pod_ASCENDDAVID121-DEEPSEEK_V3-tpot50-kv_len4096-heterogeneous.csv
```

## 4. Output Format

### 4.1 AFD CSV

New columns added:

| Column | Description |
|--------|-------------|
| `deployment_mode` | "Homogeneous" or "Heterogeneous" |
| `device_type_attn` | Device type for attention |
| `device_type_ffn` | Device type for FFN |

File naming:
- Homogeneous: `{device_type}-{model}-tpot{tpot}-kv_len{kv_len}.csv`
- Heterogeneous: `{device_type1}_{device_type2}-{model}-tpot{tpot}-kv_len{kv_len}-heterogeneous.csv`

### 4.2 DeepEP CSV

#### Homogeneous Mode
| Column | Description |
|--------|-------------|
| `deployment_mode` | "Homogeneous" |
| `device_type1` | Device type |
| `device_type2` | Device type (same as device_type1) |

#### Heterogeneous Mode
| Column | Description |
|--------|-------------|
| `device1_die` | Dies on device_type1 |
| `device2_die` | Dies on device_type2 |
| `total_die` | device1_die + device2_die |
| `throughput_device1` | DeepEP throughput on device_type1 |
| `throughput_device2` | DeepEP throughput on device_type2 |
| `weighted_throughput` | Weighted average throughput |
| `deployment_mode` | "Heterogeneous" |
| `device_type1` | Device type for attention |
| `device_type2` | Device type for FFN |

## 5. Implementation Details

### 5.1 Files Modified

| File | Changes |
|------|---------|
| `conf/config.py` | Added heterogeneous parameters |
| `src/model/base.py` | Added `hw_type` parameter |
| `src/model/deepseekv3_decode.py` | Updated MLP, MoE to use `hw_type="ffn"` |
| `src/model/qwen235_decode.py` | Updated MoE to use `hw_type="ffn"` |
| `src/model/deepseekv2_lite_decode.py` | Updated MLP, MoE to use `hw_type="ffn"` |
| `src/search/afd.py` | Heterogeneous mode support |
| `src/search/deepep.py` | Heterogeneous mode support (runs homogeneous separately) |
| `src/cli/main.py` | Added CLI arguments |
| `src/visualization/throughput.py` | Heterogeneous visualization support |
| `src/visualization/pipeline.py` | Heterogeneous pipeline Gantt chart support |
| `webapp/backend/main.py` | Webapp backend heterogeneous support |
| `webapp/frontend/components/RunExperimentTab/RunForm.vue` | Heterogeneous UI controls |
| `webapp/frontend/components/ResultsTab/CsvSelector.vue` | Heterogeneous CSV selection |
| `webapp/frontend/components/ConfigurationTab/index.vue` | Heterogeneous config display |
| `webapp/frontend/components/ConfigurationTab/HardwareConfig.vue` | Added label prop |

### 5.2 New Files

| File | Purpose |
|------|---------|
| `examples/deepseek/afd_hetero.py` | Heterogeneous AFD example |
| `docs/plans/heterogeneous-design.md` | This design document |

### 5.3 Backward Compatibility

All changes are backward compatible:
- Default `deployment_mode="Homogeneous"` preserves original behavior
- Existing API calls without new parameters work unchanged
- All 31 existing tests pass

## 6. Testing

### 6.1 Test Results

```
$ python -m pytest tests/ -v
============================= test session starts =============================
collected 31 items

tests/e2e/test_cli_subprocess.py::test_cli_deepep_minimal_grid PASSED
tests/e2e/test_cli_subprocess.py::test_cli_help_exits_zero PASSED
tests/e2e/test_search_smoke.py::test_afd_search_writes_csv PASSED
tests/e2e/test_search_smoke.py::test_deepep_search_writes_csv PASSED
...
============================= 31 passed in 1.88s ==============================
```

### 6.2 Manual Testing

```bash
# Test Heterogeneous AFD
python -c "
from conf.config import Config
from src.search.afd import AfdSearch

config = Config(
    serving_mode='AFD',
    model_type='deepseek-ai/DeepSeek-V3',
    device_type='Ascend_A3Pod',
    device_type2='Ascend_David121',
    deployment_mode='Heterogeneous',
    ...
)
afd_search = AfdSearch(config)
afd_search.deployment()
"
# Output: data/afd/mbn2/ASCENDA3_Pod_ASCENDDAVID121-DEEPSEEK_V3-tpot50-kv_len4096-heterogeneous.csv
```

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. DeepEP heterogeneous mode runs homogeneous DeepEP separately (not truly heterogeneous)
2. Communication costs between different device types use the attention device's bandwidth
3. No support for more than 2 device types

### 7.2 Future Enhancements

1. Model inter-device communication costs more accurately
2. Support arbitrary number of device types
3. Add comparison charts between homogeneous and heterogeneous deployments
4. Add interactive heterogeneous visualization in webapp

## 8. References

- [AFD Documentation](../search/AFD.md)
- [DeepEP Documentation](../search/DeepEP.md)
- [Configuration Guide](../conf/configuration.md)

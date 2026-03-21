# Visualization

Light LLM Simulator produces visualization assets from generated CSV results and exposes them in two ways:

- CLI scripts under [`src/visualization/`](../../src/visualization/)
- The browser Visualizations tab served by FastAPI

The visualizations are static PNG images generated from simulator output. There is no interactive charting layer in the current implementation.

## Output Layout

Generated images are written under `data/images/`:

- `data/images/throughput/`
- `data/images/pipeline/mbn2/`
- `data/images/pipeline/mbn3/`
- `data/images/pipeline/deepep/`

Generated CSV inputs are read from:

- `data/afd/mbn2/best/`
- `data/afd/mbn3/best/`
- `data/deepep/`

CSV filenames follow:

```text
{DeviceType.name}-{ModelType.name}-tpot{tpot}-kv_len{kv_len}.csv
```

Example:

```text
ASCENDA3_Pod-DEEPSEEK_V3-tpot50-kv_len4096.csv
```

## CLI Tools

### Throughput Charts

Script: [`src/visualization/throughput.py`](../../src/visualization/throughput.py)

This script generates two families of throughput charts:

- throughput vs. total dies for a specific `(device, model, tpot, kv_len)`
- AFD-over-DeepEP improvement vs. `kv_len` and `tpot` for a fixed `total_die`

#### Example

```bash
python src/visualization/throughput.py \
  --model_type deepseek-ai/DeepSeek-V3 \
  --device_type Ascend_A3Pod \
  --tpot_list 20 50 70 100 150 \
  --kv_len_list 2048 4096 8192 16384 131072 \
  --micro_batch_num 2 3 \
  --total_die 128 \
  --min_die 16 \
  --max_die 768
```

#### Output

- `data/images/throughput/{DeviceType.name}-{ModelType.name}-mbn{micro_batch_num}-total_die{total_die}.png`
- `data/images/throughput/{DeviceType.name}-{ModelType.name}-tpot{tpot}-kv_len{kv_len}.png`

#### Notes

- `throughput_vs_dies()` expects all three CSVs to exist for the same file name:
  - `data/deepep/`
  - `data/afd/mbn2/best/`
  - `data/afd/mbn3/best/`
- If one of those files is missing, the script raises `FileNotFoundError` for that chart.

### Pipeline Charts

Script: [`src/visualization/pipeline.py`](../../src/visualization/pipeline.py)

This script generates Gantt-style pipeline charts for the available serving modes by reading one CSV filename across:

- `data/deepep/`
- `data/afd/mbn2/best/`
- `data/afd/mbn3/best/`

#### Example

```bash
python src/visualization/pipeline.py \
  --file_name ASCENDA3_Pod-DEEPSEEK_V3-tpot50-kv_len4096.csv
```

#### Output

- `data/images/pipeline/deepep/{file_stem}-total_die{total_die}.png`
- `data/images/pipeline/mbn2/{file_stem}-total_die{total_die}.png`
- `data/images/pipeline/mbn3/{file_stem}-total_die{total_die}.png`

#### Notes

- The script skips missing CSV inputs per serving mode instead of failing the whole run.
- One image is produced per matching `total_die` row in the source CSV.

## Web Visualizations Tab

The browser UI includes a Visualizations tab implemented in [`ThroughputCharts.vue`](../../webapp/frontend/components/VisualizationsTab/ThroughputCharts.vue).

### How it works

1. The user selects or seeds:
   - `device_type`
   - `model_type`
   - `total_die`
   - `tpot`
   - `kv_len`
2. The frontend calls:

```text
GET /api/results
```

3. The backend returns lists of image URLs under `/data/images/...`
4. The UI renders those PNGs directly

The tab is seeded from shared app state, so using `View Charts` from the Results table carries the first selected row's `total_die` into the visualization view.

### Important behavior

- The UI is static-image based only. There is no zoom, pan, or export flow.
- [`webapp/backend/main.py`](../../webapp/backend/main.py) filters `/api/results` to return only image files that actually exist on disk.
- Some parameter combinations legitimately return fewer images than others if only part of the visualization set has been generated.

## End-to-End Flow

Typical visualization workflow:

1. Run a simulation from the CLI or web UI
2. Confirm CSV outputs exist under `data/afd/...` or `data/deepep/`
3. Generate images with:
   - [`src/visualization/throughput.py`](../../src/visualization/throughput.py)
   - [`src/visualization/pipeline.py`](../../src/visualization/pipeline.py)
4. Open the web UI and use the Visualizations tab, or inspect the PNG files directly

## Limitations

- Visualizations are pre-generated assets, not live chart computations
- Missing CSV inputs lead to missing charts
- Some chart families depend on both AFD and DeepEP results existing for the same parameter set
- The web UI does not offer chart export beyond the static PNG files already saved under `data/images/`

---
name: visualization
description: Run visualization scripts (throughput and pipeline) for LLM simulator results. Use this skill when the user asks to "visualization", "visualize", "run visualization", "generate charts", "plot results", or mentions creating charts or graphs from the search data.
---

# LLM Visualization Runner

This skill runs the visualization scripts to generate throughput comparison charts and pipeline Gantt charts.

## When to use

Trigger this skill when the user asks to:
- "visualization"
- "visualize"
- "run visualization"
- "generate charts"
- "plot results"
- Any similar request to create visualizations from search data

## What it does

Runs two visualization scripts with three hardware configurations each:

### Hardware Configurations

1. **David120 + David100**: `--device_type1 "Ascend_David120" --device_type2 "Ascend_David100"`
2. **David120 + 910B2**: `--device_type1 "Ascend_David120" --device_type2 "Ascend_910b2"`
3. **David100 + 910B2**: `--device_type1 "Ascend_David100" --device_type2 "Ascend_910b2"`

### Scripts

**Throughput Charts** (`src/visualization/throughput.py`):
- Generates throughput improvement comparison charts
- Compares AFD vs DeepEP performance

**Pipeline Charts** (`src/visualization/pipeline.py`):
- Generates pipeline Gantt charts
- Shows attention, dispatch, MoE, and combine timing

## Execution

Run all configurations in parallel:

```bash
cd <repo-root>

# Throughput visualization
python src/visualization/throughput.py --device_type1 "Ascend_David120" --device_type2 "Ascend_David100"

python src/visualization/throughput.py --device_type1 "Ascend_David120" --device_type2 "Ascend_910b2"

python src/visualization/throughput.py --device_type1 "Ascend_David100" --device_type2 "Ascend_910b2"

# Pipeline visualization
python src/visualization/pipeline.py --device_type1 "Ascend_David120" --device_type2 "Ascend_David100"

python src/visualization/pipeline.py --device_type1 "Ascend_David120" --device_type2 "Ascend_910b2"

python src/visualization/pipeline.py --device_type1 "Ascend_David100" --device_type2 "Ascend_910b2"
```

## Output

- **Throughput charts**: `data/images/throughput/`
  - Naming: `{DEVICE1}-{DEVICE2}-{MODEL}-mbn{MBN}-total_die{DIE}.png`

- **Pipeline charts**: `data/images/pipeline/`
  - DeepEP: `data/images/pipeline/deepep/`
  - AFD mbn2: `data/images/pipeline/mbn2/`
  - AFD mbn3: `data/images/pipeline/mbn3/`

## Upload to Google Drive

After generating visualizations, only upload **throughput images** to Google Drive:

```bash
python .claude/skills/visualization/upload_throughput.py
```

This script will:
1. Create the nested folder structure `data/images/throughput` in Google Drive if it doesn't exist
2. Upload all `.png` files from local `data/images/throughput/` to Google Drive

**Google Drive folder**: https://drive.google.com/drive/folders/17EJN3WPwGO4D-31q6rCNZySFlWDU3uHJ

After completion:
1. Inform user which visualizations were generated
2. Where to find the output images locally
3. Run `python upload_throughput.py` to upload throughput images to Google Drive
4. Optionally list the generated files

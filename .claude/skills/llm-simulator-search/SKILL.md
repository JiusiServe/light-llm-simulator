---
name: llm-simulator-search
description: Run LLM inference serving search (DeepEP and AFD modes) with predefined hardware configurations. Use this skill when the user asks to "run src.cli.main", "run the search", "run DeepEP search", "run AFD search", or mentions running the main CLI for performance exploration.
---

# LLM Simulator Search Runner

This skill runs the Light LLM Simulator's search with six predefined configurations (3 DeepEP + 3 AFD).

## When to use

Trigger this skill when the user asks to:
- "run src.cli.main"
- "run the search"
- "run the main CLI"
- "run the simulation"
- Any similar request to execute the search

## What it does

Runs `src/cli/main.py` six times with these configurations:

### DeepEP Configurations

**DeepEP 1: David120 + David100**
```
--serving_mode "DeepEP"
--device_type1 "Ascend_David120"
--device_type2 "Ascend_David100"
--min_die1 8 --max_die1 288 --die_step1 8
--min_die2 8 --max_die2 288 --die_step2 8
```

**DeepEP 2: David120 + 910B2**
```
--serving_mode "DeepEP"
--device_type1 "Ascend_David120"
--device_type2 "Ascend_910b2"
--min_die1 8 --max_die1 288 --die_step1 8
--min_die2 8 --max_die2 288 --die_step2 8
```

**DeepEP 3: David100 + 910B2**
```
--serving_mode "DeepEP"
--device_type1 "Ascend_David100"
--device_type2 "Ascend_910b2"
--min_die1 8 --max_die1 288 --die_step1 8
--min_die2 8 --max_die2 288 --die_step2 8
```

### AFD Configurations

**AFD 1: David120 + David100**
```
--serving_mode "AFD"
--device_type1 "Ascend_David120"
--device_type2 "Ascend_David100"
--min_die1 8 --max_die1 288 --die_step1 8
--min_die2 8 --max_die2 288 --die_step2 8
```

**AFD 2: David120 + 910B2**
```
--serving_mode "AFD"
--device_type1 "Ascend_David120"
--device_type2 "Ascend_910b2"
--min_die1 8 --max_die1 288 --die_step1 8
--min_die2 8 --max_die2 288 --die_step2 8
```

**AFD 3: David100 + 910B2**
```
--serving_mode "AFD"
--device_type1 "Ascend_David100"
--device_type2 "Ascend_910b2"
--min_die1 8 --max_die1 288 --die_step1 8
--min_die2 8 --max_die2 288 --die_step2 8
```

## Execution

Run all six configurations in parallel using the Bash tool:

```bash
cd <repo-root>

# DeepEP runs
python src/cli/main.py --serving_mode "DeepEP" --device_type1 "Ascend_David120" --device_type2 "Ascend_David100" --min_die1 8 --max_die1 288 --die_step1 8 --min_die2 8 --max_die2 288 --die_step2 8

python src/cli/main.py --serving_mode "DeepEP" --device_type1 "Ascend_David120" --device_type2 "Ascend_910b2" --min_die1 8 --max_die1 288 --die_step1 8 --min_die2 8 --max_die2 288 --die_step2 8

python src/cli/main.py --serving_mode "DeepEP" --device_type1 "Ascend_David100" --device_type2 "Ascend_910b2" --min_die1 8 --max_die1 288 --die_step1 8 --min_die2 8 --max_die2 288 --die_step2 8

# AFD runs
python src/cli/main.py --serving_mode "AFD" --device_type1 "Ascend_David120" --device_type2 "Ascend_David100" --min_die1 8 --max_die1 288 --die_step1 8 --min_die2 8 --max_die2 288 --die_step2 8

python src/cli/main.py --serving_mode "AFD" --device_type1 "Ascend_David120" --device_type2 "Ascend_910b2" --min_die1 8 --max_die1 288 --die_step1 8 --min_die2 8 --max_die2 288 --die_step2 8

python src/cli/main.py --serving_mode "AFD" --device_type1 "Ascend_David100" --device_type2 "Ascend_910b2" --min_die1 8 --max_die1 288 --die_step1 8 --min_die2 8 --max_die2 288 --die_step2 8
```

## Output

- DeepEP results: `data/deepep/`
- AFD results: `data/afd/`

Files follow naming pattern: `{DEVICE1}-{DEVICE2}-DEEPSEEK_V3-tpot{TPOT}-kv_len{KVLEN}.csv`

After completion:
1. Inform user which serving modes and hardware combinations were run
2. Where to find the output files
3. Upload results to Google Drive folder
4. Optionally list the generated files

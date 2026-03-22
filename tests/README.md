# Tests

Uses [pytest](https://pytest.org/). Layout is inspired by [ai-dynamo/aiconfigurator `tests/`](https://github.com/ai-dynamo/aiconfigurator/tree/main/tests).

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Commands

```bash
# Fast checks (unit + stable smoke)
MPLBACKEND=agg python -m pytest -m "unit or build"

# Unit only
MPLBACKEND=agg python -m pytest -m unit

# All tests (including e2e)
MPLBACKEND=agg python -m pytest
```

- **`unit`**: config, hardware, operators, `get_model`, CLI `--help`.
- **`e2e` / `build`**: AFD/DeepEP smoke under `tmp_path` (no writes to repo `data/`) and a minimal CLI run with `LIGHT_LLM_SKIP_POST_PLOTS=1` so visualization is skipped.

If you see matplotlib GUI errors, keep `MPLBACKEND=agg`.

# Webapp API Smoke Test Plan

## Summary

Add a lightweight backend-only webapp smoke suite under `pytest` and include it in the fast `build` subset. The suite exercises the FastAPI routes used by the webapp, avoids any JS or browser tooling, and isolates filesystem and process side effects with temp directories and monkeypatches.

## Scope

In scope:

- FastAPI route smoke coverage for `webapp/backend/main.py`
- Inclusion in `pytest -m "unit or build"`
- Dependency manifest fixes required to run the webapp and its tests in a clean environment
- Test documentation updates

Out of scope:

- Browser or frontend component tests
- Playwright, Jest, Vitest, Cypress, or any JS toolchain
- Runtime behavior changes to the webapp API
- Refactors to frontend code or backend route structure

## Important Interface Notes

No public HTTP API changes are planned.

Existing interfaces covered by the suite:

- `GET /`
- `POST /api/run`
- `GET /api/status/{run_id}`
- `GET /api/logs/{run_id}`
- `GET /api/results`
- `GET /api/model_config`
- `GET /api/hardware_config`
- `GET /api/constants`
- `GET /api/fetch_csv_results`

Developer-facing changes:

- New smoke test module: `tests/e2e/test_webapp_api_smoke.py`
- New declared dependencies in `requirements.txt` and `requirements-dev.txt`
- Updated docs in `tests/README.md` and `README.md`

## Planned Changes

### 1. Add missing dependency declarations

Update `requirements.txt` to declare the webapp runtime packages already imported by `webapp/backend/main.py`:

- `fastapi`
- `uvicorn`

Update `requirements-dev.txt` to declare the API test client package used directly by the new tests:

- `httpx>=0.28,<1`

### 2. Add a backend smoke test module

Create `tests/e2e/test_webapp_api_smoke.py` with:

- `pytest.mark.e2e`
- `pytest.mark.build`

### 3. Use an `httpx` ASGI test harness

Do not use `fastapi.testclient.TestClient`.

Use:

- `httpx.ASGITransport(app=webapp.backend.main.app)`
- `httpx.AsyncClient(transport=..., base_url="http://testserver")`
- `asyncio.run(...)` to keep the tests plain `pytest` tests without adding `pytest-asyncio`

### 4. Isolate filesystem and subprocess side effects

Use monkeypatching and temp directories for:

- `webapp.backend.main.RUN_ROOT`
- `webapp.backend.main.__file__`
- `webapp.backend.main._run_process`
- `webapp.backend.main.uuid4`

Defaults:

- Redirect run output to `tmp_path / "runs"`
- Redirect repo-root-derived `data/` lookups by monkeypatching `__file__`
- Create the minimum temp fixture files needed per test
- Mock `_run_process` so `/api/run` background execution writes a fake `output.log` and `.done`

### 5. Cover the selected API scope

Tests included:

- `test_root_serves_index_html`
- `test_constants_endpoint_returns_expected_values`
- `test_model_config_endpoint_success_and_invalid`
- `test_hardware_config_endpoint_success_and_invalid`
- `test_run_status_and_logs_lifecycle_with_mocked_background_task`
- `test_status_and_logs_404_for_missing_run`
- `test_results_endpoint_returns_only_existing_image_urls`
- `test_fetch_csv_results_returns_filtered_rows_for_afd`
- `test_fetch_csv_results_returns_rows_for_deepep`
- `test_fetch_csv_results_errors_for_invalid_mode_and_missing_file`

## Test Execution Contract

The suite must pass under:

```bash
MPLBACKEND=agg python -m pytest -m "unit or build"
```

Focused invocation:

```bash
python -m pytest tests/e2e/test_webapp_api_smoke.py
```

## Documentation Updates

Update `tests/README.md` to mention:

- the new webapp API smoke suite
- that it is included in the `build` subset
- the focused command for running it directly

Update `README.md` to align dependency instructions with actual webapp requirements while keeping the existing webapp launch command unchanged.

## Acceptance Criteria

- A clean environment can install the declared dependencies and run the webapp backend tests successfully.
- The new smoke suite runs in the `build` subset.
- No test writes repo-tracked files.
- No test launches a real simulator subprocess.
- No JS or browser test tooling is introduced.

## Assumptions And Defaults

- The initial webapp test scope is backend API smoke only.
- The suite belongs in the fast CI subset and uses `pytest.mark.build`.
- `httpx.AsyncClient` plus `ASGITransport` is required because `fastapi.testclient.TestClient` is incompatible with `httpx 0.28.x` here.
- File-backed route tests redirect repo-root-dependent paths via monkeypatched `__file__` rather than writing into repo `data/`.
- Run execution is mocked at `_run_process`; the tests validate route wiring and argument construction, not the underlying simulator search implementation.

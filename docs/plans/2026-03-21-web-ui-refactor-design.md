# Web UI Refactor Design

**Date:** 2026-03-21
**Status:** Implemented on `feature/web-ui-refactor-impl`

## Problem Statement

The Light LLM Simulator has an accurate AFD simulation engine but suffers from poor user experience and layout:

1. No real-time progress feedback during simulations
2. Confusing parameter inputs with no documentation
3. Four-panel layout requires horizontal scrolling
4. Poor result visualization and comparison capabilities

## Solution Overview

Refactor the web frontend to Vue.js 3 (via CDN) with a minimal FastAPI adjustment for image URL filtering. This provides:

- Tab-based navigation with responsive layouts
- Two-column desktop layouts for Run and Configuration tabs
- Real-time simulation progress (requires INFO logging)
- CSV file selector with filterable/sortable results within selected file
- Static image visualizations (no interactive zoom/pan)
- Improved parameter UX with grouped inputs; tooltips are optional polish

## Architecture

### Tech Stack
- **Frontend:** Vue 3 (Composition API) via CDN
- **SFC Loading:** vue3-sfc-loader (browser-based .vue compilation)
- **Charts:** Static images loaded from `/data/images/` (backend-generated PNGs)
- **Backend:** FastAPI (still calls CLI as subprocess; `/api/results` now filters missing image URLs)
- **Persistence:** localStorage for tab state, run history, CSV selection

### Component Structure
```
App
├── TabManager (handles tab switching, renders child components)
├── RunExperimentTab
│   ├── RunForm (parameter inputs, grouped sections, optional tooltips)
│   └── RunStatus (real-time progress + log streaming)
├── ConfigurationTab
│   ├── ModelConfig (displays model specs)
│   └── HardwareConfig (displays hardware specs)
├── ResultsTab
│   ├── CsvSelector (select which CSV file to view)
│   ├── ResultsFilter (filters within selected CSV: die range, throughput range)
│   └── ResultsTable (sortable, paginated, row selection, view-charts action)
└── VisualizationsTab
    └── ThroughputCharts (static image display)
```

### State Management
- Shared runtime lives in `webapp/frontend/app.js`
- `window.LightLLMRuntime` exposes:
  - `Vue`
  - `useStore()` for singleton navigation/run/CSV state
  - `useApi()` for FastAPI calls
- Reactive refs are still used for component-local state
- `watch()` drives config refreshes and run-status polling

### API Layer
- `app.js` owns the `fetchJson()` wrapper and exposes `useApi()`
- Centralized error handling for `/api/*` calls
- Shared CSV selection is seeded from the latest started run so Results and Visualizations follow the current simulation context

## Tab Design

### Run Experiment Tab
- Collapsible parameter sections; tooltips are optional polish
- Real-time progress bar and log streaming (poll `/logs/{run_id}`)
- Phase indicators and optional heuristic ETA
- Run history in localStorage
- Desktop layout keeps `RunForm` and `RunStatus` side by side so progress remains visible while editing parameters

**Important:** Phase detection requires backend to log progress at INFO level. The default WARNING level will suppress phase marker messages.

### Configuration Tab
- Model specs from `/api/model_config`
- Hardware specs from `/api/hardware_config`
- Formatted display with human-readable units
- Desktop layout shows model and hardware panels side by side

### Results Tab

**Design Approach: Filter Within Selected CSV File**

The Results tab is designed to filter and sort WITHIN a SINGLE selected CSV file. This is a necessary limitation because:

1. The backend `/fetch_csv_results` endpoint returns rows from ONE specific CSV file
2. CSV files are named by pattern: `{DeviceType.name}-{ModelType.name}-tpot{tpot}-kv_len{kv_len}.csv`
3. The backend does not provide an API to list available CSV files
4. CSV data contains performance metrics but NOT metadata like model_type, device_type

**Components:**
- **CsvSelector:** Select which CSV file to view (by serving mode, device type, model type, TPOT, KV length, micro batch number)
  - Uses exact enum names because backend uses those strings directly in filenames
- **ResultsFilter:** Filter within the loaded CSV:
  - Min/Max Total Die
  - Min/Max Throughput (tokens/die/s)
- **ResultsTable:** Sortable, paginated data grid with row selection
- **Charts handoff:** One or more selected rows can navigate to Visualizations, and the current static-image tab initializes its controls from the current CSV selection plus the first selected row's `total_die`

**Workflow:**
1. User selects CSV file using CsvSelector (matches parameters used to run a simulation)
2. CSV data loads and displays in ResultsTable
3. User can filter/sort within that single CSV's data
4. To view different simulation results, user selects a different CSV file

### Visualizations Tab
- **Static image display only** - loads pre-generated chart images from `/data/images/`
- Throughput charts: view generated PNG images
- Parameters (`device_type`, `model_type`, `total_die`, `tpot`, `kv_len`) select which pre-generated chart to display
- No interactive zoom/pan (requires backend API changes not in scope)
- No export functionality (requires backend API changes not in scope)

**Note:** The current backend returns static image URLs only, and now filters the `/api/results` response down to files that actually exist on disk. Interactive features like zoom/pan and export would still require new API endpoints that provide raw chart data or generate images on-demand.

## File Structure

```
webapp/
├── backend/
│   └── main.py (FastAPI app; `/api/results` filters missing image URLs)
└── frontend/
    ├── index.html (Vue app mount point only, loads app.js)
    ├── index_old.html (backup of current implementation)
    ├── app.js (Vue app entry, shared runtime, and vue3-sfc-loader configuration)
    ├── components/
    │   ├── TabManager.vue
    │   ├── RunExperimentTab/
    │   │   ├── RunForm.vue
    │   │   ├── RunStatus.vue
    │   │   └── index.vue
    │   ├── ConfigurationTab/
    │   │   ├── ModelConfig.vue
    │   │   ├── HardwareConfig.vue
    │   │   └── index.vue
    │   ├── ResultsTab/
    │   │   ├── CsvSelector.vue
    │   │   ├── ResultsFilter.vue
    │   │   ├── ResultsTable.vue
    │   │   └── index.vue
    │   └── VisualizationsTab/
    │       ├── ThroughputCharts.vue
    │       └── index.vue
    └── styles/
        └── main.css
```

## Testing Strategy

### Manual Testing Checklist
- All tabs switch without errors
- Run form submits and returns run_id
- Progress polling updates UI every 3 seconds (requires LOG_LEVEL=INFO)
- CSV selector loads correct file or shows 404 for missing files
- Results filter works within loaded CSV (die range, throughput range)
- Sortable table columns maintain order
- View Charts button enables with 1+ selected row
- Selected row opens Visualizations tab and seeds `total_die`
- Chart images render for available data
- Mobile/tablet responsiveness check

### Regression Prevention
- Keep current `index.html` as backup
- Verify all existing API endpoints still work
- Run simulator CLI to confirm search behavior remains unchanged
- Verify `/api/results` returns only URLs for image files that exist on disk

### Cross-Browser Testing
- Chrome (primary)
- Firefox
- Safari (Mac)

## Implementation Notes

- **Backend mostly unchanged** - CLI orchestration and CSV endpoints are the same; `/api/results` was tightened to filter missing image files
- **No build step required** - Vue 3 via CDN with vue3-sfc-loader
- **Static asset URLs** - FastAPI serves `index.html` at `/` and mounts `webapp/frontend/` at `/static`, so CSS, JS, and the top-level `.vue` loader path should use absolute `/static/...` URLs
- **Runtime wiring** - Shared API/store helpers live in `app.js` and are exposed via `window.LightLLMRuntime` to avoid local JS-module import issues inside browser-compiled SFCs
- **State sharing** - useStore uses singleton pattern (module-level refs) to ensure all components access to same state including CSV selection
- **Progress tracking** - Requires `LOG_LEVEL=INFO` environment variable for phase detection to work
- **CSV schema** - ResultsTable dynamically adapts to whatever columns exist in actual CSV data
- **Visualizations** - Static images only from `/data/images/`, no interactive features in scope
- **Progressive migration** - Tabs can be implemented incrementally if needed

## Known Limitations (Backend Constraints)

Since the backend remains intentionally lightweight, the following features still have limitations:

1. **Progress/Phase Detection:** Only works when backend logs at INFO level. Default WARNING level suppresses phase markers.

2. **Interactive Pipeline Charts:** Not supported - backend returns static PNG images only. Zoom/pan would require:
   - New API endpoint returning raw chart data
   - Frontend chart library with interactive capabilities
   - Or client-side image processing (complex)

3. **Chart Export:** Not supported - would require:
   - New API endpoint to generate images on-demand
   - Or client-side canvas/image manipulation

4. **Results Across Runs:** Not supported - the Results tab can only view and filter within ONE CSV file at a time. To view results from different simulation configurations, users select different CSV files using the CSV selector.

5. **Real-time Results:** Results only available after simulation completes. Live progress comes from log parsing, not a dedicated progress API.

These limitations are acceptable for the initial refactor and can be addressed in future backend work.

## Future Enhancement Opportunities

If backend changes become possible, consider:

1. **Results listing API** - Endpoint to list all available CSV files with metadata
2. **Results query API** - Query across multiple CSV files or return aggregated results
3. **Chart data API** - Return raw chart data (JSON) instead of static images
4. **Progress API** - Dedicated endpoint for real-time progress updates with phase information
5. **Export API** - Endpoint to generate PNG/SVG charts on-demand with custom parameters

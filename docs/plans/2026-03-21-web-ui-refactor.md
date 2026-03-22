# Web UI Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor web frontend to Vue.js 3 SPA with tab-based navigation, real-time progress, filterable results (within selected CSV), and static image visualizations while keeping the FastAPI backend lightweight.

**Architecture:** Vue 3 (Composition API) via CDN with browser-based SFC loader, component-based structure, shared runtime in `app.js`, and reactive state management.

**Tech Stack:** Vue 3, vue3-sfc-loader (browser-based SFC compilation), localStorage, FastAPI

**Build Approach:** Vue Single-File Components (.vue) compiled on-demand in browser using vue3-sfc-loader. No package.json or npm build step required.

**Results Tab Design:** Filter and sort within a SINGLE selected CSV file. The backend does not provide a way to query available CSV files or filter across runs/models/devices. Each simulation produces CSV files named by pattern `{DeviceType.name}-{ModelType.name}-tpot{tpot}-kv_len{kvLen}.csv`.

## Implementation Status

The frontend described in this plan has now been implemented on branch `feature/web-ui-refactor-impl`.

The high-level direction is still correct, but the shipped implementation differs from the original task-by-task plan in a few important ways:

- Shared frontend runtime lives in `webapp/frontend/app.js`, exposed as `window.LightLLMRuntime`
- Separate `webapp/frontend/composables/useApi.js`, `useStore.js`, and `useLocalStorage.js` were not kept in the final implementation
- Run and Configuration use responsive multi-column desktop layouts rather than strictly single-column layouts
- `RunStatus` stays mounted beside `RunForm`, so progress remains visible without tab switching
- FastAPI received one small backend adjustment: `/api/results` now filters out image URLs that do not exist on disk

## Historical Note

The detailed tasks below are preserved as the original implementation plan. They are useful for context, but they still mention intermediate files and structures that were simplified during implementation. Treat the actual code under `webapp/frontend/` and `webapp/backend/main.py` as the source of truth.

---

## Prerequisites

### Task 0: Create Worktree and Verify Environment

**Step 1: Create git worktree**

```bash
git worktree add ../llm-ui-refactor -b feature/web-ui-refactor
cd ../llm-ui-refactor
```

**Note:** All file edits and commits below should happen in the worktree checkout (`../llm-ui-refactor`). If you prefer absolute paths, use the corresponding worktree path instead of the original clone.

**Step 2: Verify backend starts**

```bash
cd ../llm-ui-refactor
uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 &
```

Expected: "Uvicorn running on http://127.0.0.1:8000" or similar

**Step 3: Access current UI to verify baseline**

Visit: http://127.0.0.1:8000

Expected: Current 4-panel layout loads successfully

**Step 4: Kill backend process**

```bash
pkill -f uvicorn
```

**Step 5: Commit worktree setup**

```bash
cd ../llm-ui-refactor
git add -A
git commit -m "chore: setup worktree for web UI refactor"
```

---

## Foundation Layer

### Task 1: Backup Current Implementation

**Files:**
- Rename: `webapp/frontend/index.html` → `webapp/frontend/index_old.html`

**Step 1: Backup current HTML**

```bash
cd ../llm-ui-refactor
mv webapp/frontend/index.html webapp/frontend/index_old.html
```

**Step 2: Verify backup exists**

```bash
ls -la webapp/frontend/ | grep index_old.html
```

Expected: `index_old.html` file listed

**Step 3: Commit**

```bash
git add webapp/frontend/index_old.html
git commit -m "refactor: backup current HTML implementation"
```

---

### Task 2: Create New HTML Entry Point

**Files:**
- Create: `webapp/frontend/index.html`

**Step 1: Write Vue 3 CDN-based HTML (no inline app)**

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Light LLM Simulator</title>
  <link rel="stylesheet" href="/static/styles/main.css">
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/static/app.js"></script>
</body>
</html>
```

**Note:** The backend serves `index.html` at `/`, but frontend files under `webapp/frontend/` are mounted at `/static`. Use absolute `/static/...` URLs for CSS, `app.js`, and the top-level SFC loader entrypoint.

**Step 2: Create styles directory**

```bash
mkdir -p webapp/frontend/styles
```

**Step 3: Commit**

```bash
git add webapp/frontend/
git commit -m "feat: create Vue 3 HTML entry point"
```

---

### Task 3: Create CSS Foundation

**Files:**
- Create: `webapp/frontend/styles/main.css`

**Step 1: Write base styles**

```css
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background-color: #f5f5f7;
  color: #1f2937;
  padding: 20px;
}

#app {
  max-width: 1400px;
  margin: 0 auto;
}

/* Tab Navigation */
.tab-bar {
  display: flex;
  gap: 4px;
  margin-bottom: 20px;
  border-bottom: 2px solid #e5e7eb;
  padding-bottom: 2px;
}

.tab-btn {
  padding: 12px 24px;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  color: #64748b;
  border-radius: 6px 6px 0 0;
  transition: all 0.2s;
}

.tab-btn:hover {
  background-color: #f3f4f6;
}

.tab-btn.active {
  background-color: #3b82f6;
  color: white;
}

/* Tab Content */
.tab-content {
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  min-height: 600px;
}

/* Forms */
.field {
  margin-bottom: 16px;
}

.field label {
  display: block;
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 6px;
  color: #374151;
}

.field input,
.field select {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 14px;
  transition: border-color 0.2s;
}

.field input:focus,
.field select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
}

.btn {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  transition: all 0.2s;
}

.btn-primary {
  background-color: #3b82f6;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: #2563eb;
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Progress */
.progress-bar {
  height: 8px;
  background-color: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
  margin: 16px 0;
}

.progress-fill {
  height: 100%;
  background-color: #10b981;
  transition: width 0.3s ease;
}

/* Tables */
.data-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 16px;
}

.data-table th {
  text-align: left;
  padding: 12px;
  background-color: #f8fafc;
  border-bottom: 2px solid #e5e7eb;
  cursor: pointer;
  user-select: none;
}

.data-table th:hover {
  background-color: #f1f5f9;
}

.data-table td {
  padding: 12px;
  border-bottom: 1px solid #e5e7eb;
}

.data-table tr:hover {
  background-color: #f8fafc;
}

/* Loading States */
.loading {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
  color: #64748b;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

**Step 2: Commit**

```bash
git add webapp/frontend/styles/main.css
git commit -m "feat: add CSS foundation"
```

---

## Composables Layer

### Task 4: Create API Composable

**Files:**
- Create: `webapp/frontend/composables/useApi.js`

**Step 1: Write API wrapper**

```javascript
const API_BASE = '/api';

export function useApi() {
  const fetchJson = async (url, options = {}) => {
    try {
      const response = await fetch(`${API_BASE}${url}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options
      });
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('API call failed:', error);
      throw error;
    }
  };

  return {
    // Run simulation
    startRun: async (payload) => {
      return fetchJson('/run', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
    },

    // Get run status
    getStatus: async (runId) => {
      return fetchJson(`/status/${runId}`);
    },

    // Get logs
    getLogs: async (runId) => {
      return fetchJson(`/logs/${runId}`);
    },

    // Get model config
    getModelConfig: async (modelType) => {
      const params = new URLSearchParams({ model_type: modelType });
      return fetchJson(`/model_config?${params}`);
    },

    // Get hardware config
    getHardwareConfig: async (deviceType) => {
      const params = new URLSearchParams({ device_type: deviceType });
      return fetchJson(`/hardware_config?${params}`);
    },

    // Get results images
    getResults: async (params) => {
      const query = new URLSearchParams(params);
      return fetchJson(`/results?${query}`);
    },

    // Fetch rows from a single CSV file
    fetchCsvResults: async (params) => {
      const query = new URLSearchParams(params);
      return fetchJson(`/fetch_csv_results?${query}`);
    }
  };
}
```

**Step 2: Commit**

```bash
git add webapp/frontend/composables/useApi.js
git commit -m "feat: add API composable"
```

---

### Task 5: Create Store Composable (Singleton Pattern)

**Files:**
- Create: `webapp/frontend/composables/useStore.js`

**Step 1: Write store for global state with singleton pattern**

```javascript
import { ref } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';

// IMPORTANT: Module-level refs for shared state across all components
const STORAGE_KEY_TAB = 'llm-sim-current-tab';
const STORAGE_KEY_HISTORY = 'llm-sim-run-history';
const STORAGE_KEY_CSV_SELECTION = 'llm-sim-csv-selection';

// Singleton state - declared outside of function so all calls share the same refs
const currentTab = ref(localStorage.getItem(STORAGE_KEY_TAB) || 'run');
const runHistory = ref(JSON.parse(localStorage.getItem(STORAGE_KEY_HISTORY) || '[]'));
const csvSelection = ref({
  servingMode: 'AFD',
  deviceType: 'ASCENDA3_Pod',
  modelType: 'DEEPSEEK_V3',
  tpot: 50,
  kvLen: 4096,
  microBatchNum: 3,
  totalDie: 128
});
const tabs = ['run', 'config', 'results', 'visualizations'];

export function useStore() {
  const setTab = (tab) => {
    currentTab.value = tab;
    localStorage.setItem(STORAGE_KEY_TAB, tab);
  };

  const addRun = (runId, params) => {
    runHistory.value.unshift({ id: runId, params, timestamp: Date.now() });
    if (runHistory.value.length > 20) {
      runHistory.value = runHistory.value.slice(0, 20);
    }
    localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(runHistory.value));
  };

  const clearHistory = () => {
    runHistory.value = [];
    localStorage.removeItem(STORAGE_KEY_HISTORY);
  };

  const setCsvSelection = (selection) => {
    csvSelection.value = { ...csvSelection.value, ...selection };
    localStorage.setItem(STORAGE_KEY_CSV_SELECTION, JSON.stringify(csvSelection.value));
  };

  const getCsvSelection = () => {
    const stored = localStorage.getItem(STORAGE_KEY_CSV_SELECTION);
    if (stored) {
      csvSelection.value = JSON.parse(stored);
    }
    return csvSelection.value;
  };

  return {
    // Tabs (shared refs)
    currentTab,
    tabs,
    setTab,
    // History (shared refs)
    runHistory,
    addRun,
    clearHistory,
    // CSV selection (shared refs)
    csvSelection,
    setCsvSelection,
    getCsvSelection
  };
}
```

**Step 2: Commit**

```bash
git add webapp/frontend/composables/useStore.js
git commit -m "feat: add store composable with singleton pattern and CSV selection"
```

---

### Task 6: Create LocalStorage Composable

**Files:**
- Create: `webapp/frontend/composables/useLocalStorage.js`

**Step 1: Write localStorage utilities**

```javascript
import { ref, watch } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';

export function useLocalStorage(key, defaultValue = null) {
  const stored = localStorage.getItem(key);
  const state = ref(stored ? JSON.parse(stored) : defaultValue);

  watch(state, (newValue) => {
    localStorage.setItem(key, JSON.stringify(newValue));
  });

  return state;
}

export function useSessionStorage(key, defaultValue = null) {
  const stored = sessionStorage.getItem(key);
  const state = ref(stored ? JSON.parse(stored) : defaultValue);

  watch(state, (newValue) => {
    sessionStorage.setItem(key, JSON.stringify(newValue));
  });

  return state;
}
```

**Step 2: Commit**

```bash
git add webapp/frontend/composables/useLocalStorage.js
git commit -m "feat: add localStorage composable"
```

---

## Components - Core

### Task 7: Create TabManager Component

**Files:**
- Create: `webapp/frontend/components/TabManager.vue`

**Step 1: Write tab navigation component with registered child components**

```html
<template>
  <div>
    <div class="tab-bar">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        :class="['tab-btn', { active: currentTab === tab.id }]"
        @click="setTab(tab.id)"
      >
        {{ tab.label }}
      </button>
    </div>
    <div class="tab-content">
      <RunExperimentTab v-if="currentTab === 'run'" />
      <ConfigurationTab v-if="currentTab === 'config'" />
      <ResultsTab v-if="currentTab === 'results'" />
      <VisualizationsTab v-if="currentTab === 'visualizations'" />
    </div>
  </div>
</template>

<script>
import { useStore } from '../composables/useStore.js';
import RunExperimentTab from './RunExperimentTab/index.vue';
import ConfigurationTab from './ConfigurationTab/index.vue';
import ResultsTab from './ResultsTab/index.vue';
import VisualizationsTab from './VisualizationsTab/index.vue';

export default {
  components: {
    RunExperimentTab,
    ConfigurationTab,
    ResultsTab,
    VisualizationsTab
  },
  setup() {
    const { currentTab, setTab } = useStore();

    const tabs = [
      { id: 'run', label: 'Run Experiment' },
      { id: 'config', label: 'Configuration' },
      { id: 'results', label: 'Results' },
      { id: 'visualizations', label: 'Visualizations' }
    ];

    return { currentTab, setTab, tabs };
  }
}
</script>
```

**Step 2: Commit**

```bash
git add webapp/frontend/components/TabManager.vue
git commit -m "feat: add TabManager component with registered child components"
```

---

### Task 8: Create RunForm Component

**Files:**
- Create: `webapp/frontend/components/RunExperimentTab/RunForm.vue`

**Step 1: Write parameter form component**

```html
<template>
  <div class="run-form">
    <h3>Simulation Parameters</h3>

    <!-- Basic Parameters -->
    <div class="section">
      <h4>Basic</h4>
      <div class="field">
        <label>Serving Mode</label>
        <select v-model="form.serving_mode">
          <option value="AFD">AFD</option>
          <option value="DeepEP">DeepEP</option>
        </select>
      </div>
      <div class="field">
        <label>Model Type</label>
        <select v-model="form.model_type">
          <option v-for="model in modelOptions" :key="model.value" :value="model.value">
            {{ model.label }}
          </option>
        </select>
      </div>
      <div class="field">
        <label>Device Type</label>
        <select v-model="form.device_type">
          <option v-for="device in deviceOptions" :key="device.value" :value="device.value">
            {{ device.label }}
          </option>
        </select>
      </div>
    </div>

    <!-- Search Range -->
    <div class="section">
      <h4>Search Range</h4>
      <div class="field-row">
        <div class="field">
          <label>Min Attention Batch Size</label>
          <input type="number" v-model.number="form.min_attn_bs" min="2" />
        </div>
        <div class="field">
          <label>Max Attention Batch Size</label>
          <input type="number" v-model.number="form.max_attn_bs" min="2" />
        </div>
      </div>
      <div class="field-row">
        <div class="field">
          <label>Min Die</label>
          <input type="number" v-model.number="form.min_die" min="16" />
        </div>
        <div class="field">
          <label>Max Die</label>
          <input type="number" v-model.number="form.max_die" min="16" />
        </div>
      </div>
      <div class="field">
        <label>Die Step</label>
        <input type="number" v-model.number="form.die_step" min="1" />
      </div>
    </div>

    <!-- Simulation Targets -->
    <div class="section">
      <h4>Simulation Targets</h4>
      <div class="field">
        <label>TPOT Targets (comma separated)</label>
        <input v-model="tpotInput" placeholder="20,50,70,100,150" />
      </div>
      <div class="field">
        <label>KV Length (comma separated)</label>
        <input v-model="kvLenInput" placeholder="2048,4096,8192" />
      </div>
      <div class="field">
        <label>Micro Batch Numbers (comma separated)</label>
        <input v-model="mbnInput" placeholder="2,3" />
      </div>
    </div>

    <!-- Advanced -->
    <details class="section">
      <summary><h4>Advanced</h4></summary>
      <div class="field-row">
        <div class="field">
          <label>Next N (Multi-Token Prediction)</label>
          <input type="number" v-model.number="form.next_n" min="1" />
        </div>
        <div class="field">
          <label>Multi-Token Ratio</label>
          <input type="number" v-model.number="form.multi_token_ratio" min="0" max="1" step="0.01" />
        </div>
      </div>
      <div class="field-row">
        <div class="field">
          <label>Attention Tensor Parallel</label>
          <input type="number" v-model.number="form.attn_tensor_parallel" min="1" />
        </div>
        <div class="field">
          <label>FFN Tensor Parallel</label>
          <input type="number" v-model.number="form.ffn_tensor_parallel" min="1" />
        </div>
      </div>
    </details>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <button class="btn btn-primary" @click="handleSubmit" :disabled="isSubmitting">
      {{ isSubmitting ? 'Starting...' : 'Start Run' }}
    </button>
  </div>
</template>

<script>
import { ref } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useApi } from '../../composables/useApi.js';
import { useStore } from '../../composables/useStore.js';

const MODEL_OPTIONS = [
  { value: 'deepseek-ai/DeepSeek-V3', label: 'DeepSeek V3' },
  { value: 'Qwen/Qwen3-235B-A22B', label: 'Qwen3-235B-A22B' },
  { value: 'deepseek-ai/DeepSeek-V2-Lite', label: 'DeepSeek V2 Lite' }
];

const DEVICE_OPTIONS = [
  { value: 'Ascend_910b2', label: 'Ascend 910B2' },
  { value: 'Ascend_910b3', label: 'Ascend 910B3' },
  { value: 'Ascend_910b4', label: 'Ascend 910B4' },
  { value: 'Ascend_A3Pod', label: 'Ascend A3Pod' },
  { value: 'Ascend_David121', label: 'Ascend David121' },
  { value: 'Ascend_David120', label: 'Ascend David120' },
  { value: 'Nvidia_A100_SXM', label: 'Nvidia A100 SXM' },
  { value: 'Nvidia_H100_SXM', label: 'Nvidia H100 SXM' }
];

export default {
  setup() {
    const api = useApi();
    const { setTab, addRun: addToHistory } = useStore();

    const form = ref({
      serving_mode: 'AFD',
      model_type: MODEL_OPTIONS[0].value,
      device_type: DEVICE_OPTIONS[0].value,
      min_attn_bs: 2,
      max_attn_bs: 1000,
      min_die: 16,
      max_die: 768,
      die_step: 16,
      next_n: 1,
      multi_token_ratio: 0.7,
      attn_tensor_parallel: 1,
      ffn_tensor_parallel: 1
    });

    const tpotInput = ref('50');
    const kvLenInput = ref('4096');
    const mbnInput = ref('3');

    const parseList = (str) => {
      if (!str) return [];
      return str.split(',').map(s => s.trim()).filter(Boolean).map(Number);
    };

    const isSubmitting = ref(false);
    const error = ref(null);

    const handleSubmit = async () => {
      isSubmitting.value = true;
      error.value = null;

      try {
        const payload = {
          ...form.value,
          tpot: parseList(tpotInput.value),
          kv_len: parseList(kvLenInput.value),
          micro_batch_num: parseList(mbnInput.value)
        };

        const result = await api.startRun(payload);
        addToHistory(result.run_id, payload);
        setTab('results');
      } catch (err) {
        error.value = err.message;
      } finally {
        isSubmitting.value = false;
      }
    };

    return {
      form,
      tpotInput,
      kvLenInput,
      mbnInput,
      modelOptions: MODEL_OPTIONS,
      deviceOptions: DEVICE_OPTIONS,
      isSubmitting,
      error,
      handleSubmit,
      parseList
    };
  }
}
</script>

<style scoped>
.run-form {
  max-width: 600px;
}

.section {
  margin-bottom: 24px;
  padding-bottom: 24px;
  border-bottom: 1px solid #e5e7eb;
}

.section:last-child {
  border-bottom: none;
}

h3 {
  margin-bottom: 20px;
  color: #1f2937;
}

h4 {
  margin-bottom: 12px;
  color: #374151;
  font-size: 14px;
}

.field-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

details {
  cursor: pointer;
}

details summary {
  list-style: none;
  outline: none;
}

details summary h4 {
  display: inline-block;
}

details summary::-webkit-details-marker {
  display: none;
}

details summary::after {
  content: ' ▼';
  font-size: 12px;
  margin-left: 8px;
}

details[open] summary::after {
  content: ' ▲';
}

.error-message {
  background-color: #fee2e2;
  color: #991b1b;
  padding: 12px;
  border-radius: 6px;
  margin-top: 16px;
}
</style>
```

**Step 2: Commit**

```bash
git add webapp/frontend/components/RunExperimentTab/RunForm.vue
git commit -m "feat: add RunForm component with fixed naming"
```

---

### Task 9: Create RunStatus Component (FIXED: react to active run changes)

**Files:**
- Create: `webapp/frontend/components/RunExperimentTab/RunStatus.vue`

**Step 1: Write progress/status component**

```html
<template>
  <div class="run-status">
    <h3>Simulation Progress</h3>

    <div v-if="runId" class="status-card">
      <div class="status-header">
        <span class="run-id">Run ID: {{ runId }}</span>
        <button class="btn btn-sm" @click="stopPolling" v-if="!isDone">
          Cancel
        </button>
      </div>

      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
      </div>

      <div class="phase-indicator">{{ currentPhase }}</div>

      <div class="log-container">
        <div v-for="(log, idx) in logs" :key="idx" class="log-line">
          {{ log }}
        </div>
        <div v-if="logs.length === 0" class="log-placeholder">Waiting for logs...</div>
      </div>

      <div v-if="isDone" class="done-message">
        ✓ Simulation Complete
      </div>
    </div>

    <div v-else class="no-run">
      No active simulation
    </div>
  </div>
</template>

<script>
import { ref, computed, watch, onUnmounted } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useApi } from '../../composables/useApi.js';
import { useStore } from '../../composables/useStore.js';

export default {
  setup() {
    const api = useApi();
    const { runHistory } = useStore();
    const activeRunId = computed(() => runHistory.value[0]?.id || null);

    const runId = ref(null);
    const isDone = ref(false);
    const logs = ref([]);
    const currentPhase = ref('Initializing');
    const progressPercent = ref(0);

    let pollInterval = null;

    const phases = [
      { pattern: /search_attn_bs/i, label: 'Finding optimal attention batch size' },
      { pattern: /searching/i, label: 'Searching die allocations' },
      { pattern: /throughput/i, label: 'Generating visualizations' }
    ];

    const detectPhase = (logText) => {
      for (const phase of phases) {
        if (phase.pattern.test(logText)) {
          currentPhase.value = phase.label;
          return;
        }
      }
    };

    const pollLogs = async () => {
      if (!runId.value) return;

      try {
        const data = await api.getLogs(runId.value);
        logs.value = (data.log || '').split('\n').filter(Boolean);

        // Detect current phase
        if (logs.value.length > 0) {
          detectPhase(logs.value[logs.value.length - 1]);
        }

        // Update progress based on log length
        progressPercent.value = Math.min(logs.value.length * 2, 95);

      } catch (err) {
        console.error('Polling failed:', err);
      }
    };

    const pollStatus = async () => {
      if (!runId.value) return;

      try {
        const data = await api.getStatus(runId.value);

        if (data.done) {
          isDone.value = true;
          progressPercent.value = 100;
          currentPhase.value = 'Complete';
          stopPolling();
        }
      } catch (err) {
        console.error('Status polling failed:', err);
      }
    };

    const startPolling = () => {
      stopPolling();
      if (!runId.value) return;

      pollLogs();
      pollStatus();
      pollInterval = setInterval(() => {
        pollLogs();
        pollStatus();
      }, 3000);
    };

    const stopPolling = () => {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    };

    watch(activeRunId, (nextRunId) => {
      stopPolling();
      runId.value = nextRunId;
      logs.value = [];
      isDone.value = false;
      currentPhase.value = nextRunId ? 'Initializing' : 'Idle';
      progressPercent.value = 0;

      if (nextRunId) {
        startPolling();
      }
    }, { immediate: true });

    onUnmounted(() => {
      stopPolling();
    });

    return {
      runId,
      isDone,
      logs,
      currentPhase,
      progressPercent,
      stopPolling
    };
  }
}
</script>

<style scoped>
.run-status {
  max-width: 600px;
}

.status-card {
  background: #f8fafc;
  border-radius: 8px;
  padding: 16px;
  margin-top: 16px;
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.run-id {
  font-weight: 600;
  color: #374151;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 12px;
}

.progress-bar {
  height: 12px;
  background-color: #e5e7eb;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 12px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #10b981);
  transition: width 0.3s ease;
}

.phase-indicator {
  font-size: 13px;
  color: #64748b;
  margin-bottom: 12px;
}

.log-container {
  background: #1f2937;
  color: #f0fdf4;
  border-radius: 6px;
  padding: 12px;
  max-height: 200px;
  overflow-y: auto;
  font-family: 'Monaco', 'Courier New', monospace;
  font-size: 12px;
}

.log-line {
  padding: 4px 0;
  border-bottom: 1px solid #374151;
}

.log-placeholder {
  color: #64748b;
  text-align: center;
  padding: 24px;
}

.done-message {
  text-align: center;
  padding: 16px;
  background-color: #d1fae5;
  color: #065f46;
  border-radius: 6px;
  font-weight: 600;
}

.no-run {
  text-align: center;
  padding: 40px;
  color: #9ca3af;
}
</style>
```

**Note:** `RunStatus` must watch `activeRunId` so a newly started run replaces the previous run ID, clears old logs, and restarts polling.

**Step 2: Commit**

```bash
git add webapp/frontend/components/RunExperimentTab/RunStatus.vue
git commit -m "feat: add RunStatus component with reactive polling"
```

---

### Task 10: Create RunExperimentTab Container

**Files:**
- Create: `webapp/frontend/components/RunExperimentTab/index.vue`

**Step 1: Write container component**

```html
<template>
  <div class="run-experiment-tab">
    <RunForm />
    <RunStatus v-if="activeRunId" />
  </div>
</template>

<script>
import { computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useStore } from '../../composables/useStore.js';

import RunForm from './RunForm.vue';
import RunStatus from './RunStatus.vue';

export default {
  components: { RunForm, RunStatus },
  setup() {
    const { runHistory } = useStore();
    const activeRunId = computed(() => runHistory.value[0]?.id);

    return { activeRunId };
  }
}
</script>

<style scoped>
.run-experiment-tab {
  max-width: 600px;
}
</style>
```

**Step 2: Commit**

```bash
git add webapp/frontend/components/RunExperimentTab/index.vue
git commit -m "feat: add RunExperimentTab container"
```

---

### Task 11: Create ConfigurationTab Components

**Files:**
- Create: `webapp/frontend/components/ConfigurationTab/ModelConfig.vue`
- Create: `webapp/frontend/components/ConfigurationTab/HardwareConfig.vue`
- Create: `webapp/frontend/components/ConfigurationTab/index.vue`

**Step 1: Write ModelConfig component**

```html
<template>
  <div class="config-section">
    <h3>Model Configuration</h3>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>

    <div v-else-if="error" class="error">
      {{ error }}
    </div>

    <div v-else-if="config" class="config-grid">
      <div class="config-item">
        <span class="label">Model Type</span>
        <span class="value">{{ config.model_type || 'N/A' }}</span>
      </div>
      <div class="config-item">
        <span class="label">Hidden Size</span>
        <span class="value">{{ config.hidden_size }}</span>
      </div>
      <div class="config-item">
        <span class="label">Layers</span>
        <span class="value">{{ config.num_layers }}</span>
      </div>
      <div class="config-item">
        <span class="label">Attention Heads</span>
        <span class="value">{{ config.num_attention_heads || config.num_heads }}</span>
      </div>
      <div class="config-item">
        <span class="label">KV Heads</span>
        <span class="value">{{ config.kv_heads }}</span>
      </div>
      <div class="config-item">
        <span class="label">Head Size</span>
        <span class="value">{{ config.head_size }}</span>
      </div>
      <div class="config-item">
        <span class="label">Model Size</span>
        <span class="value">{{ config.model_size_b }}B</span>
      </div>
      <div class="config-item">
        <span class="label">Intermediate Size</span>
        <span class="value">{{ config.intermediate_size }}</span>
      </div>
      <div class="config-item" v-if="config.kv_lora_rank">
        <span class="label">KV LoRA Rank</span>
        <span class="value">{{ config.kv_lora_rank }}</span>
      </div>
      <div class="config-item" v-if="config.n_routed_experts">
        <span class="label">Routed Experts</span>
        <span class="value">{{ config.n_routed_experts }}</span>
      </div>
      <div class="config-item" v-if="config.num_experts_per_tok">
        <span class="label">Experts Per Token</span>
        <span class="value">{{ config.num_experts_per_tok }}</span>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useApi } from '../../composables/useApi.js';

export default {
  setup() {
    const api = useApi();

    const config = ref(null);
    const loading = ref(true);
    const error = ref(null);

    const loadConfig = async () => {
      loading.value = true;
      error.value = null;

      try {
        config.value = await api.getModelConfig('deepseek-ai/DeepSeek-V3');
      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    };

    onMounted(loadConfig);

    return { config, loading, error };
  }
}
</script>

<style scoped>
.config-section {
  max-width: 600px;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.config-item {
  background: #f8fafc;
  padding: 12px;
  border-radius: 6px;
}

.config-item .label {
  display: block;
  font-size: 12px;
  color: #64748b;
  margin-bottom: 4px;
}

.config-item .value {
  font-weight: 600;
  color: #1f2937;
}
</style>
```

**Step 2: Write HardwareConfig component**

```html
<template>
  <div class="config-section">
    <h3>Hardware Configuration</h3>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>

    <div v-else-if="error" class="error">
      {{ error }}
    </div>

    <div v-else-if="config" class="config-grid">
      <div class="config-item">
        <span class="label">Device Type</span>
        <span class="value">{{ config.device_type || 'N/A' }}</span>
      </div>
      <div class="config-item">
        <span class="label">Dies Per Node</span>
        <span class="value">{{ config.num_dies_per_node }}</span>
      </div>
      <div class="config-item">
        <span class="label">HBM Capacity</span>
        <span class="value">{{ (config.aichip_memory / GB_2_BYTE).toFixed(2) }} GB</span>
      </div>
      <div class="config-item">
        <span class="label">Intra-node Bandwidth</span>
        <span class="value">{{ (config.intra_node_bandwidth / GB_2_BYTE).toFixed(2) }} GB/s</span>
      </div>
      <div class="config-item">
        <span class="label">Inter-node Bandwidth</span>
        <span class="value">{{ (config.inter_node_bandwidth / GB_2_BYTE).toFixed(2) }} GB/s</span>
      </div>
      <div class="config-item">
        <span class="label">Local Memory Bandwidth</span>
        <span class="value">{{ (config.local_memory_bandwidth / TB_2_BYTE).toFixed(2) }} TB/s</span>
      </div>
      <div class="config-item">
        <span class="label">Cube FLOPS (FP16)</span>
        <span class="value">{{ (config.cube_flops_fp16 / TB_2_BYTE).toFixed(2) }} TFLOPS</span>
      </div>
      <div class="config-item">
        <span class="label">Cube FLOPS (INT8)</span>
        <span class="value">{{ (config.cube_flops_int8 / TB_2_BYTE).toFixed(2) }} TFLOPS</span>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useApi } from '../../composables/useApi.js';

const GB_2_BYTE = 1073741824;
const TB_2_BYTE = 1099511627776;

export default {
  setup() {
    const api = useApi();

    const config = ref(null);
    const loading = ref(true);
    const error = ref(null);

    const loadConfig = async () => {
      loading.value = true;
      error.value = null;

      try {
        config.value = await api.getHardwareConfig('Ascend_A3Pod');
      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    };

    onMounted(loadConfig);

    return { config, loading, error, GB_2_BYTE, TB_2_BYTE };
  }
}
</script>

<style scoped>
.config-section {
  max-width: 600px;
  margin-top: 16px;
}
</style>
```

**Step 3: Write container component**

```html
<template>
  <div class="configuration-tab">
    <ModelConfig />
    <HardwareConfig />
  </div>
</template>

<script>
import ModelConfig from './ModelConfig.vue';
import HardwareConfig from './HardwareConfig.vue';

export default {
  components: { ModelConfig, HardwareConfig }
}
</script>
```

**Step 4: Commit**

```bash
git add webapp/frontend/components/ConfigurationTab/
git commit -m "feat: add ConfigurationTab components"
```

---

## Components - Results & Visualizations

### Task 12: Create Results Tab Components (FIXED: CSV selector with enum names, filter reactivity, stable comparison indices)

**Files:**
- Create: `webapp/frontend/components/ResultsTab/CsvSelector.vue`
- Create: `webapp/frontend/components/ResultsTab/ResultsFilter.vue`
- Create: `webapp/frontend/components/ResultsTab/ResultsTable.vue`
- Create: `webapp/frontend/components/ResultsTab/index.vue`

**Step 1: Write CsvSelector component (FIXED: use enum names)**

```html
<template>
  <div class="csv-selector">
    <h4>Select Results CSV</h4>
    <div class="selector-grid">
      <div class="field">
        <label>Serving Mode</label>
        <select v-model="selection.servingMode">
          <option value="AFD">AFD</option>
          <option value="DeepEP">DeepEP</option>
        </select>
      </div>

      <div class="field">
        <label>Device Type</label>
        <select v-model="selection.deviceType">
          <option v-for="device in deviceOptions" :key="device.value" :value="device.value">
            {{ device.label }}
          </option>
        </select>
      </div>

      <div class="field">
        <label>Model Type</label>
        <select v-model="selection.modelType">
          <option v-for="model in modelOptions" :key="model.value" :value="model.value">
            {{ model.label }}
          </option>
        </select>
      </div>

      <div class="field">
        <label>TPOT</label>
        <select v-model="selection.tpot">
          <option v-for="tp in tpotOptions" :key="tp" :value="tp">
            {{ tp }}ms
          </option>
        </select>
      </div>

      <div class="field">
        <label>KV Length</label>
        <select v-model="selection.kvLen">
          <option v-for="kv in kvLenOptions" :key="kv" :value="kv">
            {{ kv }}
          </option>
        </select>
      </div>

      <div class="field">
        <label>Micro Batch Number</label>
        <select v-model="selection.microBatchNum">
          <option v-for="mbn in mbnOptions" :key="mbn" :value="mbn">
            MBN {{ mbn }}
          </option>
        </select>
      </div>
    </div>

    <button class="btn btn-primary" @click="handleLoadCsv" :disabled="isLoading">
      {{ isLoading ? 'Loading...' : 'Load Results' }}
    </button>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>
  </div>
</template>

<script>
import { ref } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useApi } from '../../composables/useApi.js';
import { useStore } from '../../composables/useStore.js';

// IMPORTANT: Use exact Enum.name values because backend uses them literally in filenames
const MODEL_OPTIONS = [
  { value: 'DEEPSEEK_V3', label: 'DeepSeek V3' },
  { value: 'QWEN3_235B', label: 'Qwen3-235B-A22B' },
  { value: 'DEEPSEEK_V2_LITE', label: 'DeepSeek V2 Lite' }
];

const DEVICE_OPTIONS = [
  { value: 'ASCEND910B2', label: 'Ascend 910B2' },
  { value: 'ASCEND910B3', label: 'Ascend 910B3' },
  { value: 'ASCEND910B4', label: 'Ascend 910B4' },
  { value: 'ASCENDA3_Pod', label: 'Ascend A3Pod' },
  { value: 'ASCENDDAVID121', label: 'Ascend David121' },
  { value: 'ASCENDDAVID120', label: 'Ascend David120' },
  { value: 'NvidiaA100SXM', label: 'Nvidia A100 SXM' },
  { value: 'NvidiaH100SXM', label: 'Nvidia H100 SXM' }
];

const TPOT_OPTIONS = [20, 50, 70, 100, 150];
const KV_LEN_OPTIONS = [2048, 4096, 8192, 16384, 131072];
const MBN_OPTIONS = [2, 3];

export default {
  setup(props, { emit }) {
    const api = useApi();
    const store = useStore();

    // Initialize selection from store
    const selection = ref({ ...store.getCsvSelection() });

    const isLoading = ref(false);
    const error = ref(null);

    const deviceOptions = DEVICE_OPTIONS;
    const modelOptions = MODEL_OPTIONS;
    const tpotOptions = TPOT_OPTIONS;
    const kvLenOptions = KV_LEN_OPTIONS;
    const mbnOptions = MBN_OPTIONS;

    const handleLoadCsv = async () => {
      isLoading.value = true;
      error.value = null;

      try {
        const data = await api.fetchCsvResults({
          serving_mode: selection.value.servingMode,
          device_type: selection.value.deviceType,
          model_type: selection.value.modelType,
          tpot: selection.value.tpot,
          kv_len: selection.value.kvLen,
          micro_batch_num: selection.value.microBatchNum
        });

        // Update store
        store.setCsvSelection(selection.value);

        // Emit loaded data to parent
        emit('csv-loaded', data);
      } catch (err) {
        // 404 is expected if CSV doesn't exist yet
        if (err.message.includes('404') || err.message.includes('not found')) {
          error.value = 'CSV file not found. Run a simulation first, or check that CSV files exist in data/afd/ or data/deepep/.';
        } else {
          error.value = err.message;
        }
      } finally {
        isLoading.value = false;
      }
    };

    return {
      selection,
      isLoading,
      error,
      deviceOptions,
      modelOptions,
      tpotOptions,
      kvLenOptions,
      mbnOptions,
      handleLoadCsv
    };
  },
  emits: ['csv-loaded']
}
</script>

<style scoped>
.csv-selector {
  background: #f8fafc;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 16px;
}

h4 {
  margin-bottom: 12px;
  color: #374151;
}

.selector-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.error-message {
  background-color: #fee2e2;
  color: #991b1b;
  padding: 12px;
  border-radius: 6px;
  margin-top: 12px;
}
</style>
```

**Step 2: Write ResultsFilter component**

```html
<template>
  <div class="results-filter">
    <div class="filter-header">
      <h4>Filters</h4>
      <button class="btn btn-sm" @click="resetFilters">Reset</button>
    </div>

    <div class="filter-grid">
      <div class="field">
        <label>Min Total Die</label>
        <input type="number" v-model.number="filters.min_die" min="0" />
      </div>

      <div class="field">
        <label>Max Total Die</label>
        <input type="number" v-model.number="filters.max_die" min="0" />
      </div>

      <div class="field">
        <label>Min Throughput</label>
        <input type="number" v-model.number="filters.min_throughput" step="0.01" />
      </div>

      <div class="field">
        <label>Max Throughput</label>
        <input type="number" v-model.number="filters.max_throughput" step="0.01" />
      </div>
    </div>
  </div>
</template>

<script>
import { ref, watch } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';

export default {
  setup(props, { emit }) {
    const filters = ref({
      min_die: null,
      max_die: null,
      min_throughput: null,
      max_throughput: null
    });

    const resetFilters = () => {
      filters.value = {
        min_die: null,
        max_die: null,
        min_throughput: null,
        max_throughput: null
      };
    };

    // Emit filter change whenever filters change
    watch(filters, () => {
      emit('filter-change', filters.value);
    }, { deep: true });

    return {
      filters,
      resetFilters
    };
  },
  emits: ['filter-change']
}
</script>

<style scoped>
.results-filter {
  background: #f8fafc;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 16px;
}

.filter-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.filter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
}
</style>
```

**Step 3: Write ResultsTable component (FIXED: reactive filtering, stable comparison indices)**

```html
<template>
  <div class="results-table-container">
    <div v-if="data.length === 0" class="empty-state">
      No results to display. Load a CSV file first.
    </div>

    <div v-else class="table-wrapper">
      <table class="data-table">
        <thead>
          <tr>
            <th class="select-col">
              <input type="checkbox" :checked="allSelected" @change="toggleAll" />
            </th>
            <!-- Dynamic columns from actual CSV data -->
            <th v-for="col in columns" :key="col" @click="sortBy(col)">
              {{ formatColumnName(col) }} {{ sortCol === col ? sortDirIcon(col) : '' }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, idx) in paginatedData" :key="row.__stableId || idx">
            <td class="select-col">
              <input type="checkbox" :checked="selectedRows.has(row.__stableId)" @change="toggleSelect(row.__stableId)" />
            </td>
            <td v-for="col in columns" :key="col">
              {{ formatValue(row[col]) }}
            </td>
          </tr>
        </tbody>
      </table>

      <div class="pagination">
        <button class="btn btn-sm" :disabled="page === 1" @click="setPage(page - 1)">Previous</button>
        <span>Page {{ page }} of {{ totalPages }}</span>
        <button class="btn btn-sm" :disabled="page === totalPages" @click="setPage(page + 1)">Next</button>
      </div>

      <button class="btn btn-primary compare-btn" :disabled="selectedRows.size < 1" @click="emitCompare">
        View Charts ({{ selectedRows.size }})
      </button>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';

const PAGE_SIZE = 20;

export default {
  setup(props, { emit }) {
    const page = ref(1);
    const sortCol = ref('total_die');
    const sortDir = ref('desc');
    const selectedRows = ref(new Set());

    // Get actual columns from data
    const columns = computed(() => {
      if (props.data.length === 0) return [];
      return Object.keys(props.data[0]);
    });

    // Add stable IDs to each row for reliable comparison
    const dataWithStableIds = computed(() => {
      return props.data.map((row, idx) => ({
        ...row,
        __stableId: idx
      }));
    });

    const filteredData = computed(() => {
      let rows = [...dataWithStableIds.value];
      const f = props.filters || {};

      if (f.min_die !== null && f.min_die !== undefined) {
        rows = rows.filter(r => Number(r.total_die) >= Number(f.min_die));
      }
      if (f.max_die !== null && f.max_die !== undefined) {
        rows = rows.filter(r => Number(r.total_die) <= Number(f.max_die));
      }
      if (f.min_throughput !== null && f.min_throughput !== undefined) {
        rows = rows.filter(r => Number(r['throughput(tokens/die/s)']) >= Number(f.min_throughput));
      }
      if (f.max_throughput !== null && f.max_throughput !== undefined) {
        rows = rows.filter(r => Number(r['throughput(tokens/die/s)']) <= Number(f.max_throughput));
      }

      return rows;
    });

    const sortedData = computed(() => {
      return [...filteredData.value].sort((a, b) => {
        const aVal = a[sortCol.value];
        const bVal = b[sortCol.value];
        const dir = sortDir.value === 'asc' ? 1 : -1;
        const aNum = Number(aVal);
        const bNum = Number(bVal);
        if (!Number.isNaN(aNum) && !Number.isNaN(bNum)) {
          return (aNum - bNum) * dir;
        }
        return String(aVal).localeCompare(String(bVal)) * dir;
      });
    });

    const paginatedData = computed(() => {
      const start = (page.value - 1) * PAGE_SIZE;
      const end = start + PAGE_SIZE;
      return sortedData.value.slice(start, end);
    });

    const totalPages = computed(() => Math.max(1, Math.ceil(filteredData.value.length / PAGE_SIZE)));

    const sortBy = (col) => {
      if (sortCol.value === col) {
        sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc';
      } else {
        sortCol.value = col;
        sortDir.value = 'desc';
      }
    };

    const sortDirIcon = (col) => {
      if (sortCol.value !== col) return '';
      return sortDir.value === 'asc' ? '↑' : '↓';
    };

    const toggleSelect = (stableId) => {
      if (selectedRows.value.has(stableId)) {
        selectedRows.value.delete(stableId);
      } else {
        selectedRows.value.add(stableId);
      }
    };

    const toggleAll = () => {
      if (allSelected.value) {
        selectedRows.value.clear();
      } else {
        paginatedData.value.forEach(row => selectedRows.value.add(row.__stableId));
      }
    };

    const allSelected = computed(() => {
      return paginatedData.value.length > 0 &&
             paginatedData.value.every(row => selectedRows.value.has(row.__stableId));
    });

    const setPage = (p) => {
      page.value = p;
      selectedRows.value.clear();
    };

    watch(() => props.filters, () => {
      page.value = 1;
      selectedRows.value.clear();
    }, { deep: true });

    watch(() => props.data, () => {
      page.value = 1;
      selectedRows.value.clear();
    });

    const formatValue = (val) => {
      if (typeof val !== 'number') return val;
      return val.toFixed(2);
    };

    const formatColumnName = (col) => {
      // Convert snake_case to more readable format
      return col
        .replace(/_/g, ' ')
        .replace(/\b\w/g, word => word.charAt(0).toUpperCase() + word.slice(1))
        .replace('tokens die s', '(tokens/die/s)');
    };

    const emitCompare = () => {
      const selectedData = dataWithStableIds.value
        .filter(row => selectedRows.value.has(row.__stableId))
        .map(({ __stableId, ...row }) => row);
      emit('compare', selectedData);
    };

    return {
      page,
      paginatedData,
      totalPages,
      sortCol,
      sortBy,
      sortDirIcon,
      selectedRows,
      toggleSelect,
      toggleAll,
      allSelected,
      setPage,
      emitCompare,
      columns,
      formatValue,
      formatColumnName
    };
  },
  props: {
    data: {
      type: Array,
      default: () => []
    },
    filters: {
      type: Object,
      default: () => ({})
    }
  },
  emits: ['compare']
}
</script>

<style scoped>
.table-wrapper {
  overflow-x: auto;
}

.select-col {
  width: 40px;
}

.data-table th {
  cursor: pointer;
  user-select: none;
}

.data-table th:hover {
  background-color: #e2e8f0;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  margin: 16px 0;
}

.compare-btn {
  margin-top: 16px;
  width: 100%;
}

.empty-state {
  text-align: center;
  padding: 40px;
  color: #64748b;
}
</style>
```

**Step 4: Write container component (FIXED: proper imports, comparison)**

```html
<template>
  <div class="results-tab">
    <CsvSelector @csv-loaded="onCsvLoaded" />
    <ResultsFilter v-if="csvData.length > 0" @filter-change="onFilterChange" />

    <div class="results-meta">
      <span v-if="csvData.length > 0">
        Showing {{ filteredCount }} of {{ csvData.length }} results
      </span>
      <span v-else>Select a CSV file to load results.</span>
    </div>

    <ResultsTable
      v-if="csvData.length > 0"
      :data="csvData"
      :filters="currentFilters"
      @compare="onCompare"
    />
  </div>
</template>

<script>
import { ref, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useStore } from '../../composables/useStore.js';
import CsvSelector from './CsvSelector.vue';
import ResultsFilter from './ResultsFilter.vue';
import ResultsTable from './ResultsTable.vue';

export default {
  components: { CsvSelector, ResultsFilter, ResultsTable },
  setup() {
    const { setTab, setCsvSelection } = useStore();

    const csvData = ref([]);
    const currentFilters = ref({});
    const filteredCount = computed(() => {
      let rows = [...csvData.value];
      const f = currentFilters.value || {};

      if (f.min_die !== null && f.min_die !== undefined) {
        rows = rows.filter(r => Number(r.total_die) >= Number(f.min_die));
      }
      if (f.max_die !== null && f.max_die !== undefined) {
        rows = rows.filter(r => Number(r.total_die) <= Number(f.max_die));
      }
      if (f.min_throughput !== null && f.min_throughput !== undefined) {
        rows = rows.filter(r => Number(r['throughput(tokens/die/s)']) >= Number(f.min_throughput));
      }
      if (f.max_throughput !== null && f.max_throughput !== undefined) {
        rows = rows.filter(r => Number(r['throughput(tokens/die/s)']) <= Number(f.max_throughput));
      }

      return rows.length;
    });

    const onCsvLoaded = (data) => {
      csvData.value = data || [];
      currentFilters.value = {};
    };

    const onFilterChange = (filters) => {
      currentFilters.value = filters;
    };

    const onCompare = (selected) => {
      const firstSelected = selected[0];
      if (!firstSelected || firstSelected.total_die == null) return;

      // Current Visualizations tab only needs one total_die seed; use the first selected row.
      setCsvSelection({ totalDie: Number(firstSelected.total_die) });
      setTab('visualizations');
    };

    return {
      csvData,
      currentFilters,
      filteredCount,
      onCsvLoaded,
      onFilterChange,
      onCompare
    };
  }
}
</script>

<style scoped>
.results-meta {
  margin-bottom: 12px;
  color: #64748b;
  font-size: 14px;
}
</style>
```

**Step 5: Commit**

```bash
git add webapp/frontend/components/ResultsTab/
git commit -m "feat: add ResultsTab with CSV selector, reactive filtering, stable comparison"
```

---

### Task 13: Create Visualizations Tab Components (FIXED: add all required parameters)

**Files:**
- Create: `webapp/frontend/components/VisualizationsTab/ThroughputCharts.vue`
- Create: `webapp/frontend/components/VisualizationsTab/index.vue`

**Step 1: Write ThroughputCharts component (FIXED: include tpot and kvLen)**

```html
<template>
  <div class="throughput-charts">
    <div class="chart-controls">
      <div class="field">
        <label>Device Type</label>
        <select v-model="params.deviceType">
          <option v-for="d in deviceOptions" :key="d.value" :value="d.value">
            {{ d.label }}
          </option>
        </select>
      </div>
      <div class="field">
        <label>Model Type</label>
        <select v-model="params.modelType">
          <option v-for="m in modelOptions" :key="m.value" :value="m.value">
            {{ m.label }}
          </option>
        </select>
      </div>
      <div class="field">
        <label>Total Die</label>
        <select v-model="params.totalDie">
          <option v-for="d in dieOptions" :key="d" :value="d">{{ d }}</option>
        </select>
      </div>
      <div class="field">
        <label>TPOT</label>
        <select v-model="params.tpot">
          <option v-for="tp in tpotOptions" :key="tp" :value="tp">
            {{ tp }}ms
          </option>
        </select>
      </div>
      <div class="field">
        <label>KV Length</label>
        <select v-model="params.kvLen">
          <option v-for="kv in kvLenOptions" :key="kv" :value="kv">
            {{ kv }}
          </option>
        </select>
      </div>
      <button class="btn btn-sm" @click="loadCharts">Load Charts</button>
    </div>

    <div v-if="error" class="error">{{ error }}</div>

    <div class="chart-container">
      <div v-if="loading" class="loading">
        <div class="spinner"></div>
      </div>
      <div v-else-if="chartUrls.length > 0">
        <h3>Throughput Comparison</h3>
        <img
          v-for="(url, idx) in chartUrls"
          :key="idx"
          :src="url"
          alt="Throughput chart"
          class="chart-image"
          @error="onImageError($event, url)"
        />
      </div>
      <div v-else class="empty-charts">
        No charts available for the selected parameters.
        <p>Charts are generated when running simulations. The device_type, model_type, total_die, tpot, and kv_len parameters above select which pre-generated chart to display.</p>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useStore } from '../../composables/useStore.js';
import { useApi } from '../../composables/useApi.js';

export default {
  setup() {
    const api = useApi();
    const { getCsvSelection } = useStore();

    const params = ref({
      deviceType: 'ASCENDA3_Pod',
      modelType: 'DEEPSEEK_V3',
      totalDie: 128,
      tpot: 50,
      kvLen: 4096
    });

    const deviceOptions = [
      { value: 'ASCEND910B2', label: '910B2' },
      { value: 'ASCEND910B3', label: '910B3' },
      { value: 'ASCEND910B4', label: '910B4' },
      { value: 'ASCENDA3_Pod', label: 'A3Pod' },
      { value: 'ASCENDDAVID121', label: 'David121' },
      { value: 'ASCENDDAVID120', label: 'David120' },
      { value: 'NvidiaA100SXM', label: 'A100 SXM' },
      { value: 'NvidiaH100SXM', label: 'H100 SXM' }
    ];

    const modelOptions = [
      { value: 'DEEPSEEK_V3', label: 'DeepSeek V3' },
      { value: 'QWEN3_235B', label: 'Qwen3-235B' },
      { value: 'DEEPSEEK_V2_LITE', label: 'DeepSeek V2 Lite' }
    ];

    const dieOptions = [64, 128, 256, 384, 512, 768];

    const tpotOptions = [20, 50, 70, 100, 150];
    const kvLenOptions = [2048, 4096, 8192, 16384, 131072];

    const chartUrls = ref([]);
    const loading = ref(false);
    const error = ref(null);

    const loadCharts = async () => {
      loading.value = true;
      error.value = null;

      try {
        // IMPORTANT: Send ALL required parameters including tpot and kvLen
        const data = await api.getResults({
          device_type: params.value.deviceType,
          model_type: params.value.modelType,
          total_die: params.value.totalDie,
          tpot: params.value.tpot,
          kv_len: params.value.kvLen
        });
        chartUrls.value = data.throughput_images || [];
      } catch (err) {
        // 404 is expected if charts don't exist yet
        if (err.message.includes('404')) {
          chartUrls.value = [];
        } else {
          error.value = err.message;
        }
      } finally {
        loading.value = false;
      }
    };

    const onImageError = (event, url) => {
      console.warn('Failed to load chart:', url);
      event.target.style.display = 'none';
    };

    // Auto-load on mount using CSV selection
    onMounted(() => {
      const selection = getCsvSelection();
      if (selection.deviceType) params.value.deviceType = selection.deviceType;
      if (selection.modelType) params.value.modelType = selection.modelType;
      if (selection.totalDie) params.value.totalDie = selection.totalDie;
      params.value.tpot = selection.tpot;
      params.value.kvLen = selection.kvLen;
      loadCharts();
    });

    return {
      params,
      deviceOptions,
      modelOptions,
      dieOptions,
      tpotOptions,
      kvLenOptions,
      chartUrls,
      loading,
      error,
      loadCharts,
      onImageError
    };
  }
}
</script>

<style scoped>
.throughput-charts {
  max-width: 900px;
}

.chart-controls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: flex-end;
  padding: 16px;
  background: #f8fafc;
  border-radius: 8px;
  margin-bottom: 16px;
}

.chart-controls .field {
  margin-bottom: 0;
}

.chart-container {
  min-height: 400px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.chart-image {
  max-width: 100%;
  margin-top: 16px;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.empty-charts {
  color: #64748b;
  padding: 40px;
  text-align: center;
}
</style>
```

**Step 2: Write container component**

```html
<template>
  <div class="visualizations-tab">
    <ThroughputCharts />
  </div>
</template>

<script>
import ThroughputCharts from './ThroughputCharts.vue';

export default {
  components: { ThroughputCharts }
}
</script>
```

**Step 3: Commit**

```bash
git add webapp/frontend/components/VisualizationsTab/
git commit -m "feat: add VisualizationsTab with all required parameters"
```

---

## Application Integration

### Task 14: Create Main App Entry (FIXED: proper SFC loader pattern)

**Files:**
- Create: `webapp/frontend/app.js`

**Step 1: Write Vue app entry point with proper SFC loader configuration**

```javascript
import * as Vue from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { loadModule } from 'https://unpkg.com/vue3-sfc-loader/dist/vue3-sfc-loader.esm.js';

const options = {
  moduleCache: {
    vue: Vue
  },
  async getFile(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load ${url}`);
    return await res.text();
  },
  addStyle(textContent) {
    const style = document.createElement('style');
    style.textContent = textContent;
    document.head.appendChild(style);
  }
};

const app = Vue.createApp({
  components: {
    TabManager: Vue.defineAsyncComponent(() =>
      loadModule('/static/components/TabManager.vue', options)
    )
  },
  template: '<TabManager />'
});

app.mount('#app');
```

**Note:** Uses the `defineAsyncComponent(() => loadModule(..., options))` pattern from the vue3-sfc-loader README. `moduleCache.vue` must receive the imported Vue module object, not a URL string. Because FastAPI mounts `webapp/frontend/` at `/static`, the top-level SFC path should be an absolute `/static/...` URL.

**Step 2: Commit**

```bash
git add webapp/frontend/app.js
git commit -m "feat: create Vue app entry point with working SFC loader setup"
```

---

## Testing & Validation

### Task 15: Manual Smoke Tests

**Step 1: Start backend**

```bash
cd ../llm-ui-refactor

# IMPORTANT: Set LOG_LEVEL to INFO for progress indicators to work
LOG_LEVEL=INFO uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 &
```

Expected: Server starts on port 8000

**Step 2: Open browser to http://127.0.0.1:8000**

Checklist:
- [ ] Page loads without console errors
- [ ] Tab navigation works
- [ ] Run form submits successfully
- [ ] Configuration tab displays model/hardware specs
- [ ] Results tab CSV selector works
- [ ] Results tab filters work within selected CSV
- [ ] Visualizations tab loads charts

**Step 3: Run a test simulation**

Fill form and click "Start Run"

Checklist:
- [ ] Run ID displayed
- [ ] Progress bar updates
- [ ] Logs stream to UI
- [ ] "Done" status appears

**Step 4: Navigate to Results and load CSV**

Checklist:
- [ ] CSV selector works (loads CSV file by enum name matching backend)
- [ ] Filter controls work within CSV data
- [ ] Table sorting works
- [ ] Row selection works
- [ ] View Charts button enables with 1+ selection
- [ ] Selected row opens Visualizations tab and seeds `total_die`

**Step 5: Stop backend**

```bash
pkill -f uvicorn
```

**Step 6: Document test results**

Create `docs/plans/2026-03-21-web-ui-refactor-test-results.md` with findings.

**Step 7: Commit**

```bash
git add docs/plans/2026-03-21-web-ui-refactor-test-results.md
git commit -m "test: add manual smoke test results"
```

---

### Task 16: Cross-Browser Testing

**Step 1: Test in Firefox**

Visit http://127.0.0.1:8000 in Firefox

Checklist:
- [ ] All tabs functional
- [ ] Charts render
- [ ] No console errors

**Step 2: Test on Mobile (viewport resize)**

Open DevTools, toggle device toolbar to mobile

Checklist:
- [ ] Layout adapts to narrow screens
- [ ] Forms remain usable
- [ ] Tables scroll horizontally if needed

**Step 3: Commit issues found**

```bash
git add .
git commit -m "test: cross-browser and mobile testing"
```

---

## Final Steps

### Task 17: Clean up and Merge Preparation

**Step 1: Remove index_old.html (optional, after validation)**

```bash
git rm webapp/frontend/index_old.html
git commit -m "chore: remove backup HTML after validation"
```

**Step 2: Update CLAUDE.md**

Add new section for web UI development workflow.

**Step 3: Prepare for merge**

```bash
# Push to remote
git push origin feature/web-ui-refactor

# Create pull request via GitHub UI or gh CLI
gh pr create --title "Refactor Web UI to Vue.js" --body "Implements tab-based navigation, real-time progress, filterable results"
```

---

## Implementation Notes

- All commits follow conventional commits: `feat:`, `chore:`, `test:`, `fix:`
- Vue 3 loaded via CDN - no npm/package.json build step required
- **Browser-based SFC loader:** Uses `vue3-sfc-loader` with `defineAsyncComponent(() => loadModule(..., options))`. `moduleCache.vue` must be the imported Vue module object.
- **Static asset URLs:** FastAPI serves `index.html` at `/` and mounts `webapp/frontend/` at `/static`, so CSS, `app.js`, and the top-level `.vue` loader entrypoint should use absolute `/static/...` URLs.
- **Nested SFC imports:** If a child `.vue` or `../composables/...` import 404s during integration, update `getFile()` to resolve relative URLs against the parent module URL rather than assuming root-relative fetches.
- Backend API endpoints remain unchanged
- Progress polling uses existing `/logs/{run_id}` endpoint (returns raw log text as `{"log": "..."}`)
- Results use existing `/fetch_csv_results` endpoint (returns JSON array of CSV rows from one CSV file at a time)
- Charts load static images from `/data/images/` path (no new API endpoint)
- Each component is self-contained with scoped styles
- Use `useStore()` for cross-component state sharing (singleton pattern ensures shared refs including CSV selection)
- Use `useApi()` for all API calls (centralized error handling)

**IMPORTANT: LOG_LEVEL Requirement for Progress Tracking**

For RunStatus component's phase detection to work correctly:
- Start backend with `LOG_LEVEL=INFO` set as an environment variable
- The default `LOG_LEVEL=WARNING` will suppress phase marker messages that UI relies on
- Example: `LOG_LEVEL=INFO uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000`

**Backend API Gaps - Frontend-Only Behavior:**

The current backend API doesn't directly support some UX features. These will be implemented with frontend-only behavior:
- **Progress bars and phase indicators:** Frontend parses log text from `/logs/{run_id}` endpoint using regex patterns to infer progress/phase. Requires LOG_LEVEL=INFO to work.
- **ETA (optional heuristic):** Not required for the initial refactor. If added later, frontend can estimate from search parameters (die range, batch counts), but backend does not provide a true ETA.
- **Results tab CSV selection:** Backend does not provide a way to list available CSV files. Frontend allows user to select which CSV file to view via parameters (`serving_mode`, `device_type`, `model_type`, `tpot`, `kv_len`, `micro_batch_num`).
- **Pipeline visualizations:** Static images only - no interactive zoom/pan supported (would require new dynamic chart API or client-side processing).
- **Export functionality:** Not supported by static image approach (would require new backend endpoint or client-side image generation).

**Results Tab Design Note:**

The Results tab filters and sorts WITHIN A SINGLE selected CSV file. It does NOT filter across different runs/models/devices because:
1. The backend `/fetch_csv_results` endpoint returns rows from ONE specific CSV file based on parameters
2. CSV files are named by pattern: `{DeviceType.name}-{ModelType.name}-tpot{tpot}-kv_len{kvLen}.csv`
3. CSV files contain performance data (total_die, throughput, etc.) but NOT metadata columns like model_type, device_type

To view results from different simulation configurations, users select different CSV files using the CSV selector. The CSV selector must use exact enum names (for example `ASCENDA3_Pod`, `DEEPSEEK_V3`, `QWEN3_235B`) because the backend uses those strings directly in CSV and image filenames.

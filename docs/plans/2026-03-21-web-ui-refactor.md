# Web UI Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor web frontend to Vue.js 3 SPA with tab-based navigation, real-time progress, filterable results, and interactive visualizations while keeping FastAPI backend unchanged.

**Architecture:** Vue 3 (Composition API) via CDN with browser-based SFC compiler, component-based structure, reactive state management, Chart.js for visualizations.

**Tech Stack:** Vue 3, @vue/compiler-sfc (browser-based), Chart.js, localStorage, FastAPI (unchanged)

**Build Approach:** Vue Single-File Components (.vue) compiled on-demand in browser using @vue/compiler-sfc. No package.json or npm build step required.

---

## Prerequisites

### Task 0: Create Worktree and Verify Environment

**Step 1: Create git worktree**

```bash
git worktree add ../llm-ui-refactor -b feature/web-ui-refactor
cd ../llm-ui-refactor
```

**Step 2: Verify backend starts**

```bash
cd /Users/elle/claude-code/light-llm-simulator
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
cd /Users/elle/claude-code/light-llm-simulator
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

**Step 1: Write Vue 3 CDN-based HTML with browser compiler**

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Light LLM Simulator</title>
  <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
  <script src="https://unpkg.com/@vue/compiler-sfc@3/dist/compiler-sfc.esm-browser.prod.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="styles/main.css">
</head>
<body>
  <div id="app"></div>
  <script type="module" src="app.js"></script>
</body>
</html>
```

**Note:** The `@vue/compiler-sfc` script loads the Vue Single-File Component (SFC) compiler that runs in the browser, enabling .vue components without a build step.

**Step 2: Create styles directory**

```bash
mkdir -p /Users/elle/claude-code/light-llm-simulator/webapp/frontend/styles
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

    // Fetch CSV results
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

### Task 5: Create Store Composable

**Files:**
- Create: `webapp/frontend/composables/useStore.js`

**Step 1: Write store for global state**

```javascript
import { ref, computed, watch } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';

const STORAGE_KEY_TAB = 'llm-sim-current-tab';
const STORAGE_KEY_HISTORY = 'llm-sim-run-history';

export function useStore() {
  // Tab state
  const currentTab = ref(localStorage.getItem(STORAGE_KEY_TAB) || 'run');
  const tabs = ['run', 'config', 'results', 'visualizations'];

  const setTab = (tab) => {
    currentTab.value = tab;
    localStorage.setItem(STORAGE_KEY_TAB, tab);
  };

  // Run history
  const runHistory = ref(JSON.parse(localStorage.getItem(STORAGE_KEY_HISTORY) || '[]'));

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

  // Initialize from storage
  currentTab.value = localStorage.getItem(STORAGE_KEY_TAB) || 'run';

  return {
    // Tabs
    currentTab,
    tabs,
    setTab,
    isTabActive: (tab) => computed(() => currentTab.value === tab),
    // History
    runHistory,
    addRun,
    clearHistory
  };
}
```

**Step 2: Commit**

```bash
git add webapp/frontend/composables/useStore.js
git commit -m "feat: add store composable"
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

**Step 1: Write tab navigation component**

```html
<template>
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
    <slot></slot>
  </div>
</template>

<script>
import { useStore } from '../composables/useStore.js';

export default {
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
git commit -m "feat: add TabManager component"
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

    <button class="btn btn-primary" @click="startRun" :disabled="isSubmitting">
      {{ isSubmitting ? 'Starting...' : 'Start Run' }}
    </button>
  </div>
</template>

<script>
import { ref, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
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
    const { startRun, addRun } = useApi();
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

    const startRun = async () => {
      isSubmitting.value = true;
      error.value = null;

      try {
        const payload = {
          ...form.value,
          tpot: parseList(tpotInput.value),
          kv_len: parseList(kvLenInput.value),
          micro_batch_num: parseList(mbnInput.value)
        };

        const result = await startRun(payload);
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
      startRun,
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
git commit -m "feat: add RunForm component"
```

---

### Task 9: Create RunStatus Component

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
import { ref, computed, onMounted, onUnmounted } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useStore } from '../../composables/useStore.js';

export default {
  setup() {
    const { runHistory } = useStore();
    const activeRunId = computed(() => runHistory.value[0]?.id || null);

    const runId = ref(activeRunId.value);
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
        const response = await fetch(`/api/logs/${runId.value}`);
        const data = await response.json();
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
        const response = await fetch(`/api/status/${runId.value}`);
        const data = await response.json();

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

    const stopPolling = () => {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    };

    onMounted(() => {
      if (runId.value) {
        pollLogs();
        pollStatus();
        pollInterval = setInterval(() => {
          pollLogs();
          pollStatus();
        }, 3000);
      }
    });

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

**Step 2: Commit**

```bash
git add webapp/frontend/components/RunExperimentTab/RunStatus.vue
git commit -m "feat: add RunStatus component with live polling"
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
import { ref, watch, onMounted } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useStore } from '../../composables/useStore.js';

export default {
  setup() {
    const { useApi } = require('../../composables/useApi.js');
    const { getModelConfig } = useApi();

    const { currentTab } = useStore();
    const selectedModel = computed(() => {
      // Default to DeepSeek V3 when in config tab
      return 'deepseek-ai/DeepSeek-V3';
    });

    const config = ref(null);
    const loading = ref(true);
    const error = ref(null);

    const loadConfig = async () => {
      if (currentTab.value !== 'config') return;

      loading.value = true;
      error.value = null;

      try {
        config.value = await getModelConfig(selectedModel.value);
      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    };

    onMounted(loadConfig);
    watch(currentTab, loadConfig);

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
    const { getHardwareConfig, getConstants } = useApi();

    const config = ref(null);
    const loading = ref(true);
    const error = ref(null);

    const loadConfig = async () => {
      loading.value = true;
      error.value = null;

      try {
        config.value = await getHardwareConfig('Ascend_A3Pod');
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

### Task 12: Create Results Tab Components

**Files:**
- Create: `webapp/frontend/components/ResultsTab/ResultsFilter.vue`
- Create: `webapp/frontend/components/ResultsTab/ResultsTable.vue`
- Create: `webapp/frontend/components/ResultsTab/index.vue`

**Step 1: Write ResultsFilter component**

```html
<template>
  <div class="results-filter">
    <div class="filter-header">
      <h4>Filters</h4>
      <button class="btn btn-sm" @click="resetFilters">Reset</button>
    </div>

    <div class="filter-grid">
      <div class="field">
        <label>Model</label>
        <select v-model="filters.model_type">
          <option value="">All Models</option>
          <option value="DEEPSEEK_V3">DeepSeek V3</option>
          <option value="QWEN3_235B">Qwen3-235B</option>
          <option value="DEEPSEEK_V2_LITE">DeepSeek V2 Lite</option>
        </select>
      </div>

      <div class="field">
        <label>Device</label>
        <select v-model="filters.device_type">
          <option value="">All Devices</option>
          <option value="ASCEND910B2">Ascend 910B2</option>
          <option value="ASCEND910B3">Ascend 910B3</option>
          <option value="ASCEND910B4">Ascend 910B4</option>
          <option value="ASCENDA3_Pod">Ascend A3Pod</option>
          <option value="ASCENDDAVID121">Ascend David121</option>
          <option value="ASCENDDAVID120">Ascend David120</option>
          <option value="NvidiaA100SXM">Nvidia A100 SXM</option>
          <option value="NvidiaH100SXM">Nvidia H100 SXM</option>
        </select>
      </div>

      <div class="field">
        <label>Serving Mode</label>
        <select v-model="filters.serving_mode">
          <option value="">All</option>
          <option value="AFD">AFD</option>
          <option value="DeepEP">DeepEP</option>
        </select>
      </div>

      <div class="field">
        <label>TPOT</label>
        <select v-model="filters.tpot">
          <option value="">All</option>
          <option v-for="tp in [20, 50, 70, 100, 150]" :key="tp" :value="tp">
            {{ tp }}ms
          </option>
        </select>
      </div>

      <div class="field">
        <label>KV Length</label>
        <select v-model="filters.kv_len">
          <option value="">All</option>
          <option v-for="kv in [2048, 4096, 8192, 16384, 131072]" :key="kv" :value="kv">
            {{ kv }}
          </option>
        </select>
      </div>

      <div class="field">
        <label>Min Die</label>
        <input type="number" v-model.number="filters.min_die" min="0" />
      </div>

      <div class="field">
        <label>Max Die</label>
        <input type="number" v-model.number="filters.max_die" min="0" />
      </div>
    </div>

    <div class="filter-count">
      Showing {{ filteredCount }} of {{ totalCount }} results
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';

export default {
  setup(props, { emit }) {
    const filters = ref({
      model_type: '',
      device_type: '',
      serving_mode: '',
      tpot: '',
      kv_len: '',
      min_die: null,
      max_die: null
    });

    const resetFilters = () => {
      filters.value = {
        model_type: '',
        device_type: '',
        serving_mode: '',
        tpot: '',
        kv_len: '',
        min_die: null,
        max_die: null
      };
    };

    watch(filters, () => {
      emit('filter-change', filters.value);
    }, { deep: true });

    return {
      filters,
      resetFilters,
      filteredCount: computed(() => props.filteredData?.length || 0),
      totalCount: computed(() => props.allData?.length || 0)
    };
  },
  props: {
    allData: Array,
    filteredData: Array
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

.filter-count {
  font-size: 13px;
  color: #64748b;
  text-align: right;
  margin-top: 8px;
}
</style>
```

**Step 2: Write ResultsTable component**

```html
<template>
  <div class="results-table-container">
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>

    <div v-else-if="error" class="error">
      {{ error }}
    </div>

    <div v-else-if="data.length === 0" class="empty-state">
      No results found. Try adjusting filters or run a new simulation.
    </div>

    <div v-else class="table-wrapper">
      <table class="data-table">
        <thead>
          <tr>
            <th class="select-col">
              <input type="checkbox" :checked="allSelected" @change="toggleAll" />
            </th>
            <th @click="sortBy('attn_bs')">
              Attention BS {{ sortCol === 'attn_bs' ? sortDirIcon('attn_bs') : '' }}
            </th>
            <th @click="sortBy('total_die')">
              Total Die {{ sortCol === 'total_die' ? sortDirIcon('total_die') : '' }}
            </th>
            <th @click="sortBy('throughput')">
              Throughput {{ sortCol === 'throughput' ? sortDirIcon('throughput') : '' }}
            </th>
            <th @click="sortBy('e2e_time')">
              E2E Time (ms) {{ sortCol === 'e2e_time' ? sortDirIcon('e2e_time') : '' }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in paginatedData" :key="row.run_id || rowIndex">
            <td class="select-col">
              <input type="checkbox" :checked="selectedRows.has(rowIndex)" @change="toggleSelect(rowIndex)" />
            </td>
            <td>{{ row.attn_bs }}</td>
            <td>{{ row.total_die }}</td>
            <td>{{ row.throughput?.toFixed(2) }}</td>
            <td>{{ row.e2e_time?.toFixed(2) }}</td>
          </tr>
        </tbody>
      </table>

      <div class="pagination">
        <button class="btn btn-sm" :disabled="page === 1" @click="setPage(page - 1)">Previous</button>
        <span>Page {{ page }} of {{ totalPages }}</span>
        <button class="btn btn-sm" :disabled="page === totalPages" @click="setPage(page + 1)">Next</button>
      </div>

      <button class="btn btn-primary compare-btn" :disabled="selectedRows.size < 2" @click="emitCompare">
        Compare Selected ({{ selectedRows.size }})
      </button>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';

const PAGE_SIZE = 20;

export default {
  setup(props, { emit }) {
    const page = ref(1);
    const sortCol = ref('throughput');
    const sortDir = ref('desc');
    const selectedRows = ref(new Set());

    const sortedData = computed(() => {
      return [...props.data].sort((a, b) => {
        const aVal = a[sortCol.value];
        const bVal = b[sortCol.value];
        const dir = sortDir.value === 'asc' ? 1 : -1;
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return (aVal - bVal) * dir;
        }
        return String(aVal).localeCompare(String(bVal)) * dir;
      });
    });

    const paginatedData = computed(() => {
      const start = (page.value - 1) * PAGE_SIZE;
      const end = start + PAGE_SIZE;
      return sortedData.value.slice(start, end);
    });

    const totalPages = computed(() => Math.ceil(props.data.length / PAGE_SIZE));

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

    const toggleSelect = (idx) => {
      if (selectedRows.value.has(idx)) {
        selectedRows.value.delete(idx);
      } else {
        selectedRows.value.add(idx);
      }
    };

    const toggleAll = () => {
      if (allSelected.value) {
        selectedRows.value.clear();
      } else {
        sortedData.value.forEach((_, idx) => selectedRows.value.add(idx));
      }
    };

    const allSelected = computed(() => {
      return paginatedData.value.length > 0 &&
             paginatedData.value.every((_, idx) => selectedRows.value.has(idx + (page.value - 1) * PAGE_SIZE));
    });

    const setPage = (p) => {
      page.value = p;
      selectedRows.value.clear();
    };

    const emitCompare = () => {
      const selectedData = Array.from(selectedRows.value).map(idx => paginatedData.value[idx]);
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
      emitCompare
    };
  },
  props: {
    data: Array
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

**Step 3: Write container component**

```html
<template>
  <div class="results-tab">
    <ResultsFilter
      :all-data="allResults"
      :filtered-data="filteredResults"
      @filter-change="onFilterChange"
    />
    <ResultsTable
      :data="filteredResults"
      @compare="onCompare"
    />
  </div>
</template>

<script>
import { ref } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useApi } from '../../composables/useApi.js';
import ResultsFilter from './ResultsFilter.vue';
import ResultsTable from './ResultsTable.vue';

export default {
  components: { ResultsFilter, ResultsTable },
  setup() {
    const { fetchCsvResults } = useApi();

    const allResults = ref([]);
    const filteredResults = ref([]);
    const loading = ref(true);
    const error = ref(null);

    const onFilterChange = (newFiltered) => {
      filteredResults.value = newFiltered;
    };

    const loadResults = async () => {
      try {
        // Load default results
        const data = await fetchCsvResults({
          device_type: 'ASCENDA3_Pod',
          model_type: 'DEEPSEEK_V3',
          tpot: 50,
          kv_len: 4096,
          serving_mode: 'AFD',
          micro_batch_num: 3
        });
        allResults.value = data;
        filteredResults.value = data;
      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    };

    const onCompare = (selected) => {
      // Store comparison data for visualizations tab
      sessionStorage.setItem('comparison-data', JSON.stringify(selected));
      // Navigate to visualizations tab
      window.location.hash = '#visualizations';
    };

    // Load on mount
    loadResults();

    return {
      allResults,
      filteredResults,
      loading,
      error,
      onFilterChange,
      onCompare
    };
  }
}
</script>
</script>
```

**Step 4: Commit**

```bash
git add webapp/frontend/components/ResultsTab/
git commit -m "feat: add ResultsTab with filter and sort"
```

---

### Task 13: Create Visualizations Tab Components

**Files:**
- Create: `webapp/frontend/components/VisualizationsTab/ThroughputCharts.vue`
- Create: `webapp/frontend/components/VisualizationsTab/index.vue`

**Step 1: Write ThroughputCharts component**

```html
<template>
  <div class="throughput-charts">
    <div class="chart-controls">
      <div class="field">
        <label>Device Type</label>
        <select v-model="params.device_type">
          <option v-for="d in deviceOptions" :key="d.value" :value="d.value">
            {{ d.label }}
          </option>
        </select>
      </div>
      <div class="field">
        <label>Model Type</label>
        <select v-model="params.model_type">
          <option v-for="m in modelOptions" :key="m.value" :value="m.value">
            {{ m.label }}
          </option>
        </select>
      </div>
      <div class="field">
        <label>Total Die</label>
        <select v-model="params.total_die">
          <option v-for="d in dieOptions" :key="d" :value="d">{{ d }}</option>
        </select>
      </div>
      <button class="btn btn-sm" @click="loadCharts">Load Charts</button>
    </div>

    <div v-if="error" class="error">{{ error }}</div>

    <div class="chart-container">
      <canvas ref="chartCanvas" v-show="!loading"></canvas>
      <div v-if="loading" class="loading">
        <div class="spinner"></div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, nextTick } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useApi } from '../../composables/useApi.js';

export default {
  setup() {
    const { getResults } = useApi();

    const params = ref({
      device_type: 'ASCENDA3_Pod',
      model_type: 'DEEPSEEK_V3',
      total_die: 128
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

    const chartCanvas = ref(null);
    const loading = ref(false);
    const error = ref(null);
    let chartInstance = null;

    const loadCharts = async () => {
      loading.value = true;
      error.value = null;

      try {
        const data = await getResults(params.value);

        await nextTick();

        if (chartCanvas.value && data.throughput_images) {
          // Create Chart.js instance
          if (chartInstance) {
            chartInstance.destroy();
          }

          // Load images directly (simpler approach)
          const container = chartCanvas.value.parentElement;
          container.innerHTML = '<h3>Throughput Comparison</h3>';

          data.throughput_images.forEach(url => {
            const img = document.createElement('img');
            img.src = url;
            img.alt = 'Throughput chart';
            img.style.maxWidth = '100%';
            img.style.marginTop = '16px';
            img.onerror = () => {
              const msg = document.createElement('div');
              msg.className = 'missing-chart';
              msg.textContent = `Chart not found: ${url}`;
              container.appendChild(msg);
            };
            container.appendChild(img);
          });
        }

      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    };

    // Auto-load on mount
    onMounted(loadCharts);

    return {
      params,
      deviceOptions,
      modelOptions,
      dieOptions,
      chartCanvas,
      loading,
      error,
      loadCharts
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

.missing-chart {
  color: #dc2626;
  padding: 16px;
  background: #fee2e2;
  border-radius: 6px;
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
</script>
```

**Step 3: Commit**

```bash
git add webapp/frontend/components/VisualizationsTab/
git commit -m "feat: add VisualizationsTab with throughput charts"
```

---

## Application Integration

### Task 14: Create Main App Entry

**Files:**
- Create: `webapp/frontend/app.js`

**Step 1: Write Vue app entry point with SFC compiler**

```javascript
import { createApp } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { compileTemplate } from 'https://unpkg.com/@vue/compiler-sfc@3/dist/compiler-sfc.esm-browser.prod.js';
import TabManager from './components/TabManager.vue';
import RunExperimentTab from './components/RunExperimentTab/index.vue';
import ConfigurationTab from './components/ConfigurationTab/index.vue';
import ResultsTab from './components/ResultsTab/index.vue';
import VisualizationsTab from './components/VisualizationsTab/index.vue';

// Component cache for compiled SFCs
const componentCache = new Map();

async function loadComponent(name, path) {
  if (componentCache.has(name)) {
    return componentCache.get(name);
  }

  const response = await fetch(path);
  const source = await response.text();
  const { descriptor } = compileTemplate({ source });

  const component = descriptor.setup;
  componentCache.set(name, component);
  return component;
}

const components = {
  TabManager: () => loadComponent('TabManager', './components/TabManager.vue'),
  RunExperimentTab: () => loadComponent('RunExperimentTab', './components/RunExperimentTab/index.vue'),
  ConfigurationTab: () => loadComponent('ConfigurationTab', './components/ConfigurationTab/index.vue'),
  ResultsTab: () => loadComponent('ResultsTab', './components/ResultsTab/index.vue'),
  VisualizationsTab: () => loadComponent('VisualizationsTab', './components/VisualizationsTab/index.vue')
};

const app = createApp({
  components,
  template: `
    <div>
      <TabManager>
        <template #default="{ currentTab }">
          <RunExperimentTab v-if="currentTab === 'run'" />
          <ConfigurationTab v-if="currentTab === 'config'" />
          <ResultsTab v-if="currentTab === 'results'" />
          <VisualizationsTab v-if="currentTab === 'visualizations'" />
        </template>
      </TabManager>
    </div>
  `
});

app.mount('#app');
```

**Note:** Uses `@vue/compiler-sfc` to compile .vue components on-demand in the browser. Components are cached after first load.

**Step 2: Commit**

```bash
git add webapp/frontend/app.js
git commit -m "feat: create Vue app entry point"
```

---

## Testing & Validation

### Task 15: Manual Smoke Tests

**Step 1: Start backend**

```bash
cd /Users/elle/claude-code/light-llm-simulator
uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 &
```

Expected: Server starts on port 8000

**Step 2: Open browser to http://127.0.0.1:8000**

Checklist:
- [ ] Page loads without console errors
- [ ] Tab navigation works
- [ ] Run form submits successfully
- [ ] Configuration tab displays model/hardware specs
- [ ] Results tab loads sample data
- [ ] Visualizations tab loads charts

**Step 3: Run a test simulation**

Fill form and click "Start Run"

Checklist:
- [ ] Run ID displayed
- [ ] Progress bar updates
- [ ] Logs stream to UI
- [ ] "Done" status appears

**Step 4: Navigate to Results and fetch**

Checklist:
- [ ] Filter controls work
- [ ] Table sorting works
- [ ] Row selection works
- [ ] Comparison button enables with 2+ selections

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
- **Browser-based SFC compiler:** Uses `@vue/compiler-sfc` to compile .vue components on-demand, no build step
- Backend API endpoints remain unchanged
- Progress polling uses existing `/logs/{run_id}` endpoint (returns raw log text as `{"log": "..."}`)
- Results use existing `/fetch_csv_results` endpoint (returns JSON array of CSV rows)
- Charts load images from `/data/images/` static path (no new API endpoint)
- Each component is self-contained with scoped styles
- Use `useStore()` for cross-component state sharing
- Use `useApi()` for all API calls (centralized error handling)

**Backend API Gaps - Frontend-Only Behavior:**
The current backend API doesn't directly support some UX features. These will be implemented with frontend-only behavior:
- **Progress bars and phase indicators:** Frontend parses log text from `/logs/{run_id}` endpoint using regex patterns to infer progress/phase
- **ETA (estimated time remaining):** Frontend estimates based on search parameters (die range, batch counts) - not provided by backend
- **Interactive pipeline zoom/pan:** Not supported by current static image approach - would require new dynamic chart API or Chart.js configuration with raw CSV data instead of images
- **Real-time status updates:** Uses existing `/status/{run_id}` polling every 3 seconds

import * as Vue from "https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js";
import { loadModule } from "https://unpkg.com/vue3-sfc-loader/dist/vue3-sfc-loader.esm.js";

const API_BASE = "/api";
const STORAGE_KEY_TAB = "llm-sim-current-tab";
const STORAGE_KEY_HISTORY = "llm-sim-run-history";
const STORAGE_KEY_CSV_SELECTION = "llm-sim-csv-selection";
const DEFAULT_CSV_SELECTION = {
  servingMode: "AFD",
  deploymentMode: "Heterogeneous",
  deviceType: "ASCENDDAVID120",
  deviceType2: "ASCEND910B2",
  modelType: "DEEPSEEK_V3",
  tpot: 50,
  kvLen: 4096,
  microBatchNum: 3,
  totalDie: 128,
};

const LEGACY_DEFAULT_CSV_SELECTION = {
  ...DEFAULT_CSV_SELECTION,
  deviceType: "ASCENDA3_Pod",
};

const MODEL_VALUE_TO_NAME = {
  "deepseek-ai/DeepSeek-V3": "DEEPSEEK_V3",
  "deepseek-ai/DeepSeek-V3-2": "DEEPSEEK_V3_2",
  "Qwen/Qwen3-235B-A22B": "QWEN3_235B",
  "deepseek-ai/DeepSeek-V2-Lite": "DEEPSEEK_V2_LITE",
};

const DEVICE_VALUE_TO_NAME = {
  Ascend_910b2: "ASCEND910B2",
  Ascend_910b3: "ASCEND910B3",
  Ascend_910b4: "ASCEND910B4",
  Ascend_A3Pod: "ASCENDA3_Pod",
  Ascend_David121: "ASCENDDAVID121",
  Ascend_David120: "ASCENDDAVID120",
  Nvidia_A100_SXM: "NvidiaA100SXM",
  Nvidia_H100_SXM: "NvidiaH100SXM",
};

function parseStoredJson(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch (error) {
    console.warn(`Failed to parse localStorage for ${key}:`, error);
    return fallback;
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `API error: ${response.status} ${response.statusText}${detail ? ` - ${detail}` : ""}`,
    );
  }

  return await response.json();
}

const currentTab = Vue.ref(localStorage.getItem(STORAGE_KEY_TAB) || "run");
const runHistory = Vue.ref(parseStoredJson(STORAGE_KEY_HISTORY, []));
const storedCsvSelection = parseStoredJson(STORAGE_KEY_CSV_SELECTION, {});

function normalizeCsvSelection(selection = {}) {
  return {
    ...DEFAULT_CSV_SELECTION,
    ...selection,
    tpot: Number(selection.tpot ?? DEFAULT_CSV_SELECTION.tpot),
    kvLen: Number(selection.kvLen ?? DEFAULT_CSV_SELECTION.kvLen),
    microBatchNum: Number(selection.microBatchNum ?? DEFAULT_CSV_SELECTION.microBatchNum),
    totalDie: Number(selection.totalDie ?? DEFAULT_CSV_SELECTION.totalDie),
  };
}

function firstListValue(value, fallback) {
  if (Array.isArray(value) && value.length > 0) {
    return value[0];
  }
  return value ?? fallback;
}

function deriveCsvSelectionFromRunParams(params = {}) {
  return {
    servingMode: params.serving_mode || DEFAULT_CSV_SELECTION.servingMode,
    deviceType: DEVICE_VALUE_TO_NAME[params.device_type] || params.device_type || DEFAULT_CSV_SELECTION.deviceType,
    modelType: MODEL_VALUE_TO_NAME[params.model_type] || params.model_type || DEFAULT_CSV_SELECTION.modelType,
    tpot: Number(firstListValue(params.tpot, DEFAULT_CSV_SELECTION.tpot)),
    kvLen: Number(firstListValue(params.kv_len, DEFAULT_CSV_SELECTION.kvLen)),
    microBatchNum: Number(
      params.serving_mode === "DeepEP"
        ? 1
        : firstListValue(params.micro_batch_num, DEFAULT_CSV_SELECTION.microBatchNum),
    ),
  };
}

function shouldSeedFromLatestRun(storedSelection, latestRun) {
  if (!latestRun?.params) {
    return false;
  }

  if (!storedSelection || Object.keys(storedSelection).length === 0) {
    return true;
  }

  const normalizedStored = normalizeCsvSelection(storedSelection);
  const matchesCurrentDefault =
    normalizedStored.servingMode === DEFAULT_CSV_SELECTION.servingMode &&
    normalizedStored.deviceType === DEFAULT_CSV_SELECTION.deviceType &&
    normalizedStored.modelType === DEFAULT_CSV_SELECTION.modelType &&
    normalizedStored.tpot === DEFAULT_CSV_SELECTION.tpot &&
    normalizedStored.kvLen === DEFAULT_CSV_SELECTION.kvLen &&
    normalizedStored.microBatchNum === DEFAULT_CSV_SELECTION.microBatchNum;

  const matchesLegacyDefault =
    normalizedStored.servingMode === LEGACY_DEFAULT_CSV_SELECTION.servingMode &&
    normalizedStored.deviceType === LEGACY_DEFAULT_CSV_SELECTION.deviceType &&
    normalizedStored.modelType === LEGACY_DEFAULT_CSV_SELECTION.modelType &&
    normalizedStored.tpot === LEGACY_DEFAULT_CSV_SELECTION.tpot &&
    normalizedStored.kvLen === LEGACY_DEFAULT_CSV_SELECTION.kvLen &&
    normalizedStored.microBatchNum === LEGACY_DEFAULT_CSV_SELECTION.microBatchNum;

  return matchesCurrentDefault || matchesLegacyDefault;
}

const initialCsvSelection = normalizeCsvSelection(storedCsvSelection);
if (shouldSeedFromLatestRun(storedCsvSelection, runHistory.value[0])) {
  Object.assign(initialCsvSelection, deriveCsvSelectionFromRunParams(runHistory.value[0].params));
}

const csvSelection = Vue.ref(initialCsvSelection);

function persistCsvSelection(selection) {
  csvSelection.value = normalizeCsvSelection({
    ...csvSelection.value,
    ...selection,
  });
  localStorage.setItem(STORAGE_KEY_CSV_SELECTION, JSON.stringify(csvSelection.value));
}

function useStore() {
  return {
    currentTab,
    tabs: ["run", "config", "results", "visualizations"],
    setTab(tab) {
      currentTab.value = tab;
      localStorage.setItem(STORAGE_KEY_TAB, tab);
    },
    runHistory,
    addRun(runId, params) {
      runHistory.value.unshift({
        id: runId,
        params,
        timestamp: Date.now(),
      });

      if (runHistory.value.length > 20) {
        runHistory.value = runHistory.value.slice(0, 20);
      }

      localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(runHistory.value));
      persistCsvSelection(deriveCsvSelectionFromRunParams(params));
    },
    clearHistory() {
      runHistory.value = [];
      localStorage.removeItem(STORAGE_KEY_HISTORY);
    },
    csvSelection,
    setCsvSelection(selection) {
      persistCsvSelection(selection);
    },
    getCsvSelection() {
      return csvSelection.value;
    },
  };
}

function useApi() {
  return {
    startRun(payload) {
      return fetchJson("/run", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },
    getStatus(runId) {
      return fetchJson(`/status/${runId}`);
    },
    getLogs(runId) {
      return fetchJson(`/logs/${runId}`);
    },
    getModelConfig(modelType) {
      return fetchJson(`/model_config?${new URLSearchParams({ model_type: modelType })}`);
    },
    getHardwareConfig(deviceType) {
      return fetchJson(`/hardware_config?${new URLSearchParams({ device_type: deviceType })}`);
    },
    getResults(params) {
      return fetchJson(`/results?${new URLSearchParams(params)}`);
    },
    fetchCsvResults(params) {
      return fetchJson(`/fetch_csv_results?${new URLSearchParams(params)}`);
    },
    getConstants() {
      return fetchJson("/constants");
    },
  };
}

window.LightLLMRuntime = {
  Vue,
  useApi,
  useStore,
};

const options = {
  moduleCache: {
    vue: Vue,
  },
  async getFile(url) {
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`Could not load ${url}`);
    }
    return await response.text();
  },
  addStyle(textContent) {
    const style = Object.assign(document.createElement("style"), {
      textContent,
    });
    document.head.appendChild(style);
  },
};

const app = Vue.createApp({
  components: {
    TabManager: Vue.defineAsyncComponent(() =>
      loadModule("/static/components/TabManager.vue", options),
    ),
  },
  template: "<TabManager />",
});

app.mount("#app");

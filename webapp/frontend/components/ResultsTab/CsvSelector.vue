<template>
  <section class="csv-selector">
    <div class="section-header">
      <div>
        <p class="eyebrow">Results</p>
        <h3>Select CSV File</h3>
      </div>
      <button class="btn btn-primary" :disabled="isLoading" @click="handleLoadCsv">
        {{ isLoading ? 'Loading...' : 'Load Results' }}
      </button>
    </div>

    <div class="selection-grid">
      <label class="field">
        <span>Serving Mode</span>
        <select v-model="selection.servingMode">
          <option v-for="mode in servingModes" :key="mode.value" :value="mode.value">
            {{ mode.label }}
          </option>
        </select>
      </label>

      <label class="field">
        <span>Deployment Mode</span>
        <select v-model="selection.deploymentMode">
          <option v-for="mode in deploymentModes" :key="mode.value" :value="mode.value">
            {{ mode.label }}
          </option>
        </select>
      </label>

      <label class="field">
        <span>{{ deviceTypeLabel }}</span>
        <select v-model="selection.deviceType">
          <option v-for="device in deviceOptions" :key="device.value" :value="device.value">
            {{ device.label }}
          </option>
        </select>
      </label>

      <label class="field" v-if="selection.deploymentMode === 'Heterogeneous'">
        <span>{{ deviceType2Label }}</span>
        <select v-model="selection.deviceType2">
          <option v-for="device in deviceOptions" :key="device.value" :value="device.value">
            {{ device.label }}
          </option>
        </select>
      </label>

      <label class="field">
        <span>Model Type</span>
        <select v-model="selection.modelType">
          <option v-for="model in modelOptions" :key="model.value" :value="model.value">
            {{ model.label }}
          </option>
        </select>
      </label>

      <label class="field">
        <span>TPOT</span>
        <select v-model="selection.tpot">
          <option v-for="tpot in tpotOptions" :key="tpot" :value="tpot">
            {{ tpot }} ms
          </option>
        </select>
      </label>

      <label class="field">
        <span>KV Length</span>
        <select v-model="selection.kvLen">
          <option v-for="kv in kvLenOptions" :key="kv" :value="kv">
            {{ kv }}
          </option>
        </select>
      </label>

      <label class="field">
        <span>Micro Batch Num</span>
        <select v-model="selection.microBatchNum">
          <option v-for="mbn in microBatchOptions" :key="mbn" :value="mbn">
            MBN {{ mbn }}
          </option>
        </select>
      </label>
    </div>

    <p class="selection-note">
      Uses exact enum names for backend file lookups and keeps the current selection in shared app state.
    </p>

    <div v-if="error" class="error-banner">
      {{ error }}
    </div>
  </section>
</template>

<script>
const DEFAULT_SELECTION = {
  servingMode: 'AFD',
  deploymentMode: 'Heterogeneous',
  deviceType: 'ASCENDDAVID120',
  deviceType2: 'ASCEND910B2',
  modelType: 'DEEPSEEK_V3',
  tpot: 50,
  kvLen: 4096,
  microBatchNum: 3,
  totalDie: 128
};

const SERVING_MODES = [
  { value: 'AFD', label: 'AFD' },
  { value: 'DeepEP', label: 'DeepEP' }
];

const DEPLOYMENT_MODES = [
  { value: 'Homogeneous', label: 'Homogeneous' },
  { value: 'Heterogeneous', label: 'Heterogeneous' }
];

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
  { value: 'ASCENDDAVID100', label: 'Ascend David100' },
  { value: 'ASCENDDAVID120', label: 'Ascend David120' },
  { value: 'ASCENDDAVID121', label: 'Ascend David121' },
  { value: 'NvidiaA100SXM', label: 'Nvidia A100 SXM' },
  { value: 'NvidiaH100SXM', label: 'Nvidia H100 SXM' }
];

const TPOT_OPTIONS = [20, 50, 70, 100, 150];
const KV_LEN_OPTIONS = [2048, 4096, 8192, 16384, 131072];
const MICRO_BATCH_OPTIONS = [2, 3];

function createSelection() {
  return { ...DEFAULT_SELECTION };
}

function normalizeSelection(selection, includeTotalDie = false) {
  const normalized = {
    servingMode: selection.servingMode || DEFAULT_SELECTION.servingMode,
    deploymentMode: selection.deploymentMode || DEFAULT_SELECTION.deploymentMode,
    deviceType: selection.deviceType || DEFAULT_SELECTION.deviceType,
    deviceType2: selection.deviceType2 || DEFAULT_SELECTION.deviceType2,
    modelType: selection.modelType || DEFAULT_SELECTION.modelType,
    tpot: Number(selection.tpot) || DEFAULT_SELECTION.tpot,
    kvLen: Number(selection.kvLen) || DEFAULT_SELECTION.kvLen,
    microBatchNum: Number(selection.microBatchNum) || DEFAULT_SELECTION.microBatchNum
  };

  if (includeTotalDie) {
    normalized.totalDie = Number(selection.totalDie) || DEFAULT_SELECTION.totalDie;
  }

  return normalized;
}

export default {
  emits: ['csv-loaded'],
  setup(props, { emit }) {
    const { ref, watch, onMounted, computed } = window.LightLLMRuntime.Vue;
    const { useApi, useStore } = window.LightLLMRuntime;
    const api = useApi();
    const { getCsvSelection, setCsvSelection } = useStore();
    const selection = ref({
      ...createSelection(),
      ...normalizeSelection(getCsvSelection(), true)
    });
    const isLoading = ref(false);
    const error = ref(null);

    const deviceTypeLabel = computed(() => {
      if (selection.value.servingMode === 'AFD' && selection.value.deploymentMode === 'Heterogeneous') {
        return 'Device Type (Attention)';
      }
      return 'Device Type';
    });

    const deviceType2Label = computed(() => {
      if (selection.value.servingMode === 'AFD' && selection.value.deploymentMode === 'Heterogeneous') {
        return 'Device Type 2 (FFN)';
      }
      return 'Device Type 2';
    });

    watch(
      selection,
      (value) => {
        setCsvSelection(normalizeSelection(value, true));
      },
      { deep: true }
    );

    const handleLoadCsv = async () => {
      isLoading.value = true;
      error.value = null;

      const params = normalizeSelection(selection.value);
      try {
        const data = await api.fetchCsvResults({
          serving_mode: params.servingMode,
          deployment_mode: params.deploymentMode,
          device_type: params.deviceType,
          device_type2: params.deviceType2,
          model_type: params.modelType,
          tpot: String(params.tpot),
          kv_len: String(params.kvLen),
          micro_batch_num: String(params.microBatchNum)
        });
        setCsvSelection(params);
        emit('csv-loaded', data || []);
      } catch (err) {
        if (String(err.message).includes('404')) {
          error.value = 'CSV file not found. Run a simulation first or pick a different CSV selection.';
        } else {
          error.value = err.message;
        }
      } finally {
        isLoading.value = false;
      }
    };

    onMounted(() => {
      handleLoadCsv();
    });

    return {
      selection,
      isLoading,
      error,
      deviceTypeLabel,
      deviceType2Label,
      servingModes: SERVING_MODES,
      deploymentModes: DEPLOYMENT_MODES,
      modelOptions: MODEL_OPTIONS,
      deviceOptions: DEVICE_OPTIONS,
      tpotOptions: TPOT_OPTIONS,
      kvLenOptions: KV_LEN_OPTIONS,
      microBatchOptions: MICRO_BATCH_OPTIONS,
      handleLoadCsv
    };
  }
};
</script>

<style scoped>
.csv-selector {
  border: 1px solid #d8e0ea;
  border-radius: 18px;
  padding: 20px;
  background:
    linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.95)),
    radial-gradient(circle at top right, rgba(13, 148, 136, 0.08), transparent 35%);
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
  min-width: 0;
  width: 100%;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
}

.eyebrow {
  margin: 0 0 4px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 11px;
  color: #0f766e;
  font-weight: 700;
}

h3 {
  margin: 0;
  font-size: 20px;
  color: #0f172a;
}

.btn {
  appearance: none;
  border: 0;
  border-radius: 999px;
  padding: 10px 16px;
  font-weight: 700;
  cursor: pointer;
}

.btn-primary {
  background: linear-gradient(135deg, #0f766e, #2563eb);
  color: #fff;
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.selection-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
  min-width: 0;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.field span {
  font-size: 13px;
  font-weight: 700;
  color: #334155;
}

select {
  width: 100%;
  min-width: 0;
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  padding: 10px 12px;
  background: #fff;
  color: #0f172a;
}

.selection-note {
  margin: 14px 0 0;
  font-size: 13px;
  color: #475569;
}

.error-banner {
  margin-top: 14px;
  border-radius: 12px;
  border: 1px solid #fecaca;
  background: #fff1f2;
  color: #9f1239;
  padding: 12px 14px;
  font-size: 14px;
}

@media (max-width: 720px) {
  .section-header {
    flex-direction: column;
    align-items: stretch;
  }

  .btn-primary {
    width: 100%;
  }
}
</style>

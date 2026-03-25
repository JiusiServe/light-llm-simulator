<template>
  <section class="throughput-charts">
    <div class="section-header">
      <div>
        <p class="eyebrow">Visualizations</p>
        <h3>Static Throughput Charts</h3>
      </div>
      <button class="btn btn-primary" :disabled="loading" @click="loadCharts">
        {{ loading ? 'Loading...' : 'Load Charts' }}
      </button>
    </div>

    <div class="chart-controls">
      <label class="field">
        <span>Deployment Mode</span>
        <select v-model="params.deploymentMode">
          <option v-for="mode in deploymentModes" :key="mode.value" :value="mode.value">
            {{ mode.label }}
          </option>
        </select>
      </label>

      <label class="field">
        <span>{{ deviceTypeLabel }}</span>
        <select v-model="params.deviceType">
          <option v-for="device in deviceOptions" :key="device.value" :value="device.value">
            {{ device.label }}
          </option>
        </select>
      </label>

      <label class="field" v-if="params.deploymentMode === 'Heterogeneous'">
        <span>{{ deviceType2Label }}</span>
        <select v-model="params.deviceType2">
          <option v-for="device in deviceOptions" :key="device.value" :value="device.value">
            {{ device.label }}
          </option>
        </select>
      </label>

      <label class="field">
        <span>Model Type</span>
        <select v-model="params.modelType">
          <option v-for="model in modelOptions" :key="model.value" :value="model.value">
            {{ model.label }}
          </option>
        </select>
      </label>

      <label class="field">
        <span>Total Die</span>
        <select v-model="params.totalDie">
          <option v-for="die in dieOptions" :key="die" :value="die">
            {{ die }}
          </option>
        </select>
      </label>

      <label class="field">
        <span>TPOT</span>
        <select v-model="params.tpot">
          <option v-for="tpot in tpotOptions" :key="tpot" :value="tpot">
            {{ tpot }} ms
          </option>
        </select>
      </label>

      <label class="field">
        <span>KV Length</span>
        <select v-model="params.kvLen">
          <option v-for="kv in kvLenOptions" :key="kv" :value="kv">
            {{ kv }}
          </option>
        </select>
      </label>
    </div>

    <div v-if="error" class="error-banner">
      {{ error }}
    </div>

    <div class="chart-area">
      <section class="chart-section">
        <div class="chart-title">
          <h4>Throughput Images</h4>
          <span>Backend generated</span>
        </div>

        <div v-if="throughputImages.length === 0 && !loading" class="empty-state">
          No throughput images returned for the selected parameters.
        </div>

        <div v-else class="image-grid">
          <figure v-for="url in throughputImages" :key="url" class="image-card">
            <template v-if="!missingImages.has(url)">
              <img :src="url" alt="Throughput chart" @error="markMissing(url)" />
            </template>
            <template v-else>
              <div class="img-missing">
                <strong>Missing image</strong>
                <div class="img-url">{{ url }}</div>
              </div>
            </template>
          </figure>
        </div>
      </section>

      <section class="chart-section">
        <div class="chart-title">
          <h4>Pipeline Images</h4>
          <span>Backend generated</span>
        </div>

        <div v-if="pipelineImages.length === 0 && !loading" class="empty-state">
          No pipeline images returned for the selected parameters.
        </div>

        <div v-else class="image-grid">
          <figure v-for="url in pipelineImages" :key="url" class="image-card">
            <template v-if="!missingImages.has(url)">
              <img :src="url" alt="Pipeline chart" @error="markMissing(url)" />
            </template>
            <template v-else>
              <div class="img-missing">
                <strong>Missing image</strong>
                <div class="img-url">{{ url }}</div>
              </div>
            </template>
          </figure>
        </div>
      </section>
    </div>
  </section>
</template>

<script>
const DEFAULT_PARAMS = {
  deviceType: 'ASCENDA3_Pod',
  deviceType2: 'ASCENDDAVID121',
  modelType: 'DEEPSEEK_V3',
  totalDie: 128,
  tpot: 50,
  kvLen: 4096,
  servingMode: 'AFD',
  microBatchNum: 3,
  deploymentMode: 'Homogeneous'
};

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

const MODEL_OPTIONS = [
  { value: 'DEEPSEEK_V3', label: 'DeepSeek V3' },
  { value: 'QWEN3_235B', label: 'Qwen3-235B-A22B' },
  { value: 'DEEPSEEK_V2_LITE', label: 'DeepSeek V2 Lite' }
];

const DEPLOYMENT_MODES = [
  { value: 'Homogeneous', label: 'Homogeneous' },
  { value: 'Heterogeneous', label: 'Heterogeneous' }
];

const DIE_OPTIONS = [64, 128, 256, 384, 512, 768];
const TPOT_OPTIONS = [20, 50, 70, 100, 150];
const KV_LEN_OPTIONS = [2048, 4096, 8192, 16384, 131072];

export default {
  setup() {
    const { ref, reactive, onMounted, computed } = window.LightLLMRuntime.Vue;
    const { useApi, useStore } = window.LightLLMRuntime;
    const api = useApi();
    const { getCsvSelection, setCsvSelection } = useStore();
    const params = reactive({
      ...DEFAULT_PARAMS,
      ...getCsvSelection()
    });
    const throughputImages = ref([]);
    const pipelineImages = ref([]);
    const loading = ref(false);
    const error = ref(null);
    const missingImages = reactive(new Set());

    const deviceTypeLabel = computed(() => {
      if (params.servingMode === 'AFD' && params.deploymentMode === 'Heterogeneous') {
        return 'Device Type (Attention)';
      }
      return 'Device Type';
    });

    const deviceType2Label = computed(() => {
      if (params.servingMode === 'AFD' && params.deploymentMode === 'Heterogeneous') {
        return 'Device Type 2 (FFN)';
      }
      return 'Device Type 2';
    });

    const markMissing = (url) => {
      missingImages.add(url);
    };

    const loadCharts = async () => {
      loading.value = true;
      error.value = null;
      missingImages.clear();

      try {
        const apiParams = {
          device_type: params.deviceType,
          model_type: params.modelType,
          total_die: String(params.totalDie),
          tpot: String(params.tpot),
          kv_len: String(params.kvLen),
          deployment_mode: params.deploymentMode
        };
        if (params.deploymentMode === 'Heterogeneous') {
          apiParams.device_type2 = params.deviceType2;
        }
        const data = await api.getResults(apiParams);
        throughputImages.value = Array.isArray(data.throughput_images) ? data.throughput_images : [];
        pipelineImages.value = Array.isArray(data.pipeline_images) ? data.pipeline_images : [];
        setCsvSelection({
          servingMode: params.servingMode || 'AFD',
          deploymentMode: params.deploymentMode,
          deviceType: params.deviceType,
          deviceType2: params.deviceType2,
          modelType: params.modelType,
          totalDie: Number(params.totalDie),
          tpot: Number(params.tpot),
          kvLen: Number(params.kvLen),
          microBatchNum: Number(params.microBatchNum) || 3
        });
      } catch (err) {
        error.value = err.message;
        throughputImages.value = [];
        pipelineImages.value = [];
      } finally {
        loading.value = false;
      }
    };

    onMounted(() => {
      Object.assign(params, {
        ...DEFAULT_PARAMS,
        ...getCsvSelection()
      });
      loadCharts();
    });

    return {
      params,
      throughputImages,
      pipelineImages,
      loading,
      error,
      missingImages,
      markMissing,
      loadCharts,
      deviceTypeLabel,
      deviceType2Label,
      deviceOptions: DEVICE_OPTIONS,
      modelOptions: MODEL_OPTIONS,
      deploymentModes: DEPLOYMENT_MODES,
      dieOptions: DIE_OPTIONS,
      tpotOptions: TPOT_OPTIONS,
      kvLenOptions: KV_LEN_OPTIONS
    };
  }
};
</script>

<style scoped>
.throughput-charts {
  display: grid;
  gap: 16px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.eyebrow {
  margin: 0 0 4px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 11px;
  color: #7c3aed;
  font-weight: 700;
}

h3 {
  margin: 0;
  font-size: 22px;
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
  background: linear-gradient(135deg, #7c3aed, #2563eb);
  color: #fff;
  box-shadow: 0 10px 24px rgba(124, 58, 237, 0.22);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.chart-controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  padding: 18px 20px;
  border: 1px solid #d8e0ea;
  border-radius: 18px;
  background: linear-gradient(180deg, #ffffff, #f8fafc);
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field span {
  font-size: 13px;
  font-weight: 700;
  color: #334155;
}

select {
  width: 100%;
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  padding: 10px 12px;
  background: #fff;
  color: #0f172a;
}

.error-banner {
  border: 1px solid #fecaca;
  background: #fff1f2;
  color: #9f1239;
  border-radius: 12px;
  padding: 12px 14px;
}

.chart-area {
  display: grid;
  gap: 16px;
}

.chart-section {
  border: 1px solid #d8e0ea;
  border-radius: 18px;
  background: #fff;
  padding: 18px 20px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
}

.chart-title {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 12px;
}

.chart-title h4 {
  margin: 0;
  font-size: 17px;
  color: #0f172a;
}

.chart-title span {
  color: #64748b;
  font-size: 13px;
}

.image-grid {
  display: grid;
  gap: 12px;
}

.image-card {
  margin: 0;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  background: linear-gradient(180deg, #f8fafc, #ffffff);
  overflow: hidden;
}

.image-card img {
  display: block;
  width: 100%;
  height: auto;
}

.empty-state {
  border: 1px dashed #cbd5e1;
  border-radius: 14px;
  padding: 24px;
  color: #64748b;
  text-align: center;
  background: #f8fafc;
}

.img-missing {
  padding: 16px;
  color: #9f1239;
  background: #fff1f2;
}

.img-url {
  margin-top: 6px;
  color: #64748b;
  font-size: 12px;
  word-break: break-all;
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

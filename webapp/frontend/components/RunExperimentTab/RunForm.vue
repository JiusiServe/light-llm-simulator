<template>
  <div class="run-form">
    <h3>Simulation Parameters</h3>

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
        <label>Deployment Mode</label>
        <select v-model="form.deployment_mode">
          <option value="Homogeneous">Homogeneous</option>
          <option value="Heterogeneous">Heterogeneous</option>
        </select>
      </div>
      <div class="field">
        <label>{{ deviceTypeLabel }}</label>
        <select v-model="form.device_type">
          <option v-for="device in deviceOptions" :key="device.value" :value="device.value">
            {{ device.label }}
          </option>
        </select>
      </div>
      <div class="field" v-if="form.deployment_mode === 'Heterogeneous'">
        <label>{{ deviceType2Label }}</label>
        <select v-model="form.device_type2">
          <option v-for="device in deviceOptions" :key="device.value" :value="device.value">
            {{ device.label }}
          </option>
        </select>
      </div>
    </div>

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
          <label>{{ minDieLabel }}</label>
          <input type="number" v-model.number="form.min_die" min="16" />
        </div>
        <div class="field">
          <label>{{ maxDieLabel }}</label>
          <input type="number" v-model.number="form.max_die" min="16" />
        </div>
      </div>
      <div class="field">
        <label>{{ dieStepLabel }}</label>
        <input type="number" v-model.number="form.die_step" min="1" />
      </div>
      <template v-if="form.deployment_mode === 'Heterogeneous'">
        <div class="field-row">
          <div class="field">
            <label>{{ minDie2Label }}</label>
            <input type="number" v-model.number="form.min_die2" min="16" />
          </div>
          <div class="field">
            <label>{{ maxDie2Label }}</label>
            <input type="number" v-model.number="form.max_die2" min="16" />
          </div>
        </div>
        <div class="field">
          <label>{{ dieStep2Label }}</label>
          <input type="number" v-model.number="form.die_step2" min="1" />
        </div>
      </template>
    </div>

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
const MODEL_OPTIONS = [
  { value: 'deepseek-ai/DeepSeek-V3', label: 'DeepSeek V3' },
  { value: 'deepseek-ai/DeepSeek-V3-2', label: 'DeepSeek V3.2' },
  { value: 'Qwen/Qwen3-235B-A22B', label: 'Qwen3-235B-A22B' },
  { value: 'deepseek-ai/DeepSeek-V2-Lite', label: 'DeepSeek V2 Lite' }
];

const DEVICE_OPTIONS = [
  { value: 'Ascend_910b2', label: 'Ascend 910B2' },
  { value: 'Ascend_910b3', label: 'Ascend 910B3' },
  { value: 'Ascend_910b4', label: 'Ascend 910B4' },
  { value: 'Ascend_A3Pod', label: 'Ascend A3Pod' },
  { value: 'Ascend_David100', label: 'Ascend David100' },
  { value: 'Ascend_David120', label: 'Ascend David120' },
  { value: 'Ascend_David121', label: 'Ascend David121' },
  { value: 'Nvidia_A100_SXM', label: 'Nvidia A100 SXM' },
  { value: 'Nvidia_H100_SXM', label: 'Nvidia H100 SXM' }
];

export default {
  setup() {
    const { ref } = window.LightLLMRuntime.Vue;
    const { useApi, useStore } = window.LightLLMRuntime;
    const api = useApi();
    const { addRun: addToHistory } = useStore();

    const form = ref({
      serving_mode: 'AFD',
      model_type: MODEL_OPTIONS[0].value,
      device_type: DEVICE_OPTIONS[0].value,
      deployment_mode: 'Homogeneous',
      device_type2: DEVICE_OPTIONS[1].value,
      min_attn_bs: 2,
      max_attn_bs: 1000,
      min_die: 16,
      max_die: 768,
      die_step: 16,
      min_die2: 16,
      max_die2: 768,
      die_step2: 16,
      next_n: 1,
      multi_token_ratio: 0.7,
      attn_tensor_parallel: 1,
      ffn_tensor_parallel: 1
    });

    const tpotInput = ref('50');
    const kvLenInput = ref('4096');
    const mbnInput = ref('3');
    const isSubmitting = ref(false);
    const error = ref(null);

    // Computed labels based on serving mode and deployment mode
    const { computed } = window.LightLLMRuntime.Vue;

    const deviceTypeLabel = computed(() => {
      if (form.value.serving_mode === 'AFD' && form.value.deployment_mode === 'Heterogeneous') {
        return 'Device Type (Attention)';
      }
      return 'Device Type';
    });

    const minDieLabel = computed(() => {
      if (form.value.serving_mode === 'AFD' && form.value.deployment_mode === 'Heterogeneous') {
        return 'Min Die (Attention)';
      }
      return 'Min Die';
    });

    const maxDieLabel = computed(() => {
      if (form.value.serving_mode === 'AFD' && form.value.deployment_mode === 'Heterogeneous') {
        return 'Max Die (Attention)';
      }
      return 'Max Die';
    });

    const dieStepLabel = computed(() => {
      if (form.value.serving_mode === 'AFD' && form.value.deployment_mode === 'Heterogeneous') {
        return 'Die Step (Attention)';
      }
      return 'Die Step';
    });

    const minDie2Label = computed(() => {
      if (form.value.serving_mode === 'AFD' && form.value.deployment_mode === 'Heterogeneous') {
        return 'Min Die 2 (FFN)';
      }
      return 'Min Die 2';
    });

    const maxDie2Label = computed(() => {
      if (form.value.serving_mode === 'AFD' && form.value.deployment_mode === 'Heterogeneous') {
        return 'Max Die 2 (FFN)';
      }
      return 'Max Die 2';
    });

    const dieStep2Label = computed(() => {
      if (form.value.serving_mode === 'AFD' && form.value.deployment_mode === 'Heterogeneous') {
        return 'Die Step 2 (FFN)';
      }
      return 'Die Step 2';
    });

    const deviceType2Label = computed(() => {
      if (form.value.serving_mode === 'AFD' && form.value.deployment_mode === 'Heterogeneous') {
        return 'Device Type (FFN)';
      }
      return 'Device Type 2';
    });

    const parseList = (value) => {
      if (!value) return [];
      return value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => Number(item))
        .filter((item) => !Number.isNaN(item));
    };

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
      } catch (err) {
        error.value = err && err.message ? err.message : 'Failed to start run';
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
      deviceTypeLabel,
      minDieLabel,
      maxDieLabel,
      dieStepLabel,
      deviceType2Label,
      minDie2Label,
      maxDie2Label,
      dieStep2Label
    };
  }
};
</script>

<style scoped>
.run-form {
  width: 100%;
  min-width: 0;
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

.field {
  margin-bottom: 10px;
  padding: 6px;
  background: #fff;
  border-radius: 5px;
  display: flex;
  flex-direction: column;
}

.field-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

label {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 4px;
}

input,
select {
  width: 100%;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-sizing: border-box;
}

button {
  width: auto;
  padding: 10px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
}

button:hover {
  background-color: #0056b3;
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

@media (max-width: 720px) {
  .field-row {
    grid-template-columns: 1fr;
  }

  button {
    width: 100%;
  }
}
</style>

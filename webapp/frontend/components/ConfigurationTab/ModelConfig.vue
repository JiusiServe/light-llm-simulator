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
const DEFAULT_MODEL_TYPE = 'deepseek-ai/DeepSeek-V3';

export default {
  props: {
    modelType: {
      type: String,
      default: DEFAULT_MODEL_TYPE
    }
  },
  setup(props) {
    const { ref, watch } = window.LightLLMRuntime.Vue;
    const { useApi } = window.LightLLMRuntime;
    const api = useApi();

    const config = ref(null);
    const loading = ref(true);
    const error = ref(null);

    const loadConfig = async () => {
      loading.value = true;
      error.value = null;

      try {
        const data = await api.getModelConfig(props.modelType);
        config.value = Object.assign({ model_type: props.modelType }, data);
      } catch (err) {
        error.value = err && err.message ? err.message : 'Failed to load model config';
      } finally {
        loading.value = false;
      }
    };

    watch(() => props.modelType, loadConfig, { immediate: true });

    return { config, loading, error };
  }
};
</script>

<style scoped>
.config-section {
  width: 100%;
  min-width: 0;
}

.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 24px;
}

.spinner {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 3px solid #d1d5db;
  border-top-color: #3b82f6;
  animation: spin 0.8s linear infinite;
}

.error {
  background-color: #fee2e2;
  color: #991b1b;
  padding: 12px;
  border-radius: 6px;
  margin-top: 12px;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.config-item {
  background: #f8fafc;
  border: 1px solid #d8e0ea;
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

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>

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
        <span class="value">{{ formatScaled(config.aichip_memory, GB_2_BYTE, 'GB') }}</span>
      </div>
      <div class="config-item">
        <span class="label">Intra-node Bandwidth</span>
        <span class="value">{{ formatScaled(config.intra_node_bandwidth, GB_2_BYTE, 'GB/s') }}</span>
      </div>
      <div class="config-item">
        <span class="label">Inter-node Bandwidth</span>
        <span class="value">{{ formatScaled(config.inter_node_bandwidth, GB_2_BYTE, 'GB/s') }}</span>
      </div>
      <div class="config-item">
        <span class="label">Local Memory Bandwidth</span>
        <span class="value">{{ formatScaled(config.local_memory_bandwidth, TB_2_BYTE, 'TB/s') }}</span>
      </div>
      <div class="config-item">
        <span class="label">BWSIO Bandwidth</span>
        <span class="value">{{ formatScaled(config.bwsio_memory_bandwidth, GB_2_BYTE, 'GB/s') }}</span>
      </div>
      <div class="config-item">
        <span class="label">On-chip Buffer</span>
        <span class="value">{{ formatScaled(config.onchip_buffer_size, MB_2_BYTE, 'MB') }}</span>
      </div>
      <div class="config-item">
        <span class="label">Cube FLOPS (FP16)</span>
        <span class="value">{{ formatScaled(config.cube_flops_fp16, TB_2_BYTE, 'TFLOPS') }}</span>
      </div>
      <div class="config-item">
        <span class="label">Cube FLOPS (INT8)</span>
        <span class="value">{{ formatScaled(config.cube_flops_int8, TB_2_BYTE, 'TFLOPS') }}</span>
      </div>
      <div class="config-item">
        <span class="label">Vector FLOPS (FP16)</span>
        <span class="value">{{ formatScaled(config.vector_flops_fp16, TB_2_BYTE, 'TFLOPS') }}</span>
      </div>
      <div class="config-item">
        <span class="label">Vector FLOPS (INT8)</span>
        <span class="value">{{ formatScaled(config.vector_flops_int8, TB_2_BYTE, 'TFLOPS') }}</span>
      </div>
    </div>
  </div>
</template>

<script>
const GB_2_BYTE = 1073741824;
const MB_2_BYTE = 1048576;
const TB_2_BYTE = 1099511627776;
const DEFAULT_DEVICE_TYPE = 'Ascend_A3Pod';

export default {
  props: {
    deviceType: {
      type: String,
      default: DEFAULT_DEVICE_TYPE
    }
  },
  setup(props) {
    const { ref, watch } = window.LightLLMRuntime.Vue;
    const { useApi } = window.LightLLMRuntime;
    const api = useApi();

    const config = ref(null);
    const loading = ref(true);
    const error = ref(null);

    const formatScaled = (value, divisor, unit) => {
      if (value === null || value === undefined) return 'N/A';
      return `${(Number(value) / divisor).toFixed(2)} ${unit}`;
    };

    const loadConfig = async () => {
      loading.value = true;
      error.value = null;

      try {
        const data = await api.getHardwareConfig(props.deviceType);
        config.value = Object.assign({ device_type: props.deviceType }, data);
      } catch (err) {
        error.value = err && err.message ? err.message : 'Failed to load hardware config';
      } finally {
        loading.value = false;
      }
    };

    watch(() => props.deviceType, loadConfig, { immediate: true });

    return {
      config,
      loading,
      error,
      GB_2_BYTE,
      MB_2_BYTE,
      TB_2_BYTE,
      formatScaled
    };
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

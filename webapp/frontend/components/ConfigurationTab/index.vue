<template>
  <div class="configuration-tab">
    <div class="selection-summary">
      <template v-if="selection.deployment_mode === 'Heterogeneous'">
        Showing configuration for {{ selection.model_type }} on {{ selection.device_type }} and {{ selection.device_type2 }}
      </template>
      <template v-else>
        Showing configuration for {{ selection.model_type }} on {{ selection.device_type }}
      </template>
    </div>
    <ModelConfig :model-type="selection.model_type" />
    <HardwareConfig :device-type="selection.device_type" :label="device1Label" />
    <HardwareConfig v-if="selection.deployment_mode === 'Heterogeneous' && selection.device_type2" :device-type="selection.device_type2" :label="device2Label" />
  </div>
</template>

<script>
import ModelConfig from './ModelConfig.vue';
import HardwareConfig from './HardwareConfig.vue';

const DEFAULT_SELECTION = {
  model_type: 'deepseek-ai/DeepSeek-V3',
  device_type: 'Ascend_A3Pod',
  deployment_mode: 'Homogeneous',
  device_type2: null
};

export default {
  components: { ModelConfig, HardwareConfig },
  setup() {
    const { computed } = window.LightLLMRuntime.Vue;
    const { useStore } = window.LightLLMRuntime;
    const { runHistory } = useStore();

    const selection = computed(() => ({
      ...DEFAULT_SELECTION,
      ...(runHistory.value[0]?.params || {})
    }));

    const device1Label = computed(() => {
      if (selection.value.serving_mode === 'AFD' && selection.value.deployment_mode === 'Heterogeneous') {
        return 'Device 1 (Attention)';
      }
      if (selection.value.deployment_mode === 'Heterogeneous') {
        return 'Device 1';
      }
      return 'Device';
    });

    const device2Label = computed(() => {
      if (selection.value.serving_mode === 'AFD' && selection.value.deployment_mode === 'Heterogeneous') {
        return 'Device 2 (FFN)';
      }
      return 'Device 2';
    });

    return { selection, device1Label, device2Label };
  }
};
</script>

<style scoped>
.configuration-tab {
  display: grid;
  gap: 20px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-items: start;
  width: 100%;
  min-width: 0;
}

.selection-summary {
  color: #475569;
  grid-column: 1 / -1;
  margin-bottom: 4px;
}

@media (max-width: 1080px) {
  .configuration-tab {
    grid-template-columns: 1fr;
  }
}
</style>

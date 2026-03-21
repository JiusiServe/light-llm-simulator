<template>
  <div class="configuration-tab">
    <div class="selection-summary">
      Showing configuration for {{ selection.model_type }} on {{ selection.device_type }}
    </div>
    <ModelConfig :model-type="selection.model_type" />
    <HardwareConfig :device-type="selection.device_type" />
  </div>
</template>

<script>
import ModelConfig from './ModelConfig.vue';
import HardwareConfig from './HardwareConfig.vue';

const DEFAULT_SELECTION = {
  model_type: 'deepseek-ai/DeepSeek-V3',
  device_type: 'Ascend_A3Pod'
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

    return { selection };
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

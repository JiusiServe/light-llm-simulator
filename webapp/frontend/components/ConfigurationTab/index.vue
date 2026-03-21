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
import { computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useStore } from '../../composables/useStore.js';
import ModelConfig from './ModelConfig.vue';
import HardwareConfig from './HardwareConfig.vue';

const DEFAULT_SELECTION = {
  model_type: 'deepseek-ai/DeepSeek-V3',
  device_type: 'Ascend_A3Pod'
};

export default {
  components: { ModelConfig, HardwareConfig },
  setup() {
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
  max-width: 700px;
}

.selection-summary {
  color: #475569;
  margin-bottom: 16px;
}
</style>

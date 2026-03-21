<template>
  <div>
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
      <RunExperimentTab v-if="currentTab === 'run'" />
      <ConfigurationTab v-else-if="currentTab === 'config'" />
      <ResultsTab v-else-if="currentTab === 'results'" />
      <VisualizationsTab v-else-if="currentTab === 'visualizations'" />
      <div v-else class="muted-state">Unknown tab selection.</div>
    </div>
  </div>
</template>

<script>
import RunExperimentTab from "./RunExperimentTab/index.vue";
import ConfigurationTab from "./ConfigurationTab/index.vue";
import ResultsTab from "./ResultsTab/index.vue";
import VisualizationsTab from "./VisualizationsTab/index.vue";

export default {
  components: {
    RunExperimentTab,
    ConfigurationTab,
    ResultsTab,
    VisualizationsTab,
  },
  setup() {
    const { useStore } = window.LightLLMRuntime;
    const { currentTab, setTab } = useStore();

    const tabs = [
      { id: "run", label: "Run Experiment" },
      { id: "config", label: "Configuration" },
      { id: "results", label: "Results" },
      { id: "visualizations", label: "Visualizations" },
    ];

    return {
      currentTab,
      setTab,
      tabs,
    };
  },
};
</script>

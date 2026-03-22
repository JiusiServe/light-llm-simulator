<template>
  <div class="run-experiment-tab">
    <RunForm />
    <RunStatus />
  </div>
</template>

<script>
import RunForm from './RunForm.vue';
import RunStatus from './RunStatus.vue';

export default {
  components: { RunForm, RunStatus },
  setup() {
    return {};
  }
};
</script>

<style scoped>
.run-experiment-tab {
  display: grid;
  gap: 24px;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.9fr);
  align-items: start;
}

@media (max-width: 1080px) {
  .run-experiment-tab {
    grid-template-columns: 1fr;
  }
}
</style>

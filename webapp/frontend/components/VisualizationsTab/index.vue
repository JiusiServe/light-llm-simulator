<template>
  <section class="visualizations-tab">
    <ThroughputCharts />
  </section>
</template>

<script>
import ThroughputCharts from './ThroughputCharts.vue';

export default {
  components: {
    ThroughputCharts
  }
};
</script>

<style scoped>
.visualizations-tab {
  display: grid;
  gap: 16px;
}
</style>

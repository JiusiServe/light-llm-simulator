<template>
  <section class="results-tab">
    <CsvSelector @csv-loaded="onCsvLoaded" />

    <ResultsFilter v-if="csvData.length > 0" @filter-change="onFilterChange" />

    <div class="results-meta">
      <span v-if="csvData.length > 0">
        Showing {{ filteredCount }} of {{ csvData.length }} results
      </span>
      <span v-else>Select a CSV file to load results.</span>
    </div>

    <ResultsTable
      v-if="csvData.length > 0"
      :data="csvData"
      :filters="currentFilters"
      @compare="onCompare"
    />
  </section>
</template>

<script>
import CsvSelector from './CsvSelector.vue';
import ResultsFilter from './ResultsFilter.vue';
import ResultsTable from './ResultsTable.vue';

function isBlank(value) {
  return value === null || value === undefined || value === '';
}

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}


export default {
  components: {
    CsvSelector,
    ResultsFilter,
    ResultsTable
  },
  setup() {
    const { ref, computed } = window.LightLLMRuntime.Vue;
    const { useStore } = window.LightLLMRuntime;
    const { setCsvSelection, setTab } = useStore();
    const csvData = ref([]);
    const currentFilters = ref({});

    const filteredCount = computed(() => {
      let rows = [...csvData.value];
      const filters = currentFilters.value || {};

      if (!isBlank(filters.min_die)) {
        rows = rows.filter((row) => toNumber(row.total_die) >= toNumber(filters.min_die));
      }
      if (!isBlank(filters.max_die)) {
        rows = rows.filter((row) => toNumber(row.total_die) <= toNumber(filters.max_die));
      }
      if (!isBlank(filters.min_throughput)) {
        rows = rows.filter(
          (row) => toNumber(row['throughput(tokens/die/s)']) >= toNumber(filters.min_throughput)
        );
      }
      if (!isBlank(filters.max_throughput)) {
        rows = rows.filter(
          (row) => toNumber(row['throughput(tokens/die/s)']) <= toNumber(filters.max_throughput)
        );
      }

      return rows.length;
    });

    const onCsvLoaded = (data) => {
      csvData.value = Array.isArray(data) ? data : [];
      currentFilters.value = {};
    };

    const onFilterChange = (filters) => {
      currentFilters.value = filters || {};
    };

    const onCompare = (selectedRows) => {
      const firstSelected = selectedRows[0];
      if (!firstSelected || firstSelected.total_die == null) {
        return;
      }

      setCsvSelection({
        totalDie: Number(firstSelected.total_die)
      });
      setTab('visualizations');
    };

    return {
      csvData,
      currentFilters,
      filteredCount,
      onCsvLoaded,
      onFilterChange,
      onCompare
    };
  }
};
</script>

<style scoped>
.results-tab {
  display: grid;
  gap: 16px;
  min-width: 0;
  width: 100%;
}

.results-tab > * {
  min-width: 0;
  width: 100%;
}

.results-meta {
  border-left: 4px solid #0f766e;
  background: linear-gradient(90deg, rgba(15, 118, 110, 0.08), rgba(37, 99, 235, 0.05));
  color: #334155;
  padding: 12px 14px;
  border-radius: 12px;
  font-size: 14px;
}
</style>

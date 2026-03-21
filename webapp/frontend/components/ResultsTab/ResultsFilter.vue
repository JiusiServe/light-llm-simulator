<template>
  <section class="results-filter">
    <div class="section-header">
      <div>
        <p class="eyebrow">Refine</p>
        <h4>Filter Loaded CSV</h4>
      </div>
      <button class="btn btn-ghost" @click="resetFilters">Reset</button>
    </div>

    <div class="filter-grid">
      <label class="field">
        <span>Min Total Die</span>
        <input v-model="filters.min_die" type="number" min="0" placeholder="Any" />
      </label>

      <label class="field">
        <span>Max Total Die</span>
        <input v-model="filters.max_die" type="number" min="0" placeholder="Any" />
      </label>

      <label class="field">
        <span>Min Throughput</span>
        <input v-model="filters.min_throughput" type="number" step="0.01" placeholder="Any" />
      </label>

      <label class="field">
        <span>Max Throughput</span>
        <input v-model="filters.max_throughput" type="number" step="0.01" placeholder="Any" />
      </label>
    </div>
  </section>
</template>

<script>
function createFilters() {
  return {
    min_die: null,
    max_die: null,
    min_throughput: null,
    max_throughput: null
  };
}

function toNumberOrNull(value) {
  if (value === null || value === undefined || value === '') {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeFilters(filters) {
  return {
    min_die: toNumberOrNull(filters.min_die),
    max_die: toNumberOrNull(filters.max_die),
    min_throughput: toNumberOrNull(filters.min_throughput),
    max_throughput: toNumberOrNull(filters.max_throughput)
  };
}

export default {
  emits: ['filter-change'],
  setup(props, { emit }) {
    const { reactive, watch } = window.LightLLMRuntime.Vue;
    const filters = reactive(createFilters());

    const resetFilters = () => {
      Object.assign(filters, createFilters());
    };

    watch(
      filters,
      () => {
        emit('filter-change', normalizeFilters(filters));
      },
      { deep: true, immediate: true }
    );

    return {
      filters,
      resetFilters
    };
  }
};
</script>

<style scoped>
.results-filter {
  border: 1px solid #d8e0ea;
  border-radius: 18px;
  padding: 18px 20px;
  background: linear-gradient(180deg, #ffffff, #f8fafc);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
  min-width: 0;
  width: 100%;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 14px;
}

.eyebrow {
  margin: 0 0 4px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 11px;
  color: #2563eb;
  font-weight: 700;
}

h4 {
  margin: 0;
  font-size: 17px;
  color: #0f172a;
}

.btn {
  appearance: none;
  border: 0;
  border-radius: 999px;
  padding: 8px 14px;
  font-weight: 700;
  cursor: pointer;
}

.btn-ghost {
  background: #e2e8f0;
  color: #0f172a;
}

.filter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  min-width: 0;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.field span {
  font-size: 13px;
  font-weight: 700;
  color: #334155;
}

input {
  width: 100%;
  min-width: 0;
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  padding: 10px 12px;
  background: #fff;
  color: #0f172a;
}

@media (max-width: 720px) {
  .section-header {
    align-items: stretch;
  }

  .btn-ghost {
    width: 100%;
  }
}
</style>

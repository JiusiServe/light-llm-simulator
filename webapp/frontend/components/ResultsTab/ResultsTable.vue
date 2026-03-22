<template>
  <section class="results-table">
    <div v-if="data.length === 0" class="empty-state">
      No results to display. Load a CSV file first.
    </div>

    <div v-else class="table-shell">
      <div v-if="filteredData.length === 0" class="empty-state">
        No rows match the current filters.
      </div>

      <template v-else>
        <div class="table-scroll">
          <table class="data-table">
            <thead>
              <tr>
                <th class="select-col">
                  <input type="checkbox" :checked="allSelected" @change="toggleAll" />
                </th>
                <th v-for="col in columns" :key="col" @click="sortBy(col)">
                  <span>{{ formatColumnName(col) }}</span>
                  <span class="sort-indicator">{{ sortIndicator(col) }}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in paginatedData" :key="row.__stableId">
                <td class="select-col">
                  <input
                    type="checkbox"
                    :checked="selectedRows.has(row.__stableId)"
                    @change="toggleSelect(row.__stableId)"
                  />
                </td>
                <td v-for="col in columns" :key="col">
                  {{ formatValue(row[col]) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="table-footer">
          <div class="pagination">
            <button class="btn btn-ghost" :disabled="page === 1" @click="setPage(page - 1)">
              Previous
            </button>
            <span>Page {{ page }} of {{ totalPages }}</span>
            <button class="btn btn-ghost" :disabled="page === totalPages" @click="setPage(page + 1)">
              Next
            </button>
          </div>

          <button class="btn btn-primary compare-btn" :disabled="selectedRows.size < 1" @click="emitCompare">
            View Charts ({{ selectedRows.size }})
          </button>
        </div>
      </template>
    </div>
  </section>
</template>

<script>
const PAGE_SIZE = 20;

function isBlank(value) {
  return value === null || value === undefined || value === '';
}

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}

export default {
  props: {
    data: {
      type: Array,
      default: () => []
    },
    filters: {
      type: Object,
      default: () => ({})
    }
  },
  emits: ['compare'],
  setup(props, { emit }) {
    const { reactive, ref, computed, watch } = window.LightLLMRuntime.Vue;
    const page = ref(1);
    const sortCol = ref('total_die');
    const sortDir = ref('desc');
    const selectedRows = reactive(new Set());

    const columns = computed(() => {
      if (!props.data.length) {
        return [];
      }
      return Object.keys(props.data[0]).filter((col) => col !== '__stableId');
    });

    const dataWithStableIds = computed(() =>
      props.data.map((row, index) => ({
        ...row,
        __stableId: index
      }))
    );

    const rowByStableId = computed(() => {
      return new Map(dataWithStableIds.value.map((row) => [row.__stableId, row]));
    });

    const filteredData = computed(() => {
      let rows = [...dataWithStableIds.value];
      const f = props.filters || {};

      if (!isBlank(f.min_die)) {
        rows = rows.filter((row) => toNumber(row.total_die) >= toNumber(f.min_die));
      }
      if (!isBlank(f.max_die)) {
        rows = rows.filter((row) => toNumber(row.total_die) <= toNumber(f.max_die));
      }
      if (!isBlank(f.min_throughput)) {
        rows = rows.filter(
          (row) => toNumber(row['throughput(tokens/die/s)']) >= toNumber(f.min_throughput)
        );
      }
      if (!isBlank(f.max_throughput)) {
        rows = rows.filter(
          (row) => toNumber(row['throughput(tokens/die/s)']) <= toNumber(f.max_throughput)
        );
      }

      return rows;
    });

    const sortedData = computed(() => {
      const rows = [...filteredData.value];
      const direction = sortDir.value === 'asc' ? 1 : -1;

      return rows.sort((a, b) => {
        const aVal = a[sortCol.value];
        const bVal = b[sortCol.value];
        const aNum = toNumber(aVal);
        const bNum = toNumber(bVal);

        if (!Number.isNaN(aNum) && !Number.isNaN(bNum)) {
          return (aNum - bNum) * direction;
        }

        return String(aVal).localeCompare(String(bVal)) * direction;
      });
    });

    const paginatedData = computed(() => {
      const start = (page.value - 1) * PAGE_SIZE;
      return sortedData.value.slice(start, start + PAGE_SIZE);
    });

    const totalPages = computed(() => Math.max(1, Math.ceil(filteredData.value.length / PAGE_SIZE)));

    const allSelected = computed(() => {
      return paginatedData.value.length > 0 && paginatedData.value.every((row) => selectedRows.has(row.__stableId));
    });

    const sortBy = (column) => {
      if (sortCol.value === column) {
        sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc';
      } else {
        sortCol.value = column;
        sortDir.value = 'desc';
      }
    };

    const sortIndicator = (column) => {
      if (sortCol.value !== column) {
        return '';
      }
      return sortDir.value === 'asc' ? '↑' : '↓';
    };

    const setPage = (nextPage) => {
      page.value = Math.min(Math.max(1, nextPage), totalPages.value);
    };

    const toggleSelect = (stableId) => {
      if (selectedRows.has(stableId)) {
        selectedRows.delete(stableId);
      } else {
        selectedRows.add(stableId);
      }
    };

    const toggleAll = () => {
      if (allSelected.value) {
        paginatedData.value.forEach((row) => selectedRows.delete(row.__stableId));
        return;
      }

      paginatedData.value.forEach((row) => selectedRows.add(row.__stableId));
    };

    const emitCompare = () => {
      const selectedData = Array.from(selectedRows)
        .map((stableId) => rowByStableId.value.get(stableId))
        .filter(Boolean)
        .map(({ __stableId, ...row }) => row);

      emit('compare', selectedData);
    };

    watch(
      () => props.filters,
      () => {
        page.value = 1;
        selectedRows.clear();
      },
      { deep: true }
    );

    watch(
      () => props.data,
      () => {
        page.value = 1;
        selectedRows.clear();
      }
    );

    const formatColumnName = (column) => {
      const [base, suffix] = column.split('(');
      const readableBase = base
        .replace(/_/g, ' ')
        .trim()
        .replace(/\b\w/g, (char) => char.toUpperCase());

      if (!suffix) {
        return readableBase;
      }

      return `${readableBase} (${suffix.replace(/\)$/, '').trim()})`;
    };

    const formatValue = (value) => {
      if (isBlank(value)) {
        return '-';
      }

      const parsed = typeof value === 'number' ? value : toNumber(value);
      if (!Number.isNaN(parsed)) {
        return Number.isInteger(parsed) ? String(parsed) : parsed.toFixed(2);
      }

      return value;
    };

    return {
      page,
      columns,
      paginatedData,
      totalPages,
      selectedRows,
      allSelected,
      sortBy,
      sortIndicator,
      setPage,
      toggleSelect,
      toggleAll,
      emitCompare,
      formatColumnName,
      formatValue,
      filteredData
    };
  }
};
</script>

<style scoped>
.results-table {
  border: 1px solid #d8e0ea;
  border-radius: 18px;
  padding: 18px 20px;
  background: #fff;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
  min-width: 0;
  width: 100%;
}

.empty-state {
  padding: 28px;
  text-align: center;
  color: #64748b;
  border: 1px dashed #cbd5e1;
  border-radius: 16px;
  background: linear-gradient(180deg, #f8fafc, #fff);
}

.table-scroll {
  max-width: 100%;
  min-width: 0;
  overflow-x: auto;
  border-radius: 14px;
}

.data-table {
  width: max-content;
  border-collapse: collapse;
  min-width: 100%;
}

.data-table th,
.data-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #e2e8f0;
  text-align: left;
  vertical-align: top;
  font-size: 14px;
  color: #0f172a;
}

.data-table th {
  position: sticky;
  top: 0;
  background: #f8fafc;
  z-index: 1;
  cursor: pointer;
  user-select: none;
  font-size: 13px;
  font-weight: 800;
  color: #334155;
  white-space: nowrap;
}

.data-table th:hover {
  background: #eef2ff;
}

.select-col {
  width: 40px;
  text-align: center !important;
}

.sort-indicator {
  margin-left: 6px;
  color: #2563eb;
}

.table-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 16px;
}

.pagination {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #475569;
  font-size: 14px;
}

.btn {
  appearance: none;
  border: 0;
  border-radius: 999px;
  padding: 10px 14px;
  font-weight: 700;
  cursor: pointer;
}

.btn-ghost {
  background: #e2e8f0;
  color: #0f172a;
}

.btn-primary {
  background: linear-gradient(135deg, #0f766e, #2563eb);
  color: #fff;
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
}

.btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.compare-btn {
  min-width: 160px;
}

input[type='checkbox'] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

@media (max-width: 720px) {
  .table-footer {
    flex-direction: column;
    align-items: stretch;
  }

  .pagination {
    justify-content: space-between;
  }

  .compare-btn {
    width: 100%;
  }
}
</style>

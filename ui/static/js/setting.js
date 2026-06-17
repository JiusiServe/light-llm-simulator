// Copyright (c) 2025 Huawei. All rights reserved.
let datasetConfig = [];

const SettingsApp = {

     async populateSelect(apiEndpoint, selectId, detailsId, fetchDetailsFunc) {
        try {
            const response = await fetch(apiEndpoint);
            const items = await response.json();
            const selectElement = document.getElementById(selectId);

            selectElement.innerHTML = '';
            items.forEach(item => {
                const option = document.createElement('option');
                option.value = item.value;
                option.textContent = item.label;
                selectElement.appendChild(option);
            });

            if (fetchDetailsFunc) {
                selectElement.onchange = () => fetchDetailsFunc(selectId, detailsId);
            }
        } catch (error) {
            SettingsApp.showCustomAlert(`Error fetching data from ${apiEndpoint}:`, error);
        }
    },

    attachSectionHeaderEvents() {
        document.querySelectorAll('.section-header').forEach(header => {
            header.addEventListener('click', () => {
                const content = header.nextElementSibling;
                const toggle = header.querySelector('.toggle');
                content.classList.toggle('active');
                toggle.textContent = content.classList.contains('active') ? '▼' : '▶';

                if (header.textContent.includes('Hardware Topology') && content.classList.contains('active')) {

                }
            });
        });
    },

    async  loadDatasets() {
        try {
            const response = await fetch("/static/config/dataset_config.json");
            const data = await response.json();

            datasetConfig = data.datasets;

            const select = document.getElementById("dataset-select");
            select.innerHTML = "";

            datasetConfig.forEach(ds => {
                const option = document.createElement("option");
                option.value = ds.name;
                option.textContent = ds.name;
                select.appendChild(option);
            });

            if (datasetConfig.length > 0) {
                SettingsApp.showDatasetInfo();
            }

        } catch (err) {
            console.error("Failed loading dataset config:", err);
        }
    },

     toggleDataConfig() {
        const mode = document.getElementById("data-mode").value;
        document.getElementById("random-config").style.display = mode === "random" ? "block" : "none";
        document.getElementById("real-data-config").style.display = mode === "real" ? "block" : "none";

        if (mode === "real") {
            SettingsApp.loadDatasets();
        }
    },

     showDatasetInfo() {
        const selectedName = document.getElementById("dataset-select").value;
        const dataset = datasetConfig.find(d => d.name === selectedName);

        const container = document.getElementById("dataset-details");
        container.innerHTML = "";

        if (!dataset) return;

        const ul = document.createElement("ul");

        Object.entries(dataset).forEach(([key, val]) => {
            const li = document.createElement("li");
            li.innerHTML = `<strong>${key}:</strong> ${val}`;
            ul.appendChild(li);
        });

        container.appendChild(ul);
    },

    initApp() {
        this.populateSelect('/api/models', 'model-type');
        this.populateSelect('/api/npus', 'npu-type', 'npu-details', this.fetchNpuDetails)
            .then(() => this.fetchNpuDetails('npu-type', 'npu-details'));
        SettingsApp.attachSectionHeaderEvents();
        SettingsApp.toggleQueueRange()
    },

    showCustomAlert(message) {
      document.getElementById('custom-alert-message').innerText = message;
      document.getElementById('custom-alert-overlay').style.display = 'flex';
    },

    closeCustomAlert() {
      document.getElementById('custom-alert-overlay').style.display = 'none';
    },

    async fetchNpuDetails(npuId, detailsId) {
        const npuType = document.getElementById(npuId).value;

        try {
            const response = await fetch(`/get_npu_details?npuType=${npuType}`);
            if (!response.ok) {
                throw new Error('Failed to fetch npu details');
            }
            const modelDetails = await response.json();

            const detailsHtml = `
                <ul>
                    <li style="font-family: Arial, sans-serif; font-size: 1.3em"
                        data-npu-memory="${modelDetails.npu_memory}">
                        <strong>Npu Memory (GB):</strong> ${modelDetails.npu_memory}
                    </li>
                    <li style="font-family: Arial, sans-serif; font-size: 1.3em"
                        data-npu-flops="${modelDetails.npu_flops}">
                        <strong>Npu Flops (TB):</strong> ${modelDetails.npu_flops}
                    </li>
                    <li style="font-family: Arial, sans-serif; font-size: 1.3em"
                        data-intra-node-bw="${modelDetails.intra_node_bandwidth}">
                        <strong>Intra Node BW (GB/s):</strong> ${modelDetails.intra_node_bandwidth}
                    </li>
                    <li style="font-family: Arial, sans-serif; font-size: 1.3em"
                        data-inter-node-bw="${modelDetails.inter_node_bandwidth}">
                        <strong>Inter Node BW (GB/s):</strong> ${modelDetails.inter_node_bandwidth}
                    </li>
                    <li style="font-family: Arial, sans-serif; font-size: 1.3em"
                        data-local-memory-bw="${modelDetails.local_memory_bandwidth}">
                        <strong>Local Memory BW (GB):</strong> ${modelDetails.local_memory_bandwidth}
                    </li>
                </ul>
            `;
            document.getElementById(detailsId).innerHTML = detailsHtml;
        } catch (error) {
            document.getElementById(detailsId).innerText = 'Failed to fetch npu details.';
        }
    },

    toggleQueueRange() {
        const scheduler = document.getElementById("scheduler-type").value;
        const wrapper = document.getElementById("queue-range-wrapper");

        if (scheduler === "ewsjf") {
            wrapper.style.display = "block";
        } else {
            wrapper.style.display = "none";
        }
    },

    getSettings() {
        const settings = {};

        // Model Configuration
        settings.modelType = document.getElementById("model-type")?.value || null;

        // Hardware
        settings.numNodes = parseInt(document.getElementById("num-nodes")?.value || 0, 10);
        settings.npusPerNode = parseInt(document.getElementById("npus-per-node")?.value || 0, 10);
        settings.npuType = document.getElementById("npu-type")?.value || null;

        // Data Config
        settings.dataMode = document.getElementById("data-mode")?.value || "random";

        if (settings.dataMode === "random") {
            settings.randomConfig = {
                inputLength: parseInt(document.getElementById("random-input-len")?.value || 0, 10),
                outputLength: parseInt(document.getElementById("random-output-len")?.value || 0, 10),
                rangeRatio: parseFloat(document.getElementById("random-range-ratio")?.value || 0),
            };
        } else if (settings.dataMode === "real") {
            settings.realData = {
                dataset: document.getElementById("dataset-select")?.value || null,
            };
        }

        settings.schedulerType = document.getElementById("scheduler-type")?.value || "ewsjf";
        if (settings.schedulerType === "ewsjf") {
            settings.queueRange = parseInt(document.getElementById("queue-range")?.value || 0, 10)
        } else if (settings.schedulerType === "fcfs") {

        }

        // General numeric settings
        settings.numPrompts = parseInt(document.getElementById("num-prompts")?.value || 0, 10);
        settings.rate = parseInt(document.getElementById("rate")?.value || 0, 10);
        settings.chunkSize = parseInt(document.getElementById("chunk-size")?.value || 0, 10);

        // Mode
        settings.mode = document.getElementById("mode")?.value || null;

        //parallel
        settings.pp = parseInt(document.getElementById("pp")?.value || 0, 10);
        settings.tp = parseInt(document.getElementById("tp")?.value || 0, 10);

        return settings;
    }
};

document.addEventListener('DOMContentLoaded', () => {
    SettingsApp.initApp();
});
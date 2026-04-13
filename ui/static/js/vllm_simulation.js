function createModernDashboard() {
    const results = document.getElementById("results");
    results.classList.add("dashboard-root");

    const leftColumn = document.createElement("div");
    leftColumn.className = "left-column";
    results.appendChild(leftColumn);

    const rightColumn = document.createElement("div");
    rightColumn.className = "right-column";
    results.appendChild(rightColumn);

    const metricsCard = document.createElement("div");
    metricsCard.className = "metrics-card";
    metricsCard.innerHTML = `
        <div class="table-container">
            <table>
                <tbody>
                    <tr><th colspan="2">Serving Benchmark Result</th></tr>
                    <tr><td>Duration</td><td id="durationValue">-</td></tr>
                    <tr><td>Completed</td><td id="completedValue">-</td></tr>
                    <tr><td>Total Input Tokens</td><td id="totalInputTokensValue">-</td></tr>
                    <tr><td>Total Output Tokens</td><td id="totalOutputTokensValue">-</td></tr>
                    <tr><td>Request Throughput</td><td id="requestThroughputValue">-</td></tr>
                    <tr><td>Output Throughput</td><td id="outputThroughputValue">-</td></tr>
                    <tr><td>Total Token Throughput</td><td id="totalTokenThroughputValue">-</td></tr>
                    
                    <tr><th colspan="2">Time to First Token</th></tr>
                    <tr><td>Mean TTFT (ms)</td><td id="meanTTFTValue">-</td></tr>
                    <tr><td>Median TTFT (ms)</td><td id="medianTTFTValue">-</td></tr>
                    <tr><td>P99 TTFT (ms)</td><td id="p99TTFTValue">-</td></tr>
                    
                    <tr><th colspan="2">Time per Output Token</th></tr>
                    <tr><td>Mean TPOT (ms)</td><td id="meanTPOTValue">-</td></tr>
                    <tr><td>Median TPOT (ms)</td><td id="medianTPOTValue">-</td></tr>
                    <tr><td>P99 TPOT (ms)</td><td id="p99TPOTValue">-</td></tr>
                    
                    <tr><th colspan="2">Inter-token Latency</th></tr>
                    <tr><td>Mean ITL (ms)</td><td id="meanITLValue">-</td></tr>
                    <tr><td>Median ITL (ms)</td><td id="medianITLValue">-</td></tr>
                    <tr><td>P99 ITL (ms)</td><td id="p99ITLValue">-</td></tr>
                </tbody>
            </table>
        </div>
    `;
    leftColumn.appendChild(metricsCard);

    const dynamicTableCard = document.createElement("div");
    dynamicTableCard.className = "graph-card";
    dynamicTableCard.style.overflowY = "auto";

    dynamicTableCard.innerHTML = `
        <table id="dynamicTable">
            <thead>
                <tr>
                    <th>Avg prompt throughput</th>
                    <th>Avg generation throughput</th>
                    <th>Running</th>
                    <th>Waiting</th>
                    <th>GPU KV cache usage</th>
                    <th>Prefix cache hit rate</th>
                </tr>
            </thead>
            <tbody>
            </tbody>
        </table>
    `;
    rightColumn.appendChild(dynamicTableCard);

    // Log
    const logCard = document.createElement("div");
    logCard.className = "log-card";
    logCard.id = "logBox";
    leftColumn.appendChild(logCard);

    // Live Graph Container
    const graphCard = document.createElement("div");
    graphCard.className = "live-graph-card";
    graphCard.innerHTML = `
        <canvas id="liveChartCanvas"></canvas>
    `;
    rightColumn.appendChild(graphCard);

    const ctx = document.getElementById("liveChartCanvas").getContext("2d");
    liveChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: "Avg Prompt Throughput",
                    data: [],
                    borderColor: "#007bff",
                    tension: 0.4
                },
                {
                    label: "Avg Generation Throughput",
                    data: [],
                    borderColor: "#ff5733",
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            animation: false,
            plugins: {
                legend: {
                    labels: {
                        font: {
                            size: 26
                        }
                    }
                },
                title: {
                    display: true,
                    text: "Live Throughput",
                    font: {
                        size: 28
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Step Update",
                        font: {
                            size: 24
                        }
                    },
                    ticks: {
                        font: {
                            size: 22
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "Tokens/sec",
                        font: {
                            size: 24
                        }
                    },
                    ticks: {
                        font: {
                            size: 22
                        }
                    }
                }
            }
        }
    });

    function updateGraph(data) {
        const timestamp = step_update;
        step_update += 1

        liveChart.data.labels.push(timestamp);
        liveChart.data.datasets[0].data.push(data.avgPromptThroughput);
        liveChart.data.datasets[1].data.push(data.avgGenerationThroughput);

        if (liveChart.data.labels.length > 50) {
            liveChart.data.labels.shift();
            liveChart.data.datasets[0].data.shift();
            liveChart.data.datasets[1].data.shift();
        }

        liveChart.update();
    }

    function formatNumber(value) {
        if (typeof value === "number") {
            return value.toFixed(2);
        }
        return value;
    }

    function updateMetrics(data) {
        document.getElementById("durationValue").textContent = formatNumber(data.duration);
        document.getElementById("completedValue").textContent = data.completed;
        document.getElementById("totalInputTokensValue").textContent = data.total_input_tokens;
        document.getElementById("totalOutputTokensValue").textContent = data.total_output_tokens;
        document.getElementById("requestThroughputValue").textContent = formatNumber(data.request_throughput);
        document.getElementById("outputThroughputValue").textContent = formatNumber(data.output_throughput);
        document.getElementById("totalTokenThroughputValue").textContent = formatNumber(data.total_token_throughput);

        document.getElementById("meanTTFTValue").textContent = formatNumber(data.mean_ttft_ms);
        document.getElementById("medianTTFTValue").textContent = formatNumber(data.median_ttft_ms);
        document.getElementById("p99TTFTValue").textContent = formatNumber(data.p99_ttft_ms);

        document.getElementById("meanTPOTValue").textContent = formatNumber(data.mean_tpot_ms);
        document.getElementById("medianTPOTValue").textContent = formatNumber(data.median_tpot_ms);
        document.getElementById("p99TPOTValue").textContent = formatNumber(data.p99_tpot_ms);

        document.getElementById("meanITLValue").textContent = formatNumber(data.mean_itl_ms);
        document.getElementById("medianITLValue").textContent = formatNumber(data.median_itl_ms);
        document.getElementById("p99ITLValue").textContent = formatNumber(data.p99_itl_ms);
    }

    function addDynamicRow(data) {
        const tbody = document.getElementById("dynamicTable").querySelector("tbody");
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${data.avgPromptThroughput}</td>
            <td>${data.avgGenerationThroughput}</td>
            <td>${data.running}</td>
            <td>${data.waiting}</td>
            <td>${data.gpuKVCacheUsage}</td>
            <td>${data.prefixCacheHitRate}</td>
        `;
        tbody.appendChild(row);
    }

    function log(message) {
        const div = document.getElementById("logBox");
        div.innerHTML += `
            <div class="log-line" >
                ${message}
            </div>`;
        div.scrollTop = div.scrollHeight;
    }

    return { updateMetrics, addDynamicRow, log , updateGraph};
}

let dashboard = null;
let step_update = 0;

window.socket = io();

socket.on('connect', () => {
});

socket.on('connect_error', (error) => {
});

socket.on('metrics', (data) => {
    dashboard.updateMetrics(data);
});

socket.on('throughput_details', (data) => {
    const values = [
        data.avgPromptThroughput,
        data.avgGenerationThroughput,
        data.running,
        data.waiting,
        data.gpuKVCacheUsage,
        data.prefixCacheHitRate
    ].map(Number);

    if (values.every(v => v === 0)) {
        return;
    }
    dashboard.addDynamicRow(data);
    dashboard.updateGraph(data);
});

socket.on('log', (msg) => {
    dashboard.log(msg);
});


async function runVLLM() {
    const container = document.getElementById("results");
    container.innerHTML = "";
    container.style.display = "";
    container.style.gridTemplateColumns = "";
    container.style.gap = "";
    container.style.padding = "";

    let data = SettingsApp.getSettings()

    dashboard = createModernDashboard();

    const response = await fetch("/run_vllm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    return await response.json();
}
const dopConfigs = [
    { pp: 1, tp: 1 },
    { pp: 1, tp: 8 },
    { pp: 2, tp: 4 },
    { pp: 4, tp: 2 },
    { pp: 8, tp: 1 }
];
const batchSizes = [1, 10, 100];
const metrics = ["TTFT",  "throughput_prefill","prefill_memory", "TBT",  "throughput_decode","decode_memory"];
const title = ["TTFT", "Prefill Throughput",  "Prefill Memory", "TBT", "Decode Throughput",  "Decode Memory"];

async function fetchKpi(data_from_ui) {
    const response = await fetch("/get_kpi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data_from_ui)
    });

    return await response.json();
}

function setupResultsGrid() {
    const container = document.getElementById("results");
    container.innerHTML = "";
    container.style.display = "grid";
    container.style.gridTemplateColumns = "repeat(3, 1fr)";
    container.style.gap = "20px";
    container.style.padding = "10px";
}

async function runCostModel() {
    setupResultsGrid();
    showLoader();

    let data_from_ui  = SettingsApp.getSettings()
    const results = {};

    metrics.forEach(m => results[m] = { labels: [], datasets: {} });

    let totalSteps = batchSizes.length * dopConfigs.length;
    let currentStep = 0;

    for (const batch of batchSizes) {
        for (const cfg of dopConfigs) {
            currentStep++;

            const percent = Math.round((currentStep / totalSteps) * 100);
            updateLoader(percent);

            const label = `pp=${cfg.pp},tp=${cfg.tp}`;
            data_from_ui["pp"] = cfg.pp;
            data_from_ui["tp"] = cfg.tp;
            data_from_ui["bs"] = batch;
            const data = await fetchKpi(data_from_ui);

            metrics.forEach(m => {
                if (!results[m].datasets[batch]) {
                    results[m].datasets[batch] = [];
                }
                results[m].labels.push(label);
                results[m].datasets[batch].push(data[m]);
            });
        }
    }
    hideLoader();
    renderCharts(results);
}

function renderCharts(results) {
    const container = document.getElementById("results");
    container.innerHTML = "";

    const colors = {
        1: "red",
        10: "blue",
        100: "green"
    };

    title.forEach((titleText, index) => {
        const metric = metrics[index];

        const wrapper = document.createElement("div");
        wrapper.style.position = "relative";
        wrapper.style.border = "1px solid #ccc";
        wrapper.style.padding = "10px";
        wrapper.style.borderRadius = "8px";
        wrapper.style.background = "#fafafa";
        wrapper.style.height = "40vh";
        wrapper.style.display = "flex";
        wrapper.style.flexDirection = "column";

        const canvas = document.createElement("canvas");
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        wrapper.appendChild(canvas);

        const iconContainer = document.createElement("div");
        iconContainer.style.position = "absolute";
        iconContainer.style.top = "5px";
        iconContainer.style.right = "5px";
        iconContainer.style.display = "flex";
        iconContainer.style.gap = "10px";

        const expandBtn = document.createElement("span");
        expandBtn.innerHTML = "🔍";
        expandBtn.style.cursor = "pointer";

        const downloadBtn = document.createElement("span");
        downloadBtn.innerHTML = "⬇️";
        downloadBtn.style.cursor = "pointer";

        iconContainer.appendChild(expandBtn);
        iconContainer.appendChild(downloadBtn);
        wrapper.appendChild(iconContainer);

        container.appendChild(wrapper);

        const chart = new Chart(canvas, {
            type: "line",
            data: {
                labels: [...new Set(results[metric].labels)],
                datasets: batchSizes.map(batch => ({
                    label: `Batch ${batch}`,
                    data: results[metric].datasets[batch],
                    borderWidth: 3,
                    borderColor: colors[batch],
                    backgroundColor: colors[batch],
                    tension: 0.3
                }))
            },
            options: {
                maintainAspectRatio: false,
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            font: { size: 18 }
                        }
                    },
                    title: {
                        display: true,
                        text: titleText,
                        font: { size: 22, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "PP / TP Configuration",
                            font: { size: 18 }
                        },
                        ticks: {
                            font: { size: 16 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: `Value (${titleText})`,
                            font: { size: 18 }
                        },
                        ticks: {
                            font: { size: 16 }
                        }
                    }
                }
            }
        });

        downloadBtn.onclick = () => {
            const link = document.createElement("a");
            link.download = `${titleText}.png`;
            link.href = chart.toBase64Image();
            link.click();
        };

        expandBtn.onclick = () => {

            const overlay = document.createElement("div");
            Object.assign(overlay.style, {
                position: "fixed",
                top: "0",
                left: "0",
                width: "100vw",
                height: "100vh",
                background: "rgba(0,0,0,0.7)",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                zIndex: "9999",
                cursor: "pointer"
            });

            const modalBox = document.createElement("div");
            Object.assign(modalBox.style, {
                width: "90vw",
                height: "90vh",
                background: "#ffffff",
                borderRadius: "12px",
                padding: "20px",
                boxSizing: "border-box",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                cursor: "default"
            });

            const modalCanvas = document.createElement("canvas");
            Object.assign(modalCanvas.style, {
                width: "100%",
                height: "100%"
            });

            modalBox.appendChild(modalCanvas);
            overlay.appendChild(modalBox);
            document.body.appendChild(overlay);

            new Chart(modalCanvas, chart.config);

            overlay.addEventListener("click", (event) => {
                if (event.target === overlay) {
                    overlay.remove();
                }
            });
        };



    });
}

function showLoader() {
    const loader = document.createElement("div");
    loader.id = "loadingOverlay";
    Object.assign(loader.style, {
        position: "fixed",
        top: "0",
        left: "0",
        width: "100vw",
        height: "100vh",
        background: "rgba(0,0,0,0.6)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
        fontSize: "2rem",
        color: "white",
        zIndex: "9999",
    });

    const text = document.createElement("div");
    text.id = "loadingText";
    text.innerText = "Loading... 0%";

    const spinner = document.createElement("div");
    Object.assign(spinner.style, {
        width: "80px",
        height: "80px",
        border: "10px solid #ccc",
        borderTop: "10px solid #4CAF50",
        borderRadius: "50%",
        animation: "spin 1s linear infinite",
        marginBottom: "20px"
    });

    loader.appendChild(spinner);
    loader.appendChild(text);
    document.body.appendChild(loader);

    const style = document.createElement("style");
    style.innerHTML = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
}

function updateLoader(percent) {
    const text = document.getElementById("loadingText");
    if (text) text.innerText = `Loading... ${percent}%`;
}

function hideLoader() {
    const loader = document.getElementById("loadingOverlay");
    if (loader) loader.remove();
}





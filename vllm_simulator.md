# vLLM PredSim Project

## Overview

This project is a simulator built on top of the vLLM system. It mimics the behavior of a real vLLM run, similar to what occurs in vLLM itself, but without using a GPU or executing the actual model. Instead, it relies on a cost model that estimates execution times and resource usage, allowing for efficient testing, benchmarking, and scheduling experiments without the need for full model inference.

### Environment Setup
#### Create and activate a virtual environment:

```bash
python3.12 -m venv vllm_predsim
source vllm_predsim/bin/activate
pip install --upgrade pip
```

### Installing vLLM (Required Version)
This project uses a development build of vLLM installed directly from the vLLM Git repository at a specific commit.

```bash
git clone https://github.com/sidikbro/vllm
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install -e .  

pip install flask flask_socketio
```

## Integrating PredSim Changes into VLLM
### Step 1: Clone the project and copy the UI and scheduler folders

Clone the light-llm-simulator repository (branch `predsimv1`):
```bash
git clone https://github.com/JiusiServe/light-llm-simulator.git
cd light-llm-simulator
git checkout predsimv1 
```

Copy the ui,sched folders into your VLLM project:
```bash
cp -r ui path/to/vllm/ui
cp -r vllm/v1/core/sched path/to/vllm/vllm/v1/core/sched

```
This will create the ui folder inside your VLLM project.

### Step 2: Replace modified files in VLLM

For all modified files in the PredSim repository, copy them into the corresponding paths inside your VLLM repository, replacing the original files.

Here is the list of files to copy:
```bash
examples/online_serving/api_client.py
vllm/benchmarks/serve.py
vllm/config/__init__.py
vllm/config/scheduler.py
vllm/distributed/parallel_state.py
vllm/engine/arg_utils.py
vllm/entrypoints/openai/api_server.py
vllm/v1/worker/cost_model.py
vllm/v1/worker/mock_gpu_model_runner.py
vllm/v1/worker/mock_model_runner.py
vllm/v1/worker/simulator_config.py
```
Copy each file or folder into the corresponding location in the VLLM project:

```bash
cp path/to/predsim/serve.py path/to/vllm/benchmarks/serve.py
cp path/to/predsim/__init__.py path/to/vllm/config/__init__.py
cp path/to/predsim/scheduler.py path/to/vllm/config/scheduler.py
# Repeat for each file/folder in the list
```

## Running the App

To start the application, activate the virtual environment and run the main entry file located in the `ui` directory from the VLLM project root.

```bash
python ui/app.py
```





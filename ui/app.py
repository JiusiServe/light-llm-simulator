import json
import os
import re
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO

from config.hw_config import DeviceType, HWConf, HardwareTopology
from config.constants import TB , GB
from config.models import ModelType
import vllm.v1.worker.simulator_config
from vllm.v1.core.sched.ewsjf_scheduler.scheduler_cls import SCHEDULER_CLS
from vllm.v1.worker.cost_model import CostModel, ParallelismConfig, ChunkContext
from huggingface_hub import login
import subprocess
import time
import socket
import subprocess
import threading
import sys

from vllm.config import (
    VllmConfig,
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    SchedulerConfig,
    DeviceConfig,
    LoadConfig,
    CompilationConfig
)

login("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")

class SimulatorApp:
    def __init__(self, host="0.0.0.0", port=5000, debug=True):
        self.host = host
        self.port = port
        self.debug = debug
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.ANSI_ESCAPE = re.compile(r'\x1B\[[0-9;]*[A-Za-z]')
        self.THROUGHPUT_PATTERN = re.compile(
        r"Avg prompt throughput: ([\d.]+) tokens/s, "
        r"Avg generation throughput: ([\d.]+) tokens/s, "
        r"Running: (\d+) reqs, "
        r"Waiting: (\d+) reqs, "
        r"GPU KV cache usage: ([\d.]+)%, "
        r"Prefix cache hit rate: ([\d.]+)%"
        )

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route('/api/models', methods=['GET'])
        def get_models():
            models = [{"value": model.value, "label": model.name.replace('_', '-')} for model in ModelType]
            return jsonify(models)

        @self.app.route('/get_npu_details', methods=['GET'])
        def get_npu_details():
            npu_type_str = request.args.get('npuType')
            if not npu_type_str:
                return jsonify({'error': 'Card type is required'}), 400

            try:
                npu_type = DeviceType(npu_type_str)
                ascend_config = HWConf.create(npu_type)
                return jsonify({
                    'npu_memory': ascend_config.npu_memory / GB,
                    'npu_flops': ascend_config.npu_flops_fp16 / TB,
                    'intra_node_bandwidth': ascend_config.intra_node_bandwidth / GB,
                    'inter_node_bandwidth': ascend_config.inter_node_bandwidth / GB,
                    'local_memory_bandwidth': ascend_config.local_memory_bandwidth / GB
                })
            except ValueError:
                return jsonify({'error': 'Invalid model type'}), 400

        @self.app.route('/api/npus', methods=['GET'])
        def get_npus():
            npus = [{"value": device.value, "label": device.name.replace('_', '-')} for device in DeviceType]
            return jsonify(npus)

        @self.app.route('/get_kpi', methods=['POST'])
        def get_kpi_for_dop():
            data = request.get_json()
            hw_topology = HardwareTopology.create(
                number_of_ranks=data["numNodes"],
                npus_per_rank=data["npusPerNode"],
                ascend_type=DeviceType(data["npuType"]),
            )

            hw_topology.compute_util = 0.6
            hw_topology.mem_bw_util = 0.8

            model_type = ModelType(data["modelType"])
            model_config = ModelConfig(
                model=model_type.value,
                tokenizer=model_type.value,
                dtype="float16",
                trust_remote_code=True
            )
            cache_config = CacheConfig()
            parallel_config = ParallelConfig(
                tensor_parallel_size=data["tp"],
                pipeline_parallel_size=data["pp"],
                data_parallel_size=1,
            )

            scheduler_config = SchedulerConfig(
                max_num_batched_tokens=data["chunkSize"],
                enable_chunked_prefill=True
            )

            device_config = DeviceConfig()
            load_config = LoadConfig()
            compilation_config = CompilationConfig()

            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                parallel_config=parallel_config,
                scheduler_config=scheduler_config,
                device_config=device_config,
                load_config=load_config,
                compilation_config=compilation_config
            )

            opt = CostModel(model_config, hw_topology, vllm_config)

            calculate_vllm_bench_ranges(data)
            infer_config = ChunkContext(batch_size=data["bs"], chunk_size_act=data["randomConfig"]["maxInputLength"],
                                           output_length=data["randomConfig"]["maxOutputLength"], stage='None')

            return opt.estimate_cost_of(infer_config)

        def calculate_vllm_bench_ranges(data):
            random_input_len = data["randomConfig"]["inputLength"]
            random_output_len = data["randomConfig"]["outputLength"]
            random_range_ratio = data["randomConfig"]["rangeRatio"]
            input_min = int(random_input_len * (1 - random_range_ratio))
            input_max = int(random_input_len * (1 + random_range_ratio))

            output_min = int(random_output_len * (1 - random_range_ratio))
            output_max = int(random_output_len * (1 + random_range_ratio))

            data["randomConfig"]["minInputLength"] = input_min
            data["randomConfig"]["maxInputLength"] = input_max
            data["randomConfig"]["minOutputLength"] = output_min
            data["randomConfig"]["maxOutputLength"] = output_max

        @self.app.route('/run_vllm', methods=['POST'])
        def run_vllm():
            data = request.get_json()

            calculate_vllm_bench_ranges(data)

            project_root = Path(__file__).resolve().parent.parent
            api_dir = project_root / "vllm" / "entrypoints" / "openai"
            benchmark_dir = project_root / "benchmarks"

            api_cmd = [
                sys.executable,
                "api_server.py",
                "--port", "10000",
                "--model", str(data["modelType"]),
                "--pipeline-parallel-size", str(data["pp"]),
                "--number-of-ranks", str(data["numNodes"]),
                "--npus-per-rank", str(data["npusPerNode"]),
                "--device-name-config", str(data["npuType"]),
                "--distributed-executor-backend", "mp",
                "--enforce-eager",
                "--max-num-batched-tokens", str(data["chunkSize"]),
            ]

            if str(data['schedulerType']) == 'ewsjf':
                external_parameters = {'min_input_length': data["randomConfig"]["minInputLength"],
                                       'max_input_length': data["randomConfig"]["maxInputLength"],
                                       'step_size': data["queueRange"]}

                base_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(base_dir, "..", "vllm", "v1", "core", "sched", "ewsjf_scheduler", "config.json")

                config_path = os.path.normpath(config_path)

                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(external_parameters, f, indent=4)

                api_cmd.extend(['--scheduler-cls', SCHEDULER_CLS['ewsjf']])

            api_process = subprocess.Popen(
                api_cmd,
                cwd=api_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False
            )

            self.stream_logs(api_process, self.socketio)
            server_pid = api_process.pid
            self.socketio.emit("log", f"Server PID: {server_pid}")

            self.socketio.emit("log", "Waiting for server to start...")
            server_started = False
            for i in range(60):
                if api_process.poll() is not None:
                    self.socketio.emit("log", "Server process died!")
                    return jsonify({"status": "Server process died"}), 500

                ret = subprocess.run(
                    ["curl", "-s", f"http://localhost:10000/health"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if ret.returncode == 0:
                    self.socketio.emit("log", "Server is ready!")
                    server_started = True
                    break

                self.socketio.emit("log", f"Still waiting... ({i + 1}/60)")
                time.sleep(10)

            if not server_started:
                self.socketio.emit("log", "Server failed to start within 10 minutes")
                api_process.terminate()
                return jsonify({"status": "Server failed to start"}), 500

            self.socketio.emit("log", "Running benchmark...")
            benchmark_cmd = [
                "vllm", "bench", "serve",
                "--port", "10000",
                "--model", str(data["modelType"]),
                "--dataset-name", "random",
                "--num-prompts", str(data["numPrompts"]),
                "--random-input-len", str(data["randomConfig"]["inputLength"]),
                "--random-output-len", str(data["randomConfig"]["outputLength"]),
                "--request-rate", str(data["rate"]),
                "--random-range-ratio", str(data["randomConfig"]["rangeRatio"]),
                "--ignore-eos"
            ]

            benchmark_process = subprocess.Popen(
                benchmark_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False
            )

            self.stream_logs(benchmark_process, self.socketio)
            benchmark_process.wait()
            self.socketio.emit("log", "Benchmark finished.")

            result_dir = project_root / "ui" / "benchmark_output"
            os.makedirs(result_dir, exist_ok=True)

            file_name = os.path.join(result_dir, "benchmark_result.json")

            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.socketio.emit("metrics", data)


            api_process.terminate()
            api_process.wait()
            self.socketio.emit("log", "API server terminated.")

            return jsonify({"status": "Benchmark finished successfully"})

    def run(self):
        print("Starting Simulator App...")
        self.app.run(host=self.host, port=self.port, debug=self.debug)

    def stream_logs(self, process, socketio):
        def read_stream(stream, prefix):
            if stream == None:
                return

            for line in iter(stream.readline, b""):
                raw = line.decode(errors='ignore').strip()
                clean_text = self.ANSI_ESCAPE.sub("", raw)

                match = self.THROUGHPUT_PATTERN.search(clean_text)
                if match:
                    data = {
                        "avgPromptThroughput": float(match.group(1)),
                        "avgGenerationThroughput": float(match.group(2)),
                        "running": int(match.group(3)),
                        "waiting": int(match.group(4)),
                        "gpuKVCacheUsage": float(match.group(5)),
                        "prefixCacheHitRate": float(match.group(6))
                    }
                    socketio.emit("throughput_details", data)
                    continue

                print(clean_text)

            stream.close()

        threading.Thread(target=read_stream, args=(process.stdout, "stdout"), daemon=True).start()
        threading.Thread(target=read_stream, args=(process.stderr, "stderr"), daemon=True).start()

if __name__ == "__main__":
    sim = SimulatorApp()
    sim.run()

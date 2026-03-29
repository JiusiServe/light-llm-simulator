import time
import types
import os
import gc

from torch import nn

from ui.config.hw_config import DeviceType, HardwareTopology
from vllm.distributed import get_pp_group
from vllm.model_executor import set_random_seed
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

import torch

from vllm.model_executor.model_loader.utils import set_default_torch_dtype, initialize_model
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.cost_model import CostModel
from vllm.v1.worker.mock_gpu_model_runner import GPUModelRunner_
from vllm.v1.engine.core import EngineCore
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config, unify_kv_cache_configs)
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment, Worker
from .cost_model import ParallelismConfig


class GPURunnerModelWrapper(nn.Module):
    def __init__(self, model, vllm_config, model_config):
        super().__init__()
        self.model = model.model
        self.model_config = model_config
        self.vllm_config = vllm_config
        hw_topology = HardwareTopology.create(number_of_ranks=model_config.number_of_ranks,
                                              npus_per_rank=model_config.npus_per_rank,
                                              ascend_type=DeviceType(model_config.device_name_config))
        
        hw_topology.compute_util = 0.6
        hw_topology.mem_bw_util = 0.8
        self.cost_model = CostModel(model_config, hw_topology, vllm_config)
        self.hidden_states = torch.randn(model_config.max_model_len, model_config.get_hidden_size(),
                                 dtype=model_config.dtype)

    def forward_dummy(self, input_ids=None, inputs_embeds=None, positions=None, intermediate_tensors=None, **kwargs):
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is not None:
            ### computed, scheduled, total
            workload = attn_metadata['cost']            

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.model.get_input_embeddings(input_ids)
            residual = self.hidden_states[:input_ids.shape[0], :]
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for idx, layer in enumerate(self.model.layers[self.model.start_layer:self.model.end_layer]):
            if idx in self.model.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + (residual if residual is not None else 0))

        module_names = [name for name, _ in self.model.layers[0].named_modules() if name]
        if attn_metadata is not None:
            start_time = time.perf_counter()
            time_to_finish = self.cost_model.get_workload_timing(workload, module_names)
            end_time = time.perf_counter()
            time_to_finish = time_to_finish * (self.model.end_layer - self.model.start_layer) - (end_time - start_time)
            if time_to_finish > 0.0:
                time.sleep(time_to_finish)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

def mock_load_model(self, vllm_config, model_config):    
    hidden_dim = model_config.hf_config.hidden_size
    num_layers = model_config.hf_config.num_hidden_layers
    vocab_size = model_config.hf_config.vocab_size

    device = torch.device("cpu")

    with set_default_torch_dtype(model_config.dtype):
        with device:
            model = initialize_model(vllm_config=vllm_config,
                                     model_config=model_config)

    model_wrapper = GPURunnerModelWrapper(model, vllm_config, model_config)
    model.forward = model_wrapper.forward_dummy

    print(f"[Mock] Using MockModel (hidden_dim={hidden_dim}, "
          f"num_layers={num_layers}, vocab_size={vocab_size})")

    return model.eval()

BaseModelLoader.load_model = mock_load_model

def mock_determine_available_memory(self) -> int:
    hw_topology = HardwareTopology.create(number_of_ranks=self.vllm_config.model_config.number_of_ranks,
                                          npus_per_rank=self.vllm_config.model_config.npus_per_rank,
                                          ascend_type=DeviceType(self.vllm_config.model_config.device_name_config))
    device_memory = hw_topology.hw_conf.npu_memory

    num_params = sum(p.numel() for p in self.model_runner.model.parameters())
    model_memory_bytes = num_params * self.model_runner.dtype.itemsize

    available_kv_cache_memory = device_memory - model_memory_bytes
    return int(available_kv_cache_memory)

Worker.determine_available_memory = mock_determine_available_memory

def mock_compile_or_warm_up_model(self) -> None:
    # Reset the seed to ensure that the random state is not affected by
    # the model initialization and profiling.
    set_random_seed(self.model_config.seed)
    self.model_runner.warming_up_model()

def mock_init_device(self):
    if self.device_config.device.type == "cuda":
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    else:
        raise RuntimeError(
            f"Not support device type: {self.device_config.device}")

    self.device = torch.device("cpu")

    gc.collect()

    # Initialize the distributed environment.
    init_worker_distributed_environment(self.vllm_config, self.rank,
                                        self.distributed_init_method,
                                        self.local_rank,
                                        current_platform.dist_backend)
    # Set random seed.
    set_random_seed(self.model_config.seed)

    # Construct the model runner
    self.model_runner: GPUModelRunner_ = GPUModelRunner_(
        self.vllm_config, self.device)

    if self.rank == 0:
        # If usage stat is enabled, collect relevant info.
        report_usage_stats(self.vllm_config)

Worker.compile_or_warm_up_model = mock_compile_or_warm_up_model
Worker.init_device = mock_init_device







import sys
from unittest.mock import MagicMock, patch
import importlib.machinery
import os

# Step 1: Mock C extensions
sys.modules['vllm._C'] = MagicMock()
sys.modules['vllm._moe_C'] = MagicMock()

# Step 2: Create Triton language mock with extra attribute
triton_lang_mock = MagicMock()
triton_lang_mock.__spec__ = importlib.machinery.ModuleSpec("triton.language", None)
triton_lang_mock.__name__ = "triton.language"

# Add extra.libdevice structure
triton_lang_mock.extra = MagicMock()
triton_lang_mock.extra.libdevice = MagicMock()
triton_lang_mock.extra.cuda = MagicMock()
triton_lang_mock.extra.cuda.libdevice = MagicMock()

# Step 3: Create main Triton mock
from vllm.triton_utils.importing import TritonPlaceholder

triton_mock = TritonPlaceholder()
triton_mock.__spec__ = importlib.machinery.ModuleSpec("triton", None)
triton_mock.backends = MagicMock()
triton_mock.backends.compiler = MagicMock()
triton_mock.backends.compiler.AttrsDescriptor = MagicMock()
triton_mock.language = triton_lang_mock

# Step 4: Mock ALL triton submodules in sys.modules
sys.modules['triton'] = triton_mock
sys.modules['triton.language'] = triton_lang_mock
sys.modules['triton.language.extra'] = triton_lang_mock.extra
sys.modules['triton.language.extra.libdevice'] = triton_lang_mock.extra.libdevice
sys.modules['triton.language.extra.cuda'] = triton_lang_mock.extra.cuda
sys.modules['triton.language.extra.cuda.libdevice'] = triton_lang_mock.extra.cuda.libdevice
sys.modules['triton.backends'] = triton_mock.backends
sys.modules['triton.backends.compiler'] = triton_mock.backends.compiler

# Step 5: Mock torch.cuda methods
import torch

torch.cuda.get_device_capability = MagicMock(return_value=(8, 0))
torch.cuda.get_device_properties = MagicMock()
torch.cuda.device_count = MagicMock(return_value=8)
torch.cuda.is_available = MagicMock(return_value=True)


# Add this to your mocking setup
def mock_cuda_get_device_properties(device, properties):
    """Mock implementation that returns fake device properties"""
    result = {}
    if "name" in properties:
        result["name"] = "NVIDIA H100"
    if "total_memory" in properties:
        result["total_memory"] = 80 * 1024 ** 3  # 80GB
    return result


# Mock torch.cuda.current_stream BEFORE torch.ops
mock_stream = MagicMock()
mock_stream.cuda_stream = 0
torch.cuda.current_stream = MagicMock(return_value=mock_stream)

# Mock torch.cuda.device context manager
torch.cuda.device = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

# Step 6: Mock torch.ops._C
if not hasattr(torch.ops, '_C'):
    torch.ops._C = MagicMock()
torch.ops._C.cutlass_scaled_mm_supports_fp8 = MagicMock(return_value=True)
torch.ops._C.cutlass_scaled_mm_supports_fp4 = MagicMock(return_value=True)
torch.ops._C.cutlass_scaled_mm_supports_block_fp8 = MagicMock(return_value=True)
torch.ops._C.cutlass_group_gemm_supported = MagicMock(return_value=True)
torch.ops._C.cutlass_sparse_scaled_mm_supported = MagicMock(return_value=True)
torch.ops._C.silu_and_mul = MagicMock()
torch.ops._C.mul_and_silu = MagicMock()
torch.ops._C.gelu_and_mul = MagicMock()
torch.ops._C.gelu_tanh_and_mul = MagicMock()
torch.ops._C.gelu_new = MagicMock()
torch.ops._C.gelu_fast = MagicMock()
torch.ops._C.gelu_quick = MagicMock()
torch.ops._C.fatrelu_and_mul = MagicMock()
torch.ops._C.swigluoai_and_mul = MagicMock()
torch.ops._C.rms_norm = MagicMock()
torch.ops._C.fused_add_rms_norm = MagicMock()
torch.ops._C.static_scaled_fp8_quant = MagicMock()
torch.ops._C.dynamic_scaled_fp8_quant = MagicMock()
torch.ops._C.dynamic_per_token_scaled_fp8_quant = MagicMock()
torch.ops._C.rms_norm_static_fp8_quant = MagicMock()
torch.ops._C.fused_add_rms_norm_static_fp8_quant = MagicMock()
torch.ops._C.rms_norm_dynamic_per_token_quant = MagicMock()
torch.ops._C.scaled_fp4_quant = MagicMock()

# Step 7: Set environment variables
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork" #"spawn"
os.environ["VLLM_DISTRIBUTED_BACKEND"] = "gloo"
os.environ["VLLM_USE_V1"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# Patch it before importing vLLM
import vllm.utils

vllm.utils.cuda_get_device_properties = mock_cuda_get_device_properties

##############################
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["NCCL_DEBUG"] = "TRACE"  # Even though using Gloo, this might help
os.environ["GLOO_LOG_LEVEL"] = "TRACE"
############################################
# Step 8: Set up platform mock
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability
import vllm.platforms

from vllm.utils import GiB_bytes

mock_platform = CudaPlatform()
mock_platform.get_device_capability = lambda device_id=0: DeviceCapability(major=8, minor=0)
mock_platform.device_count = lambda: 4
mock_platform.get_device_total_memory = lambda device_id=0: 32 * GiB_bytes
# Add this line to force Gloo backend
mock_platform.get_current_memory_usage = lambda device=None: 2 * GiB_bytes
mock_platform.dist_backend = "gloo"
vllm.platforms._current_platform = mock_platform
mock_platform.__iter__ = lambda self: iter([])  # To avoid iteration issues

# Step 9: Mock NCCL library - THIS IS THE KEY ADDITION
# Create mock NCCL library with all required operations
mock_nccl_lib = MagicMock()
mock_nccl_lib.ncclGetRawVersion = MagicMock(return_value=22703)
mock_nccl_lib.ncclGetVersion = MagicMock(return_value="2.27.3")
mock_nccl_lib.ncclGetUniqueId = MagicMock(return_value=MagicMock())
mock_nccl_lib.ncclCommInitRank = MagicMock(return_value=MagicMock())
mock_nccl_lib.ncclAllReduce = MagicMock(return_value=0)
mock_nccl_lib.ncclSend = MagicMock(return_value=0)
mock_nccl_lib.ncclRecv = MagicMock(return_value=0)
mock_nccl_lib.ncclAllGather = MagicMock(return_value=0)
mock_nccl_lib.ncclReduceScatter = MagicMock(return_value=0)
mock_nccl_lib.ncclBroadcast = MagicMock(return_value=0)
mock_nccl_lib.ncclGroupStart = MagicMock(return_value=0)
mock_nccl_lib.ncclGroupEnd = MagicMock(return_value=0)
mock_nccl_lib.ncclReduce = MagicMock(return_value=0)

# Patch NCCLLibrary class to return our mock
# This needs to be done as a context manager or permanent patch
from unittest.mock import patch
import vllm.distributed.device_communicators.pynccl as pynccl_module

# Store original NCCLLibrary
original_nccl_library = pynccl_module.NCCLLibrary

# Replace NCCLLibrary with a function that returns our mock
pynccl_module.NCCLLibrary = lambda *args, **kwargs: mock_nccl_lib

# Also patch it in pynccl_wrapper
import vllm.distributed.device_communicators.pynccl_wrapper as pynccl_wrapper_module

pynccl_wrapper_module.NCCLLibrary = lambda *args, **kwargs: mock_nccl_lib

from conf.model_config import ModelConfig, ModelType
from conf.hardware_config import HardwareTopology
from src.module.base import BaseModule
from src.module.deepseekv3_decode import (
    DeepSeekV3DecodeAttn,
    DeepSeekV3DecodeMLP,
    DeepSeekV3DecodeMoe,
)


def get_model(
    model_type: str,
    model_config: ModelConfig,
    aichip_config: HardwareTopology,
    search_config
)-> BaseModule:
    assert(model_type in ModelType), f"unsupport model {model_type}"

    if model_type == ModelType.DEEPSEEK_V3:
        attn = DeepSeekV3DecodeAttn(model_config, aichip_config, search_config)
        mlp = DeepSeekV3DecodeMLP(model_config, aichip_config, search_config)
        moe = DeepSeekV3DecodeMoe(model_config, aichip_config, search_config)
        model = {"attn": attn, "mlp": mlp, "moe": moe}
    return model

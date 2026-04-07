from conf.model_config import ModelType
from conf.config import Config
from src.model.base import BaseModule
from src.model.deepseekv3_decode import (
    DeepSeekV3DecodeEmbedding,
    DeepSeekV3DecodeAttn,
    DeepSeekV3DecodeMLP,
    DeepSeekV3DecodeMoe,
    DeepSeekV3DecodeLMHead
)
from src.model.qwen235_decode import (
    Qwen235DecodeEmbedding,
    Qwen235DecodeAttn,
    Qwen235DecodeMoe,
    Qwen235DecodeLMHead
)
from src.model.deepseekv2_lite_decode import (
    DeepSeekV2LiteDecodeEmbedding,
    DeepSeekV2LiteDecodeAttn,
    DeepSeekV2LiteDecodeMLP,
    DeepSeekV2LiteDecodeMoe,
    DeepSeekV2LiteDecodeLMHead
)


def get_model(
    config: Config
)-> BaseModule:
    '''
    Description:
        Get all modules of the specified model.
    Args:
        config: The configuration of the search task.
    Returns:
        A dictionary that contains all modules of the specified model.
    '''
    assert(config.model_type in ModelType), f"unsupport model {config.model_type}"

    if config.model_type == ModelType.DEEPSEEK_V3:
        embedding = DeepSeekV3DecodeEmbedding(config)
        attn = DeepSeekV3DecodeAttn(config)
        mlp = DeepSeekV3DecodeMLP(config)
        moe = DeepSeekV3DecodeMoe(config)
        lm_head = DeepSeekV3DecodeLMHead(config)
        model = {"embedding": embedding, "attn": attn, "mlp": mlp, "moe": moe, "lm_head": lm_head}
    if config.model_type == ModelType.QWEN3_235B:
        embedding = Qwen235DecodeEmbedding(config)
        attn = Qwen235DecodeAttn(config)
        moe = Qwen235DecodeMoe(config)
        lm_head = Qwen235DecodeLMHead(config)
        model = {"embedding": embedding, "attn": attn, "moe": moe, "lm_head": lm_head}
    if config.model_type == ModelType.DEEPSEEK_V2_LITE:
        embedding = DeepSeekV2LiteDecodeEmbedding(config)
        attn = DeepSeekV2LiteDecodeAttn(config)
        mlp = DeepSeekV2LiteDecodeMLP(config)
        moe = DeepSeekV2LiteDecodeMoe(config)
        lm_head = DeepSeekV2LiteDecodeLMHead(config)
        model = {"embedding": embedding, "attn": attn, "mlp": mlp, "moe": moe, "lm_head": lm_head}
    return model

def get_attention_family(
    model_type: str,
)-> str:
    '''
    Description:
        Get the attention mechanism of the specified model.
    Args:
        model_type: The type of the model.
    Returns:
        The attention mechanism of the specified model.
    '''
    assert(model_type in ModelType), f"unsupport model {model_type}"
    if model_type == ModelType.DEEPSEEK_V3 or model_type == ModelType.DEEPSEEK_V2_LITE:
        return "MLA"
    if model_type == ModelType.QWEN3_235B:
        return "GQA"

from enum import Enum


class ModelType(Enum):
    """
    Enumeration of supported model types with their HuggingFace model identifiers.

    This enum provides a centralized way to reference different LLM models
    and ensures type safety when working with model configurations.
    """
    Llama3_8B = "meta-llama/Meta-Llama-3-8B-Instruct"
    Llama3_70B = "meta-llama/Meta-Llama-3-70B-Instruct"
    Qwen2_5_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
    Qwen2_5_7B = "Qwen/Qwen2.5-7B-Instruct"

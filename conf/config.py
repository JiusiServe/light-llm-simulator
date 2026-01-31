from conf.model_config import ModelConfig, ModelType
from conf.hardware_config import HWConf, DeviceType
from conf.common import MIN_ROUTED_EXPERT_PER_DIE
import math


class Config:
    def __init__(
        self,
        serving_mode: str,
        model_type: ModelType,
        device_type1: DeviceType,
        device_type2: DeviceType,
        min_attn_bs: int,
        max_attn_bs: int,
        min_die1: int,
        max_die1: int,
        min_die2: int,
        max_die2: int,
        die_step1: int,
        die_step2: int,
        tpot:list[int],
        kv_len: list[int],
        micro_batch_num: list[int],
        next_n: int,
        multi_token_ratio: float,
        attn_tensor_parallel: int,
        ffn_tensor_parallel: int
    ) -> None:
        """
        Initialize a Config object.
        A Config object contains all configurations of the search task.
        TODO:
        Allow passing in a yaml file to do patch

        Args:
            serving_mode: The serving mode of the task.
            model_type: The type of the model.
            device_type1: The type of the device1.
                          For AFD, it is the device type to explore attention module.
            device_type2: The type of the device2.
                          For AFD, it is the device type to explore ffn module.
            min_attn_bs: The min number of attention batch size to explore.
            max_attn_bs: The max number of attention batch size to explore.
            min_die1: The min number of die to explore for device_type1.
            max_die1: The max number of die to explore for device_type1.
            min_die2: The min number of die to explore for device_type2.
            max_die2: The max number of die to explore for device_type2.
            die_step1: The step size of the die to explore for device_type1.
            die_step2: The step size of the die to explore for device_type2.
            tpot: The target TPOT.
            kv_len: The input sequence length.
            micro_batch_num: The micro batch number.
            next_n: Predict the next n tokens through the MTP(Multi-Token Prediction) technique.
            multi_token_ratio: The acceptance rate of the additionally predicted token.
            attn_tensor_parallel: Number of dies used for tensor model parallelism.
            ffn_tensor_parallel: Number of dies used for tensor model parallelism.
        """
        self.serving_mode = serving_mode
        model_type = ModelType(model_type)
        self.model_type = model_type
        self.model_config = ModelConfig.create_model_config(model_type)
        self.device_type1 = DeviceType(device_type1)
        self.device_type2 = DeviceType(device_type2)
        self.aichip_config1 = HWConf.create(self.device_type1)
        self.aichip_config2 = HWConf.create(self.device_type2)
        self.min_attn_bs = min_attn_bs
        self.max_attn_bs = max_attn_bs
        self.min_die1 = min_die1
        self.max_die1 = max_die1
        self.min_die2 = min_die2
        self.max_die2 = max_die2
        self.die_step1 = die_step1
        self.die_step2 = die_step2
        self.tpot = tpot
        self.kv_len = kv_len
        self.micro_batch_num = micro_batch_num
        self.seq_len = next_n + 1
        self.multi_token_ratio = multi_token_ratio
        self.attn_tensor_parallel = attn_tensor_parallel
        self.ffn_tensor_parallel = ffn_tensor_parallel
        self.attn_bs = min_attn_bs
        self.ffn_bs = self.attn_bs * self.model_config.num_experts_per_tok
        self.attn_die = min_die1
        self.ffn_die = min_die2
        self.routed_expert_per_die = max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.model_config.n_routed_experts / self.ffn_die)
            )

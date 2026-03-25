from conf.model_config import ModelConfig, ModelType
from conf.hardware_config import HWConf, DeviceType
from conf.common import MIN_ROUTED_EXPERT_PER_DIE
import math
from typing import Optional


class Config:
    def __init__(
        self,
        serving_mode: str,
        model_type: ModelType,
        device_type: DeviceType,
        min_attn_bs: int,
        max_attn_bs: int,
        min_die: int,
        max_die: int,
        die_step: int,
        tpot: list[int],
        kv_len: list[int],
        micro_batch_num: list[int],
        next_n: int,
        multi_token_ratio: float,
        attn_tensor_parallel: int,
        ffn_tensor_parallel: int,
        deployment_mode: str = "Homogeneous",
        device_type2: Optional[str] = None,
        min_die2: Optional[int] = None,
        max_die2: Optional[int] = None,
        die_step2: Optional[int] = None
    ) -> None:
        """
        Initialize a Config object.
        A Config object contains all configurations of the search task.
        TODO:
        Allow passing in a yaml file to do patch

        Args:
            serving_mode: The serving mode of the task ("AFD" or "DeepEP").
            model_type: The type of the model.
            device_type: The type of the device (for attention in Heterogeneous mode).
            min_attn_bs: The min number of attention batch size to explore.
            max_attn_bs: The max number of attention batch size to explore.
            min_die: The min number of die to explore (for attention in Heterogeneous mode).
            max_die: The max number of die to explore (for attention in Heterogeneous mode).
            die_step: The step size of the die to explore.
            tpot: The target TPOT.
            kv_len: The input sequence length.
            micro_batch_num: The micro batch number.
            next_n: Predict the next n tokens through the MTP(Multi-Token Prediction) technique.
            multi_token_ratio: The acceptance rate of the additionally predicted token.
            attn_tensor_parallel: Number of dies used for tensor model parallelism.
            ffn_tensor_parallel: Number of dies used for tensor model parallelism.
            deployment_mode: "Homogeneous" or "Heterogeneous".
            device_type2: The second device type for FFN in Heterogeneous mode.
            min_die2: The min number of FFN dies to explore in Heterogeneous mode.
            max_die2: The max number of FFN dies to explore in Heterogeneous mode.
            die_step2: The step size for FFN dies in Heterogeneous mode.
        """
        self.serving_mode = serving_mode

        # Validate deployment_mode
        valid_deployment_modes = ("Homogeneous", "Heterogeneous")
        if deployment_mode not in valid_deployment_modes:
            raise ValueError(f"Invalid deployment_mode: {deployment_mode}. Must be one of {valid_deployment_modes}")
        self.deployment_mode = deployment_mode

        model_type = ModelType(model_type)
        self.model_type = model_type
        self.model_config = ModelConfig.create_model_config(model_type)

        # Device type for attention (device1)
        self.device_type = DeviceType(device_type)
        self.aichip_config = HWConf.create(self.device_type)

        # Heterogeneous mode support
        if deployment_mode == "Heterogeneous":
            if device_type2 is None:
                raise ValueError("device_type2 is required for Heterogeneous deployment mode")
            self.device_type2 = DeviceType(device_type2)
            self.aichip_config2 = HWConf.create(self.device_type2)
        else:
            self.device_type2 = self.device_type
            self.aichip_config2 = self.aichip_config

        # Backward compatible aliases for module access
        self.aichip_config_attn = self.aichip_config
        self.aichip_config_ffn = self.aichip_config2

        # Die search parameters
        self.min_attn_bs = min_attn_bs
        self.max_attn_bs = max_attn_bs
        self.min_die = min_die  # attn_die min in Heterogeneous mode
        self.max_die = max_die  # attn_die max in Heterogeneous mode
        self.die_step = die_step

        # FFN die search parameters for Heterogeneous mode
        if deployment_mode == "Heterogeneous":
            self.min_die2 = min_die2 if min_die2 is not None else min_die
            self.max_die2 = max_die2 if max_die2 is not None else max_die
            self.die_step2 = die_step2 if die_step2 is not None else die_step
        else:
            self.min_die2 = min_die
            self.max_die2 = max_die
            self.die_step2 = die_step

        self.tpot = tpot
        self.kv_len = kv_len
        self.micro_batch_num = micro_batch_num
        self.seq_len = next_n + 1
        self.multi_token_ratio = multi_token_ratio
        self.attn_tensor_parallel = attn_tensor_parallel
        self.ffn_tensor_parallel = ffn_tensor_parallel
        self.attn_bs = min_attn_bs
        self.ffn_bs = self.attn_bs * self.model_config.num_experts_per_tok
        self.attn_die = min_die
        self.ffn_die = min_die
        self.routed_expert_per_die = max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.model_config.n_routed_experts / self.ffn_die)
            )

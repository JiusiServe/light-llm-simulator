from abc import ABC, abstractmethod
from typing import List, Literal
from conf.config import Config


class BaseModule(ABC):
    """
    Description:
        Base class for all model modules.
        A model is composed of multiple modules.
        For example, a DeepSeek V3 model is composed of an attention module, an MLP module and a MoE module.
        Inherit from this class to create a new module.
    Attributes:
        config: The configuration of the search task.
        model_config: The configuration of the model. It is a attribute of the config.
        aichip_config: The configuration of the hardware. It is a attribute of the config.
        e2e_time: The end-to-end time of the module.It is the max of compute and memory time.
        compute_time: The compute time of the module.
        memory_time: The memory time of the module.
    """
    def __init__(
        self,
        config: Config,
        hw_type: Literal["attn", "ffn"] = "attn"
    ) -> None:
        self.config = config
        self.model_config = config.model_config
        # Select the appropriate hardware config based on module type
        # In Heterogeneous mode: attn modules use device_type, ffn modules use device_type2
        # In Homogeneous mode: both use device_type
        if hw_type == "attn":
            self.aichip_config = config.aichip_config_attn
        else:  # ffn
            self.aichip_config = config.aichip_config_ffn

        self.e2e_time: float = 0.0
        self.compute_time: float = 0.0
        self.memory_time: float = 0.0

        self.ops: List = []

    def __call__(self):
        self._execute_ops()
        self._aggregate_times()

    @abstractmethod
    def _build_ops(self):
        """
        Description:
            Build the list that contains all operators for the module.
        """
        pass

    def _execute_ops(self):
        """
        Description:
            Execute all operators for the module.
        """
        for op in self.ops:
            op()

    @abstractmethod
    def _aggregate_times(self):
        """
        Description:
            Aggregate the times of the operators for the module to obtain 
            the module's memory time, compute time and end-to-end time.
        """
        pass

from abc import ABC, abstractmethod
from typing import List
from conf.model_config import ModelConfig
from conf.hardware_config import HardwareTopology


class BaseModule(ABC):
    def __init__(
        self,
        model_config: ModelConfig,
        aichip_config: HardwareTopology,
        search_config
    ):
        self.model_config = model_config
        self.aichip_config = aichip_config
        self.search_config = search_config

        self.e2e_time: float = 0.0
        self.compute_time: float = 0.0
        self.memory_time: float = 0.0

        self.ops: List = []

    def __call__(self):
        self._execute_ops()
        self._aggregate_times()

    @abstractmethod
    def _build_ops(self):
        pass

    def _execute_ops(self):
        for op in self.ops:
            op()

    @abstractmethod
    def _aggregate_times(self):
        pass

from dataclasses import dataclass
from config.config_utils import overwrite_config

from typing import Any


@dataclass
class PAVPU:
    sliding_threshold: bool
    static_theshold_value: float
    window: int


@dataclass
class Metrics:
    pavpu: PAVPU

    @staticmethod
    @overwrite_config
    def load_config(pavpu: dict[str, Any]) -> "Metrics":
        return Metrics(pavpu=PAVPU(**pavpu))

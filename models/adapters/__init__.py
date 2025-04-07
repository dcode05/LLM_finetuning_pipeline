"""
Adapters module for parameter-efficient fine-tuning.
"""

from models.adapters.base import BaseAdapter
from models.adapters.lora_adapter import LoRAAdapter
from models.adapters.qlora_adapter import QLoRAAdapter
from models.adapters.prompt_tuning_adapter import PromptTuningAdapter
from models.adapters.prefix_tuning_adapter import PrefixTuningAdapter

__all__ = [
    "BaseAdapter",
    "LoRAAdapter",
    "QLoRAAdapter",
    "PromptTuningAdapter",
    "PrefixTuningAdapter",
] 
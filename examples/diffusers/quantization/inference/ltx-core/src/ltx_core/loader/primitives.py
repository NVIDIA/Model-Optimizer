# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Alexey Kravtsov
from collections import namedtuple
from dataclasses import dataclass
from typing import Protocol

import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_keys_ops import SDKeyOps


@dataclass(frozen=True)
class StateDict:
    sd: dict
    device: torch.device
    size: int
    dtype: torch.dtype

    def footprint(self) -> tuple[int, torch.device]:
        return self.size, self.device


class StateDictLoader(Protocol):
    def metadata(self, path: str) -> dict:
        """
        Load metadata from path
        """

    def load(
        self, path: str | list[str], sd_key_ops: SDKeyOps | None = None, device: torch.device | None = None
    ) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_key_ops
        """


class ModelBuilderProtocol(Protocol):
    def meta_model(self, config: dict, module_ops: list[ModuleOps] | None = None) -> torch.nn.Module: ...

    def build(self, dtype: torch.dtype | None = None) -> torch.nn.Module:
        """
        Build the model
        Args:
            dtype: Target dtype for the model, if None, uses the dtype of the model_path model
        Returns:
            Model instance
        """
        ...


class LoRAAdaptableProtocol(Protocol):
    def lora(self, lora_path: str, strength: float) -> "LoRAAdaptableProtocol":
        pass


LoraPathStrengthAndKeyOps = namedtuple("LoraPathStrengthAndKeyOps", ["path", "strength", "sd_key_ops"])
LoraStateDictWithStrength = namedtuple("LoraStateDictWithStrength", ["state_dict", "strength"])

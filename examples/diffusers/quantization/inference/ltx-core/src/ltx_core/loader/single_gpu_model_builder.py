# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Alexey Kravtsov
from dataclasses import dataclass, field, replace

import torch

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndKeyOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_keys_ops import SDKeyOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator


@dataclass(frozen=True)
class SingleGPUModelBuilder(ModelBuilderProtocol, LoRAAdaptableProtocol):
    model_class_configurator: ModelConfigurator
    model_path: str | tuple[str, ...]
    model_sd_key_ops: SDKeyOps | None = None
    loras: tuple[LoraPathStrengthAndKeyOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)

    def lora(
        self, lora_path: str, strength: float = 1.0, sd_key_ops: SDKeyOps | None = None
    ) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndKeyOps(lora_path, strength, sd_key_ops)))

    def model_config(self) -> dict:
        first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
        return self.model_loader.metadata(first_shard_path)

    def meta_model(self, config: dict, module_ops: list[ModuleOps] | None = None) -> torch.nn.Module:
        with torch.device("meta"):
            model = self.model_class_configurator.from_config(config)
        if module_ops is not None and len(module_ops) > 0:
            # TODO: apply module ops to the model
            raise NotImplementedError("Module ops are not implemented yet")
        return model

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_key_ops: SDKeyOps | None = None
    ) -> StateDict:
        state_dict = registry.get(paths, sd_key_ops)
        if state_dict is None:
            state_dict = self.model_loader.load(paths, sd_key_ops=sd_key_ops, device=device)
            registry.add(paths, sd_key_ops=sd_key_ops, state_dict=state_dict)
        return state_dict

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.nn.Module:
        device = torch.device("cuda") if device is None else device
        config = self.model_config()
        meta_model = self.meta_model(config)
        model_paths = self.model_path if isinstance(self.model_path, tuple) else [self.model_path]
        model_state_dict = self.load_sd(
            model_paths, sd_key_ops=self.model_sd_key_ops, registry=self.registry, device=device
        )
        dtype = model_state_dict.dtype if dtype is None else dtype

        lora_strengths = [lora.strength for lora in self.loras]
        if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
            sd = {key: value.to(dtype=dtype) for key, value in model_state_dict.sd.items()}
            meta_model.load_state_dict(sd, strict=False, assign=True)
            return meta_model

        lora_state_dicts = [
            self.load_sd([lora.path], sd_key_ops=lora.sd_key_ops, registry=self.registry, device=device)
            for lora in self.loras
        ]
        lora_sd_and_strengths = [
            LoraStateDictWithStrength(sd, strength)
            for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
        ]
        final_sd = apply_loras(model_state_dict, lora_sd_and_strengths, dtype)
        meta_model.load_state_dict(final_sd.sd, strict=False, assign=True)
        return meta_model

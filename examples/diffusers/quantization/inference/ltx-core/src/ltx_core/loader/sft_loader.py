# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Alexey Kravtsov
import json

import safetensors
import torch

from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.sd_keys_ops import CompulsoryContent, ContentReplacement, SDKeyOps


def _apply_sd_key_ops(name: str, sd_key_ops: SDKeyOps) -> str:
    filters = [content for content in sd_key_ops.mapping if isinstance(content, CompulsoryContent)]
    valid = any(name.startswith(f.prefix) and name.endswith(f.suffix) for f in filters)
    valid = valid if len(filters) > 0 else True
    if not valid:
        return None

    for replacements in sd_key_ops.mapping:
        if not isinstance(replacements, ContentReplacement):
            continue
        before, after = replacements
        if before in name:
            name = name.replace(before, after)
    return name


class SafetensorsStateDictLoader(StateDictLoader):
    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_key_ops: SDKeyOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_key_ops
        """
        sd = {}
        size = 0
        dtype = None
        if device is None:
            device = torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        for shard_path in model_paths:
            with safetensors.safe_open(shard_path, framework="pt") as f:
                safetensor_keys = f.keys()
                for name in safetensor_keys:
                    expected_name = name if sd_key_ops is None else _apply_sd_key_ops(name, sd_key_ops)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)
                    size += value.nbytes
                    dtype = value.dtype
                    sd[expected_name] = value

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)


class SafetensorsModelStateDictLoader(StateDictLoader):
    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict:
        with safetensors.safe_open(path, framework="pt") as f:
            return json.loads(f.metadata()["config"])

    def load(
        self, path: str | list[str], sd_key_ops: SDKeyOps | None = None, device: torch.device | None = None
    ) -> StateDict:
        return self.weight_loader.load(path, sd_key_ops, device)

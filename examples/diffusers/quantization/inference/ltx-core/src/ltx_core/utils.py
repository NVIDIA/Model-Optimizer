# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Amit Pintz.

from typing import Any

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square (RMS) normalize `x` over its last dimension.

    Thin wrapper around `torch.nn.functional.rms_norm` that infers the normalized
    shape and forwards `weight` and `eps`.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def check_config_value(config: dict, key: str, expected: Any) -> None:  # noqa: ANN401
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")

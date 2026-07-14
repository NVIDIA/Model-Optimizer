# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

from ...block_config import BlockConfig
from .auto_model_descriptor import AutoModelDescriptor

__all__ = ["automodel_patcher"]


def _as_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


@contextmanager
def automodel_patcher(
    descriptor: AutoModelDescriptor,
    block_configs: Iterable[BlockConfig] | None,
):
    """Temporarily patch native AutoModel decoder block construction."""
    classes = _as_tuple(descriptor.decoder_layer_cls())
    originals = {cls: cls.__init__ for cls in classes}
    block_configs = tuple(block_configs or ())
    try:
        for cls, orig_init in originals.items():
            cls.__init__ = descriptor.make_patched_init(orig_init, block_configs)
        yield
    finally:
        for cls, orig_init in originals.items():
            cls.__init__ = orig_init

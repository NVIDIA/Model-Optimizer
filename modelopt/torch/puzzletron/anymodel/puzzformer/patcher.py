# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# mypy: ignore-errors

import copy
import inspect
import threading
from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Any, Dict, List

from transformers import PretrainedConfig

from ...block_config import BlockConfig, maybe_cast_block_configs
from ..model_descriptor.base import ModelDescriptor

__all__ = [
    "deci_x_patcher",
    "override_config_with_block_configs",
]

_PATCHER_LOCK = threading.RLock()


def _get_variable_from_stack(names: list[str]) -> Any:
    """Search the call stack for a variable with one of the given names."""
    f = inspect.currentframe().f_back
    while f:
        for name in names:
            if name in f.f_locals:
                return f.f_locals[name]
        f = f.f_back
    raise RuntimeError(f"{names} not found in caller stack")


@contextmanager
def deci_x_patcher(
    model_descriptor: ModelDescriptor,
    block_configs: List[BlockConfig | dict] | None = None,
):
    """Context manager that patches decoder layer __init__ for heterogeneous per-layer configs.

    This is the core mechanism that enables AnyModel to work with any HuggingFace model.
    It patches the decoder layer class(es) to read per-layer block_configs and apply
    layer-specific overrides (e.g., different intermediate_size per layer).

    Args:
        model_descriptor: The model descriptor that defines which classes to patch
            and how to map block_configs to layer overrides.
        block_configs: Optional list of BlockConfig (one per layer). If not provided,
            will try to read from config.block_configs during model initialization.

    Example:
        >>> with deci_x_patcher(LlamaModelDescriptor, block_configs):
        ...     model = AutoModelForCausalLM.from_config(config)
    """
    decoder_layer_classes = model_descriptor.decoder_layer_cls()  # Now a list of classes
    if not isinstance(decoder_layer_classes, list):
        decoder_layer_classes = [decoder_layer_classes]

    block_configs = maybe_cast_block_configs(block_configs)

    with _PATCHER_LOCK:
        orig_inits = []
        for cls in decoder_layer_classes:
            orig_inits.append(cls.__init__)

        def _orig_init_for(instance):
            for cls, orig_init in zip(decoder_layer_classes, orig_inits):
                if isinstance(instance, cls):
                    return orig_init
            raise TypeError(
                f"{type(instance).__name__} is not an instance of a patched decoder layer class"
            )

        @wraps(orig_inits[0])
        def _patched_decoder_layer_init(self, config, *args, **kwargs):
            _block_configs = block_configs or getattr(config, "block_configs", None)
            if _block_configs is None:
                return _orig_init_for(self)(self, config, *args, **kwargs)

            _block_configs = maybe_cast_block_configs(_block_configs)
            layer_idx = _get_variable_from_stack(["layer_idx", "idx"])
            # Native models can reuse an HF decoder-layer class for auxiliary
            # layers outside the backbone, such as Qwen3.5 MTP depths.  Those
            # indices intentionally have no Puzzletron block config and must
            # retain their native construction.
            if not isinstance(layer_idx, int) or not 0 <= layer_idx < len(_block_configs):
                return _orig_init_for(self)(self, config, *args, **kwargs)
            _block_config = _block_configs[layer_idx]
            override_block_config = model_descriptor.block_config_to_layer_overrides(
                _block_config
            )
            _config = override_config_with_block_configs(config, override_block_config)
            model_descriptor.patch_layer_config(_config, _block_config, layer_idx)
            # Keep the copied config aligned with the typed context. NemotronH no-op
            # hooks read back through decoder_layer.config.block_configs after init.
            _config.block_configs = _block_configs
            _orig_init_for(self)(self, _config, *args, **kwargs)

            # Apply no-op post-init
            attention_like_no_op = any(
                (subblock := _block_config.get_subblock(kind)) is not None and subblock.no_op
                for kind in ("attention", "mamba")
            )
            mlp_like_no_op = any(
                (subblock := _block_config.get_subblock(kind)) is not None and subblock.no_op
                for kind in ("ffn", "moe")
            )
            if attention_like_no_op:
                if not model_descriptor.attn_no_op_supported():
                    raise NotImplementedError(
                        f"attn no-op not supported for `{model_descriptor.__class__.__name__}`, "
                        "please implement the method: `attn_no_op_post_init()`"
                    )
                model_descriptor.attn_no_op_post_init(decoder_layer=self)

            if mlp_like_no_op:
                if not model_descriptor.mlp_no_op_supported():
                    raise NotImplementedError(
                        f"mlp no-op not supported for `{model_descriptor.__class__.__name__}`, "
                        "please implement the method: `mlp_no_op_post_init()`"
                    )
                model_descriptor.mlp_no_op_post_init(decoder_layer=self)

        with ExitStack() as stack:
            # Patch every decoder layer class. The patch is process-global, so keep
            # construction serialized when runtime-stat workers build benchmark
            # models concurrently.
            for orig_init, cls in zip(orig_inits, decoder_layer_classes):
                stack.callback(setattr, cls, "__init__", orig_init)  # Restore on exit
                cls.__init__ = _patched_decoder_layer_init
            yield


def override_config_with_block_configs(
    config: PretrainedConfig, block_configs: Dict[str, Any]
) -> PretrainedConfig:
    """Create a copy of config with block_config overrides applied."""
    _config = copy.deepcopy(config)
    # Model initialization requires fails with None in case of no-ops
    _config_overrides = {k: v for k, v in block_configs.items() if v is not None}
    _config.update(_config_overrides)
    return _config

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""High-level API to sparsify the weights and the attention of a model."""

from collections.abc import Iterator
from typing import Any

from torch import nn

from modelopt.torch.opt.conversion import ModeloptStateManager, apply_mode
from modelopt.torch.opt.mode import ModeLike, get_mode_config
from modelopt.torch.opt.searcher import ForwardLoop, SearchConfig
from modelopt.torch.utils import restore_training_modes, unwrap_model

from .attention_sparsity.config import (
    CalibrationConfig,
    SolAttnAttributeConfig,
    SparseAttentionAttributeConfig,
    SparseAttentionConfig,
    VSAAttributeConfig,
)
from .attention_sparsity.conversion import _check_methods_and_backends, is_attn_sparsified
from .attention_sparsity.methods import get_sparse_method
from .attention_sparsity.mode import SparseAttentionModeRegistry
from .attention_sparsity.model_sparsify import calibrate
from .weight_sparsity import sparsification as weight_sparsification
from .weight_sparsity.mode import SparsityModeRegistry
from .weight_sparsity.module import SparseModule

__all__ = ["calibrate", "export", "sparsify"]

_REGISTRIES = (SparsityModeRegistry, SparseAttentionModeRegistry)


def sparsify(
    model: nn.Module,
    mode: ModeLike | dict[str, Any] | SparseAttentionConfig,
    config: SearchConfig | None = None,
    *,
    forward_loop: ForwardLoop | None = None,
) -> nn.Module:
    """Sparsify the weights and/or the attention of a model in-place.

    Example::

        model = mts.sparsify(model, "sparsegpt", config={"data_loader": data_loader})
        model = mts.sparsify(model, mts.SKIP_SOFTMAX_CALIB)  # calibrates on RULER data

    Args:
        model: The model to sparsify.
        mode: A mode, a list of modes, or a list of ``(mode, mode_config)`` tuples. Modes are
            applied and then searched or calibrated one at a time, in order:

            * ``"sparse_magnitude"``: 2:4 weight sparsity by weight magnitude.
            * ``"sparsegpt"``: 2:4 weight sparsity with SparseGPT weight updates.
            * ``"sparse_attention"``: sparse attention configured by a
              :class:`SparseAttentionConfig <modelopt.torch.sparsity.attention_sparsity.config.SparseAttentionConfig>`
              such as ``mts.SKIP_SOFTMAX_CALIB``, which may also be passed directly as ``mode``.
        config: Search options such as ``forward_loop``, ``data_loader``, ``collect_func`` and
            ``verbose``, plus algorithm-specific ones such as SparseGPT's ``device``. Each mode
            receives the options its algorithm accepts and runs its own pass over the data, so a
            ``data_loader`` shared by several modes must be re-iterable.
        forward_loop: Shorthand for ``config["forward_loop"]``. Sparse attention calibration
            accepts a ``forward_loop`` or ``data_loader`` only with a decode target sparsity of 0;
            without either, it calibrates on generated RULER data.

    Returns:
        The sparsified model.
    """
    if isinstance(mode, (dict, SparseAttentionConfig)):
        mode = [("sparse_attention", dict(mode))]
    model = unwrap_model(model, force_unwrap=True)
    config = dict(config or {})
    if forward_loop is not None:
        if config.get("forward_loop") is not None:
            raise ValueError("Pass `forward_loop` either as an argument or in `config`, not both.")
        config["forward_loop"] = forward_loop
    if config.get("forward_loop") is not None and config.get("data_loader") is not None:
        raise ValueError("Pass either `forward_loop` or `data_loader`, not both.")
    has_data = config.get("forward_loop") is not None or config.get("data_loader") is not None

    # Validate all modes, their order and the config before modifying the model.
    modes = get_mode_config(mode)
    if len(modes) > 1 and isinstance(config.get("data_loader"), Iterator):
        raise ValueError("A `data_loader` used by several modes cannot be a one-shot iterator.")
    manager = ModeloptStateManager()  # a scratch copy of the model's modes to check the new ones
    if ModeloptStateManager.is_converted(model):
        manager.state_dict().extend(ModeloptStateManager(model).state_dict())
    searchers = []
    for m, mode_config in modes:
        registry = next((r for r in _REGISTRIES if m in r), None)
        if registry is None:
            raise ValueError(f"{m} is not a sparsity mode.")
        mode_cfg = registry[m].config_class(**mode_config)
        manager.add_mode(m, mode_cfg, {})
        if registry is SparseAttentionModeRegistry:
            _check_attention_config(mode_cfg, has_data)
        searchers.append(registry[m].search_algorithm())
    supported_keys = set().union(*(s.default_search_config for s in searchers))
    if unsupported_keys := config.keys() - supported_keys:
        raise ValueError(f"Unsupported config keys: {sorted(unsupported_keys)}")

    training_modes = {module: module.training for module in model.modules()}
    # The next mode freezes the metadata of the previous one, so each mode is searched right away.
    for (m, mode_config), searcher in zip(modes, searchers):
        model = apply_mode(model, [(m, mode_config)])
        search_keys = searcher.default_search_config.keys()
        try:
            searcher.search(model, {}, (), {k: v for k, v in config.items() if k in search_keys})
        finally:
            # A search gives every module the training mode of the model; restore them before the
            # next search runs its forward loop.
            restore_training_modes(model, training_modes)
    return model


def _check_attention_config(config: SparseAttentionConfig, has_data: bool) -> None:
    """Validate what conversion and calibration would otherwise check only after changing the model."""
    _check_methods_and_backends(config.sparse_cfg)
    for pattern, cfg in config.sparse_cfg.items():
        if pattern == "calibration":
            calib = CalibrationConfig(**cfg)
            if has_data and calib.target_sparse_ratio.get("decode", 0) > 0:
                raise ValueError(
                    "Decode calibration runs on generated RULER data; set its target sparsity to 0"
                    " to calibrate with `forward_loop` or `data_loader`."
                )
        else:
            attribute_cfg = {k: v for k, v in cfg.items() if k != "calibration"}
            config_class = {
                "vsa": VSAAttributeConfig,
                "sol_attn": SolAttnAttributeConfig,
            }.get(attribute_cfg.get("method", "flash_skip_softmax"), SparseAttentionAttributeConfig)
            get_sparse_method(config_class(**attribute_cfg).method)  # raises if not registered


def export(model: nn.Module) -> nn.Module:
    """Export a sparse model to a regular model by folding the weight sparsity masks into the weights.

    Export after fine-tuning: the masks are no longer enforced afterwards. Sparse attention needs no
    export and stays applied.
    """
    if is_attn_sparsified(model) and not any(isinstance(m, SparseModule) for m in model.modules()):
        return model
    return weight_sparsification.export(model)

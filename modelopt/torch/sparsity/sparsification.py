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

from typing import Any

from torch import nn

from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.opt.mode import ModeLike, get_mode_config
from modelopt.torch.opt.searcher import ForwardLoop, SearchConfig

from .attention_sparsity.config import SparseAttentionConfig
from .attention_sparsity.conversion import is_attn_sparsified
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
            receives the options its algorithm accepts.
        forward_loop: Shorthand for ``config["forward_loop"]``. Sparse attention calibration
            accepts a ``forward_loop`` or ``data_loader`` only with a decode target sparsity of 0;
            without either, it calibrates on generated RULER data.

    Returns:
        The sparsified model.
    """
    if isinstance(mode, (dict, SparseAttentionConfig)):
        mode = [("sparse_attention", dict(mode))]
    config = dict(config or {})
    if forward_loop is not None:
        if config.get("forward_loop") is not None:
            raise ValueError("Pass `forward_loop` either as an argument or in `config`, not both.")
        config["forward_loop"] = forward_loop
    if config.get("forward_loop") is not None and config.get("data_loader") is not None:
        raise ValueError("Pass either `forward_loop` or `data_loader`, not both.")

    # Validate all modes and config keys before modifying the model.
    modes = get_mode_config(mode)
    searchers = []
    for m, mode_config in modes:
        registry = next((r for r in _REGISTRIES if m in r), None)
        if registry is None:
            raise ValueError(f"{m} is not a sparsity mode.")
        registry[m].config_class(**mode_config)
        searchers.append(registry[m].search_algorithm())
    supported_keys = set().union(*(s.default_search_config for s in searchers))
    if unsupported_keys := config.keys() - supported_keys:
        raise ValueError(f"Unsupported config keys: {sorted(unsupported_keys)}")

    # The next mode freezes the metadata of the previous one, so each mode is searched right away.
    for (m, mode_config), searcher in zip(modes, searchers):
        model = apply_mode(model, [(m, mode_config)])
        search_keys = searcher.default_search_config.keys()
        searcher.search(model, {}, (), {k: v for k, v in config.items() if k in search_keys})
    return model


def export(model: nn.Module) -> nn.Module:
    """Export a sparse model to a regular model by folding the weight sparsity masks into the weights.

    Export after fine-tuning: the masks are no longer enforced afterwards. Sparse attention needs no
    export and stays applied.
    """
    if is_attn_sparsified(model) and not any(isinstance(m, SparseModule) for m in model.modules()):
        return model
    return weight_sparsification.export(model)

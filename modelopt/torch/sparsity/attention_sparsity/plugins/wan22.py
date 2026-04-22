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

"""Plugin for Wan 2.2 video diffusion models with VSA support.

Wan 2.2 (``WanTransformer3DModel`` from diffusers) uses standard diffusers
``Attention`` modules whose ``AttnProcessor2_0`` calls
``F.scaled_dot_product_attention``.  VSA's default SDPA patch in
``SparseAttentionModule._forward_with_vsa_sdpa_patch`` therefore intercepts
the right call — we only need to tell VSA the post-patchify ``(T, H, W)``.

This plugin installs a forward pre-hook on every ``WanTransformer3DModel``
that:

1. Reads ``hidden_states`` shape ``(B, C, T, H, W)`` from the transformer
   input.
2. Divides by ``model.config.patch_size = (p_t, p_h, p_w)`` — same
   computation diffusers does internally (see
   ``WanTransformer3DModel.forward``: ``post_patch_num_frames = num_frames // p_t``
   etc.).
3. Propagates the resulting shape to every ``SparseAttentionModule`` in
   the transformer whose method is VSA, via ``method.set_video_shape()``.

Self-attention layers (``attn1``) then see a valid ``video_shape`` when the
SDPA patch fires.  Cross-attention (``attn2``) is skipped by VSA's
``can_apply_vsa`` guard since Q/K lengths differ.
"""

import logging

import torch.nn as nn

from ..sparse_attention import SparseAttentionModule
from . import CUSTOM_MODEL_PLUGINS

logger = logging.getLogger(__name__)


def _is_wan22_model(model: nn.Module) -> bool:
    """Detect a Wan 2.2 transformer by class name.

    Wan 2.1 / 2.2 both use ``WanTransformer3DModel`` in diffusers — matching
    by name keeps the plugin decoupled from the diffusers import.
    """
    if type(model).__name__ == "WanTransformer3DModel":
        return True
    return any(type(m).__name__ == "WanTransformer3DModel" for m in model.modules())


def _find_wan22_transformers(model: nn.Module) -> list[nn.Module]:
    """Return every ``WanTransformer3DModel`` reachable from ``model``.

    The 14B model is a ``WanPipeline`` with ``transformer`` and
    ``transformer_2``, so we return every match.
    """
    if type(model).__name__ == "WanTransformer3DModel":
        return [model]
    return [m for m in model.modules() if type(m).__name__ == "WanTransformer3DModel"]


def _get_patch_size(transformer: nn.Module) -> tuple[int, int, int] | None:
    """Read ``patch_size`` from the transformer's config."""
    config = getattr(transformer, "config", None)
    if config is None:
        return None
    patch_size = getattr(config, "patch_size", None)
    if patch_size is None:
        return None
    try:
        p_t, p_h, p_w = patch_size
    except (TypeError, ValueError):
        return None
    return (int(p_t), int(p_h), int(p_w))


def _extract_hidden_states(args: tuple, kwargs: dict):
    """Pick out the ``hidden_states`` argument regardless of call style."""
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    return args[0] if len(args) > 0 else None


def _make_wan22_video_shape_hook(transformer: nn.Module):
    """Create the per-transformer forward pre-hook.

    Closes over the specific ``transformer`` so it can walk its own
    submodules, independent of other Wan 2.2 transformers in the same
    pipeline.
    """
    patch_size = _get_patch_size(transformer)
    if patch_size is None:
        logger.debug("Wan 2.2 transformer has no config.patch_size; hook inert")

        def _noop(module, args, kwargs):
            return None

        return _noop

    p_t, p_h, p_w = patch_size

    def _hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
        hidden_states = _extract_hidden_states(args, kwargs)
        if hidden_states is None or hidden_states.ndim != 5:
            return

        _, _, num_frames, height, width = hidden_states.shape
        video_shape = (num_frames // p_t, height // p_h, width // p_w)
        if any(d <= 0 for d in video_shape):
            logger.debug(
                f"Wan 2.2 VSA hook: invalid video_shape {video_shape} for "
                f"input {(num_frames, height, width)} / patch {patch_size}; skipping"
            )
            return

        # Also expose on the transformer for debugging / external inspection.
        module._vsa_video_shape = video_shape

        # Propagate to every VSA method instance in this transformer.
        for sub in module.modules():
            if not isinstance(sub, SparseAttentionModule):
                continue
            method = getattr(sub, "_sparse_method_instance", None)
            if method is None:
                continue
            if getattr(method, "name", None) != "vsa":
                continue
            method.set_video_shape(video_shape)

    return _hook


def register_wan22_vsa(model: nn.Module) -> int:
    """Install a VSA ``video_shape`` pre-hook on every Wan 2.2 transformer.

    Idempotent: the hook is re-registered on each call because
    ``plugins/__init__.py`` stores callbacks in a set — re-invoking after
    ``mtsa.sparsify`` is safe, but we guard against double-registration by
    tagging the transformer with ``_vsa_hook_registered``.
    """
    transformers = _find_wan22_transformers(model)
    if not transformers:
        return 0

    registered = 0
    for transformer in transformers:
        if getattr(transformer, "_vsa_hook_registered", False):
            continue
        hook = _make_wan22_video_shape_hook(transformer)
        transformer.register_forward_pre_hook(hook, with_kwargs=True)
        transformer._vsa_hook_registered = True
        registered += 1
        logger.info(f"Registered Wan 2.2 VSA video_shape hook on {type(transformer).__name__}")

    return registered


def register_wan22_on_the_fly(model: nn.Module) -> bool:
    """Plugin entry point: install the Wan 2.2 VSA hook if applicable."""
    if not _is_wan22_model(model):
        return False
    num = register_wan22_vsa(model)
    if num > 0:
        logger.info(f"Installed VSA video_shape hook on {num} Wan 2.2 transformer(s)")
        return True
    return False


CUSTOM_MODEL_PLUGINS.add(register_wan22_on_the_fly)

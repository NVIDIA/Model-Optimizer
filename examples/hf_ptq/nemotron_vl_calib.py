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

"""Nemotron VL calibration helpers.

Nemotron Nano VL v2 remote-code wrapper `forward()` is not ideal to call during PTQ calibration because it may:
- Call `torch.distributed.get_rank()` unconditionally
- Assume `past_key_values` exists in the language model output

Instead, we run a "safe multimodal forward" that exercises:
- Vision encoder feature extraction (C-RADIOv2-H)
- Insertion of vision embeddings into token embeddings at `img_context_token_id`
- Language model forward pass (to trigger quantizer calibration)
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any

import torch


def _accepted_kwargs(fn, kwargs: dict) -> dict:
    """Drop keys ``fn`` cannot accept, so a wrapper with a narrow forward signature does not
    trade one ``AttributeError`` for a ``TypeError``. Callables taking ``**kwargs`` are left alone.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def safe_nemotron_vl_forward(full_model: torch.nn.Module, batch: dict[str, Any]) -> None:
    """Run a minimal multimodal forward for Nemotron VL that avoids wrapper output packaging."""
    pixel_values = batch.get("pixel_values")
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    position_ids = batch.get("position_ids")
    image_flags = batch.get("image_flags")

    if pixel_values is None or input_ids is None:
        return

    # Nemotron Nano VL v2 expects `image_flags` in forward(), but the processor doesn't always emit it.
    # `pixel_values` is flattened across batch*images, so `image_flags` should align with pixel_values.shape[0].
    if image_flags is None and torch.is_tensor(pixel_values):
        image_flags = torch.ones(
            (pixel_values.shape[0], 1), device=pixel_values.device, dtype=torch.long
        )
    if image_flags is None:
        return

    # Match the model's preferred vision dtype (usually bf16).
    vision_dtype = None
    with contextlib.suppress(AttributeError, TypeError):
        vision_dtype = getattr(full_model.vision_model.config, "torch_dtype", None)
    if vision_dtype is None:
        with contextlib.suppress(AttributeError, TypeError):
            vision_dtype = getattr(full_model.language_model.config, "torch_dtype", None)
    if (
        vision_dtype is not None
        and torch.is_tensor(pixel_values)
        and pixel_values.dtype != vision_dtype
    ):
        pixel_values = pixel_values.to(dtype=vision_dtype)

    # InternVL-style Nemotron VL exposes `extract_feature(pixel_values)`; the newer omni wrappers
    # (e.g. NemotronH_Omni_Reasoning_V3 / nemotron_h_omni) do not -- they merge the vision
    # embeddings via masked_scatter inside their own forward, and do not necessarily define
    # `img_context_token_id` either. Hand the batch to the model's own forward before touching
    # anything InternVL-specific, carrying the dtype-cast pixel_values.
    if not hasattr(full_model, "extract_feature"):
        # Only the batch's own entries, plus the dtype-cast pixel_values. The synthesized
        # `image_flags` above is deliberately NOT injected: it is built on pixel_values.device,
        # while a sharded wrapper runs its vision tower elsewhere and indexes image_embeds with
        # it, so a synthesized tensor raises "indices should be ... on the same device".
        # Wrappers that need image_flags either receive the batch's own or build their own.
        fallback = {k: v for k, v in batch.items() if v is not None}
        fallback["pixel_values"] = pixel_values
        fallback["use_cache"] = False
        full_model(**_accepted_kwargs(full_model.forward, fallback))
        return

    # Token embeddings
    inputs_embeds = full_model.language_model.get_input_embeddings()(input_ids)
    image_flags_s = image_flags.squeeze(-1)

    b, n, c = inputs_embeds.shape
    flat_embeds = inputs_embeds.reshape(b * n, c)
    flat_ids = input_ids.reshape(b * n)
    selected = flat_ids == full_model.img_context_token_id
    # A bool mask must sit on the device of the tensor it indexes, not the one it was derived
    # from. Under a sharded device_map the embedding table can run on a different GPU than the
    # batch, so `selected` -- built from `input_ids` -- follows `flat_embeds`. Both the fast
    # path and the except-branch retry below index with it.
    if selected.device != flat_embeds.device:
        selected = selected.to(flat_embeds.device)

    vit_embeds = full_model.extract_feature(pixel_values)
    # Same rule for the image filter: the vision tower returns on its own device, while
    # `image_flags` arrived with the batch (or was synthesized on `pixel_values.device`).
    if image_flags_s.device != vit_embeds.device:
        image_flags_s = image_flags_s.to(vit_embeds.device)
    vit_embeds = vit_embeds[image_flags_s == 1]
    # The merge target lives on the embedding device; bring the filtered vision embeddings over.
    if vit_embeds.device != flat_embeds.device:
        vit_embeds = vit_embeds.to(flat_embeds.device)
    try:
        flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, c)
    except Exception:
        vit_embeds = vit_embeds.reshape(-1, c)
        n_token = selected.sum()
        flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds[:n_token]

    inputs_embeds = flat_embeds.reshape(b, n, c)

    # `inputs_embeds` comes off the embedding table, so under a sharded device_map it can sit on a
    # different device than the mask/ids that came in with the batch. Align them to the embedding
    # device so the LLM call starts consistent; accelerate's hooks handle later blocks.
    _lm_dev = getattr(full_model.language_model.get_input_embeddings().weight, "device", None)
    if _lm_dev is not None:
        if attention_mask is not None and attention_mask.device != _lm_dev:
            attention_mask = attention_mask.to(_lm_dev)
        if position_ids is not None and position_ids.device != _lm_dev:
            position_ids = position_ids.to(_lm_dev)

    # LLM forward (drives activation stats)
    full_model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=False,
    )

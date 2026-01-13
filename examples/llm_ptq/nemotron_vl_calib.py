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
from typing import Any

import torch


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
    with contextlib.suppress(Exception):
        vision_dtype = getattr(full_model.vision_model.config, "torch_dtype", None)
    if vision_dtype is None:
        with contextlib.suppress(Exception):
            vision_dtype = getattr(full_model.language_model.config, "torch_dtype", None)
    if (
        vision_dtype is not None
        and torch.is_tensor(pixel_values)
        and pixel_values.dtype != vision_dtype
    ):
        pixel_values = pixel_values.to(dtype=vision_dtype)

    # Token embeddings
    inputs_embeds = full_model.language_model.get_input_embeddings()(input_ids)
    image_flags_s = image_flags.squeeze(-1)

    b, n, c = inputs_embeds.shape
    flat_embeds = inputs_embeds.reshape(b * n, c)
    flat_ids = input_ids.reshape(b * n)
    selected = flat_ids == full_model.img_context_token_id

    # Vision embeddings
    vit_embeds = full_model.extract_feature(pixel_values)
    vit_embeds = vit_embeds[image_flags_s == 1]
    try:
        flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, c)
    except Exception:
        vit_embeds = vit_embeds.reshape(-1, c)
        n_token = selected.sum()
        flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds[:n_token]

    inputs_embeds = flat_embeds.reshape(b, n, c)

    # LLM forward (drives activation stats)
    full_model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=False,
    )

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

"""Tests for the Qwen-Image SVDQuant diffusers export path.

Covers the three pieces added for Qwen support:
- the Qwen branch of ``generate_diffusion_dummy_inputs`` (validated by running the
  dummy forward on a real tiny ``QwenImageTransformer2DModel``),
- the strict-failure mode of ``_fuse_qkv_linears_diffusion``,
- promotion of quantizer-owned SVDQuant tensors to clean module-level keys that
  survive ``hide_quantizers_from_state_dict``.
"""

from functools import partial

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export.diffusers_utils import (
    generate_diffusion_dummy_forward_fn,
    generate_diffusion_dummy_inputs,
    hide_quantizers_from_state_dict,
)
from modelopt.torch.export.unified_export_hf import (
    _fuse_qkv_linears_diffusion,
    _promote_quantizer_tensors_to_module,
)


class _MLP(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _forward_loop(model, data):
    for batch in data:
        model(batch)


def _quantize(model: nn.Module, algorithm=None, dim: int = 64) -> nn.Module:
    cfg = mtq.INT8_SMOOTHQUANT_CFG.copy()
    if algorithm is not None:
        cfg["algorithm"] = algorithm
    data = [torch.randn(2, dim) for _ in range(2)]
    mtq.quantize(model, cfg, partial(_forward_loop, data=data))
    return model


def test_qwen_dummy_inputs_drive_real_transformer_forward():
    """The Qwen dummy inputs must actually drive a real tiny Qwen transformer."""
    pytest.importorskip("diffusers")
    from _test_utils.torch.diffusers_models import get_tiny_qwen_image_transformer

    transformer = get_tiny_qwen_image_transformer().to("cpu", torch.float32).eval()

    inputs = generate_diffusion_dummy_inputs(transformer, torch.device("cpu"), torch.float32)
    assert inputs is not None
    for key in (
        "hidden_states",
        "encoder_hidden_states",
        "encoder_hidden_states_mask",
        "img_shapes",
        "txt_seq_lens",
    ):
        assert key in inputs, f"missing Qwen dummy input '{key}'"
    assert inputs["hidden_states"].shape[-1] == transformer.config.in_channels
    assert inputs["encoder_hidden_states"].shape[-1] == transformer.config.joint_attention_dim

    # Strongest check: the generated dummy inputs run through the real model.
    with torch.no_grad():
        generate_diffusion_dummy_forward_fn(transformer)()


def test_qwen_qkv_fusion_strict_raises_on_failed_dummy_forward():
    """strict=True turns a dummy-forward failure into a hard error; strict=False does not."""
    model = _quantize(_MLP())

    def _boom():
        raise RuntimeError("dummy forward failed")

    with pytest.raises(RuntimeError):
        _fuse_qkv_linears_diffusion(model, dummy_forward_fn=_boom, strict=True)

    # Non-strict path warns and returns without raising.
    _fuse_qkv_linears_diffusion(model, dummy_forward_fn=_boom, strict=False)


def test_svdquant_promotion_survives_hide_quantizers():
    """Promoted LoRA + pre_quant_scale land on the module under clean keys and
    survive ``hide_quantizers_from_state_dict`` (which strips the quantizers)."""
    model = _quantize(_MLP(), algorithm={"method": "svdquant", "lowrank": 8})

    _promote_quantizer_tensors_to_module(model)

    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert linears
    for module in linears:
        assert hasattr(module, "svdquant_lora_a")
        assert hasattr(module, "svdquant_lora_b")
        # INT8_SMOOTHQUANT produces a pre_quant_scale that is promoted too.
        assert hasattr(module, "pre_quant_scale")
        # Rank-consistent shapes: lora_a [rank, in], lora_b [out, rank].
        assert module.svdquant_lora_a.shape[1] == module.in_features
        assert module.svdquant_lora_b.shape[0] == module.out_features
        assert module.svdquant_lora_a.shape[0] == module.svdquant_lora_b.shape[1]

    with hide_quantizers_from_state_dict(model):
        keys = list(model.state_dict().keys())

    assert any(k.endswith(".svdquant_lora_a") for k in keys)
    assert any(k.endswith(".svdquant_lora_b") for k in keys)
    assert any(k.endswith(".pre_quant_scale") for k in keys)
    # Clean keys only: no quantizer-prefixed keys remain once quantizers are hidden.
    assert not any("weight_quantizer" in k for k in keys)
    assert not any("input_quantizer" in k for k in keys)

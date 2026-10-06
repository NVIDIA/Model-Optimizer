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

"""Export must honor the same Qwen linear-attention groups as AutoQuant."""

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import (
    _fuse_shared_input_modules,
    collect_shared_input_modules,
)


class _LinearAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
            setattr(self, name, torch.nn.Linear(32, 32))

    def forward(self, inputs):
        return sum(module(inputs) for module in self.children())


@pytest.mark.parametrize("mismatch_within_pair", [False, True])
def test_export_preserves_linear_attention_runtime_groups(mismatch_within_pair):
    model = torch.nn.Sequential()
    model.add_module("decoder", torch.nn.Module())
    model.decoder.add_module("linear_attn", _LinearAttention())
    inputs = torch.randn(1, 8, 32)
    for name, module in list(model.decoder.linear_attn.named_children()):
        fp8 = name in {"in_proj_a", "in_proj_b"} or (mismatch_within_pair and name == "in_proj_z")
        mtq.quantize(
            module, mtq.FP8_DEFAULT_CFG if fp8 else mtq.INT8_DEFAULT_CFG, lambda m: m(inputs)
        )
    groups, norms = collect_shared_input_modules(model, lambda: model.decoder.linear_attn(inputs))
    assert any(len(modules) == 4 for modules in groups.values())
    if mismatch_within_pair:
        with pytest.raises(AssertionError, match="different quantization formats"):
            _fuse_shared_input_modules(model, groups, norms)
    else:
        fused = _fuse_shared_input_modules(model, groups, norms)
        assert {frozenset(names) for names in fused.values()} == {
            frozenset({"decoder.linear_attn.in_proj_qkv", "decoder.linear_attn.in_proj_z"}),
        }

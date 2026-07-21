# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for offload-aware unified HF export helpers (CPU-only, no GPU required)."""

import pytest
import torch
import torch.nn as nn

try:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import set_module_tensor_to_device
except ImportError:
    pytest.skip("accelerate not available", allow_module_level=True)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import (
    _export_quantized_weight,
    _has_accelerate_offload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_offloaded_linear(dim: int = 16):
    """Return a Linear with a CPU-offload AlignDevicesHook attached and params on meta."""
    linear = nn.Linear(dim, dim, bias=False)
    weights_map = {"weight": linear.weight.data.clone().cpu()}
    hook = AlignDevicesHook(execution_device="cpu", offload=True, weights_map=weights_map)
    add_hook_to_module(linear, hook)
    set_module_tensor_to_device(linear, "weight", "meta")
    return linear, weights_map


# ---------------------------------------------------------------------------
# _has_accelerate_offload
# ---------------------------------------------------------------------------


def test_has_accelerate_offload_true():
    linear, _ = _make_offloaded_linear()
    assert _has_accelerate_offload(linear) is True


def test_has_accelerate_offload_false_no_hooks():
    linear = nn.Linear(16, 16)
    assert _has_accelerate_offload(linear) is False


def test_has_accelerate_offload_false_non_offload_hook():
    """A hook with offload=False should not be detected as offloaded."""
    linear = nn.Linear(16, 16)
    hook = AlignDevicesHook(execution_device="cpu", offload=False)
    add_hook_to_module(linear, hook)
    assert _has_accelerate_offload(linear) is False


def test_has_accelerate_offload_detects_nested_module():
    """Offload hook on a child module should be detected when scanning the parent."""

    class _Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.child(x)

    parent = _Parent()
    weights_map = {"weight": parent.child.weight.data.clone().cpu()}
    hook = AlignDevicesHook(execution_device="cpu", offload=True, weights_map=weights_map)
    add_hook_to_module(parent.child, hook)
    set_module_tensor_to_device(parent.child, "weight", "meta")

    assert _has_accelerate_offload(parent) is True


# ---------------------------------------------------------------------------
# _export_quantized_weight meta guard
# ---------------------------------------------------------------------------


def test_meta_guard_raises_on_meta_weight():
    """_export_quantized_weight must raise RuntimeError when weight is a meta tensor."""
    linear = nn.Linear(16, 16, bias=False)

    mtq.quantize(linear, mtq.FP8_DEFAULT_CFG, lambda m: m(torch.randn(1, 16)))

    # Manually set weight to meta to simulate what happens after hooks are removed.
    linear.weight = nn.Parameter(torch.empty(16, 16, device="meta"))

    with pytest.raises(RuntimeError, match="meta tensor"):
        _export_quantized_weight(linear, torch.float32)


def test_meta_guard_not_raised_for_real_weight():
    """No RuntimeError when weight is a real (non-meta) tensor."""
    linear = nn.Linear(32, 32, bias=False)
    mtq.quantize(linear, mtq.FP8_DEFAULT_CFG, lambda m: m(torch.randn(1, 32)))
    # Should not raise
    _export_quantized_weight(linear, torch.float32)

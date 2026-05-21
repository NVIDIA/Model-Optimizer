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

"""CPU-only unit tests for ``modelopt.torch.export.quant_utils``."""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("accelerate")

from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from accelerate.utils import set_module_tensor_to_device

from modelopt.torch.export.quant_utils import fuse_prequant_to_linear
from modelopt.torch.quantization.nn import TensorQuantizer


class LlamaMLP(nn.Module):
    """Stub class with the name expected by ``PQS_FUSE_MODULE_MAPPING``."""

    def __init__(self, hidden_size=8, intermediate_size=8):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # ``fuse_prequant_to_linear`` calls reset_amax / enable_stats_collection /
        # the quantizer forward on ``up_proj.weight_quantizer`` - it needs to exist.
        self.up_proj.weight_quantizer = TensorQuantizer()
        self.down_proj.input_quantizer = TensorQuantizer()


def _offload_weight_to_meta(linear: nn.Linear) -> dict:
    """Simulate accelerate CPU offload: move the weight to meta, attach an offload hook."""
    weights_map: dict[str, torch.Tensor] = {"weight": linear.weight.data.clone()}
    set_module_tensor_to_device(linear, "weight", "meta")
    hook = AlignDevicesHook(
        execution_device=torch.device("cpu"),
        offload=True,
        weights_map=weights_map,
        place_submodules=False,
    )
    add_hook_to_module(linear, hook)
    return weights_map


def test_fuse_prequant_to_linear_with_offloaded_weight():
    """Regression test for #795: meta-device weight on the ``fuse_into`` linear.

    Reproduces the failure mode triggered by HF ``device_map="auto"`` with tight
    ``max_memory``: the linear we mutate has its weight on meta because accelerate has
    offloaded it. The fuse path must materialize the weight via the offload hook,
    perform the multiplication, and write the fused weight back to the offload map.
    """
    torch.manual_seed(0)
    module = LlamaMLP(hidden_size=8, intermediate_size=8)

    pre_quant_scale = torch.linspace(0.1, 0.8, steps=8)
    module.down_proj.input_quantizer._pre_quant_scale = pre_quant_scale.clone()

    original_weight = module.up_proj.weight.data.clone()
    weights_map = _offload_weight_to_meta(module.up_proj)
    assert module.up_proj.weight.device.type == "meta"

    fuse_prequant_to_linear(module)

    # The hook's post_forward re-evicts the parameter to meta after the context exits.
    assert module.up_proj.weight.device.type == "meta"

    # The fused weight should have been written back to the offload weights_map.
    expected = original_weight * pre_quant_scale.view(-1, 1)
    assert torch.allclose(weights_map["weight"], expected), (
        "Offload weights_map should hold the fused weight after fuse_prequant_to_linear"
    )

    # pre_quant_scale is consumed and the module is flagged.
    assert not hasattr(module.down_proj.input_quantizer, "_pre_quant_scale")
    assert getattr(module.down_proj, "fused_with_prequant", False) is True


def test_fuse_prequant_to_linear_without_offload():
    """The non-offloaded path must still produce the in-place fused weight."""
    torch.manual_seed(0)
    module = LlamaMLP(hidden_size=8, intermediate_size=8)

    pre_quant_scale = torch.linspace(0.1, 0.8, steps=8)
    module.down_proj.input_quantizer._pre_quant_scale = pre_quant_scale.clone()
    original_weight = module.up_proj.weight.data.clone()

    fuse_prequant_to_linear(module)

    expected = original_weight * pre_quant_scale.view(-1, 1)
    assert torch.allclose(module.up_proj.weight.data, expected)
    assert not hasattr(module.down_proj.input_quantizer, "_pre_quant_scale")
    assert getattr(module.down_proj, "fused_with_prequant", False) is True

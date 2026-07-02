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

"""CPU tests for the attention-only impl-swap attach in ``quant_sparse_attn_worker``.

The attach must (1) convert only vLLM ``Attention`` modules to ``_QuantVLLMAttention`` and configure
their ``q/k/v/p_bmm_quantizer`` as dynamic block-16 NVFP4 (K/V getting the global-scale-1.0 default),
and (2) leave a realquant-style Linear untouched -- converting that Linear (as ``mtq.quantize`` would)
raises ``AssertionError`` in ``_VLLMParallelLinear._setup``, which is exactly what this path avoids.

Only ``create_parallel_state`` is patched away (it needs a live vLLM distributed group); everything
else -- the registry, the real ``_setup``, ``set_quantizer_by_cfg``, ``post_restore_vllm_attentions``
-- runs for real on CPU.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import vllm.model_executor.layers.linear as vllm_linear

import modelopt.torch.quantization.plugins.vllm as vllm_plugin
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.utils.distributed import ParallelState

# The worker module lives under examples/vllm_serve and is imported as a top-level module there.
_VLLM_SERVE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples", "vllm_serve"
)
sys.path.insert(0, os.path.abspath(_VLLM_SERVE_DIR))

from quant_sparse_attn_worker import _attach_attention_quant_impl_swap


@pytest.fixture
def _no_distributed_parallel_state(monkeypatch):
    """Convert attention on CPU without a live vLLM distributed group.

    ``_QuantVLLMAttention._setup`` calls ``create_parallel_state()`` (needs ``get_dp_group()`` /
    ``get_tp_group()``); on CPU we swap it for the same default ``_initialize_parallel_state`` uses.
    """
    monkeypatch.setattr(
        vllm_plugin, "create_parallel_state", lambda: ParallelState(data_parallel_group=None)
    )


class _RealAttention(vllm_plugin.vllm_attention.Attention):
    """Real vLLM ``Attention`` subclass with ``__init__`` bypassed (no engine needed on CPU)."""

    def __init__(self):
        nn.Module.__init__(self)
        # A trivial param so post-restore device/dtype detection succeeds.
        self.dummy = nn.Parameter(torch.zeros(1))


class _RealQuantMethod:
    """Stand-in for a realquant Linear method (e.g. ModelOptFp8LinearMethod) -- NOT unquantized."""


class _RealQuantLinear(vllm_linear.RowParallelLinear):
    """Real vLLM Linear subclass carrying a non-``UnquantizedLinearMethod`` quant method."""

    def __init__(self):
        nn.Module.__init__(self)
        self.quant_method = _RealQuantMethod()


@pytest.mark.usefixtures("_no_distributed_parallel_state")
def test_impl_swap_converts_attention_and_configures_nvfp4():
    attention = _RealAttention()
    model = nn.ModuleDict({"attn": attention})

    _attach_attention_quant_impl_swap(model)

    converted = model.attn
    assert isinstance(converted, vllm_plugin._QuantVLLMAttention)

    # All four BMM quantizers enabled and configured as dynamic block-16 NVFP4.
    for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer", "p_bmm_quantizer"):
        quantizer = getattr(converted, name)
        assert quantizer.is_enabled
        assert quantizer.is_nvfp4_dynamic
        assert quantizer.num_bits == (2, 1)
        assert (quantizer.block_sizes or {}).get(-1) == 16

    # K/V pick up the global-scale-1.0 runtime default (amax == 6 * 448 == 2688), matching
    # test_vllm_kv_default_scale.py; the dynamic Q/P quantizers get no runtime default.
    inputs = torch.tensor([-3.0, 5.0])
    for name in ("k_bmm_quantizer", "v_bmm_quantizer"):
        assert getattr(converted, name)._get_amax(inputs).item() == 2688.0
    assert not hasattr(converted.q_bmm_quantizer, "_runtime_default_amax")
    assert not hasattr(converted.p_bmm_quantizer, "_runtime_default_amax")


@pytest.mark.usefixtures("_no_distributed_parallel_state")
def test_impl_swap_leaves_realquant_linear_untouched():
    linear = _RealQuantLinear()
    model = nn.ModuleDict({"proj": linear})

    # Sanity: this Linear IS a registered QuantModule but is NOT an attention type; converting it
    # (as mtq.quantize / replace_quant_module would) trips the _VLLMParallelLinear quant_method
    # assert -- the exact failure the attention-only attach exists to avoid.
    assert type(linear) in QuantModuleRegistry
    assert not isinstance(linear, vllm_plugin._ATTENTION_TYPES)
    with pytest.raises(AssertionError):
        QuantModuleRegistry.convert(_RealQuantLinear())

    # The attach must neither convert the Linear nor raise.
    _attach_attention_quant_impl_swap(model)

    assert model.proj is linear
    assert type(model.proj) is _RealQuantLinear
    assert not isinstance(model.proj, vllm_plugin._QuantVLLMAttention)

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

import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import (
    SequentialQuantizer,
    TensorQuantizer,
    register_quant_backend,
    unregister_quant_backend,
)
from modelopt.torch.quantization.utils import (
    disable_weight_cache,
    enable_weight_cache,
    invalidate_weight_cache,
    weight_cache_scope,
)

BACKEND_CALLS = []


def _ste_round_backend(inputs, quantizer):
    BACKEND_CALLS.append(quantizer)
    return inputs + (torch.round(inputs * 8) / 8 - inputs).detach()


@pytest.fixture
def ste_round_backend():
    register_quant_backend("ste_round", _ste_round_backend)
    BACKEND_CALLS.clear()
    yield
    unregister_quant_backend("ste_round")


def _quantized_mlp(**weight_cfg):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": 8, "axis": 0, **weight_cfg},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    return mtq.quantize(model, cfg, lambda m: m(torch.randn(4, 16)))


def _accumulate(model, micro_batches):
    """Forward/backward every micro-batch, recomputing activations as Megatron does."""
    model.zero_grad()
    outputs = [checkpoint(model, x, use_reentrant=False) for x in micro_batches]
    for out in outputs:
        out.square().sum().backward()
    return [out.detach() for out in outputs], [p.grad.clone() for p in model.parameters()]


@pytest.mark.parametrize("weight_cfg", [{}, {"backend": "ste_round"}])
def test_cached_step_matches_uncached(ste_round_backend, weight_cfg):
    model = _quantized_mlp(**weight_cfg)
    micro_batches = [torch.randn(4, 16) for _ in range(3)]
    ref_outputs, ref_grads = _accumulate(model, micro_batches)

    assert enable_weight_cache(model) == 2
    BACKEND_CALLS.clear()
    with weight_cache_scope(model) as stats:
        outputs, grads = _accumulate(model, micro_batches)

    for out, ref in zip(outputs + grads, ref_outputs + ref_grads):
        assert torch.equal(out, ref)
    # Each weight is fake-quantized once, then reused by 3 forwards and 3 recomputations.
    assert (stats.misses, stats.hits) == (2, 10)
    if weight_cfg:
        assert len(BACKEND_CALLS) == 2
    assert stats.retained_bytes == 0 and stats.peak_bytes > 0


def test_weight_and_quantizer_changes_recompute():
    model = _quantized_mlp()
    enable_weight_cache(model)
    linear, x = model[0], torch.randn(4, 16)
    with weight_cache_scope(model) as stats:
        model(x)
        with torch.no_grad():
            linear.weight.mul_(0.5)  # an in-place update changes the weight's version
        # A copy is not enrolled, so it gives the uncached result.
        assert torch.equal(model(x), copy.deepcopy(model)(x))
        linear.weight_quantizer.amax = linear.weight_quantizer.amax * 0.5
        invalidate_weight_cache(model)
        assert torch.equal(model(x), copy.deepcopy(model)(x))
    assert (stats.misses, stats.invalidations) == (5, 3)


def test_weight_cache_contract():
    model = _quantized_mlp()
    with pytest.raises(ValueError, match="enable_weight_cache"), weight_cache_scope(model):
        pass
    with pytest.raises(ValueError, match="pass_through_bwd"):
        enable_weight_cache(_quantized_mlp(pass_through_bwd=False))
    learned = _quantized_mlp()
    learned[0].weight_quantizer.register_parameter("scale", nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="learned scales"):
        enable_weight_cache(learned)
    composed = _quantized_mlp()
    composed[0].weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
    with pytest.raises(ValueError, match="only TensorQuantizer"):
        enable_weight_cache(composed)

    enable_weight_cache(model)
    with pytest.raises(RuntimeError, match="boom"), weight_cache_scope(model):
        with pytest.raises(RuntimeError, match="inside weight_cache_scope"):
            enable_weight_cache(model)
        model(torch.randn(4, 16))
        with pytest.raises(RuntimeError, match="nested"), weight_cache_scope(model):
            pass
        raise RuntimeError("boom")
    # A failed step closes its scope, so the next one starts empty.
    with weight_cache_scope(model) as stats:
        model(torch.randn(4, 16))
    assert stats.misses == 2 and stats.hits == 0


def test_enrollment_by_chunk():
    chunks = [_quantized_mlp(), _quantized_mlp()]
    # Like Megatron virtual pipeline chunks: each chunk owns its quantizers and can be scoped alone.
    assert enable_weight_cache(chunks, select=lambda name, q: name.startswith("0.")) == 2
    with weight_cache_scope(chunks[1]) as stats:
        chunks[1](torch.randn(4, 16))
    assert stats.misses == 1
    with pytest.raises(ValueError, match="already enrolled"):
        enable_weight_cache(nn.Sequential(chunks[1]))

    disable_weight_cache(chunks[1])
    with pytest.raises(ValueError, match="enable_weight_cache"), weight_cache_scope(chunks[1]):
        pass
    with weight_cache_scope(chunks[0]) as stats:
        chunks[0](torch.randn(4, 16))
    assert stats.misses == 1

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

from types import SimpleNamespace

import pytest
import torch

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.ggml.backend as backend_module
import modelopt.torch.quantization.ggml.iq1_s as iq1_s_module
import modelopt.torch.quantization.ggml.iq2_xs as iq2_xs_module
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml.backend import ggml_fake_quant
from modelopt.torch.quantization.ggml.common import narrow_to_float32
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize("num_bits", ["iq1_s", "iq2_xs"])
def test_ggml_backend_via_quantize(num_bits):
    torch.manual_seed(1234)
    model = torch.nn.Linear(256, 2, bias=False)
    inputs = torch.randn(2, 256)
    unquantized_output = model(inputs).detach()
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": num_bits, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }

    mtq.quantize(model, config, forward_loop=lambda module: module(inputs))
    output = model(inputs)

    assert model.weight_quantizer.backend == "ggml"
    assert model.weight_quantizer.num_bits == num_bits
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert not torch.equal(output, unquantized_output)


def test_ggml_backend_rejects_unknown_format():
    with pytest.raises(ValueError, match="requires num_bits"):
        ggml_fake_quant(torch.ones(1, 256), SimpleNamespace(num_bits="unknown"))


def test_ggml_codecs_are_exported_from_quantization_package():
    assert mtq.quantize_iq1_s is iq1_s_module.quantize_iq1_s
    assert mtq.quantize_iq2_xs is iq2_xs_module.quantize_iq2_xs


@pytest.mark.parametrize(
    "extra_args",
    [
        {"block_chunk_size": 17},
        {"decode_chunk_size": 19},
        {"block_chunk_size": 17, "decode_chunk_size": 19},
    ],
)
def test_ggml_backend_forwards_chunk_sizes(monkeypatch, extra_args):
    """Both chunk knobs are per-quantizer tunable; they bound different loops."""
    received = {}

    def fake_quant(inputs, _quantizer, **kwargs):
        received.update(kwargs)
        return inputs

    monkeypatch.setattr(backend_module, "iq1_s_fake_quant", fake_quant)
    inputs = torch.ones(1, 256)
    quantizer = SimpleNamespace(num_bits="iq1_s", backend_extra_args=extra_args)

    assert ggml_fake_quant(inputs, quantizer) is inputs
    assert received == extra_args


def test_ggml_backend_rejects_unknown_extra_arg():
    quantizer = SimpleNamespace(num_bits="iq1_s", backend_extra_args={"unknown": 1})

    with pytest.raises(ValueError, match="Unsupported ggml backend_extra_args"):
        ggml_fake_quant(torch.ones(1, 256), quantizer)


@pytest.mark.parametrize(
    ("num_bits", "module", "fake_quant_name", "quantize_name"),
    [
        ("iq1_s", iq1_s_module, "iq1_s_fake_quant", "quantize_iq1_s"),
        ("iq2_xs", iq2_xs_module, "iq2_xs_fake_quant", "quantize_iq2_xs"),
    ],
)
def test_ggml_backend_caches_packed_weight_and_invalidates_on_change(
    monkeypatch, num_bits, module, fake_quant_name, quantize_name
):
    weight = torch.randn(1, 256)
    quantizer = SimpleNamespace(num_bits=num_bits, _quantizer_cache=None)
    original_quantize = getattr(module, quantize_name)
    call_count = 0

    def counted_quantize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_quantize(*args, **kwargs)

    monkeypatch.setattr(module, quantize_name, counted_quantize)
    fake_quant = getattr(module, fake_quant_name)

    fake_quant(weight, quantizer, block_chunk_size=1)
    fake_quant(weight, quantizer, block_chunk_size=1)
    assert call_count == 1

    fake_quant(weight, quantizer, block_chunk_size=2)
    assert call_count == 2

    with torch.no_grad():
        weight.add_(0.01)
    fake_quant(weight, quantizer, block_chunk_size=2)
    assert call_count == 3


def test_narrow_to_float32_matches_the_cuda_load_float_policy():
    """Non-finite elements become zero; finite out-of-range elements saturate."""
    largest = torch.finfo(torch.float32).max
    values = torch.tensor(
        [torch.nan, torch.inf, -torch.inf, 1e100, -1e100, 1.5], dtype=torch.float64
    )

    narrowed = narrow_to_float32(values)

    assert narrowed.dtype is torch.float32
    assert torch.equal(
        narrowed, torch.tensor([0.0, 0.0, 0.0, largest, -largest, 1.5], dtype=torch.float32)
    )


@pytest.mark.parametrize(
    ("num_bits", "module"), [("iq1_s", iq1_s_module), ("iq2_xs", iq2_xs_module)]
)
def test_ggml_weight_is_packed_once_across_forwards(monkeypatch, num_bits, module):
    """The packed weight is reused across forwards rather than re-encoded each time.

    TensorQuantizer hands the backend a fresh view of the weight on every forward, so a cache
    that checked tensor identity never hit: the codebook search reran on every forward, roughly
    100x during a generate loop and over 90% of a PTQ run's wall clock.
    """
    packer = f"quantize_{num_bits}"
    original = getattr(module, packer)
    calls = []

    def counting(weight, **kwargs):
        calls.append(tuple(weight.shape))
        return original(weight, **kwargs)

    monkeypatch.setattr(module, packer, counting)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=num_bits, block_sizes={-1: 256}, backend="ggml")
    )
    weight = torch.randn(4, 256)

    with torch.inference_mode():  # what generate() runs under
        for _ in range(5):
            quantizer(weight)

    assert calls == [(4, 256)], f"expected one pack, got {len(calls)}"


@pytest.mark.parametrize(
    ("num_bits", "module"), [("iq1_s", iq1_s_module), ("iq2_xs", iq2_xs_module)]
)
def test_ggml_decode_chunk_is_sized_independently_of_the_encode_chunk(
    monkeypatch, num_bits, module
):
    """The decode runs every forward; the encode runs once and holds the big temporaries.

    Sharing one constant between them is what made IQ2_XS four times slower end to end than
    IQ1_S, so pin that the decode gets its own, larger chunk.
    """
    seen = {}
    original = getattr(module, f"dequantize_{num_bits}")

    def recording(packed_weights, weight_shape, **kwargs):
        seen["block_chunk_size"] = kwargs["block_chunk_size"]
        return original(packed_weights, weight_shape, **kwargs)

    monkeypatch.setattr(module, f"dequantize_{num_bits}", recording)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=num_bits, block_sizes={-1: 256}, backend="ggml")
    )
    quantizer(torch.randn(4, 256))

    assert seen["block_chunk_size"] == module._DEFAULT_DECODE_CHUNK_SIZE
    assert module._DEFAULT_DECODE_CHUNK_SIZE > module._DEFAULT_BLOCK_CHUNK_SIZE


@pytest.mark.parametrize(
    ("num_bits", "module"), [("iq1_s", iq1_s_module), ("iq2_xs", iq2_xs_module)]
)
def test_ggml_decode_is_invariant_to_chunk_size(num_bits, module):
    """Chunking the decode is a memory bound, not a numerical choice."""
    torch.manual_seed(0)
    weight = torch.randn(3, 1024, dtype=torch.bfloat16)
    packed, shape = getattr(module, f"quantize_{num_bits}")(weight)
    dequantize = getattr(module, f"dequantize_{num_bits}")

    reference = dequantize(packed, shape, dtype=weight.dtype, block_chunk_size=1)
    for chunk in (2, 7, 4096):
        assert torch.equal(
            dequantize(packed, shape, dtype=weight.dtype, block_chunk_size=chunk), reference
        )

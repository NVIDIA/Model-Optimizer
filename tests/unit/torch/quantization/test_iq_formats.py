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

"""Behaviour every GGML IQ format shares, exercised identically for each of them.

The formats differ only in codebook size, payload layout and bits per weight. Anything that
should hold for one should hold for all, so the contract lives here once and is parametrized
rather than duplicated per format -- a new format is a row in ``FORMATS``.
"""

import numpy as np
import pytest
import torch
from _test_utils.torch.quantization.iq_llama_cpp_vectors import (
    expected_values,
    formats,
    packed_blocks,
)

import modelopt.torch.quantization.ggml.iq1_s as iq1_s_module
import modelopt.torch.quantization.ggml.iq2_xs as iq2_xs_module
import modelopt.torch.quantization.ggml.iq2_xxs as iq2_xxs_module
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.nn import TensorQuantizer

# name -> (module, packed bytes per block, codebook entries, bits per weight)
FORMATS = {
    "iq1_s": (iq1_s_module, 50, 2048, 1.5625),
    "iq2_xxs": (iq2_xxs_module, 66, 256, 2.0625),
    "iq2_xs": (iq2_xs_module, 74, 512, 2.3125),
}
NAMES = sorted(FORMATS)
# IQ1 grids are ternary; IQ2 grids hold the magnitudes 8, 25 and 43.
TERNARY = {"iq1_s"}


def _parts(name):
    module, block_bytes, entries, bits = FORMATS[name]
    return (
        module,
        getattr(module, f"quantize_{name}"),
        getattr(module, f"dequantize_{name}"),
        getattr(module, f"{name}_grid"),
        block_bytes,
        entries,
        bits,
    )


@pytest.mark.parametrize("name", NAMES)
def test_canonical_grid(name):
    _, _, _, grid_fn, _, entries, _ = _parts(name)
    grid = grid_fn()

    assert grid.shape == (entries, 8)
    assert grid.dtype == torch.float32
    if name in TERNARY:
        assert set(grid.unique().tolist()) == {-1.0, 0.0, 1.0}
        assert grid[0].tolist() == [-1.0] * 8
    else:
        assert set(grid.unique().tolist()) <= {8.0, 25.0, 43.0}
        assert grid[0].tolist() == [8.0] * 8


@pytest.mark.parametrize("name", NAMES)
def test_grid_normalizes_unindexed_cuda_device(monkeypatch, name):
    module, _, _, grid_fn, _, _, _ = _parts(name)
    cached = torch.empty(0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setitem(module._GRID_CACHE, torch.device("cuda", 7), cached)

    assert grid_fn("cuda") is cached


@pytest.mark.parametrize("name", NAMES)
def test_effective_bits_matches_the_payload(name):
    module, _, _, _, block_bytes, _, bits = _parts(name)
    assert getattr(module, f"{name.upper()}_BLOCK_BYTES") == block_bytes
    assert getattr(module, f"{name.upper()}_EFFECTIVE_BITS") == pytest.approx(bits)
    assert getattr(module, f"{name.upper()}_BLOCK_SIZE") == 256


@pytest.mark.parametrize("name", NAMES)
def test_round_trip_and_payload_fields(name):
    _, quantize, dequantize, _, block_bytes, _, _ = _parts(name)
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn((2, 512), generator=generator, dtype=torch.bfloat16)

    packed, shape = quantize(weight)
    reconstructed = dequantize(packed, shape)
    chunked = dequantize(packed, shape, block_chunk_size=1)

    assert packed.shape == (2, 2, block_bytes)
    assert packed.dtype == torch.uint8
    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == torch.bfloat16
    assert torch.equal(reconstructed, chunked)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.25


@pytest.mark.parametrize("name", NAMES)
def test_decode_is_invariant_to_chunk_size(name):
    """Chunking the decode is a memory bound, not a numerical choice."""
    _, quantize, dequantize, _, _, _, _ = _parts(name)
    torch.manual_seed(0)
    weight = torch.randn(3, 1024, dtype=torch.bfloat16)
    packed, shape = quantize(weight)

    reference = dequantize(packed, shape, dtype=weight.dtype, block_chunk_size=1)
    for chunk in (2, 7, 4096):
        assert torch.equal(
            dequantize(packed, shape, dtype=weight.dtype, block_chunk_size=chunk), reference
        )


@pytest.mark.parametrize("name", NAMES)
def test_zero_block_has_canonical_zero_encoding(name):
    _, quantize, dequantize, _, _, _, _ = _parts(name)
    weight = torch.zeros((2, 256), dtype=torch.bfloat16)
    packed, shape = quantize(weight)

    assert not packed.any()
    assert torch.equal(dequantize(packed, shape), weight)


@pytest.mark.parametrize("name", NAMES)
def test_underflowed_scale_has_canonical_zero_encoding(name):
    """A block whose scale rounds to zero in FP16 packs as all zero bytes.

    The magnitude has to clear every format's threshold at once: the IQ1 formats divide by a
    native max of 16.875 against the IQ2 formats' 166.6, so a weight that underflows an IQ2
    scale still lands on an FP16 subnormal for IQ1.
    """
    _, quantize, dequantize, _, _, _, _ = _parts(name)
    weight = torch.full((1, 256), -1e-8, dtype=torch.bfloat16)
    packed, shape = quantize(weight)

    assert not packed.any()
    assert torch.equal(dequantize(packed, shape), torch.zeros_like(weight))


@pytest.mark.parametrize("name", NAMES)
def test_requires_complete_last_dimension_blocks(name):
    _, quantize, _, _, _, _, _ = _parts(name)
    with pytest.raises(ValueError, match="last weight dimension"):
        quantize(torch.ones(2, 257))


@pytest.mark.parametrize("name", NAMES)
def test_treats_nonfinite_values_as_zero(name):
    _, quantize, _, _, _, _, _ = _parts(name)
    weight = torch.zeros(1, 256)
    weight[0, 0] = float("nan")
    weight[0, 1] = float("inf")
    weight[0, 2] = float("-inf")

    packed, _ = quantize(weight)
    assert not packed.any()


@pytest.mark.parametrize("name", NAMES)
def test_saturates_finite_values_above_the_float32_range(name):
    """float64 weights are accepted, so a finite value too large for float32 must saturate.

    Converting before sanitizing would turn it into infinity and then zero, which silently
    encodes a large weight as nothing and diverges from the CUDA ``load_float`` policy.
    """
    _, quantize, _, _, _, _, _ = _parts(name)
    torch.manual_seed(0)
    weight = torch.randn(1, 256, dtype=torch.float64)
    weight[0, 7] = 1e100
    saturated = weight.clone()
    saturated[0, 7] = torch.finfo(torch.float32).max
    zeroed = weight.clone()
    zeroed[0, 7] = 0.0

    packed, _ = quantize(weight)
    assert torch.equal(packed, quantize(saturated)[0])
    assert not torch.equal(packed, quantize(zeroed)[0])


@pytest.mark.parametrize("name", NAMES)
def test_rejects_invalid_shape_metadata(name):
    _, _, dequantize, _, block_bytes, _, _ = _parts(name)
    packed = torch.zeros((1, 1, block_bytes), dtype=torch.uint8)
    for weight_shape in (torch.tensor([[1, 256]]), torch.tensor([1.0, 256.0]), torch.tensor([257])):
        with pytest.raises(ValueError, match=r"weight_shape|logical weight shape"):
            dequantize(packed, weight_shape)


@pytest.mark.parametrize("name", NAMES)
def test_rejects_scalar_packed_payload(name):
    _, _, dequantize, _, _, _, _ = _parts(name)
    with pytest.raises(ValueError, match="packed_weights"):
        dequantize(torch.tensor(0, dtype=torch.uint8), torch.tensor([1, 256]))


@pytest.mark.parametrize("name", NAMES)
def test_fake_quant_has_pass_through_gradient(name):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=name, block_sizes={-1: 256}, backend="ggml")
    )
    weight = torch.randn(2, 256, requires_grad=True)
    quantizer(weight).sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))


@pytest.mark.parametrize("name", NAMES)
def test_search_is_independent_of_default_dtype(name):
    """The encoder must not inherit a global default dtype; it works in float32 throughout."""
    _, quantize, _, _, _, _, _ = _parts(name)
    torch.manual_seed(0)
    weight = torch.randn(2, 256)
    expected, _ = quantize(weight)
    try:
        torch.set_default_dtype(torch.float64)
        assert torch.equal(quantize(weight)[0], expected)
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("name", formats())
def test_decoder_matches_llama_cpp_on_captured_blocks(name):
    """Decode bytes we did not produce and match llama.cpp's own output exactly.

    A round-trip against our own encoder cannot catch a layout error that the encoder makes
    symmetrically; these blocks come from a real checkpoint, so they can.
    """
    _, _, dequantize, _, block_bytes, _, _ = _parts(name)
    blocks = packed_blocks(name)
    assert blocks.shape[1] == block_bytes

    count = blocks.shape[0]
    decoded = dequantize(
        torch.from_numpy(blocks).reshape(count, 1, block_bytes),
        torch.tensor([count, 256]),
        dtype=torch.float32,
    ).reshape(count, 256)

    assert np.array_equal(decoded.numpy(), expected_values(name))


def test_every_format_has_conformance_vectors():
    """A new format must arrive with blocks captured from a real llama.cpp checkpoint."""
    assert sorted(formats()) == NAMES


def test_error_decreases_with_bit_width():
    """More bits must buy less error, or a format's scale handling is wrong."""
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn((4, 1024), generator=generator)
    errors = []
    for name in sorted(NAMES, key=lambda n: FORMATS[n][3]):
        quantizer = TensorQuantizer(
            QuantizerAttributeConfig(num_bits=name, block_sizes={-1: 256}, backend="ggml")
        )
        errors.append(float((quantizer(weight) - weight).square().mean()))

    assert errors == sorted(errors, reverse=True), dict(zip(sorted(NAMES), errors))


def test_every_registered_format_is_covered():
    """A format registered for dispatch must also be listed here, or it escapes this contract."""
    assert sorted(IQ_FORMAT_REGISTRY) == sorted(FORMATS)

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

"""CUDA encoders for every GGML IQ format.

Parametrized over the family rather than written per format, so a behaviour asserted
for one is asserted for all. IQ1_S is the odd one out only in its entry point: it
derives the block scale in its own kernel instead of taking a precomputed one.
"""

import pytest
import torch

import modelopt.torch.quantization.ggml.iq1_s as iq1_s_module
import modelopt.torch.quantization.ggml.iq2_xs as iq2_xs_module
import modelopt.torch.quantization.ggml.iq2_xxs as iq2_xxs_module
from modelopt.torch.quantization.extensions import get_cuda_ext_ggml
from modelopt.torch.quantization.ggml import (
    IQ1_S_BLOCK_BYTES,
    IQ2_XS_BLOCK_BYTES,
    IQ2_XXS_BLOCK_BYTES,
    IQ_FORMAT_REGISTRY,
)

# module, packer name, per-block payload size, whether the packer takes precomputed scales
FORMATS = {
    "iq1_s": (iq1_s_module, "iq1_s_pack", IQ1_S_BLOCK_BYTES, False),
    "iq2_xxs": (iq2_xxs_module, "iq2_xxs_pack", IQ2_XXS_BLOCK_BYTES, True),
    "iq2_xs": (iq2_xs_module, "iq2_xs_pack", IQ2_XS_BLOCK_BYTES, True),
}


def _extension():
    extension = get_cuda_ext_ggml(raise_if_failed=True)
    assert extension is not None
    return extension


def _pack(name, weight):
    module, packer, _, takes_scales = FORMATS[name]
    grid = getattr(module, f"{name}_grid")("cuda")
    if not takes_scales:
        return getattr(_extension(), packer)(weight, grid)
    blocks = weight.contiguous().reshape(-1, 256)
    scales = getattr(module, f"_predict_{name}_scales")(blocks)
    return getattr(_extension(), packer)(weight, grid, scales)


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_cuda_pack_matches_pytorch_encoder_and_is_decodable(monkeypatch, name):
    """Byte parity with the reference encoder on a fixed small weight.

    Exact parity is asserted on this input, not in general: the two encoders evaluate the same
    squared error with different floating-point fusion, so where two local scales fall within a
    float32 ULP they can round to different sides. That happens in roughly one block in several
    thousand, costs under 1e-8 of relative reconstruction error, and favours neither encoder --
    see ``test_cuda_pack_reconstruction_matches_pytorch_at_scale``.
    """
    module, _, block_bytes, _ = FORMATS[name]
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((8, 512), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed = _pack(name, weight).reshape(8, 2, block_bytes)
    packed_again = _pack(name, weight).reshape(8, 2, block_bytes)
    monkeypatch.setattr(module, "get_cuda_ext_ggml", lambda: None)
    reference, shape = getattr(module, f"quantize_{name}")(weight)
    reconstructed = getattr(module, f"dequantize_{name}")(packed, shape)

    assert packed.shape == (8, 2, block_bytes)
    assert torch.equal(packed, packed_again)
    assert torch.equal(packed, reference)
    assert shape.device.type == "cpu"
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.25


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_cuda_pack_is_deterministic_on_one_device(name):
    """The contract reproducibility actually needs: same machine, same bytes."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    weight = torch.randn((64, 1024), generator=generator, device="cuda", dtype=torch.bfloat16)
    first = _pack(name, weight)
    for _ in range(3):
        assert torch.equal(_pack(name, weight), first)


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_cuda_pack_reconstruction_matches_pytorch_at_scale(monkeypatch, name):
    """Over many blocks the encoders may disagree on a near-tied scale, but not on quality."""
    module, _, block_bytes, _ = FORMATS[name]
    generator = torch.Generator(device="cuda").manual_seed(11)
    weight = torch.randn((128, 2048), generator=generator, device="cuda", dtype=torch.bfloat16)
    blocks = weight.shape[0] * weight.shape[1] // 256

    packed = _pack(name, weight).reshape(weight.shape[0], weight.shape[1] // 256, block_bytes)
    monkeypatch.setattr(module, "get_cuda_ext_ggml", lambda: None)
    reference, shape = getattr(module, f"quantize_{name}")(weight)

    dequantize = getattr(module, f"dequantize_{name}")
    target = weight.float()
    denominator = target.square().sum()
    cuda_error = (
        dequantize(packed, shape, dtype=torch.float32) - target
    ).square().sum() / denominator
    torch_error = (
        dequantize(reference, shape, dtype=torch.float32) - target
    ).square().sum() / denominator

    differing = int((packed != reference).any(dim=-1).sum())
    assert differing <= blocks // 1000, f"{differing} of {blocks} blocks differ"
    assert torch.isclose(cuda_error, torch_error, rtol=1e-5)


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_cuda_zero_encoding_matches_ggml_block_layout(name):
    module, _, block_bytes, _ = FORMATS[name]
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    packed = _pack(name, weight).reshape(1, 1, block_bytes)
    shape = torch.tensor(weight.shape, device="cuda")

    assert not packed.any()
    assert torch.equal(getattr(module, f"dequantize_{name}")(packed, shape), weight)


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_cuda_nonfinite_policy_matches_pytorch_encoder(monkeypatch, name):
    module, _, _, _ = FORMATS[name]
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.float32)
    weight[0, 0] = float("nan")
    weight[0, 1] = float("inf")
    weight[0, 2] = float("-inf")

    packed, _ = getattr(module, f"quantize_{name}")(weight)
    monkeypatch.setattr(module, "get_cuda_ext_ggml", lambda: None)
    reference, _ = getattr(module, f"quantize_{name}")(weight)
    assert torch.equal(packed, reference)


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_cuda_falls_back_to_pytorch_encoder(monkeypatch, name):
    """Without the extension the format still packs, through the torch search."""
    module, _, block_bytes, _ = FORMATS[name]
    monkeypatch.setattr(module, "get_cuda_ext_ggml", lambda: None)
    generator = torch.Generator(device="cuda").manual_seed(5)
    weight = torch.randn((2, 256), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed, shape = getattr(module, f"quantize_{name}")(weight)
    reconstructed = getattr(module, f"dequantize_{name}")(packed, shape)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()

    assert packed.shape == (2, 1, block_bytes)
    assert normalized_mse < 0.25


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_cuda_float64_matches_pytorch_encoder(monkeypatch, name):
    module, _, _, _ = FORMATS[name]
    generator = torch.Generator(device="cuda").manual_seed(99)
    weight = torch.randn((2, 256), generator=generator, device="cuda", dtype=torch.float64)

    packed, _ = getattr(module, f"quantize_{name}")(weight)
    monkeypatch.setattr(module, "get_cuda_ext_ggml", lambda: None)
    reference, _ = getattr(module, f"quantize_{name}")(weight)
    assert torch.equal(reference, packed)


def test_every_registered_format_is_covered():
    """A format registered for dispatch must also be listed here, or it escapes this contract."""
    assert sorted(IQ_FORMAT_REGISTRY) == sorted(FORMATS)

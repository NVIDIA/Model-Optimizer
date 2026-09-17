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


from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch

import modelopt.torch.quantization.extensions as ext

# Override default timeout as these tests JIT-compile the CUDA extensions, which is slow
pytestmark = pytest.mark.timeout(240)


# Compile extensions first so it does not count towards time used to run a test that needs it
def test_cuda_ext():
    assert ext.get_cuda_ext() is not None


def test_cuda_ext_fp8():
    assert ext.get_cuda_ext_fp8() is not None


def test_cuda_ext_mx():
    assert ext.get_cuda_ext_mx() is not None


def test_cuda_ext_iq1_s():
    assert ext.get_cuda_ext_iq1_s() is not None


def test_cuda_ext_iq2_xs():
    assert ext.get_cuda_ext_iq2_xs() is not None


def _generator():
    """Seeded generator so a failure reproduces exactly."""
    return torch.Generator(device="cuda").manual_seed(0)


class _IqFormat(NamedTuple):
    """One GGML IQ packing extension and the format constants its contract is defined by."""

    get_extension: Callable
    entries: int
    payload_bytes: int
    needs_scales: bool
    # Value alphabet the codebook is built from: signed ternary for IQ1_S, and the non-negative
    # magnitudes IQ2_XS stores (its signs live in the packed code).
    grid_values: tuple[float, ...]
    # Largest magnitude the format can represent at a block scale of 1.
    native_max: float


_IQ_EXTENSIONS = (
    pytest.param(
        _IqFormat(ext.get_cuda_ext_iq1_s, 2048, 50, False, (-1.0, 0.0, 1.0), 16.875), id="iq1_s"
    ),
    pytest.param(
        _IqFormat(ext.get_cuda_ext_iq2_xs, 512, 74, True, (1.0, 8.0, 25.0, 43.0), 166.625),
        id="iq2_xs",
    ),
)


def _grid(fmt: _IqFormat, zero: bool = False) -> torch.Tensor:
    """Codebook of ``fmt.entries`` distinct vectors drawn from the format's value alphabet."""
    if zero:
        return torch.zeros((fmt.entries, 8), device="cuda", dtype=torch.float32)
    values = torch.tensor(fmt.grid_values, device="cuda", dtype=torch.float32)
    digits = torch.arange(fmt.entries, device="cuda").unsqueeze(1) // len(fmt.grid_values) ** (
        torch.arange(8, device="cuda")
    )
    return values[digits % len(fmt.grid_values)]


def _pack(fmt: _IqFormat, extension, weight, grid, scales=None):
    if not fmt.needs_scales:
        return extension.pack(weight, grid)
    if scales is None:
        scales = torch.zeros(weight.numel() // 256, device=weight.device, dtype=torch.float16)
    return extension.pack(weight, grid, scales)


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_zero_block_layout(fmt):
    extension = fmt.get_extension(raise_if_failed=True)
    weight = torch.zeros((2, 256), device="cuda", dtype=torch.bfloat16)

    packed = _pack(fmt, extension, weight, _grid(fmt, zero=True))

    assert packed.shape == (2, fmt.payload_bytes)
    assert not packed.any()


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_encodes_non_zero_block(fmt):
    """Exercise the encode loop itself: search, reductions, and the payload writes."""
    extension = fmt.get_extension(raise_if_failed=True)
    weight = torch.randn((2, 256), device="cuda", dtype=torch.bfloat16, generator=_generator())
    scales = (weight.float().abs().amax(dim=-1) / fmt.native_max).half()

    packed = _pack(fmt, extension, weight, _grid(fmt), scales=scales)

    assert packed.shape == (2, fmt.payload_bytes)
    # The fp16 block scale lands in the first two payload bytes, and the caller supplies it
    # verbatim for IQ2_XS.
    block_scale = packed[:, :2].contiguous().view(torch.float16).flatten()
    assert (block_scale > 0).all()
    if fmt.needs_scales:
        assert torch.equal(block_scale, scales)
    # Codebook indices, signs, and local scales are written past the block scale, and two
    # different blocks must not encode identically.
    assert packed[:, 2:].any()
    assert not torch.equal(packed[0], packed[1])


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_rejects_unsupported_dtype(fmt):
    extension = fmt.get_extension(raise_if_failed=True)
    weight = torch.ones((1, 256), device="cuda").to(torch.float8_e4m3fn)

    with pytest.raises(RuntimeError, match="supports float32, float64, float16, and bfloat16"):
        _pack(fmt, extension, weight, _grid(fmt, zero=True))


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_rejects_row_straddling_input(fmt):
    extension = fmt.get_extension(raise_if_failed=True)
    weight = torch.ones((512, 384), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="innermost dimension must be a multiple of 256"):
        _pack(fmt, extension, weight, _grid(fmt, zero=True))


def test_cuda_ext_iq2_xs_rejects_non_finite_scales():
    extension = ext.get_cuda_ext_iq2_xs(raise_if_failed=True)
    fmt = _IQ_EXTENSIONS[1].values[0]
    weight = torch.ones((1, 256), device="cuda", dtype=torch.bfloat16)
    scales = torch.full((1,), float("nan"), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="scales must be finite"):
        extension.pack(weight, _grid(fmt), scales)

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


def test_cuda_ext_ggml():
    assert ext.get_cuda_ext_ggml() is not None


def _generator():
    """Seeded generator so a failure reproduces exactly."""
    return torch.Generator(device="cuda").manual_seed(0)


class _IqFormat(NamedTuple):
    """One GGML IQ packer and the format constants its contract is defined by."""

    # Name the packer is bound under on the shared GGML extension.
    packer: str
    entries: int
    payload_bytes: int
    needs_scales: bool
    # Value alphabet the codebook is built from, as GGML defines it: signed ternary bytes
    # {0x00, 0x01, 0xff} for IQ1_S, and the non-negative magnitudes {0x08, 0x19, 0x2b} for
    # IQ2_XS, whose signs live in the packed code instead.
    grid_values: tuple[float, ...]
    # Largest magnitude the format can represent at a block scale of 1.
    native_max: float


_IQ_EXTENSIONS = (
    pytest.param(_IqFormat("iq1_s_pack", 2048, 50, False, (-1.0, 0.0, 1.0), 16.875), id="iq1_s"),
    pytest.param(
        _IqFormat("iq2_xs_pack", 512, 74, True, (8.0, 25.0, 43.0), 166.625),
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
    pack = getattr(extension, fmt.packer)
    if not fmt.needs_scales:
        return pack(weight, grid)
    if scales is None:
        scales = torch.zeros(weight.numel() // 256, device=weight.device, dtype=torch.float16)
    return pack(weight, grid, scales)


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_zero_block_layout(fmt):
    extension = ext.get_cuda_ext_ggml(raise_if_failed=True)
    weight = torch.zeros((2, 256), device="cuda", dtype=torch.bfloat16)

    packed = _pack(fmt, extension, weight, _grid(fmt, zero=True))

    assert packed.shape == (2, fmt.payload_bytes)
    assert not packed.any()


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_encodes_non_zero_block(fmt):
    """Exercise the encode loop itself: search, reductions, and the payload writes."""
    extension = ext.get_cuda_ext_ggml(raise_if_failed=True)
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
    extension = ext.get_cuda_ext_ggml(raise_if_failed=True)
    weight = torch.ones((1, 256), device="cuda").to(torch.float8_e4m3fn)

    with pytest.raises(RuntimeError, match="supports float32, float64, float16, and bfloat16"):
        _pack(fmt, extension, weight, _grid(fmt, zero=True))


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_rejects_row_straddling_input(fmt):
    extension = ext.get_cuda_ext_ggml(raise_if_failed=True)
    weight = torch.ones((512, 384), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="innermost dimension must be a multiple of 256"):
        _pack(fmt, extension, weight, _grid(fmt, zero=True))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0, -1e-4])
def test_cuda_ext_iq2_xs_rejects_invalid_scales(bad):
    """A non-finite scale decodes to garbage; a negative one inverts every decoded element."""
    extension = ext.get_cuda_ext_ggml(raise_if_failed=True)
    fmt = _IQ_EXTENSIONS[1].values[0]
    weight = torch.ones((1, 256), device="cuda", dtype=torch.bfloat16)
    scales = torch.full((1,), bad, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="scales must be finite and non-negative"):
        extension.iq2_xs_pack(weight, _grid(fmt), scales)


def test_cuda_ext_iq2_xs_negative_zero_scale_packs_as_zero():
    """Negative zero is a zero scale: it must take the zero-payload branch, not search."""
    extension = ext.get_cuda_ext_ggml(raise_if_failed=True)
    fmt = _IQ_EXTENSIONS[1].values[0]
    weight = torch.randn((2, 256), device="cuda", dtype=torch.bfloat16, generator=_generator())
    grid = _random_grid(fmt)
    scales = torch.tensor([-0.0, 0.0], device="cuda", dtype=torch.float16)

    packed = extension.iq2_xs_pack(weight, grid, scales)

    assert not packed.any()


def _random_grid(fmt: _IqFormat) -> torch.Tensor:
    """Random codebook, so the optimality check below has no ties to break.

    The kernels treat the grid as opaque data, so a synthetic codebook exercises the search
    exactly as a real one does -- while keeping these tests independent of the GGML tables.
    IQ2_XS additionally requires non-negative magnitudes, since it carries signs separately.
    """
    shape = (fmt.entries, 8)
    if fmt.needs_scales:
        return torch.rand(shape, device="cuda", generator=_generator()) * fmt.grid_values[-1]
    return torch.randn(shape, device="cuda", generator=_generator())


def _decode(fmt: _IqFormat, packed: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Decode a packed payload the way GGML does, from the format definition rather than from
    the kernel's own layout code, so a misplaced field shows up as a decode mismatch.
    """
    payload = packed.cpu()
    blocks = payload.shape[0]
    grid = grid.cpu()
    # The fp16 block scale occupies the first two bytes of every IQ payload.
    d = payload[:, :2].contiguous().view(torch.float16).float()

    if not fmt.needs_scales:  # IQ1_S: 32 index bytes, then 8 uint16 of per-group metadata.
        qs = payload[:, 2:34].int().view(blocks, 8, 4)
        qh = payload[:, 34:50:2].int() | (payload[:, 35:50:2].int() << 8)
        local = (qh >> 12) & 7
        delta = torch.where((qh & 0x8000) != 0, -0.125, 0.125)
        index = qs | (((qh.unsqueeze(-1) >> (3 * torch.arange(4))) & 7) << 8)
        scale = (d * (2 * local + 1)).unsqueeze(-1).unsqueeze(-1)
        return (scale * (grid[index] + delta[..., None, None])).reshape(blocks, 256)

    # IQ2_XS: 32 uint16 codes, then 16 four-bit local scales packed two per byte.
    codes = payload[:, 2:66:2].int() | (payload[:, 3:66:2].int() << 8)
    index = codes & (fmt.entries - 1)
    stored = ((codes >> 9).unsqueeze(-1) >> torch.arange(7)) & 1
    # Only seven sign bits are stored; the eighth restores even parity over all eight.
    signs = torch.cat([stored, (stored.sum(-1) & 1).unsqueeze(-1)], dim=-1)
    nibbles = payload[:, 66:74].int()
    local = torch.stack([nibbles & 0xF, (nibbles >> 4) & 0xF], dim=-1).reshape(blocks, 16)
    scale = (d * (2 * local + 1) * 0.125).repeat_interleave(2, dim=1).unsqueeze(-1)
    return (scale * grid[index] * (1.0 - 2.0 * signs.float())).reshape(blocks, 256)


def _oracle_group_error(
    fmt: _IqFormat, values: torch.Tensor, grid: torch.Tensor, d
) -> torch.Tensor:
    """Smallest squared error each group can reach at the block scale the payload carries.

    Reproduces the kernels' objective by brute force: every local scale (and, for IQ1_S, every
    delta sign) against every codebook entry, minimised per vector and summed over the group.
    """
    vectors, choices = (4, 16) if not fmt.needs_scales else (2, 16)
    groups = 32 // vectors
    x = values.cpu().float().reshape(-1, groups, vectors, 8)
    grid, d = grid.cpu(), d.cpu()
    xnorm = x.square().sum(-1)

    errors = []
    for choice in range(choices):
        local = choice & 7 if not fmt.needs_scales else choice
        if not fmt.needs_scales:
            delta = 0.125 if choice < 8 else -0.125
            shifted = grid + delta
            scale = (d * (2 * local + 1)).reshape(-1, 1, 1, 1)
            dot = x @ shifted.T
        else:
            shifted = grid
            scale = (d * (2 * local + 1) * 0.125).reshape(-1, 1, 1, 1)
            terms = x.abs().unsqueeze(-2) * grid  # [..., entries, 8]
            dot = terms.sum(-1)
            odd = (x < 0).sum(-1, keepdim=True) % 2 != 0
            dot = torch.where(odd, dot - 2 * terms.min(-1).values, dot)
        dot = dot.reshape(*x.shape[:3], fmt.entries)
        error = xnorm.unsqueeze(-1) - 2 * scale * dot + scale.square() * shifted.square().sum(-1)
        errors.append(error.clamp_min(0).min(-1).values.sum(-1))
    return torch.stack(errors).min(0).values


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_encoding_is_optimal(fmt):
    """Round-trip the payload and check the search actually found the best codes.

    This is the test that pins the bit layout: decoding follows the GGML field positions, so a
    misplaced index, local scale, delta sign, or sign bit makes the reconstruction worse than
    the brute-force optimum rather than merely different.
    """
    extension = ext.get_cuda_ext_ggml(raise_if_failed=True)
    weight = torch.randn((4, 256), device="cuda", dtype=torch.float32, generator=_generator())
    grid = _random_grid(fmt)
    scales = (weight.abs().amax(dim=-1) / fmt.native_max).half() if fmt.needs_scales else None

    packed = _pack(fmt, extension, weight, grid, scales=scales)
    decoded = _decode(fmt, packed, grid)

    # The block scale is a fixed heuristic, so compare the search at the scale actually stored.
    d = packed[:, :2].contiguous().view(torch.float16).float()
    group_size = 8 * (2 if fmt.needs_scales else 4)
    achieved = (weight.cpu() - decoded).square().reshape(4, -1, group_size).sum(-1)
    optimal = _oracle_group_error(fmt, weight, grid, d)

    assert torch.allclose(achieved, optimal, rtol=1e-3, atol=1e-6), (
        f"max excess {(achieved - optimal).abs().max():.3e}"
    )
    # Sanity: the quantizer must be doing better than emitting zeros.
    assert achieved.sum() < 0.5 * weight.cpu().square().sum()


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_input_dtype_equivalence(fmt):
    """Every accepted input dtype carrying identical values must pack to identical bytes."""
    extension = ext.get_cuda_ext_ggml(raise_if_failed=True)
    # Multiples of 1/16 in [-4, 4) are exact in float16 and bfloat16 as well as the wider types.
    weight = torch.randint(-64, 64, (2, 256), device="cuda", generator=_generator()).float() / 16
    grid = _random_grid(fmt)
    scales = (weight.abs().amax(dim=-1) / fmt.native_max).half() if fmt.needs_scales else None

    payloads = [
        _pack(fmt, extension, weight.to(dtype), grid, scales=scales)
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16)
    ]

    for dtype, payload in zip((torch.float64, torch.float16, torch.bfloat16), payloads[1:]):
        assert torch.equal(payloads[0], payload), f"{dtype} disagrees with float32"


@pytest.mark.parametrize("fmt", _IQ_EXTENSIONS)
def test_cuda_ext_iq_non_finite_inputs_are_zeroed(fmt):
    """NaN and infinity pack as zeros; finite values too large for float32 saturate instead."""
    extension = ext.get_cuda_ext_ggml(raise_if_failed=True)
    clean = torch.randn((2, 256), device="cuda", dtype=torch.float32, generator=_generator())
    grid = _random_grid(fmt)
    scales = (clean.abs().amax(dim=-1) / fmt.native_max).half() if fmt.needs_scales else None

    spoiled = clean.clone()
    spoiled[0, 5], spoiled[0, 200], spoiled[1, 17] = float("nan"), float("inf"), float("-inf")
    zeroed = clean.clone()
    zeroed[0, 5], zeroed[0, 200], zeroed[1, 17] = 0.0, 0.0, 0.0

    assert torch.equal(
        _pack(fmt, extension, spoiled, grid, scales=scales),
        _pack(fmt, extension, zeroed, grid, scales=scales),
    )

    # A finite float64 outside the float32 range must saturate, not collapse to zero.
    huge = zeroed.double()
    huge[1, 7] = 1e100
    assert not torch.equal(
        _pack(fmt, extension, huge, grid, scales=scales),
        _pack(fmt, extension, zeroed.double(), grid, scales=scales),
    )

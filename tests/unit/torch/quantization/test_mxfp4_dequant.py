# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the standalone MXFP4-packed -> BF16 dequantization entry point.

These cover:

* A. mathematical correctness — LUT, nibble order, per-group scale,
  UE8M0 exponent math.
* B. layout correctness — DS 2D shape and rank-agnostic prefix.
* C. cross-validation against two independent references:
    - ``transformers._convert_moe_packed_tensors`` (the MX convention used
      by gpt-oss);
    - DeepSeek-V4's own ``cast_e2m1fn_to_e4m3fn`` + a trivial FP8-to-BF16
      dequant stage, which is the "lossless via FP8" path the model's
      ``inference/convert.py`` uses.
"""

import importlib.util
import os

import pytest
import torch

from modelopt.torch.quantization.qtensor.mxfp4_tensor import MXFP4QTensor

# Signed E2M1 value table indexed by 4-bit pattern (bit 3 = sign, bits 2..0 = magnitude).
E2M1_SIGNED = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _pack_fp4(indices: torch.Tensor) -> torch.Tensor:
    """Pack ``(..., K)`` tensor of 4-bit nibble indices into ``(..., K//2)`` uint8.

    Convention matches the checkpoint layout: low nibble = even element,
    high nibble = odd element (i.e. ``value[..., 2k]`` lives in the low
    4 bits of ``byte[..., k]``).
    """
    assert indices.shape[-1] % 2 == 0, "Last dim must be even"
    low = indices[..., 0::2].to(torch.uint8) & 0x0F
    high = indices[..., 1::2].to(torch.uint8) & 0x0F
    return low | (high << 4)


# --- A. mathematical correctness ---


class TestMathematicalCorrectness:
    def test_lut_exhaustive_all_16_values(self):
        """Every 4-bit pattern decodes to its E2M1 signed value at scale=2^0."""
        # 32 nibbles: pattern 0..15 twice so all 16 values appear in low and high nibble.
        indices = torch.arange(32, dtype=torch.uint8) % 16
        blocks = _pack_fp4(indices)  # shape (16,)
        scales = torch.tensor([127], dtype=torch.uint8)  # UE8M0 byte 127 -> 2^0
        out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=32, dtype=torch.float32)
        expected = torch.tensor([E2M1_SIGNED[i] for i in indices.tolist()], dtype=torch.float32)
        assert torch.equal(out, expected)

    def test_nibble_ordering_low_is_even(self):
        """Low nibble of each byte must correspond to the even element index."""
        # Byte[0] = (high=0xA, low=0x3) -> even=1.5, odd=-1.0
        # Byte[1] = (high=0xF, low=0x9) -> even=-0.5, odd=-6.0
        indices = torch.tensor([0x3, 0xA, 0x9, 0xF] + [0] * 28, dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        scales = torch.tensor([127], dtype=torch.uint8)
        out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=32, dtype=torch.float32)
        assert out[0].item() == 1.5
        assert out[1].item() == -1.0
        assert out[2].item() == -0.5
        assert out[3].item() == -6.0

    def test_per_group_scale_independence(self):
        """Two adjacent groups of 32 use their own scales; values across the group
        boundary are scaled independently."""
        # Every fp4 index is 1 (value 0.5). 64 values -> 2 groups.
        indices = torch.full((64,), 1, dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        # Group 0: 2^0; Group 1: 2^5 = 32
        scales = torch.tensor([127, 127 + 5], dtype=torch.uint8)
        out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=32, dtype=torch.float32)
        assert out[31].item() == 0.5
        assert out[32].item() == 0.5 * 32

    def test_ue8m0_scale_math(self):
        """Scale bytes decode as 2^(byte - 127). Verify at a few exponents."""
        # 32 nibbles: only first is index 1 (value 0.5), rest are 0.
        indices = torch.tensor([1] + [0] * 31, dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        for byte, expected_multiplier in [(127, 1.0), (117, 2**-10), (137, 2**10), (97, 2**-30)]:
            scales = torch.tensor([byte], dtype=torch.uint8)
            out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=32, dtype=torch.float32)
            assert out[0].item() == pytest.approx(0.5 * expected_multiplier), f"byte={byte}"


# --- B. layout correctness ---


class TestLayoutCorrectness:
    def test_ds_2d_expert_layout(self):
        """Canonical DS V4 expert-weight input: (M, K//2) bytes + (M, K//32) scales."""
        M, K, block_size = 4, 128, 32
        torch.manual_seed(0)
        indices = torch.randint(0, 16, (M, K), dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        scales = torch.randint(100, 151, (M, K // block_size), dtype=torch.uint8)
        out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=block_size)
        assert out.shape == (M, K)
        assert out.dtype == torch.bfloat16

    def test_rank_agnostic_prefix(self):
        """3D prefix gives same result as 2D-flattened call on reshaped inputs."""
        E, M, K, block_size = 3, 4, 64, 32
        torch.manual_seed(1)
        indices = torch.randint(0, 16, (E, M, K), dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        scales = torch.randint(100, 151, (E, M, K // block_size), dtype=torch.uint8)

        out_3d = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=block_size)
        out_flat = MXFP4QTensor.dequantize_packed(
            blocks.reshape(E * M, K // 2),
            scales.reshape(E * M, K // block_size),
            block_size=block_size,
        )
        assert out_3d.shape == (E, M, K)
        assert torch.equal(out_3d.reshape(E * M, K), out_flat)


# --- C. cross-validation against transformers ---


class TestCrossValidationWithTransformers:
    """The important ones. The whole point of the helper is to agree with the
    MX convention as implemented by ``transformers._convert_moe_packed_tensors``."""

    @staticmethod
    def _transformers_reference(blocks_gptoss_layout, scales):
        """Run transformers' reference but strip the GPT-OSS-specific trailing transpose."""
        transformers = pytest.importorskip("transformers")
        from transformers.integrations.mxfp4 import _convert_moe_packed_tensors
        out = _convert_moe_packed_tensors(blocks_gptoss_layout, scales, dtype=torch.bfloat16)
        # transformers returns shape (..., M, K) but with a final transpose(1,2) already
        # applied — so for input (E, M, G, 16) it outputs (E, K, M). Undo it.
        return out.transpose(-1, -2).contiguous()

    def test_matches_transformers_on_random_inputs(self):
        """Bit-identical match against transformers on random MXFP4 inputs."""
        transformers = pytest.importorskip("transformers")
        from transformers.integrations.mxfp4 import _convert_moe_packed_tensors  # noqa: F401

        E, M, K, block_size = 2, 8, 256, 32
        G = K // block_size
        torch.manual_seed(42)
        # GPT-OSS-style storage: (E, M, G, 16) packed bytes + (E, M, G) scales.
        blocks_4d = torch.randint(0, 256, (E, M, G, 16), dtype=torch.uint8)
        scales_3d = torch.randint(100, 151, (E, M, G), dtype=torch.uint8)

        ref = self._transformers_reference(blocks_4d, scales_3d)  # (E, M, K)
        # DS layout: flatten group+pack axes into one trailing dim.
        blocks_ds = blocks_4d.reshape(E, M, G * 16)
        out = MXFP4QTensor.dequantize_packed(blocks_ds, scales_3d, block_size=block_size)

        assert out.shape == ref.shape == (E, M, K)
        assert torch.equal(out, ref)

    def test_matches_transformers_on_2d_ds_shape(self):
        """DS V4 experts arrive as 2D (M, K//2). Make sure the 2D path agrees too."""
        pytest.importorskip("transformers")
        from transformers.integrations.mxfp4 import _convert_moe_packed_tensors

        M, K, block_size = 16, 128, 32
        G = K // block_size
        torch.manual_seed(7)
        blocks_2d = torch.randint(0, 256, (M, G * 16), dtype=torch.uint8)
        scales_2d = torch.randint(100, 151, (M, G), dtype=torch.uint8)

        # To feed transformers we promote to (1, M, G, 16) and undo the trailing transpose.
        blocks_4d = blocks_2d.reshape(1, M, G, 16)
        scales_3d = scales_2d.reshape(1, M, G)
        ref = _convert_moe_packed_tensors(blocks_4d, scales_3d, dtype=torch.bfloat16)
        ref = ref.transpose(-1, -2).contiguous().squeeze(0)  # -> (M, K)

        out = MXFP4QTensor.dequantize_packed(blocks_2d, scales_2d, block_size=block_size)
        assert torch.equal(out, ref)


class TestCrossValidationWithDeepSeekConvert:
    """C#8: validate against DeepSeek-V4's own ``cast_e2m1fn_to_e4m3fn`` + FP8-to-BF16
    dequant path. This is the "lossless via FP8" story the model's ``inference/convert.py``
    relies on, valid when the per-128-block scale spread is within 2^6 octaves.
    """

    @staticmethod
    def _load_convert_module():
        path = os.environ.get(
            "DSV4_CONVERT_PY",
            "/home/mxin/mxin/dsv4/models/DeepSeek-V4-Pro/inference/convert.py",
        )
        if not os.path.exists(path):
            pytest.skip(f"DeepSeek V4 convert.py not found at {path} (set DSV4_CONVERT_PY)")
        spec = importlib.util.spec_from_file_location("dsv4_convert", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _fp8_blockwise_to_bf16(fp8, fp8_scale_e8m0, block=128):
        """Minimal FP8 E4M3 × UE8M0 128x128 block-scale -> BF16 dequant."""
        out_dim, in_dim = fp8.shape
        assert fp8_scale_e8m0.shape == (out_dim // block, in_dim // block)
        scale_exp = fp8_scale_e8m0.contiguous().view(torch.uint8).to(torch.int32) - 127
        # Broadcast per-block exponent to element granularity
        scale_exp = scale_exp.repeat_interleave(block, 0).repeat_interleave(block, 1)
        return torch.ldexp(fp8.to(torch.float32), scale_exp).to(torch.bfloat16)

    def test_matches_fp4_to_fp8_to_bf16(self):
        mod = self._load_convert_module()
        cast_e2m1fn_to_e4m3fn = mod.cast_e2m1fn_to_e4m3fn

        # Smallest sizes satisfying convert.py's divisibility constraints
        out_dim, in_dim = 128, 256
        G32 = in_dim // 32
        torch.manual_seed(11)
        # DS layout: int8 packed, shape (out_dim, in_dim // 2)
        blocks = torch.randint(-128, 128, (out_dim, in_dim // 2), dtype=torch.int8)
        # Keep scales in a narrow range so that within every 128-block the 4 fp4-group scales
        # span < 2^6 -> no underflow, paths are bit-identical.
        scale_bytes = 127 + torch.randint(-3, 4, (out_dim, G32), dtype=torch.int32)
        scale_bytes = scale_bytes.clamp(1, 254).to(torch.uint8)
        scales_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)

        # Path A: our direct MXFP4 -> BF16
        out_a = MXFP4QTensor.dequantize_packed(
            blocks, scales_e8m0, block_size=32, dtype=torch.bfloat16
        )

        # Path B: DS's cast_e2m1fn_to_e4m3fn (MXFP4 -> FP8 at 128x128 blocks) then FP8 -> BF16
        fp8, fp8_scale = cast_e2m1fn_to_e4m3fn(blocks, scales_e8m0)
        out_b = self._fp8_blockwise_to_bf16(fp8, fp8_scale, block=128)

        assert out_a.shape == out_b.shape == (out_dim, in_dim)
        assert torch.equal(out_a, out_b)

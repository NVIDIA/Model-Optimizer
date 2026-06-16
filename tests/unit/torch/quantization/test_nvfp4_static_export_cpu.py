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

"""CPU round-trip tests for NVFP4 static export with extreme per-block amax (underflow/overflow)."""

from __future__ import annotations

import pytest
import torch

from modelopt.torch.export.quant_utils import QUANTIZATION_NVFP4, to_quantized_weight
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization.qtensor.nvfp4_tensor import FP4_E2M1_MAX

BLOCK_SIZE = 16
FP4_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _make_static_quantizer(
    per_block_amax: torch.Tensor, global_amax: torch.Tensor
) -> NVFP4StaticQuantizer:
    cfg = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: BLOCK_SIZE, "type": "static", "scale_bits": (4, 3)},
    )
    q = NVFP4StaticQuantizer(quant_attribute_cfg=cfg)
    q.amax = per_block_amax.clone()
    q.global_amax = global_amax.clone()
    return q


def _export_round_trip(
    weight: torch.Tensor, quantizer: NVFP4StaticQuantizer
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full export path and dequantize it back, mimicking vLLM serving.

    Returns (weight_scale_fp8, weight_scale_2_fp32, dequantized_weight_bf16).
    """
    weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(quantizer)
    weight_scale, _ = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
        quantizer, weight, weight_scale_2.to(weight.device)
    )
    packed = to_quantized_weight(
        weight,
        weight_scale,
        QUANTIZATION_NVFP4,
        weight_scale_2,
        BLOCK_SIZE,
    )
    qtensor = NVFP4QTensor(weight.shape, weight.dtype, packed)
    dequant = qtensor.dequantize(
        scale=weight_scale,
        double_scale=weight_scale_2,
        block_sizes={-1: BLOCK_SIZE},
        dtype=weight.dtype,
    )
    return weight_scale, weight_scale_2, dequant


def _layer1_routed_expert_like(
    out_dim: int, in_dim: int, *, n_outliers: int, seed: int
) -> torch.Tensor:
    """Synthesize a tensor whose block-amax distribution matches the failure case.

    The vast majority of blocks have tiny absolute values (~1e-7), and a handful
    of rows carry an outlier magnitude (~1e-1) that drives the global tensor
    amax to the FP8-normal regime. This is the distribution that produces 81%
    raw-1 FP8 block scales in the broken Ultra V3 MSE export.
    """
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_dim, in_dim, generator=g, dtype=torch.bfloat16) * 1e-7
    # Inject outliers in n_outliers distinct (row, col) positions.
    n = max(1, n_outliers)
    rows = torch.randint(0, out_dim, (n,), generator=g)
    cols = torch.randint(0, in_dim, (n,), generator=g)
    weight[rows, cols] = torch.randn(n, generator=g, dtype=torch.bfloat16) * 0.1
    return weight


def _per_block_max(weight: torch.Tensor) -> torch.Tensor:
    """Per-(out, num_blocks) absolute max, mirroring NVFP4 block-amax reduction."""
    blocks = weight.float().view(*weight.shape[:-1], -1, BLOCK_SIZE)
    return blocks.abs().amax(dim=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNVFP4StaticExportFiniteAndBounded:
    """The static export path must never emit NaN/Inf scales or dequant values.

    These tests fail fast if any saved bit pattern (weight_scale, packed weight,
    dequantized result) contains a NaN or Inf, no matter what amax the caller
    set on the static quantizer.
    """

    def test_layer1_like_distribution_no_nan(self):
        weight = _layer1_routed_expert_like(64, 256, n_outliers=4, seed=0)
        block_max = _per_block_max(weight)
        global_amax = block_max.max()
        # MSE-style: a mix of "shrunk" (multiplier 0.5) and "expanded"
        # (multiplier 2.5) per-block amax around the actual block maxima.
        mult = torch.where(
            torch.arange(block_max.numel()) % 2 == 0,
            torch.tensor(0.5),
            torch.tensor(2.5),
        ).view_as(block_max)
        amax = (block_max * mult).clamp(min=1e-30)
        q = _make_static_quantizer(amax, global_amax)

        ws, ws2, deq = _export_round_trip(weight, q)

        assert torch.isfinite(ws.float()).all(), "weight_scale (FP8) must be finite"
        assert torch.isfinite(ws2).all(), "weight_scale_2 (FP32) must be finite"
        assert torch.isfinite(deq.float()).all(), "dequantized weight must be finite"

    def test_overflow_amax_saturates_no_nan(self):
        """When _amax > _global_amax (MSE multiplier > 1 on the global-max
        block), the FP8 cast must saturate to 448, not produce NaN."""
        weight = torch.randn(8, 16, dtype=torch.bfloat16) * 1e-2
        block_max = _per_block_max(weight)
        global_amax = block_max.max()
        # Force the first block to have amax 4x the global max.
        amax = block_max.clone()
        amax[0] = global_amax * 4.0
        q = _make_static_quantizer(amax, global_amax)

        ws, ws2, deq = _export_round_trip(weight, q)

        assert torch.isfinite(ws.float()).all(), "FP8 weight_scale must saturate, not NaN"
        # FP8 e4m3fn max is 448. The byte for the overflowing block should be 448.
        assert ws[0].float().max().item() == pytest.approx(448.0)
        assert torch.isfinite(deq.float()).all()

    def test_underflow_amax_no_inf_in_dequant(self):
        """When per_block_amax / global_amax is below FP8 representable range,
        the static export must not emit Inf or NaN in the *dequantized* tensor.
        Whether the FP8 byte is 0 (natural underflow) or a clamped subnormal,
        the dequant of every weight in the affected blocks must be finite."""
        weight = torch.randn(8, 16, dtype=torch.bfloat16) * 1e-7
        block_max = _per_block_max(weight)
        global_amax = block_max.max() * 1e6  # _global_amax much larger than blocks
        amax = block_max.clamp(min=1e-30)
        q = _make_static_quantizer(amax, torch.tensor(global_amax))

        ws, ws2, deq = _export_round_trip(weight, q)

        assert torch.isfinite(ws.float()).all()
        assert torch.isfinite(ws2).all()
        assert torch.isfinite(deq.float()).all(), (
            "dequantized weight has NaN/Inf in underflow regime — this is the "
            "failure pattern that breaks vLLM serving"
        )


class TestNVFP4StaticExportRoundTripBound:
    """Round-trip dequant magnitude must stay bounded by the encoded amax.

    For every block, ``|dequant| <= 6 * weight_scale_FP8 * weight_scale_2``,
    and that product must be bounded above by ``max(_amax_block, 448 * scale_2)``
    (FP8 saturation). If any block's dequant exceeds that bound, an
    out-of-distribution outlier was synthesized by the export path itself.
    """

    def test_dequant_magnitude_within_amax(self):
        weight = _layer1_routed_expert_like(32, 128, n_outliers=4, seed=1)
        block_max = _per_block_max(weight)
        global_amax = block_max.max()
        # Use _amax = block_max (no shrinking, no expansion).
        amax = block_max.clamp(min=1e-30)
        q = _make_static_quantizer(amax, global_amax)

        ws, ws2, deq = _export_round_trip(weight, q)

        # The maximum representable magnitude per block is 6 * dequant_scale.
        # Allow a small relative tolerance for the bf16 quantization that
        # NVFP4QTensor.dequantize applies to its output (~0.4% per element).
        dequant_scale_per_block = ws.float() * ws2.float()
        expected_block_bound = 6.0 * dequant_scale_per_block  # shape (out, num_blocks)
        deq_block_max = deq.float().view(*deq.shape[:-1], -1, BLOCK_SIZE).abs().amax(dim=-1)
        violation = (deq_block_max - expected_block_bound).clamp(min=0)
        # Reject any per-block dequant magnitude that exceeds the FP4 saturation
        # bound by more than 1% (well above bf16 round-up noise) — that would
        # indicate the export synthesized out-of-distribution outliers.
        relative = violation / expected_block_bound.clamp(min=1e-30)
        max_relative = relative.max().item()
        assert max_relative <= 1e-2, (
            f"dequant block max exceeds the FP4 saturation bound by "
            f"{max_relative:.2%}. Worst block index: "
            f"{tuple(int(i) for i in (relative == relative.max()).nonzero()[0].tolist())}"
        )


class TestNVFP4StaticVsDynamicEquivalence:
    """When _amax = per_block_amax (no MSE shrink/expand), the static and
    dynamic export paths must produce bit-identical FP8 weight_scale bytes.
    Both paths apply the same lower clamp at the fp8 subnormal min (2**-9)
    so tiny-amax blocks land on 0x01 instead of underflowing to 0x00."""

    def test_static_matches_dynamic_when_amax_is_block_max(self):
        weight = _layer1_routed_expert_like(16, 64, n_outliers=2, seed=2)
        block_max = _per_block_max(weight)
        global_amax = block_max.max()

        # Static path
        q = _make_static_quantizer(block_max.clamp(min=1e-30), global_amax)
        static_ws_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(q)
        static_ws, _ = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
            q, weight, static_ws_2
        )

        # Dynamic path
        dynamic_ws, dynamic_ws_2 = NVFP4QTensor.get_weights_scaling_factor(
            weight, BLOCK_SIZE, weights_scaling_factor_2=static_ws_2.clone()
        )

        # Both paths should yield identical FP8 byte patterns when amax matches.
        assert torch.equal(static_ws.view(torch.uint8), dynamic_ws.view(torch.uint8)), (
            "static and dynamic export paths produced different FP8 "
            "weight_scale bytes for the same per-block amax — this means "
            "the static path's scale computation diverges from the dynamic path"
        )
        assert torch.allclose(static_ws_2, dynamic_ws_2)


class TestNVFP4StaticManualRoundTrip:
    """Cross-check the export path against a manual per-block computation.

    For each block: ``dequant_scale = FP8(amax * 448 / global_amax) * (global_amax / (6 * 448))``,
    and each FP4 value lies in {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. The
    dequantized weight for any element should be the FP4 value at the rounded
    ordinal, multiplied by the block's dequant_scale.
    """

    def test_single_block_matches_manual(self):
        # One block of 16 elements, mid-magnitude (FP8-normal regime).
        torch.manual_seed(3)
        weight = (torch.rand(1, BLOCK_SIZE, dtype=torch.bfloat16) - 0.5) * 0.05
        block_max = _per_block_max(weight)
        global_amax = block_max.max()
        amax = block_max.clamp(min=1e-30)
        q = _make_static_quantizer(amax, global_amax)

        ws, ws2, deq = _export_round_trip(weight, q)

        # Manual per-block dequant scale: FP8-quantized ratio * scale_2.
        dequant_scale = ws.float()[0, 0].item() * ws2.float().item()

        # Every dequant element must be FP4 value * dequant_scale, allowing for
        # the bf16 round-trip applied by NVFP4QTensor.dequantize on its output
        # (~0.4% relative). Use a tolerance that's loose enough for bf16 and
        # tight enough to catch a real off-grid value.
        deq_vals = deq.float().reshape(-1)
        grid = torch.tensor(
            [v.item() * dequant_scale for v in FP4_VALUES.float()],
            dtype=torch.float32,
        )
        for v in deq_vals.tolist():
            distance = (grid - v).abs().min().item()
            tolerance = max(abs(v) * 1e-2, 1e-12)
            assert distance <= tolerance, (
                f"dequant value {v} is not on the FP4 grid (min distance {distance:g}, "
                f"tolerance {tolerance:g}); grid = {sorted(grid.tolist())}"
            )


class TestNVFP4StaticCornerCases:
    """Edge cases that have historically caused trouble in MSE static export."""

    def test_zero_amax_block_does_not_explode(self):
        """If MSE selects amax=0 for a block (e.g., dead expert), the export
        must not emit NaN/Inf or amplify dequant magnitude."""
        weight = torch.zeros(2, BLOCK_SIZE, dtype=torch.bfloat16)
        weight[1, :] = 0.05  # one block with real values
        block_max = _per_block_max(weight)
        global_amax = block_max.max()
        amax = block_max.clone()
        amax[0] = 0.0  # explicit zero amax for the dead block
        q = _make_static_quantizer(amax, global_amax)

        ws, ws2, deq = _export_round_trip(weight, q)

        assert torch.isfinite(ws.float()).all()
        assert torch.isfinite(deq.float()).all()
        # The dead block must dequantize to all zeros (no leakage from the
        # special amax==0 substitution).
        assert torch.equal(deq[0].float(), torch.zeros_like(deq[0].float())), (
            "amax==0 block leaked nonzero dequant values"
        )

    def test_ultra_v3_layer1_distribution_byte_distribution_sane(self):
        """Sanity: in the Ultra-V3 layer-1-like regime, the export's FP8 byte
        distribution does not contain raw NaN bytes (0x7F or 0xFF)."""
        weight = _layer1_routed_expert_like(128, 512, n_outliers=8, seed=4)
        block_max = _per_block_max(weight)
        global_amax = block_max.max()
        amax = block_max.clamp(min=1e-30)
        q = _make_static_quantizer(amax, global_amax)

        ws, _, _ = _export_round_trip(weight, q)
        ws_bytes = ws.view(torch.uint8).reshape(-1)

        # FP8 e4m3fn NaN bytes are 0x7F (127) and 0xFF (255).
        nan_count = int(((ws_bytes == 127) | (ws_bytes == 255)).sum().item())
        assert nan_count == 0, f"static export emitted {nan_count} NaN FP8 weight_scale bytes"


def _make_static_quantizer_46(
    per_block_amax: torch.Tensor, global_amax: torch.Tensor
) -> NVFP4StaticQuantizer:
    """Static NVFP4 quantizer with 4/6 adaptive block scaling enabled."""
    cfg = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={
            -1: BLOCK_SIZE,
            "type": "static",
            "scale_bits": (4, 3),
            "four_over_six": True,
        },
    )
    q = NVFP4StaticQuantizer(quant_attribute_cfg=cfg)
    q.amax = per_block_amax.clone()
    q.global_amax = global_amax.clone()
    return q


class TestNVFP4StaticFourOverSixExport:
    """4/6 static export normalizes block scales by 256 (not 448) and stays finite.

    weight_scale_2 = global_amax / (6 * 256) instead of / (6 * 448). The per-block M=6/M=4
    choice is made by MSE calibration (baked into _amax); export reads _amax as-is.
    """

    def test_scale_2_normalizes_by_256(self):
        weight = _layer1_routed_expert_like(32, 128, n_outliers=4, seed=5)
        block_max = _per_block_max(weight)
        global_amax = block_max.max()
        amax = block_max.clamp(min=1e-30)

        ws2_46 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(
            _make_static_quantizer_46(amax, global_amax)
        )
        ws2_default = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(
            _make_static_quantizer(amax, global_amax)
        )

        # weight_scale_2 = global_amax / (6 * m_fp8); only m_fp8 differs (256 vs 448).
        assert torch.allclose(ws2_46, global_amax.float() / (6.0 * 256.0), rtol=1e-6)
        assert torch.allclose(ws2_46 / ws2_default, torch.tensor(448.0 / 256.0), rtol=1e-5)

    def test_export_round_trip_finite(self):
        weight = _layer1_routed_expert_like(64, 256, n_outliers=4, seed=6)
        block_max = _per_block_max(weight)
        global_amax = block_max.max()
        amax = block_max.clamp(min=1e-30)
        q = _make_static_quantizer_46(amax, global_amax)

        ws, ws2, deq = _export_round_trip(weight, q)

        assert torch.isfinite(ws.float()).all(), "weight_scale (FP8) must be finite"
        assert torch.isfinite(ws2).all(), "weight_scale_2 (FP32) must be finite"
        assert torch.isfinite(deq.float()).all(), "dequantized weight must be finite"

    def test_export_reads_amax_no_reselection(self):
        """Export reads _amax as-is (no re-selection): per-block scale == _amax / 6.

        MSE calibration bakes the M=4 choice into _amax (selected blocks × 6/4); export
        must pass it straight through.
        """
        weight = _layer1_routed_expert_like(32, 128, n_outliers=6, seed=7)
        block_max = _per_block_max(weight)
        global_amax = block_max.max()
        amax = block_max.clamp(min=1e-30)

        # Simulate MSE picking M=4 on every other block (amax scaled by 6/4 = 1.5).
        baked = amax.clone()
        baked[:, ::2] *= 1.5
        q = _make_static_quantizer_46(baked, global_amax)

        ws2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(q)
        selected, _ = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
            q, weight, ws2, keep_high_precision=True
        )
        assert torch.allclose(
            selected, (baked.float() / FP4_E2M1_MAX).view_as(selected), rtol=1e-5
        ), "Export did not reproduce the baked per-block amax."

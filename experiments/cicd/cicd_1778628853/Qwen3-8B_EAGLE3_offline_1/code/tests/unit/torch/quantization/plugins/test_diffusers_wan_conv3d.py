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

"""Tests for the quantized WanCausalConv3d wrapper in the diffusers plugin.

WanCausalConv3d applies asymmetric causal padding before the underlying Conv3D,
so the quantized subclass has to replicate that padding logic around the
quantized forward. These tests run on CPU with the NVFP4 dispatch disabled (by
only exercising the default quantized path); the NVFP4 implicit-GEMM path is
covered by the GPU tests under ``tests/gpu/torch/quantization/kernels``.
"""

import pytest
import torch

pytest.importorskip("onnx")

from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

# Triggers registration of _QuantDiffusersWanCausalConv3d.
import modelopt.torch.quantization.plugins.diffusion.diffusers  # noqa: F401
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.diffusion.diffusers import _QuantDiffusersWanCausalConv3d


def _make_quantized(in_ch: int = 4, out_ch: int = 6, padding=(1, 1, 1)) -> torch.nn.Module:
    m = WanCausalConv3d(in_ch, out_ch, kernel_size=3, padding=padding)
    m.eval()
    # Convert via the registry so the generated class picks up WanCausalConv3d
    # in its MRO (needed for ``nn.Conv3d._conv_forward`` to be reachable).
    return QuantModuleRegistry.convert(m)


def _strip_quant_state(state_dict: dict) -> dict:
    return {
        k: v
        for k, v in state_dict.items()
        if not any(
            k.startswith(p) for p in ("input_quantizer", "weight_quantizer", "output_quantizer")
        )
    }


class TestQuantWanCausalConv3dRegistration:
    def test_registered(self):
        assert WanCausalConv3d in QuantModuleRegistry
        # Registry-returned class is a generated subclass of our dm class.
        assert issubclass(QuantModuleRegistry[WanCausalConv3d], _QuantDiffusersWanCausalConv3d)

    def test_convert_preserves_type_identity(self):
        m = _make_quantized()
        mro = [c.__name__ for c in type(m).__mro__]
        assert "WanCausalConv3d" in mro
        assert "_QuantDiffusersWanCausalConv3d" in mro

    def test_no_implicit_gemm_without_nvfp4(self):
        # Default quantizer config is INT8 — must NOT route through implicit GEMM.
        m = _make_quantized()
        assert not m._should_use_implicit_gemm()


class TestQuantWanCausalConv3dForward:
    """Exercise the default quantized path (NVFP4 kernel is GPU-only).

    We disable the quantizers and assert output matches an unquantized
    ``WanCausalConv3d`` with the same weights — this verifies the causal-padding
    logic in the overridden ``forward`` is preserved after conversion.
    """

    @pytest.mark.parametrize("padding", [(1, 1, 1), (2, 0, 0), (0, 1, 1)])
    def test_matches_unquantized_no_cache(self, padding):
        torch.manual_seed(0)
        m = _make_quantized(in_ch=4, out_ch=6, padding=padding)
        m.input_quantizer.disable()
        m.weight_quantizer.disable()

        m_ref = WanCausalConv3d(4, 6, kernel_size=3, padding=padding)
        m_ref.eval()
        m_ref.load_state_dict(_strip_quant_state(m.state_dict()), strict=False)

        x = torch.randn(1, 4, 5, 6, 6)
        out = m(x)
        out_ref = m_ref(x)
        assert torch.allclose(out, out_ref, atol=1e-5), (
            f"Max diff: {(out - out_ref).abs().max().item()}"
        )

    def test_matches_unquantized_with_cache_x(self):
        """cache_x is the temporal-cache branch used during causal decoding."""
        torch.manual_seed(0)
        m = _make_quantized(padding=(1, 1, 1))  # _padding[4] == 2 > 0 → cache path active
        m.input_quantizer.disable()
        m.weight_quantizer.disable()

        m_ref = WanCausalConv3d(4, 6, kernel_size=3, padding=(1, 1, 1))
        m_ref.eval()
        m_ref.load_state_dict(_strip_quant_state(m.state_dict()), strict=False)

        x = torch.randn(1, 4, 5, 6, 6)
        cache_x = torch.randn(1, 4, 1, 6, 6)
        out = m(x, cache_x=cache_x)
        out_ref = m_ref(x, cache_x=cache_x)
        assert torch.allclose(out, out_ref, atol=1e-5)

    def test_output_quantizer_applied(self):
        """Enabling the output quantizer must change the forward output."""
        torch.manual_seed(0)
        m = _make_quantized()
        m.input_quantizer.disable()
        m.weight_quantizer.disable()
        # Output quantizer is disabled by default; enable it and check it takes effect.
        x = torch.randn(1, 4, 3, 5, 5)
        out_disabled = m(x)
        m.output_quantizer.enable()
        out_enabled = m(x)
        # INT8 default config clamps and rounds; at least some elements differ.
        assert not torch.allclose(out_disabled, out_enabled)

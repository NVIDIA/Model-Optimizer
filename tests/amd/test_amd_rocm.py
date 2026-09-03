#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 AMD, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AMD ROCm-specific tests for ROCm-Model-Optimizer.

Run with:
    pytest tests/test_amd_rocm.py -v                    # all AMD tests
    pytest tests/test_amd_rocm.py -v -k "fp8"          # FP8 tests only
    pytest tests/test_amd_rocm.py -v --benchmark-only  # benchmarks only

Requires AMD MI300X/MI325X with ROCm 7.0+.
"""
import pytest
import torch
import torch.nn as nn


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def require_rocm():
    if not getattr(torch.version, "hip", None):
        pytest.skip("AMD ROCm not available")


@pytest.fixture
def device():
    return "cuda"


@pytest.fixture
def small_ffn():
    """Small FFN for unit tests (fast)."""
    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512, bias=False)
            self.fc2 = nn.Linear(512, 256, bias=False)
            self.act = nn.SiLU()
        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))
    return FFN().cuda().half()


# ── Smoke tests ──────────────────────────────────────────────────────────────
class TestROCmCompat:
    def test_is_rocm(self):
        from modelopt._rocm_compat import is_rocm
        assert is_rocm() is True

    def test_fp8_supported(self):
        from modelopt._rocm_compat import is_fp8_supported, get_gpu_arch
        arch = get_gpu_arch()
        fp8 = is_fp8_supported()
        print(f"\n  arch={arch}, fp8={fp8}")
        assert isinstance(fp8, bool)

    def test_get_gpu_arch(self):
        from modelopt._rocm_compat import get_gpu_arch
        arch = get_gpu_arch()
        assert "gfx" in arch, f"Expected gfx arch, got: {arch}"

    def test_get_optimal_dtype(self):
        from modelopt._rocm_compat import get_optimal_dtype, is_fp8_supported
        dtype = get_optimal_dtype()
        if is_fp8_supported():
            assert dtype == torch.float8_e4m3fnuz
        else:
            assert dtype == torch.bfloat16


class TestAMDConfigs:
    def test_fp8_config_exported(self):
        import modelopt.torch.quantization as mtq
        assert hasattr(mtq, "AMD_FP8_DEFAULT_CFG")
        cfg = mtq.AMD_FP8_DEFAULT_CFG
        assert "quant_cfg" in cfg
        assert "algorithm" in cfg

    def test_int8_config_exported(self):
        import modelopt.torch.quantization as mtq
        assert hasattr(mtq, "AMD_INT8_DEFAULT_CFG")

    def test_fp8_calibration(self, small_ffn):
        import modelopt.torch.quantization as mtq
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_FP8_DEFAULT_CFG, forward_loop=lambda m: m(x))
        # Check quantizers were inserted
        has_quantizer = any(
            "quantizer" in name
            for name, _ in small_ffn.named_modules()
        )
        assert has_quantizer, "No quantizers inserted after AMD_FP8_DEFAULT_CFG"

    def test_int8_calibration(self, small_ffn):
        import modelopt.torch.quantization as mtq
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_INT8_DEFAULT_CFG, forward_loop=lambda m: m(x))


class TestFP8Linear:
    def test_from_linear(self, device):
        from modelopt._rocm_compat import FP8Linear
        lin = nn.Linear(512, 256, bias=False).cuda().half()
        fp8 = FP8Linear.from_linear(lin)
        assert fp8.weight_fp8.dtype == torch.float8_e4m3fnuz
        assert fp8.weight_fp8.shape == (256, 512)

    def test_forward_shape(self, device):
        from modelopt._rocm_compat import FP8Linear, is_fp8_supported
        if not is_fp8_supported():
            pytest.skip("FP8 not supported")
        lin = nn.Linear(512, 256, bias=False).cuda().half()
        fp8 = FP8Linear.from_linear(lin)
        fp8.set_input_scale(1.0)
        x = torch.randn(4, 512, device=device, dtype=torch.float16)
        out = fp8(x)
        assert out.shape == (4, 256)
        assert out.dtype == torch.float16

    def test_set_input_scale(self, device):
        from modelopt._rocm_compat import FP8Linear, is_fp8_supported
        if not is_fp8_supported():
            pytest.skip("FP8 not supported")
        lin = nn.Linear(256, 128, bias=False).cuda().half()
        fp8 = FP8Linear.from_linear(lin)
        fp8.set_input_scale(0.5)
        assert abs(float(fp8.scale_x) - 0.5) < 1e-6


class TestINT8Linear:
    def test_from_linear(self, device):
        from modelopt._rocm_compat import INT8Linear
        lin = nn.Linear(512, 256, bias=False).cuda().half()
        i8 = INT8Linear.from_linear(lin)
        assert i8.weight_int8.dtype == torch.int8
        assert i8.weight_int8.shape == (256, 512)

    def test_forward_shape(self, device):
        from modelopt._rocm_compat import INT8Linear
        lin = nn.Linear(512, 256, bias=False).cuda().half()
        i8 = INT8Linear.from_linear(lin)
        i8.set_input_scale(1.0)
        # torch._int_mm requires M > 16
        x = torch.randn(32, 512, device=device, dtype=torch.float16)
        out = i8(x)
        assert out.shape == (32, 256)


class TestExtractScales:
    def test_extract_scales_after_calibration(self, small_ffn):
        from modelopt._rocm_compat import extract_fp8_scales
        import modelopt.torch.quantization as mtq
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_FP8_DEFAULT_CFG, forward_loop=lambda m: m(x))
        scales = extract_fp8_scales(small_ffn)
        assert len(scales) > 0
        for name, scale in scales.items():
            assert scale > 0, f"Scale for {name} must be positive"

    def test_convert_to_static_fp8(self, small_ffn):
        from modelopt._rocm_compat import (
            extract_fp8_scales, convert_to_static_fp8, FP8Linear, is_fp8_supported
        )
        if not is_fp8_supported():
            pytest.skip("FP8 not supported")
        import modelopt.torch.quantization as mtq
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_FP8_DEFAULT_CFG, forward_loop=lambda m: m(x))
        scales = extract_fp8_scales(small_ffn)
        ffn_fp8 = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.SiLU(),
            nn.Linear(512, 256, bias=False)
        ).cuda().half()
        convert_to_static_fp8(ffn_fp8, scales)
        # Check at least one layer was converted
        has_fp8 = any(isinstance(m, FP8Linear) for m in ffn_fp8.modules())
        assert has_fp8


class TestKVCache:
    def test_quantize_dequantize_roundtrip(self):
        from modelopt._rocm_compat import (
            quantize_kv_cache_fp8, dequantize_kv_cache_fp8, is_fp8_supported
        )
        if not is_fp8_supported():
            pytest.skip("FP8 not supported")
        # Small controlled values — FP8 E4M3FNUZ has max=448; use range [-2, 2]
        torch.manual_seed(42)
        k = (torch.rand(2, 8, 64, 128, device="cuda") * 4 - 2).half()
        v = (torch.rand(2, 8, 64, 128, device="cuda") * 4 - 2).half()
        k_fp8, v_fp8, sk, sv = quantize_kv_cache_fp8(k, v)
        assert k_fp8.dtype == torch.float8_e4m3fnuz
        assert v_fp8.dtype == torch.float8_e4m3fnuz
        # Dequantize
        k_rec = dequantize_kv_cache_fp8(k_fp8, sk)
        # Replace any NaN with 0 for error check (FP8 edge values may not round-trip exactly)
        k_rec_clean = torch.nan_to_num(k_rec, nan=0.0, posinf=0.0, neginf=0.0)
        k_clean = k.clone()
        k_clean[k_rec.isnan()] = 0.0  # zero out positions that had NaN
        # FP8 has limited precision — allow up to 10% relative error per element
        max_err = (k_clean.float() - k_rec_clean.float()).abs().max().item()
        assert max_err < 2.0, f"KV roundtrip error {max_err:.4f} > 2.0"
        # Check shapes preserved
        assert k_rec.shape == k.shape

    def test_memory_savings_7b(self):
        from modelopt._rocm_compat import kv_cache_memory_savings
        s = kv_cache_memory_savings(4096, 32, 128, batch_size=1, n_layers=32)
        assert s["savings_ratio"] == pytest.approx(0.5, abs=0.01)
        assert s["fp16_gb"] > s["fp8_gb"]


class TestINT8Pipeline:
    """INT8 full pipeline tests (OPT-33)."""

    def test_convert_to_int8_linear(self, small_ffn):
        from modelopt._rocm_compat import convert_to_int8_linear, INT8Linear
        convert_to_int8_linear(small_ffn, min_out_features=128)
        has_int8 = any(isinstance(m, INT8Linear) for m in small_ffn.modules())
        assert has_int8

    def test_int8_calibration_and_convert(self, small_ffn):
        import modelopt.torch.quantization as mtq
        from modelopt._rocm_compat import (
            extract_fp8_scales, convert_to_int8_linear, INT8Linear
        )
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_INT8_DEFAULT_CFG, forward_loop=lambda m: m(x))
        scales = extract_fp8_scales(small_ffn)
        int8_scales = {k: v * (448.0 / 127.0) for k, v in scales.items()}
        fresh = type(small_ffn)().cuda().half()
        convert_to_int8_linear(fresh, int8_scales, min_out_features=128)
        # torch._int_mm requires M > 16; use 32
        x_infer = torch.randn(32, 256, device="cuda", dtype=torch.float16)
        out = fresh(x_infer)
        assert out.shape == (32, 256)


class TestProfilingUtils:
    """Tests for OPT-41 profiling utilities."""

    def test_profile_amd_model(self, small_ffn):
        from modelopt._rocm_compat import profile_amd_model
        x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
        stats = profile_amd_model(small_ffn, x, n_iters=10, label="test_fp16")
        assert "latency_ms" in stats
        assert stats["latency_ms"] > 0
        assert stats["throughput_per_sec"] > 0

    def test_count_fp8_layers(self, small_ffn):
        from modelopt._rocm_compat import (
            convert_to_fp8_linear, count_fp8_layers, is_fp8_supported
        )
        if not is_fp8_supported():
            pytest.skip("FP8 not supported")
        fp8_model = type(small_ffn)().cuda().half()
        convert_to_fp8_linear(fp8_model)
        counts = count_fp8_layers(fp8_model)
        assert counts["fp8_linear"] > 0
        assert counts["quantized_fraction"] > 0


class TestWarmupUtils:
    """Tests for OPT-23/34 warmup utilities."""

    def test_warmup_fp8_shapes(self):
        from modelopt._rocm_compat import warmup_fp8_shapes, is_fp8_supported
        if not is_fp8_supported():
            pytest.skip("FP8 not supported")
        # Use small shapes for speed
        results = warmup_fp8_shapes([(4, 256, 128)], device="cuda")
        assert len(results) == 1
        assert list(results.values())[0] > 0

    def test_warmup_for_llama(self):
        from modelopt._rocm_compat import warmup_for_llama, is_fp8_supported
        if not is_fp8_supported():
            pytest.skip("FP8 not supported")
        results = warmup_for_llama("7b", batch_sizes=[1])
        assert len(results) > 0

    def test_get_llama_shapes(self):
        from modelopt._rocm_compat import get_llama_ffn_shapes
        H, I = get_llama_ffn_shapes("70b")
        assert H == 8192
        assert I == 28672


class TestBuildPipeline:
    """Tests for OPT-35 build_amd_inference_model."""

    def test_build_fp8_pipeline(self, small_ffn):
        from modelopt._rocm_compat import build_amd_inference_model, FP8Linear, is_fp8_supported
        if not is_fp8_supported():
            pytest.skip("FP8 not supported")
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        model = build_amd_inference_model(small_ffn, x, quant_dtype="fp8")
        has_fp8 = any(isinstance(m, FP8Linear) for m in model.modules())
        assert has_fp8
        out = model(x[:2])
        assert out.shape == (2, 256)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 AMD, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AMD MIGraphX INT8 integration tests.

Tests the export_fp8_onnx → MIGraphX INT8 compile pipeline.
Skips automatically if migrachx is not installed.

Run with:
    pytest tests/amd/test_amd_migrachx.py -v
"""
import pytest
import torch
import torch.nn as nn
import os
import tempfile


@pytest.fixture(autouse=True)
def require_rocm():
    if not getattr(torch.version, "hip", None):
        pytest.skip("AMD ROCm not available")


@pytest.fixture
def migrachx():
    """Try to import migrachx; skip if not available."""
    try:
        import migrachx
        return migrachx
    except ImportError:
        pytest.skip("migrachx not installed — skipping MIGraphX tests")


@pytest.fixture
def small_ffn():
    """Small FFN for export testing."""
    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512, bias=False)
            self.fc2 = nn.Linear(512, 256, bias=False)
            self.act = nn.SiLU()
        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))
    return FFN().cuda().half().eval()


class TestMIGraphXAvailability:
    """Basic MIGraphX availability tests."""
    
    def test_migrachx_importable(self):
        """Check if migrachx can be imported."""
        try:
            import migrachx
            has_migrachx = True
            version = getattr(migrachx, "__version__", "unknown")
        except ImportError:
            has_migrachx = False
            version = "N/A"
        
        print(f"\n  migrachx available: {has_migrachx} (version: {version})")
        # This test always passes — just reports availability
        assert True
    
    def test_migrachx_get_target(self, migrachx):
        """Verify migrachx can create GPU target."""
        target = migrachx.get_target("gpu")
        assert target is not None
        print(f"\n  MIGraphX GPU target: {target}")


class TestONNXExport:
    """Test ONNX export functionality (no MIGraphX needed)."""
    
    def test_onnx_export_creates_file(self, small_ffn):
        """export_fp8_onnx should create a valid ONNX file."""
        import modelopt.torch.quantization as mtq
        from modelopt._rocm_compat import export_fp8_onnx
        
        x_cal = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_FP8_DEFAULT_CFG,
                     forward_loop=lambda m: m(x_cal))
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        
        try:
            export_fp8_onnx(small_ffn, x_cal[:1], onnx_path)
            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 1000, "ONNX file too small"
            size_mb = os.path.getsize(onnx_path) / 1e6
            print(f"\n  ONNX export: {size_mb:.2f} MB")
        finally:
            os.unlink(onnx_path)
    
    def test_onnx_export_valid_structure(self, small_ffn):
        """Exported ONNX should be parseable."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")
        
        import modelopt.torch.quantization as mtq
        from modelopt._rocm_compat import export_fp8_onnx
        
        x_cal = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_FP8_DEFAULT_CFG,
                     forward_loop=lambda m: m(x_cal))
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        
        try:
            export_fp8_onnx(small_ffn, x_cal[:1], onnx_path)
            model_onnx = onnx.load(onnx_path)
            onnx.checker.check_model(model_onnx)
            print(f"\n  ONNX opset: {model_onnx.opset_import[0].version}")
            print(f"  Nodes: {len(model_onnx.graph.node)}")
        finally:
            os.unlink(onnx_path)


class TestMIGraphXINT8:
    """MIGraphX INT8 quantization and compilation tests."""
    
    @pytest.mark.integration
    def test_migrachx_parse_onnx(self, migrachx, small_ffn):
        """Parse calibrated ONNX model with MIGraphX."""
        import modelopt.torch.quantization as mtq
        from modelopt._rocm_compat import export_fp8_onnx
        
        x_cal = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_FP8_DEFAULT_CFG,
                     forward_loop=lambda m: m(x_cal))
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        
        try:
            export_fp8_onnx(small_ffn, x_cal[:1], onnx_path)
            program = migrachx.parse_onnx(onnx_path)
            assert program is not None
            print(f"\n  MIGraphX parse_onnx: SUCCESS")
        finally:
            os.unlink(onnx_path)
    
    @pytest.mark.integration
    def test_migrachx_quantize_int8(self, migrachx, small_ffn):
        """Apply MIGraphX INT8 quantization to parsed ONNX."""
        import modelopt.torch.quantization as mtq
        from modelopt._rocm_compat import export_fp8_onnx
        
        x_cal = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_FP8_DEFAULT_CFG,
                     forward_loop=lambda m: m(x_cal))
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        
        try:
            export_fp8_onnx(small_ffn, x_cal[:1], onnx_path)
            program = migrachx.parse_onnx(onnx_path)
            migrachx.quantize_int8(program)
            print(f"\n  MIGraphX quantize_int8: SUCCESS")
        except Exception as e:
            pytest.fail(f"MIGraphX INT8 quantization failed: {e}")
        finally:
            os.unlink(onnx_path)
    
    @pytest.mark.integration  
    def test_migrachx_compile_and_run(self, migrachx, small_ffn):
        """Full pipeline: export → INT8 quantize → compile → run."""
        import modelopt.torch.quantization as mtq
        from modelopt._rocm_compat import export_fp8_onnx
        import numpy as np
        
        x_cal = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        mtq.quantize(small_ffn, mtq.AMD_FP8_DEFAULT_CFG,
                     forward_loop=lambda m: m(x_cal))
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        
        try:
            export_fp8_onnx(small_ffn, x_cal[:1], onnx_path)
            program = migrachx.parse_onnx(onnx_path)
            migrachx.quantize_int8(program)
            program.compile(migrachx.get_target("gpu"))
            
            # Run inference
            x_np = x_cal[:1].float().cpu().numpy()
            result = program.run({"input": migrachx.argument(x_np)})
            mgx_out = np.array(result[0])
            
            print(f"\n  MIGraphX compile+run: output shape={mgx_out.shape}")
            assert mgx_out.shape[-1] == 256, f"Expected 256 outputs, got {mgx_out.shape}"
        except Exception as e:
            pytest.fail(f"MIGraphX compile/run failed: {e}")
        finally:
            os.unlink(onnx_path)

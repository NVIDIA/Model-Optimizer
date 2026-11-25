# SPDX-License-Identifier: Apache-2.0
"""AMD ROCm-optimized calibration utilities for modelopt.

Provides faster calibration on MI300X/MI325X by using real FP8 hardware
dispatch instead of FP16 fake-quant simulation.
"""
from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn


def amd_calibrate(
    model: nn.Module,
    dataloader: list[torch.Tensor],
    batch_size: int = 16,
    use_fp8_forward: bool = True,
) -> nn.Module:
    """Calibrate model for AMD ROCm quantization using hardware-accurate statistics.
    
    On AMD MI300X/MI325X, uses real float8_e4m3fnuz hardware ops during
    calibration to collect accurate amax statistics that reflect actual
    FP8 quantization errors (vs FP16 fake-quant which can be less accurate).
    
    Args:
        model: Model wrapped with mtq.quantize() already applied
        dataloader: List of input tensors for calibration
        batch_size: Number of samples per calibration forward pass
        use_fp8_forward: Whether to use FP8 hardware ops (True for MI300X+)
    
    Returns:
        Calibrated model
    
    Example:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.amd_calibration import amd_calibrate
        
        mtq.quantize(model, mtq.AMD_FP8_DEFAULT_CFG,
            forward_loop=lambda m: amd_calibrate(m, cal_data))
    """
    from modelopt._rocm_compat import is_rocm, is_fp8_supported
    
    use_fp8 = use_fp8_forward and is_rocm() and is_fp8_supported()
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                if use_fp8:
                    # Cast to FP8 and back for hardware-accurate statistics
                    batch_fp8 = batch.contiguous().to(torch.float8_e4m3fnuz).to(batch.dtype)
                    model(batch_fp8)
                else:
                    model(batch)
            elif isinstance(batch, (list, tuple)):
                model(*batch)
    
    return model


def get_amd_forward_loop(
    calibration_data: torch.Tensor,
    num_steps: int = 128,
    use_fp8: bool = True,
) -> Callable:
    """Create an AMD-optimized forward loop for mtq.quantize().
    
    Args:
        calibration_data: Calibration input tensor (any batch size)
        num_steps: Number of forward passes for calibration
        use_fp8: Use FP8 hardware during calibration (recommended for MI300X)
    
    Returns:
        forward_loop callable for mtq.quantize()
    
    Example:
        mtq.quantize(model, mtq.AMD_FP8_DEFAULT_CFG,
            forward_loop=get_amd_forward_loop(x_cal, num_steps=64))
    """
    from modelopt._rocm_compat import is_rocm, is_fp8_supported
    
    _use_fp8 = use_fp8 and is_rocm() and is_fp8_supported()
    _bs = min(calibration_data.shape[0], 8)  # Calibrate in small batches
    
    def forward_loop(model: nn.Module) -> None:
        model.eval()
        with torch.no_grad():
            for i in range(0, min(num_steps * _bs, calibration_data.shape[0]), _bs):
                batch = calibration_data[i:i+_bs]
                if _use_fp8:
                    batch_fp8 = batch.contiguous().to(torch.float8_e4m3fnuz).to(batch.dtype)
                    model(batch_fp8)
                else:
                    model(batch)
    
    return forward_loop

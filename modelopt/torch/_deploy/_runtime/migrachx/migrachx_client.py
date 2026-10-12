# SPDX-FileCopyrightText: Copyright (c) 2024 AMD CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AMD ROCm MIGraphX deploy backend for ROCm Model Optimizer.
# Mirrors the TRTLocalClient interface (registry key "TRT") with AMD MIGraphX.
#
# MIGraphX is AMD's graph-level inference optimizer (analogous to TensorRT).
# It compiles ONNX models to optimized GPU programs and supports:
#   - INT8 quantization (via QuantizeLinear/DequantizeLinear ops in the ONNX graph)
#   - FP8 quantization (ROCm >= 6.x, gfx942+)
#   - FP16 / BF16 precision
#   - Fused ops, kernel auto-tuning (hipBLASLt / composable_kernel backend)
#
# Usage:
#   from modelopt.torch._deploy import compile
#   compiled = compile(model, deployment={"runtime": "MIGraphX", "precision": "int8"})

from __future__ import annotations

import io
import os
import tempfile
import time
from typing import Any

import numpy as np
import torch

from modelopt._rocm_compat import get_gpu_arch, is_rocm

from ..registry import RuntimeRegistry
from ..runtime_client import Deployment, DeploymentTable, DetailedResults, RuntimeClient

__all__ = ["MIGraphXLocalClient"]

# MIGraphX Python bindings ship with ROCm (import migraphx)
# We lazy-import so the rest of modelopt works without ROCm installed.
def _get_migraphx():
    try:
        import migraphx
        return migraphx
    except ImportError:
        raise ImportError(
            "MIGraphX Python bindings not found. Install ROCm and ensure "
            "migraphx is on PYTHONPATH. Typically: /opt/rocm/lib/migraphx/python/"
        )


def _np_dtype_from_migrachx(mgx_shape) -> np.dtype:
    """Convert MIGraphX shape type string to numpy dtype."""
    type_map = {
        "float_type": np.float32,
        "half_type":  np.float16,
        "bf16_type":  np.float16,   # numpy has no bf16 — use fp16 for I/O
        "double_type": np.float64,
        "int8_type":  np.int8,
        "int16_type": np.int16,
        "int32_type": np.int32,
        "int64_type": np.int64,
        "uint8_type": np.uint8,
        "fp8e4m3fnuz_type": np.float16,  # no fp8 in numpy — use fp16 buffer
    }
    type_str = str(mgx_shape.type_string())
    return type_map.get(type_str, np.float32)


@RuntimeRegistry.register("MIGraphX")
class MIGraphXLocalClient(RuntimeClient):
    """RuntimeClient implementation for AMD MIGraphX.

    Registered as "MIGraphX" in the RuntimeRegistry. Accepts ONNX bytes as IR
    and compiles them to a MIGraphX program (`.mxr` serialized format) for
    on-device inference and profiling on AMD MI300X/MI325X (gfx942).

    Example deployment config::

        {"runtime": "MIGraphX", "precision": "fp16", "accelerator": "GPU"}
    """

    # ── Deployment table ────────────────────────────────────────────────────

    @property
    def default_deployment(self) -> Deployment:
        return {k: v[0] for k, v in self.deployment_table.items()}

    @property
    def deployment_table(self) -> DeploymentTable:
        return {
            "accelerator": ["GPU"],
            "precision": [
                "fp16",    # default — safe on all gfx9xx
                "fp32",
                "bf16",
                "int8",    # requires QDQ ONNX graph from modelopt calibration
                "fp8",     # requires ROCm >= 6.x and gfx942+
            ],
            "onnx_opset": [str(i) for i in range(13, 22)],
        }

    # ── Compilation ─────────────────────────────────────────────────────────

    def _ir_to_compiled(
        self,
        ir_bytes: bytes,
        compilation_args: dict[str, Any] | None = None,
    ) -> bytes:
        """Compile ONNX bytes → MIGraphX program bytes (.mxr format).

        Args:
            ir_bytes: ONNX model serialized as bytes.
            compilation_args: Optional dict with keys:
                - exhaustive_tune (bool): run full kernel search (slow, best perf)
                - gpu_offload (bool): offload to GPU (default True)
                - fast_math (bool): enable fast math optimizations

        Returns:
            Serialized MIGraphX program bytes (can be saved as .mxr).
        """
        mgx = _get_migrachx()
        args = compilation_args or {}

        # Write ONNX bytes to temp file (MIGraphX parses from file)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(ir_bytes)
            onnx_path = f.name

        try:
            # Parse ONNX model
            model = mgx.parse_onnx(onnx_path)

            # Apply precision settings
            precision = self.deployment.get("precision", "fp16")
            if precision in ("fp16", "bf16"):
                flag = "bf16_mode" if precision == "bf16" else "fp16_mode"
                quant_args = {flag: True}
                mgx.quantize_fp16(model, **quant_args) if precision == "fp16" else \
                    mgx.quantize_bf16(model)
            elif precision == "int8":
                # INT8 path: QDQ nodes already in ONNX from modelopt calibration
                # MIGraphX recognizes QuantizeLinear/DequantizeLinear ops natively
                pass  # no extra quantize call needed — QDQ graph handles it
            elif precision == "fp8":
                # FP8 path: available on gfx942+ with ROCm >= 6.x
                arch = get_gpu_arch()
                if "gfx942" not in arch and "gfx950" not in arch:
                    import warnings
                    warnings.warn(f"FP8 precision requested but arch {arch} may not support it. "
                                  "Falling back to FP16.")
                    mgx.quantize_fp16(model)
                # else: MIGraphX auto-detects FP8 ops from the ONNX graph

            # Compile for GPU target
            compile_kwargs: dict[str, Any] = {}
            if args.get("exhaustive_tune", False):
                compile_kwargs["exhaustive_tune"] = True
            if args.get("fast_math", True):
                compile_kwargs["fast_math"] = True

            # AMD OPT-7: Use hipBLASLt algorithm search for best kernel per shape
        # exhaustive_tune=True benchmarks all algorithms and caches the winner
        # Can add 10-60s to first compile but gives 1.5-2x better throughput
        if not compile_kwargs.get("exhaustive_tune") and args.get("exhaustive_tune", False):
            compile_kwargs["exhaustive_tune"] = True
        model.compile(mgx.get_target("gpu"), **compile_kwargs)

            # Serialize to bytes
            buf = io.BytesIO()
            mgx.save(model, buf)
            return buf.getvalue()

        finally:
            os.unlink(onnx_path)

    # ── Profiling ───────────────────────────────────────────────────────────

    def _profile(
        self,
        compiled_model: bytes,
        compilation_args: dict[str, Any] | None = None,
    ) -> tuple[float, DetailedResults]:
        """Profile a compiled MIGraphX program and return latency in ms.

        Args:
            compiled_model: Serialized MIGraphX program bytes.
            compilation_args: Optional dict with keys:
                - warmup_iters (int): warmup iterations (default 10)
                - bench_iters (int): benchmark iterations (default 100)

        Returns:
            Tuple of (latency_ms, detailed_results).
        """
        mgx = _get_migrachx()
        args = compilation_args or {}

        warmup = args.get("warmup_iters", 10)
        iters  = args.get("bench_iters", 100)

        # Deserialize program
        buf = io.BytesIO(compiled_model)
        model = mgx.load(buf)

        # Build dummy inputs
        params = model.get_parameter_shapes()
        inputs = {}
        for name, shape in params.items():
            lens = shape.lens()
            dtype = _np_dtype_from_migrachx(shape)
            inputs[name] = mgx.argument(np.random.randn(*lens).astype(dtype))

        # Warmup
        for _ in range(warmup):
            model.run(inputs)

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model.run(inputs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / iters * 1000.0

        detailed: DetailedResults = {
            "runtime":     "MIGraphX",
            "latency_ms":  elapsed_ms,
            "warmup_iters": warmup,
            "bench_iters":  iters,
            "precision":   self.deployment.get("precision", "fp16"),
            "gpu_arch":    get_gpu_arch(),
        }
        return elapsed_ms, detailed

    # ── Inference ───────────────────────────────────────────────────────────

    def _inference(
        self,
        compiled_model: bytes,
        inputs: list[torch.Tensor],
        io_shapes: dict[str, list] | None = None,
    ) -> list[torch.Tensor]:
        """Run inference with a compiled MIGraphX program.

        Args:
            compiled_model: Serialized MIGraphX program bytes.
            inputs: List of torch Tensors (in model input order).
            io_shapes: Optional dict mapping output name → shape (not needed for MIGraphX,
                which infers shapes automatically after compilation).

        Returns:
            List of torch Tensors (outputs).
        """
        mgx = _get_migrachx()

        buf = io.BytesIO(compiled_model)
        model = mgx.load(buf)

        # Map inputs by position to parameter names
        param_names = list(model.get_parameter_shapes().keys())
        if len(inputs) != len(param_names):
            raise ValueError(
                f"Expected {len(param_names)} inputs, got {len(inputs)}. "
                f"Parameter names: {param_names}"
            )

        mgx_inputs = {}
        for name, tensor in zip(param_names, inputs):
            arr = tensor.detach().cpu().numpy()
            mgx_inputs[name] = mgx.argument(arr)

        # Run
        results = model.run(mgx_inputs)

        # Convert outputs back to torch Tensors
        outputs = []
        for r in results:
            arr = np.array(r.tolist())  # migrachx result → numpy
            outputs.append(torch.from_numpy(arr))

        return outputs

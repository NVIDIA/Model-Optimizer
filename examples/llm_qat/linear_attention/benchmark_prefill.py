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

"""Measure GDN/KDA prefill numerical-emulation overhead against the exact FLA path."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fla.ops.kda import chunk_kda

from modelopt.torch.kernels.quantization.linear_attention.fla_chunk_gated_delta_rule import (
    chunk_gated_delta_rule,
)
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    matmul_gdn,
    matmul_kda,
)
from modelopt.torch.quantization.linear_attention.matmul import LinearAttentionMatmulSites
from modelopt.torch.quantization.nn import TensorQuantizer


def main():
    """Run interleaved fixed-workload forward/backward trials and write a JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention", choices=["gdn", "kda"], default="gdn")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--solve-degrees", type=int, nargs="*", default=[])
    options = parser.parse_args()
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    shape = (options.batch, options.length, options.heads, options.dim)
    q, k = [
        F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16().requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate_shape = shape if options.attention == "kda" else shape[:-1]
    g = (-torch.rand(gate_shape, device="cuda") * 0.03).requires_grad_()
    beta = torch.rand(shape[:-1], device="cuda", requires_grad=True)
    state = (
        torch.randn(options.batch, options.heads, options.dim, options.dim, device="cuda") * 0.1
    ).requires_grad_()
    inputs = (q, k, v, g, beta)
    baseline = chunk_kda if options.attention == "kda" else chunk_gated_delta_rule
    materialized = matmul_kda if options.attention == "kda" else matmul_gdn
    variants = {
        "exact_fla": lambda: baseline(*inputs, initial_state=state, output_final_state=True)
    }
    for mode in (
        "exact_matmul",
        "fp8_operands",
        "nvfp4_operands",
        "fp16_accumulation",
        "fp8_with_state",
    ):
        sites = LinearAttentionMatmulSites()
        w = TensorQuantizer(QuantizerAttributeConfig(enable=False))
        policy = LinearAttentionConfig(backend="matmul")
        if mode in ("fp8_operands", "fp8_with_state", "nvfp4_operands"):
            attributes: dict[str, Any] = {"num_bits": (4, 3), "axis": (0, 1, 2), "type": "dynamic"}
            if mode == "nvfp4_operands":
                attributes = {
                    "num_bits": (2, 1),
                    "type": "dynamic",
                    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                }
            for quantizer in sites.modules():
                if isinstance(quantizer, TensorQuantizer):
                    quantizer.set_from_attribute_config(attributes)
                    quantizer.enable()
            w.set_from_attribute_config(attributes)
            w.enable()
        if mode == "fp8_with_state":
            w.set_from_attribute_config({"num_bits": (4, 3), "axis": (0, 1, 2), "type": "dynamic"})
            w.enable()
        if mode == "fp16_accumulation":
            policy = LinearAttentionConfig(
                backend="matmul",
                matmul={
                    name: {"accumulator_dtype": "float16", "reduction_block": 16} for name in sites
                },
            )

        def run(sites=sites, w=w, policy=policy, mode=mode):
            return materialized(
                *inputs,
                initial_state=state,
                output_final_state=True,
                sites=sites,
                w_quantizer=w,
                policy=policy,
                state_qdq=mode == "fp8_with_state",
            )

        variants[mode] = run

    for degree in options.solve_degrees:
        sites = LinearAttentionMatmulSites()
        w = TensorQuantizer(QuantizerAttributeConfig(enable=False))
        policy = LinearAttentionConfig(
            backend="matmul",
            solve={"method": "neumann", "degree": degree, "implementation": "triton"},
        )

        def run_solve(sites=sites, w=w, policy=policy):
            return materialized(
                *inputs,
                initial_state=state,
                output_final_state=True,
                sites=sites,
                w_quantizer=w,
                policy=policy,
            )

        variants[f"neumann_{degree}"] = run_solve

    def measure(fn):
        torch.cuda.synchronize()
        baseline_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, mid, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        wall_start = time.perf_counter()
        start.record()
        output, final = fn()
        mid.record()
        loss = output.float().square().mean() + final.square().mean()
        torch.autograd.grad(loss, (*inputs, state))
        end.record()
        torch.cuda.synchronize()
        return {
            "forward_ms": start.elapsed_time(mid),
            "backward_ms": mid.elapsed_time(end),
            "wall_ms": (time.perf_counter() - wall_start) * 1000,
            "peak_extra_bytes": torch.cuda.max_memory_allocated() - baseline_bytes,
        }

    for fn in variants.values():
        for _ in range(3):
            measure(fn)
    rng = random.Random(2026)
    samples = {name: [] for name in variants}
    for _ in range(options.repeats):
        order = list(variants)
        rng.shuffle(order)
        for name in order:
            samples[name].append(measure(variants[name]))
    summary = {
        name: {key: statistics.median(sample[key] for sample in values) for key in values[0]}
        for name, values in samples.items()
    }
    for name, values in samples.items():
        ratios = [
            sample["wall_ms"] / control["wall_ms"]
            for sample, control in zip(values, samples["exact_fla"])
        ]
        summary[name]["median_paired_wall_ratio"] = statistics.median(ratios)
    revision = options.revision
    if revision is None:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sources = [
        Path("modelopt/torch/quantization/linear_attention") / name
        for name in ("prefill.py", "kda.py", "matmul.py", "reference.py", "config.py", "solve.py")
    ]
    sources.append(Path("modelopt/torch/kernels/quantization/linear_attention/neumann.py"))
    result = {
        "attention": options.attention,
        "git_head": revision,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "fla_core": importlib.metadata.version("fla-core"),
        "shape": shape,
        "dtype": "bfloat16",
        "repeats": options.repeats,
        "summary": summary,
        "samples": samples,
    }
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

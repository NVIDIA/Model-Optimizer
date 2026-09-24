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

"""Compare complete prefix/suffix QAT cost under matched token/replay policies."""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    LinearAttentionMatmulSites,
    matmul_gdn,
    matmul_kda,
)
from modelopt.torch.quantization.nn import TensorQuantizer

if TYPE_CHECKING:
    from collections.abc import Callable


def main():
    """Check outputs/gradients before interleaved, warmup-excluded measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention", choices=["gdn", "kda"], default="kda")
    parser.add_argument("--length", type=int, default=257)
    parser.add_argument("--prefill", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--state-format", choices=["fp8_e4m3", "int8"], default="fp8_e4m3")
    options = parser.parse_args()
    if options.repeats < 2 or not 0 <= options.prefill < options.length:
        parser.error("Require >=2 repeats and a nonempty suffix")
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    shape = (1, options.length, options.heads, options.dim)
    q, k = [
        F.normalize(torch.randn(shape, device="cuda"), dim=-1).requires_grad_() for _ in range(2)
    ]
    v = torch.randn(shape, device="cuda", requires_grad=True)
    g = (
        -torch.rand(shape if options.attention == "kda" else shape[:-1], device="cuda") * 0.03
    ).requires_grad_()
    beta = (torch.rand(shape[:-1], device="cuda") * 0.4).requires_grad_()
    initial = (
        torch.randn(1, options.heads, options.dim, options.dim, device="cuda") * 0.1
    ).requires_grad_()
    args = q, k, v, g, beta
    sites = LinearAttentionMatmulSites()
    w = TensorQuantizer(QuantizerAttributeConfig(enable=False))
    function = matmul_kda if options.attention == "kda" else matmul_gdn
    variants: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]] = {}
    for mode in ["exact", "token", "decay", "replay"]:
        for impl in ["torch", "triton"]:
            decode: dict[str, Any] = {"implementation": impl}
            if mode == "replay":
                decode.update(mode="replay", replay={"window": 8})
            if mode == "decay":
                decode["decay_log_step"] = 1 / 256
            policy = LinearAttentionConfig(backend="matmul", decode=decode)

            def run(policy=policy, mode=mode):
                return function(
                    *args,
                    initial_state=initial,
                    output_final_state=True,
                    sites=sites,
                    w_quantizer=w,
                    policy=policy,
                    state_qdq=mode != "exact",
                    state_format=options.state_format,
                    prefill_lengths=[options.prefill],
                )

            variants[f"{mode}_{impl}"] = run
    native = chunk_kda if options.attention == "kda" else chunk_gated_delta_rule
    variants["fla_bf16"] = lambda: native(
        q.bfloat16(),
        k.bfloat16(),
        v.bfloat16(),
        g,
        beta,
        initial_state=initial,
        output_final_state=True,
    )
    probes = [torch.randn_like(v), torch.randn_like(initial)]
    correctness = {}
    for mode in ["exact", "token", "decay", "replay"]:
        results = []
        for impl in ["torch", "triton"]:
            output = variants[f"{mode}_{impl}"]()
            grads = torch.autograd.grad(
                sum((x * p).sum() for x, p in zip(output, probes)), (*args, initial)
            )
            results.append(tuple(x.detach() for x in (*output, *grads)))
        errors = [float((a - b).norm() / a.norm().clamp_min(1e-8)) for a, b in zip(*results)]
        if max(errors) > 1e-4 or not all(
            torch.isfinite(x).all() for result in results for x in result
        ):
            raise RuntimeError(f"{mode}: correctness gate failed: {errors}")
        correctness[mode] = errors

    def measure(fn):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        start = time.perf_counter()
        a.record()
        output, final = fn()
        b.record()
        torch.autograd.grad(output.square().mean() + final.square().mean(), (*args, initial))
        c.record()
        torch.cuda.synchronize()
        return {
            "forward_ms": a.elapsed_time(b),
            "backward_ms": b.elapsed_time(c),
            "wall_ms": 1000 * (time.perf_counter() - start),
            "peak_extra_bytes": torch.cuda.max_memory_allocated() - allocated,
        }

    for fn in variants.values():
        for _ in range(3):
            measure(fn)
    samples = {name: [] for name in variants}
    rng = random.Random(2026)
    for _ in range(options.repeats):
        order = list(variants)
        rng.shuffle(order)
        for name in order:
            samples[name].append(measure(variants[name]))
    summary = {
        name: {key: statistics.median(s[key] for s in values) for key in values[0]}
        for name, values in samples.items()
    }
    for mode in ["exact", "token", "decay", "replay"]:
        ratios = [
            a["wall_ms"] / b["wall_ms"]
            for a, b in zip(samples[f"{mode}_torch"], samples[f"{mode}_triton"])
        ]
        bootstrap = sorted(
            statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000)
        )
        summary[f"{mode}_triton"]["paired_speedup_vs_reference"] = {
            "median": statistics.median(ratios),
            "interval_95": [bootstrap[124], bootstrap[4874]],
        }
    sources = [
        *list(Path("modelopt/torch/quantization/linear_attention").glob("*.py")),
        Path("modelopt/torch/kernels/quantization/linear_attention/decode.py"),
        Path(__file__),
    ]
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": shape,
        "prefill": options.prefill,
        "state_format": options.state_format,
        "attention": options.attention,
        "dtype": "float32; FLA baseline casts Q/K/V to bfloat16",
        "repeats": options.repeats,
        "correctness_relative_norms": correctness,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "summary": summary,
        "samples": samples,
    }
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

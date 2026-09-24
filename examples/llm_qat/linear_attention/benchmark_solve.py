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

"""Measure solve cost and numerical error separately from complete prefill."""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch

from modelopt.torch.quantization.linear_attention import (
    LinearAttentionSolveConfig,
    triangular_inverse,
)


def main():
    """Write interleaved timing samples and exact-solve-relative errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrices", type=int, default=64)
    parser.add_argument("--degrees", nargs="+", type=int, default=[3, 7, 15, 31])
    parser.add_argument("--repeats", type=int, default=20)
    options = parser.parse_args()
    if options.matrices < 1 or options.repeats < 1:
        parser.error("matrices and repeats must be positive")
    torch.manual_seed(2026)
    torch.set_float32_matmul_precision("highest")
    lower = (torch.randn(options.matrices, 64, 64, device="cuda") * 0.04).tril(-1).requires_grad_()
    probe = torch.randn_like(lower)
    identity = torch.eye(64, device="cuda")
    policies = {"exact": LinearAttentionSolveConfig()}
    for degree in options.degrees:
        for implementation in ("torch", "triton"):
            policies[f"{implementation}_{degree}"] = LinearAttentionSolveConfig(
                method="neumann",
                degree=degree,
                implementation=implementation,
            )
    exact = triangular_inverse(lower, policies["exact"])
    exact_gradient = torch.autograd.grad((exact * probe).sum(), lower)[0]
    errors = {}
    for name, policy in policies.items():
        value = triangular_inverse(lower, policy)
        gradient = torch.autograd.grad((value * probe).sum(), lower)[0]
        errors[name] = {
            "inverse_relative_error": float((value - exact).norm() / exact.norm()),
            "gradient_relative_error": float(
                (gradient - exact_gradient).norm() / exact_gradient.norm()
            ),
            "maximum_relative_residual": float(
                (identity - (identity + lower) @ value).norm(dim=(-2, -1)).amax() / identity.norm()
            ),
        }
    del exact, exact_gradient, value, gradient

    def measure(policy):
        torch.cuda.synchronize()
        initial_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, mid, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        begin = time.perf_counter()
        start.record()
        inverse = triangular_inverse(lower, policy)
        mid.record()
        torch.autograd.grad((inverse * probe).sum(), lower)
        end.record()
        torch.cuda.synchronize()
        return {
            "forward_ms": start.elapsed_time(mid),
            "backward_ms": mid.elapsed_time(end),
            "wall_ms": 1000 * (time.perf_counter() - begin),
            "peak_extra_bytes": torch.cuda.max_memory_allocated() - initial_bytes,
        }

    for policy in policies.values():
        for _ in range(3):
            measure(policy)
    samples = {name: [] for name in policies}
    rng = random.Random(2026)
    for _ in range(options.repeats):
        order = list(policies)
        rng.shuffle(order)
        for name in order:
            samples[name].append(measure(policies[name]))
    summary = {
        name: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
        for name, rows in samples.items()
    }
    sources = [
        Path("modelopt/torch/quantization/linear_attention/solve.py"),
        Path("modelopt/torch/kernels/quantization/linear_attention/neumann.py"),
        Path(__file__),
    ]
    result = {
        "input": "strict lower Gaussian(0,0.04), FP32",
        "matrices": options.matrices,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seed": 2026,
        "policies": {name: policy.model_dump() for name, policy in policies.items()},
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "errors": errors,
        "summary": summary,
        "samples": samples,
    }
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"errors": errors, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

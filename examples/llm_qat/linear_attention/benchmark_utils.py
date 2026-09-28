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

"""Shared measurements for the linear-attention training examples."""

import statistics
import time

import torch


def benchmark_variants(variants, loss_fn, inputs, repeats, rng):
    """Measure interleaved forward/backward samples after three warmup rounds."""

    def measure(fn):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, mid, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        wall_start = time.perf_counter()
        start.record()
        output = fn()
        mid.record()
        torch.autograd.grad(loss_fn(output), inputs)
        end.record()
        torch.cuda.synchronize()
        return {
            "forward_ms": start.elapsed_time(mid),
            "backward_ms": mid.elapsed_time(end),
            "wall_ms": 1000 * (time.perf_counter() - wall_start),
            "peak_extra_bytes": torch.cuda.max_memory_allocated() - allocated,
        }

    for fn in variants.values():
        for _ in range(3):
            measure(fn)
    samples = {name: [] for name in variants}
    for _ in range(repeats):
        order = list(variants)
        rng.shuffle(order)
        for name in order:
            samples[name].append(measure(variants[name]))
    summary = {
        name: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
        for name, rows in samples.items()
    }
    return samples, summary

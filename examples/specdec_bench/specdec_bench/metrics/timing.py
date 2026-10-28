# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

from .base import Metric


class Timing(Metric):
    def __init__(self, tp_size):
        super().__init__()
        self.timing = []
        self.name = "timing"
        self.total_tokens = []
        self.tp_size = tp_size

    def process_step(self, step_outputs, request_id, turn_id):
        self.timing.append(step_outputs["token_times"])
        self.total_tokens.append(
            sum([sum([len(j) for j in i]) for i in step_outputs["output_ids"]])
        )

    def process_final(self, text_outputs):
        e2e_time = []
        ttft_time = []
        tpot_time = []
        gen_tp_time = []
        start_time = min([t[0] for t in self.timing])
        end_time = max([t[-1] for t in self.timing])
        duration = end_time - start_time
        self.out["Output Token Throughput (tokens/sec)"] = sum(self.total_tokens) / duration
        # No AIPerf equivalent -- AIPerf measures an endpoint and cannot know the
        # deployment's GPU count. Kept because it is how specdec results get compared
        # across parallelism settings.
        self.out["Output Token Throughput Per GPU (tokens/sec/gpu)"] = (
            self.out["Output Token Throughput (tokens/sec)"] / self.tp_size
        )
        self.out["Request Throughput (requests/sec)"] = len(self.timing) / duration
        self.out["Request Count (requests)"] = len(self.timing)
        for tokens, times in zip(self.total_tokens, self.timing):
            if len(times) > 1:
                e2e_time.append(times[-1] - times[0])
                ttft_time.append(times[1] - times[0])
            if len(times) > 2:
                gen_tp_time.append((tokens - 1) / (times[-1] - times[1]))
                tpot_time.extend([a - b for a, b in zip(times[1:], times[:-1])])
        self.out["Request Latency (ms)"] = compute_statistics(e2e_time, scale=_S_TO_MS)
        self.out["Time to First Token (ms)"] = compute_statistics(ttft_time, scale=_S_TO_MS)
        if tpot_time:
            self.out["Inter Token Latency (ms)"] = compute_statistics(tpot_time, scale=_S_TO_MS)
            self.out["Output Token Throughput Per User (tokens/sec/user)"] = compute_statistics(
                gen_tp_time
            )
        self.out["Output Sequence Length (tokens)"] = compute_statistics(self.total_tokens)
        for k, v in self.out.items():
            print(k, v)
        self.write()

    def clear(self):
        self.timing = []


# AIPerf's percentile set. The previous 0.25/0.5/0.75 stopped short of the tail, so
# nothing could be said about p90/p99 -- which is where serving decisions are made.
_AIPERF_QUANTILES = (0.25, 0.5, 0.75, 0.90, 0.95, 0.99)
_S_TO_MS = 1000.0


def compute_statistics(data, quantiles=_AIPERF_QUANTILES, scale=1.0):
    # Convert the data to a numpy array for easier calculations. ``scale`` converts
    # units at the boundary -- AIPerf reports latencies in milliseconds while the
    # timestamps collected here are seconds, and the key names carry the unit so the
    # conversion is visible rather than implied.
    data = np.array(data) * scale

    # Compute the statistics
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    std_val = np.std(data)

    # Compute quantiles (default: 25th, 50th, and 75th percentiles)
    quantile_vals = np.percentile(data, [q * 100 for q in quantiles])

    # Return the results as a dictionary
    stats = {
        "min": f"{min_val:.4f}",
        "max": f"{max_val:.4f}",
        "mean": f"{mean_val:.4f}",
        "std": f"{std_val:.4f}",
        "quantiles": {f"{q}": f"{v:.4f}" for q, v in zip(quantiles, quantile_vals)},
    }

    return stats

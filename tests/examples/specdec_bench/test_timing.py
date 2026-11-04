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

"""Numeric behaviour of the Timing metric.

Deliberately narrow: these cover the two things that can be silently *wrong*
rather than merely renamed -- the seconds-to-milliseconds conversion, and
inter-token latency under speculative decoding. Key names are not asserted on;
that would restate the implementation.
"""

from specdec_bench.metrics.timing import Timing, compute_statistics


def _drive(metric, token_times, n_tokens):
    """Feed one request's stream timestamps through the metric's step hook.

    ``process_step`` derives the token count from ``output_ids`` by summing
    lengths, so a single beam holding ``n_tokens`` one-token entries is the
    simplest shape that yields the intended total.
    """
    metric.process_step(
        {"token_times": token_times, "output_ids": [[[0] for _ in range(n_tokens)]]},
        request_id=0,
        turn_id=0,
    )


def test_compute_statistics_scales_units():
    """`scale` converts at the boundary; the unit lives in the key name."""
    seconds = [1.0, 2.0, 3.0]
    stats = compute_statistics(seconds, scale=1000.0)
    assert float(stats["mean"]) == 2000.0
    assert float(stats["min"]) == 1000.0
    assert float(stats["max"]) == 3000.0


def test_itl_excludes_prefill_and_is_per_token():
    """ITL must measure the generation phase only, divided by tokens.

    The request below spends 10s in prefill and then emits 4 more tokens over
    1.0s, so the honest inter-token latency is 1.0 / 4 = 0.25s = 250ms.

    Pooling raw per-step gaps instead would include the 10s prefill gap and
    average to roughly 2750ms -- an order of magnitude out, and dominated by a
    phase this metric is supposed to exclude.
    """
    metric = Timing(tp_size=1)
    # t0 request start, t1 first token (10s prefill), then 4 tokens over 1.0s.
    _drive(metric, [0.0, 10.0, 10.25, 10.5, 10.75, 11.0], n_tokens=5)
    metric.directory = None
    metric.write = lambda: None  # keep the test off the filesystem
    metric.process_final(text_outputs=[])

    itl_ms = float(metric.out["Inter Token Latency (ms)"]["mean"])
    assert abs(itl_ms - 250.0) < 1e-6, itl_ms


def test_itl_is_the_reciprocal_of_per_user_throughput():
    """The two are published side by side; AIPerf defines them as reciprocals.

    This is the speculative-decoding case: each stream chunk carries several
    accepted draft tokens, so there are fewer timestamps than tokens. Counting
    gaps instead of tokens would make ITL *rise* as the draft improves.
    """
    metric = Timing(tp_size=1)
    # 3 chunks after the first token, but 9 tokens emitted -- accept length 3.
    _drive(metric, [0.0, 1.0, 1.3, 1.6, 1.9], n_tokens=9)
    metric.directory = None
    metric.write = lambda: None
    metric.process_final(text_outputs=[])

    itl_s = float(metric.out["Inter Token Latency (ms)"]["mean"]) / 1000.0
    per_user_tps = float(metric.out["Output Token Throughput Per User (tokens/sec/user)"]["mean"])
    # compute_statistics stores values rounded to 4 decimal places, so the
    # round-trip through the reported throughput caps how exactly the reciprocal
    # can agree. 1e-5 is comfortably inside that and still far tighter than the
    # per-chunk-gap bug it guards against, which was off by a factor of ~3.
    assert abs(itl_s - 1.0 / per_user_tps) < 1e-5, (itl_s, per_user_tps)

    # 8 inter-token intervals across 0.9s of generation. This value is exact:
    # counting the 3 stream chunks instead of the 8 tokens would give 0.3s.
    assert abs(itl_s - 0.9 / 8) < 1e-9, itl_s


def test_single_token_response_does_not_divide_by_zero():
    """A one-token response has no inter-token interval; it must be skipped."""
    metric = Timing(tp_size=1)
    _drive(metric, [0.0, 1.0, 1.5], n_tokens=1)
    metric.directory = None
    metric.write = lambda: None
    metric.process_final(text_outputs=[])  # must not raise

    assert "Inter Token Latency (ms)" not in metric.out

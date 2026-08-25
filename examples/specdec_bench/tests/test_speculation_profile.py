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

"""Tests for the acceptance-length -> draft-position conversion.

The two failure modes these lock down both produce a *plausible-looking* profile,
which is why they get explicit coverage rather than relying on the end-to-end run:

  1. the off-by-two between acceptance length (counts the target's bonus token)
     and draft position;
  2. densification of a sparse histogram to a fixed-length vector.
"""

from itertools import pairwise

import pytest
from specdec_bench.metrics.acceptance_rate import AcceptanceRate
from specdec_bench.speculation_profile import build_profile, checkpoint_id, stub_profile


def _acceptance_out_from_histogram(histogram):
    """Run a length histogram through the real metric, not a reimplementation."""
    metric = AcceptanceRate()
    metric._process_lengths(dict(histogram))
    return metric.out


def test_offset_and_densification():
    # 100 steps: 40 emitted 1 token (no draft accepted), 30 emitted 2, 30 emitted 3.
    out = _acceptance_out_from_histogram({1: 40, 2: 30, 3: 30})
    out["Average_AL"] = (40 * 1 + 30 * 2 + 30 * 3) / 100  # 1.9

    profile = build_profile(out, num_speculative_tokens=3, method="eagle3")

    # P(>=1 draft accepted) = 60/100; P(>=2) = 30/100.
    assert profile["marginal_accept_rates"] == pytest.approx([0.6, 0.3, 0.0])
    # Conditional: first draft 0.6; second given first 0.3/0.6 = 0.5; third never.
    assert profile["conditional_accept_rates"] == pytest.approx([0.6, 0.5, 0.0])
    # Vector is padded to num_speculative_tokens even though length 4 never occurred.
    assert len(profile["marginal_accept_rates"]) == 3


def test_mean_consistency_identity_holds():
    """AL == 1 + sum(marginals) is the cross-check that catches a bad offset."""
    out = _acceptance_out_from_histogram({1: 40, 2: 30, 3: 30})
    out["Average_AL"] = 1.9
    profile = build_profile(out, num_speculative_tokens=3, method="eagle3")

    check = profile["validation"]["mean_consistency"]
    assert check["passed"]
    assert check["implied_mean_accept_length"] == pytest.approx(1.9)


def test_mean_consistency_flags_a_wrong_mean():
    out = _acceptance_out_from_histogram({1: 40, 2: 30, 3: 30})
    out["Average_AL"] = 3.5  # inconsistent with the histogram
    profile = build_profile(out, num_speculative_tokens=3, method="eagle3")
    assert not profile["validation"]["mean_consistency"]["passed"]


def test_marginals_are_non_increasing():
    """vLLM's synthetic sampler requires a non-increasing survival function."""
    out = _acceptance_out_from_histogram({1: 10, 2: 20, 3: 30, 4: 40})
    out["Average_AL"] = (10 + 40 + 90 + 160) / 100
    profile = build_profile(out, num_speculative_tokens=5, method="eagle3")

    marginals = profile["marginal_accept_rates"]
    assert profile["validation"]["marginal_monotonicity"]["passed"]
    assert all(a >= b for a, b in pairwise(marginals))


def test_json_string_keys_are_tolerated():
    """Profiles may be rebuilt from a round-tripped acceptance_rate.json."""
    out = _acceptance_out_from_histogram({1: 40, 2: 30, 3: 30})
    out["Average_AL"] = 1.9
    round_tripped = {
        "Conditional_Acceptance_Rate": {
            str(k): v for k, v in out["Conditional_Acceptance_Rate"].items()
        },
        "Joint_Acceptance_Rate": {str(k): v for k, v in out["Joint_Acceptance_Rate"].items()},
        "Average_AL": 1.9,
    }
    assert (
        build_profile(round_tripped, num_speculative_tokens=3)["marginal_accept_rates"]
        == build_profile(out, num_speculative_tokens=3)["marginal_accept_rates"]
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("eagle3", "chain_analytic"),
        ("EAGLE3", "chain_analytic"),
        ("dflash", "measured_per_k"),
        ("dspark", "measured_per_k"),
        (None, "measured_per_k"),
    ],
)
def test_accept_length_model_defaults_by_method(method, expected):
    """Block-parallel methods must not advertise that K extrapolates."""
    out = _acceptance_out_from_histogram({1: 50, 2: 50})
    out["Average_AL"] = 1.5
    assert (
        build_profile(out, num_speculative_tokens=2, method=method)["accept_length_model"]
        == expected
    )


def test_stub_profile_is_marked_unmeasured():
    stub = stub_profile(num_speculative_tokens=3, method="dflash")
    assert stub["measured"] is False
    assert stub["mean_accept_length"] is None
    assert len(stub["conditional_accept_rates"]) == 3


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "/lustre/fsw/portfolios/coreai/projects/x/hf-local/nvidia/MiniMax-M2.7-DFlash",
            "nvidia/MiniMax-M2.7-DFlash",
        ),
        ("/hf-local/Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
        ("nvidia/MiniMax-M2.7-DFlash", "nvidia/MiniMax-M2.7-DFlash"),
        ("bare-name", "bare-name"),
        (None, None),
        ("", None),
    ],
)
def test_checkpoint_id_strips_internal_paths(path, expected):
    """The profile is published with checkpoints, so it must not carry cluster layout."""
    assert checkpoint_id(path) == expected

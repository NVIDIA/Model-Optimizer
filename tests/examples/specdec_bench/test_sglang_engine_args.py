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

"""SGLang renames engine arguments between releases; the adapter must follow.

``sgl.Engine`` rejects an unknown keyword with a bare ``TypeError`` naming one key, so a
version skew shows up as an inscrutable crash at startup. 0.5.22 -- the first arm64 image
carrying LiLiCorr -- split ``cuda_graph_max_bs`` into decode/prefill limits, which made
the SGLang path unusable on that hardware.
"""

import pytest
from specdec_bench.models import sglang as sglang_mod
from specdec_bench.models.sglang import _adapt_engine_kwargs


@pytest.fixture
def accepted(monkeypatch):
    """Pin the set of names the 'installed' SGLang accepts."""

    def _set(names):
        monkeypatch.setattr(sglang_mod, "_accepted_engine_args", lambda: set(names))

    return _set


def test_split_argument_fans_out_to_both_replacements(accepted):
    """0.5.22 split one cuda-graph limit into decode and prefill; both must get the value."""
    accepted({"model_path", "cuda_graph_max_bs_decode", "cuda_graph_max_bs_prefill"})

    out = _adapt_engine_kwargs({"model_path": "/m", "cuda_graph_max_bs": 8})

    assert out == {
        "model_path": "/m",
        "cuda_graph_max_bs_decode": 8,
        "cuda_graph_max_bs_prefill": 8,
    }


def test_names_the_installed_version_still_has_are_left_alone(accepted):
    accepted({"model_path", "cuda_graph_max_bs"})

    out = _adapt_engine_kwargs({"model_path": "/m", "cuda_graph_max_bs": 8})

    assert out == {"model_path": "/m", "cuda_graph_max_bs": 8}


def test_unknown_performance_knob_is_dropped_rather_than_crashing(accepted):
    """Losing a perf knob changes how fast the run goes, not what it measures."""
    accepted({"model_path"})

    out = _adapt_engine_kwargs({"model_path": "/m", "enable_torch_compile": True})

    assert out == {"model_path": "/m"}


@pytest.mark.parametrize(
    "critical",
    ["speculative_draft_model_path", "speculative_algorithm", "model_path", "tp_size"],
)
def test_dropping_an_argument_that_changes_the_measurement_is_refused(accepted, critical):
    """The failure this prevents: no draft model loaded, so the run quietly benchmarks the
    target alone and reports the result as a speculative speedup."""
    accepted({"something_else"})

    with pytest.raises(ValueError, match=critical):
        _adapt_engine_kwargs({critical: "value"})


def test_adapter_is_a_no_op_when_the_accepted_names_cannot_be_read(monkeypatch):
    """Never block a run on introspection that failed -- let SGLang do the rejecting."""
    monkeypatch.setattr(sglang_mod, "_accepted_engine_args", lambda: None)
    kwargs = {"model_path": "/m", "cuda_graph_max_bs": 8}

    assert _adapt_engine_kwargs(kwargs) == kwargs

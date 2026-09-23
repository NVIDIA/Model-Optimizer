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

import copy
import runpy
from pathlib import Path

import pytest

_STUDY = runpy.run_path(
    str(
        Path(__file__).resolve().parents[4] / "examples/llm_qat/linear_attention/compare_quality.py"
    )
)


def _receipt():
    return {
        "model": "fixture",
        "model_revision": "pinned",
        "dataset": "fixture",
        "dataset_config": "raw",
        "dataset_revision": "pinned",
        "eval_split": "test",
        "train_file_sha256": "train",
        "eval_file_sha256": "test",
        "train_tokens_sha256": "train_tokens",
        "eval_tokens_sha256": "test_tokens",
        "sequence_length": 128,
        "prefill_tokens": None,
        "loss_scope": "full",
        "seed": 2026,
        "train_order": [1, 0],
        "training": "attention_only",
        "trainable_parameters": 100,
        "kda_layers": 2,
        "learning_rate": 1e-5,
        "weight_decay": 0,
        "gradient_clip": 1,
        "train_predicted_tokens": 256,
        "source_sha256": {"solve.py": "pinned"},
        "packages": {"fla-core": "0.5.1"},
        "torch": "pinned",
        "gpu": "same_device",
        "before": {"block_nll": [1.0, 2.0, 3.0, 4.0], "predicted_tokens": 512},
        "after": {"block_nll": [0.9, 1.9, 2.9, 3.9], "predicted_tokens": 512},
    }


def test_paired_quality_bound_and_failure():
    control = _receipt()
    candidate = copy.deepcopy(control)
    candidate["before"]["block_nll"] = [x + 0.01 for x in control["before"]["block_nll"]]
    candidate["after"]["block_nll"] = [x + 0.03 for x in control["after"]["block_nll"]]
    result = _STUDY["compare"](control, candidate, samples=100)["comparison"]
    assert result["before"]["mean_nll_delta"] == pytest.approx(0.01)
    assert result["before"]["upper_bound_within_margin"]
    assert not result["after"]["upper_bound_within_margin"]


@pytest.mark.parametrize(
    "field",
    [
        "eval_tokens_sha256",
        "source_sha256",
        "train_order",
        "learning_rate",
        "prefill_tokens",
        "loss_scope",
    ],
)
def test_quality_refuses_unmatched_studies(field):
    control, candidate = _receipt(), _receipt()
    candidate[field] = "different"
    with pytest.raises(ValueError, match=field):
        _STUDY["compare"](control, candidate)


def test_quality_refuses_nonfinite_values():
    control, candidate = _receipt(), _receipt()
    candidate["after"]["block_nll"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _STUDY["compare"](control, candidate, samples=10)

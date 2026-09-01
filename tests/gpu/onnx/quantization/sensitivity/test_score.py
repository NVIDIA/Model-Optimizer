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


from __future__ import annotations

import importlib
import math
import shutil

import pytest
from _test_utils.onnx.quantization.sensitivity.models import (
    SYNTHETIC_OP_SCOPE,
    build_conv_mm_ln_onnx,
    deterministic_calibration,
    get_coatnet_paths,
)

from modelopt.onnx.quantization.sensitivity import score


@pytest.fixture(scope="module")
def synthetic_onnx_path(tmp_path_factory):
    """Build the synthetic 2-Conv + 1-MatMul + 1-LayerNorm graph once per test module."""
    path = str(tmp_path_factory.mktemp("sens_synth") / "sens_synth.onnx")
    build_conv_mm_ln_onnx(path)
    return path


def test_synthetic_random_calibration_smoke(synthetic_onnx_path):
    """With ``calibration_data=None``, the synthetic-random path returns finite scores."""
    result = score(
        synthetic_onnx_path,
        calibration_data=None,
        num_synthetic_samples=8,
        metric="kl_div",
        target_precision="int8",
        granularity="op_type",
        calibration_eps=["cpu"],
        op_types_scope=SYNTHETIC_OP_SCOPE,
    )
    assert result["calibration_source"] == "synthetic"
    scores = result["scores"]
    failed = result.get("failed", [])
    assert set(scores) | set(failed) == set(SYNTHETIC_OP_SCOPE), (
        f"Op(s) missing from result. scores={scores} failed={failed}"
    )
    for name, value in scores.items():
        assert math.isfinite(value) and value >= 0.0, (
            f"Score for {name!r} should be finite non-negative, got {value}"
        )


@pytest.mark.parametrize("metric", ["kl_div", "mse", "cos"])
def test_synthetic_deterministic_ln_highest(synthetic_onnx_path, metric):
    """Synthetic graph + deterministic real inputs -- ``LayerNormalization`` scores highest of all ops."""
    result = score(
        synthetic_onnx_path,
        calibration_data=deterministic_calibration(),
        metric=metric,
        target_precision="int8",
        granularity="op_type",
        calibration_eps=["cpu"],
        op_types_scope=SYNTHETIC_OP_SCOPE,
    )
    assert result["calibration_source"] == "real"
    assert result["num_calibration_samples"] == 8
    top_op = max(result["scores"].items(), key=lambda kv: kv[1])[0]
    assert top_op == "LayerNormalization", (
        f"Expected LayerNormalization to be the top-ranked op, got '{top_op}' from {result['scores']}"
    )


def test_failed_probe_is_recorded(synthetic_onnx_path, monkeypatch):
    """A probe that inserts no Q/DQ nodes is recorded in ``failed`` and absent from ``scores``."""
    # Patch quantize() in score() to copy the input as-is, so the probe path has no Q/DQ nodes
    score_module = importlib.import_module("modelopt.onnx.quantization.sensitivity.score")

    def _fake_quantize(**kwargs):
        shutil.copy(kwargs["onnx_path"], kwargs["output_path"])

    monkeypatch.setattr(score_module, "quantize", _fake_quantize)

    # Calculate scores
    result = score(
        synthetic_onnx_path,
        calibration_data=deterministic_calibration(),
        metric="kl_div",
        target_precision="int8",
        granularity="op_type",
        calibration_eps=["cpu"],
        op_types_scope=SYNTHETIC_OP_SCOPE,
    )
    assert result["failed"], "Expected failed probes to be surfaced, got empty list"
    assert not result["scores"], (
        f"Expected empty scores when every probe fails, got {result['scores']}"
    )


def test_failed_probe_records_exceptions(synthetic_onnx_path, monkeypatch):
    """A probe whose ``quantize()`` call raises is recorded in ``failed`` and absent from ``scores``."""
    # Patch quantize() in score() to raise an issue, so every probe hits the except branch that
    # appends the target to ``failed``
    score_module = importlib.import_module("modelopt.onnx.quantization.sensitivity.score")

    def _raising_quantize(**kwargs):
        raise RuntimeError("simulated quantize failure")

    monkeypatch.setattr(score_module, "quantize", _raising_quantize)

    result = score(
        synthetic_onnx_path,
        calibration_data=deterministic_calibration(),
        metric="kl_div",
        target_precision="int8",
        granularity="op_type",
        calibration_eps=["cpu"],
        op_types_scope=SYNTHETIC_OP_SCOPE,
    )
    assert result["failed"], "Expected failed probes to be surfaced, got empty list"
    assert not result["scores"], (
        f"Expected empty scores when every probe raises, got {result['scores']}"
    )


@pytest.mark.manual(reason="CoAtNet-0 integration; ~14 min on H100, needs pre-staged fixtures")
def test_coatnet_op_type_matches_manual_groundtruth():
    """CoAtNet-0 op-type ranking surfaces the ops that ``--op_types_to_quantize Conv`` avoids.

    Top-4 = ``Add`` / ``Mul`` / ``LayerNormalization`` / ``ReduceMean`` (all > 1.5 KL); ``Conv``
    sits ~10x below. Matches the manual "Conv-only wins 82% top-1" ground truth.
    """
    onnx_path, calib_path = get_coatnet_paths()

    result = score(
        onnx_path,
        calibration_data=calib_path,
        metric="kl_div",
        target_precision="int8",
        granularity="op_type",
        calibration_eps=["cuda:0", "cpu"],
    )
    assert result["calibration_source"] == "real"
    scores = result["scores"]
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    expected_top4 = ("Add", "Mul", "LayerNormalization", "ReduceMean")
    top4 = {name for name, _ in ranked[:4]}
    assert set(expected_top4).issubset(top4), (
        f"Top-4 sensitive ops should include {expected_top4}, got {ranked}"
    )
    # Pin the quantitative threshold from the docstring.
    assert all(scores[name] > 1.5 for name in expected_top4), (
        f"Each expected top-4 op should score above 1.5 KL, got "
        f"{ {name: scores[name] for name in expected_top4} }"
    )
    assert scores["Conv"] < 0.5, (
        f"Conv score {scores['Conv']:.3f} unexpectedly high (top-4 are all > 1.5)"
    )
    for op in ("Softmax", "Gemm", "GlobalAveragePool"):
        assert scores.get(op, 0.0) < 0.001, f"{op} score {scores.get(op, 0.0):.3g} should be ~0"


@pytest.mark.manual(
    reason="CoAtNet-0 per-node integration; ~30-60 min on H100, needs pre-staged fixtures"
)
def test_coatnet_per_node_matches_manual_groundtruth():
    """CoAtNet-0 per-node ranking: LN / MHA nodes in top-10, individual Conv nodes in bottom-10."""
    onnx_path, calib_path = get_coatnet_paths()

    result = score(
        onnx_path,
        calibration_data=calib_path,
        metric="kl_div",
        target_precision="int8",
        granularity="node",
        calibration_eps=["cuda:0", "cpu"],
    )
    assert result["calibration_source"] == "real"
    ranked = sorted(result["scores"].items(), key=lambda kv: kv[1], reverse=True)
    k = 10
    assert len(ranked) >= 2 * k, "Per-node ranking is unexpectedly short."
    top_names = [name for name, _ in ranked[:k]]
    bottom_names = [name for name, _ in ranked[-k:]]
    assert any("layernorm" in n.lower() or "attn" in n.lower() for n in top_names), (
        f"Expected LN or MHA nodes in top-{k}, got {top_names}"
    )
    assert any("conv" in n.lower() for n in bottom_names), (
        f"Expected Conv nodes in bottom-{k}, got {bottom_names}"
    )

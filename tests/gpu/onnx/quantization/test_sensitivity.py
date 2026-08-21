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

"""Tests for the ONNX quantization sensitivity primitive.

Tiers:

1. Synthetic-graph unit test with real deterministic inputs -- LayerNorm scores highest.
2. CoAtNet-0 op-type integration (``@pytest.mark.slow`` + real ImageNet calibration).
3. CoAtNet-0 per-node integration (``@pytest.mark.slow_gpu`` + real ImageNet calibration).
4. Synthetic-random calibration regression guard -- LayerNorm still > Conv directionally.

Tiers 2 and 3 read a pre-staged CoAtNet-0 ONNX + calibration ``.npz`` from a fixtures directory
resolved via ``MODELOPT_ONNX_ACCURACY_MODELS_DIR`` (default ``/tmp``). Missing fixtures ``pytest.skip``
cleanly.
"""

from __future__ import annotations

import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.quantization.sensitivity import score

_INPUT_NAME = "input"
_OUTPUT_NAME = "output"
_C_IN = 8
_C_MID = 16
_H = W = 16
_MATMUL_DIM = _C_MID * _H * W
_LOGITS = 32
_FIXTURE_DIR = os.environ.get("MODELOPT_ONNX_ACCURACY_MODELS_DIR", "/tmp")
# Ops covered by the synthetic Conv+MatMul+LN graph. Passed explicitly because the score()
# default -- get_autotuner_quantizable_ops() -- excludes LayerNormalization even though the ModelOpt
# quantize() path registers it via configure_ort.
_SYNTHETIC_OP_SCOPE = ["Conv", "MatMul", "LayerNormalization"]


def _build_conv_mm_ln_onnx(path: str, opset: int = 17) -> None:
    """Build a small 2-Conv + 1-MatMul + 1-LayerNorm ONNX for deterministic sensitivity tests."""
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((_C_MID, _C_IN, 3, 3)).astype(np.float32) * 0.1
    b1 = np.zeros((_C_MID,), dtype=np.float32)
    w2 = rng.standard_normal((_C_MID, _C_MID, 3, 3)).astype(np.float32) * 0.1
    b2 = np.zeros((_C_MID,), dtype=np.float32)
    mm = rng.standard_normal((_MATMUL_DIM, _LOGITS)).astype(np.float32) * 0.05
    ln_scale = np.ones((_LOGITS,), dtype=np.float32)
    ln_bias = np.zeros((_LOGITS,), dtype=np.float32)

    initializers = [
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(w2, "w2"),
        numpy_helper.from_array(b2, "b2"),
        numpy_helper.from_array(mm, "mm_w"),
        numpy_helper.from_array(ln_scale, "ln_scale"),
        numpy_helper.from_array(ln_bias, "ln_bias"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "w1", "b1"],
            ["conv1_out"],
            name="conv_1",
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            "Conv",
            ["conv1_out", "w2", "b2"],
            ["conv2_out"],
            name="conv_2",
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            "Flatten", ["conv2_out"], ["flat_out"], name="flatten_1", axis=1
        ),
        helper.make_node(
            "MatMul", ["flat_out", "mm_w"], ["mm_out"], name="matmul_1"
        ),
        helper.make_node(
            "LayerNormalization",
            ["mm_out", "ln_scale", "ln_bias"],
            [_OUTPUT_NAME],
            name="layernorm_1",
            axis=-1,
            epsilon=1e-5,
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="sens_test_graph",
        inputs=[
            helper.make_tensor_value_info(
                _INPUT_NAME, TensorProto.FLOAT, [1, _C_IN, _H, W]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(_OUTPUT_NAME, TensorProto.FLOAT, [1, _LOGITS])
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8
    )
    onnx.save(model, path)


def _deterministic_calibration(num_samples: int = 8) -> dict[str, np.ndarray]:
    """Fixed-seed calibration data for the synthetic sensitivity graph."""
    rng = np.random.default_rng(42)
    return {_INPUT_NAME: rng.standard_normal((num_samples, _C_IN, _H, W)).astype(np.float32)}


def _assert_ln_over_conv(scores: dict[str, float]) -> None:
    """Directional invariant: LayerNormalization must rank strictly above Conv."""
    assert "LayerNormalization" in scores, f"LayerNorm missing from scores: {scores}"
    assert "Conv" in scores, f"Conv missing from scores: {scores}"
    assert scores["LayerNormalization"] > scores["Conv"], (
        f"Expected LayerNormalization > Conv, got {scores}"
    )


@pytest.mark.parametrize("metric", ["kl_div", "mse", "cos"])
def test_synthetic_deterministic_ln_highest(tmp_path, metric):
    """Tier 1: synthetic graph + deterministic real inputs -- LN scores highest of all ops."""
    onnx_path = str(tmp_path / "sens_synth.onnx")
    _build_conv_mm_ln_onnx(onnx_path)
    calib = _deterministic_calibration()

    result = score(
        onnx_path,
        calibration_data=calib,
        metric=metric,
        target_precision="int8",
        granularity="op_type",
        calibration_eps=("cpu",),
        op_types_scope=_SYNTHETIC_OP_SCOPE,
    )
    assert result["calibration_source"] == "real"
    assert result["num_calibration_samples"] == 8
    scores = result["scores"]
    assert scores, "No scores produced for synthetic graph."
    # Highest-scoring op should be LayerNormalization.
    top_op = max(scores.items(), key=lambda kv: kv[1])[0]
    assert top_op == "LayerNormalization", (
        f"Expected LayerNormalization to be the top-ranked op, got '{top_op}' from {scores}"
    )
    _assert_ln_over_conv(scores)


def test_synthetic_random_calibration_directional(tmp_path):
    """Tier 4: with ``calibration_data=None``, LN > Conv invariant still holds directionally."""
    onnx_path = str(tmp_path / "sens_synth.onnx")
    _build_conv_mm_ln_onnx(onnx_path)

    result = score(
        onnx_path,
        calibration_data=None,
        num_synthetic_samples=8,
        metric="kl_div",
        target_precision="int8",
        granularity="op_type",
        calibration_eps=("cpu",),
        op_types_scope=_SYNTHETIC_OP_SCOPE,
    )
    assert result["calibration_source"] == "synthetic"
    _assert_ln_over_conv(result["scores"])


def _require_fixture(name: str) -> str:
    """Return a fixture path or ``pytest.skip`` if it isn't staged on this host."""
    path = os.path.join(_FIXTURE_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Sensitivity fixture missing: {path}")
    return path


@pytest.mark.slow
def test_coatnet_op_type_matches_manual_groundtruth():
    """Tier 2: CoAtNet-0 op-type ranking must surface the ops that ``--op_types_to_quantize
    Conv`` implicitly avoids.

    Empirical ranking on CoAtNet-0 with 500-sample ImageNet calibration and ``kl_div``:

        Add                 2.848  <-- highest impact
        Mul                 1.890
        LayerNormalization  1.653
        ReduceMean          1.570
        BatchNormalization  0.355
        Conv                0.181
        AveragePool         0.057
        Sigmoid             0.039
        MatMul              0.015
        Relu               ~0
        Softmax            ~0
        GlobalAveragePool  ~0
        Gemm                0

    Top-4 = Add / Mul / LayerNormalization / ReduceMean are the load-bearing failures
    (residual paths, SE gating + softmax scale, norm boundaries).  Conv sits ~10x below
    the top-4 and quantizes cleanly, matching the manual "Conv-only wins 82% top-1"
    ground truth read as a quantization policy.

    Wall-clock ~14 min on H100 with 500 samples / 13 probes (~60s per probe).

    Fixtures (override root via ``MODELOPT_SENSITIVITY_FIXTURES``):
      * ``coatnet-0_rw_inpsize_1x3x224x224_opsetv_17_simplified.onnx`` -- baseline ONNX.
      * ``imagenet_calib_500.npz`` -- 500-sample ImageNet calibration dict.
    """
    onnx_path = _require_fixture(
        "coatnet-0_rw_inpsize_1x3x224x224_opsetv_17_simplified.onnx"
    )
    calib_path = _require_fixture("imagenet_calib_500.npz")

    result = score(
        onnx_path,
        calibration_data=calib_path,
        metric="kl_div",
        target_precision="int8",
        granularity="op_type",
        calibration_eps=("cuda:0", "cpu"),
    )
    assert result["calibration_source"] == "real"
    scores = result["scores"]
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    top4 = {name for name, _ in ranked[:4]}
    assert {"Add", "Mul", "LayerNormalization", "ReduceMean"}.issubset(top4), (
        f"Top-4 sensitive ops should include Add / Mul / LayerNormalization / "
        f"ReduceMean (all > 1.5 KL), got {ranked}"
    )
    # Conv sits ~10x below the top-4 -- justifies the Conv-only quantization policy.
    assert scores["Conv"] < 0.5, (
        f"Conv score {scores['Conv']:.3f} unexpectedly high (top-4 are all > 1.5)"
    )
    # These cluster at ~0 -- primitive won't recommend excluding them because there's
    # nothing to exclude.
    for op in ("Softmax", "Gemm", "GlobalAveragePool"):
        assert scores.get(op, 0.0) < 0.001, (
            f"{op} score {scores.get(op, 0.0):.3g} should be ~0"
        )


@pytest.mark.slow_gpu
def test_coatnet_per_node_matches_manual_groundtruth():
    """Tier 3: CoAtNet-0 per-node ranking (LN/MHA nodes top, Conv nodes bottom).

    Wall clock ~30-60 min; gated behind ``@pytest.mark.slow_gpu`` so default CI stays fast.
    """
    onnx_path = _require_fixture(
        "coatnet-0_rw_inpsize_1x3x224x224_opsetv_17_simplified.onnx"
    )
    calib_path = _require_fixture("imagenet_calib_500.npz")

    result = score(
        onnx_path,
        calibration_data=calib_path,
        metric="kl_div",
        target_precision="int8",
        granularity="node",
        calibration_eps=("cuda:0", "cpu"),
    )
    assert result["calibration_source"] == "real"
    ranked = sorted(result["scores"].items(), key=lambda kv: kv[1], reverse=True)
    assert len(ranked) >= 20, "Per-node ranking is unexpectedly short."
    top_k = 10
    bottom_k = 10
    top_names = [name for name, _ in ranked[:top_k]]
    bottom_names = [name for name, _ in ranked[-bottom_k:]]
    # LayerNorm / MHA subgraph nodes dominate the top of the ranking.
    assert any("layernorm" in n.lower() or "attn" in n.lower() for n in top_names), (
        f"Expected LN or MHA nodes in top-{top_k}, got {top_names}"
    )
    # Individual Conv nodes cluster at the bottom (Conv-only ground truth).
    assert any("conv" in n.lower() for n in bottom_names), (
        f"Expected Conv nodes in bottom-{bottom_k}, got {bottom_names}"
    )

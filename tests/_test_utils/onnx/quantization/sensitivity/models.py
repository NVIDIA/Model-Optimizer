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

"""Shared fixtures and synthetic-graph builders for the sensitivity test suite."""

from __future__ import annotations

import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

_INPUT_NAME = "input"
_OUTPUT_NAME = "output"
_C_IN = 8
_C_MID = 16
_H = _W = 16
_MATMUL_DIM = _C_MID * _H * _W
_LOGITS = 32

_FIXTURE_DIR = os.environ.get("MODELOPT_ONNX_ACCURACY_MODELS_DIR", "/tmp")

# Ops covered by the synthetic Conv+MatMul+LN graph. Passed explicitly by tests to constrain
# the scoring scope to the ops actually present in this small graph.
SYNTHETIC_OP_SCOPE = ["Conv", "MatMul", "LayerNormalization"]


def build_conv_mm_ln_onnx(path: str, opset: int = 17) -> None:
    """Build a 2-Conv + 1-MatMul + 1-LayerNorm ONNX for deterministic sensitivity tests."""
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
        helper.make_node("Flatten", ["conv2_out"], ["flat_out"], name="flatten_1", axis=1),
        helper.make_node("MatMul", ["flat_out", "mm_w"], ["mm_out"], name="matmul_1"),
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
        inputs=[helper.make_tensor_value_info(_INPUT_NAME, TensorProto.FLOAT, [1, _C_IN, _H, _W])],
        outputs=[helper.make_tensor_value_info(_OUTPUT_NAME, TensorProto.FLOAT, [1, _LOGITS])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
    onnx.save(model, path)


def deterministic_calibration(num_samples: int = 8) -> dict[str, np.ndarray]:
    """Fixed-seed calibration data for the synthetic sensitivity graph."""
    rng = np.random.default_rng(42)
    return {_INPUT_NAME: rng.standard_normal((num_samples, _C_IN, _H, _W)).astype(np.float32)}


def assert_ln_over_conv(scores: dict[str, float]) -> None:
    """Directional invariant: LayerNormalization must rank strictly above Conv."""
    assert "LayerNormalization" in scores, f"LayerNorm missing from scores: {scores}"
    assert "Conv" in scores, f"Conv missing from scores: {scores}"
    assert scores["LayerNormalization"] > scores["Conv"], (
        f"Expected LayerNormalization > Conv, got {scores}"
    )


def require_fixture(name: str) -> str:
    """Return a fixture path under ``MODELOPT_ONNX_ACCURACY_MODELS_DIR`` or ``pytest.skip``."""
    path = os.path.join(_FIXTURE_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Sensitivity fixture missing: {path}")
    return path


def get_coatnet_paths() -> tuple[str, str]:
    """CoAtNet-0 baseline ONNX + 500-sample ImageNet calibration; ``pytest.skip`` if missing."""
    return (
        require_fixture("coatnet-0_rw_inpsize_1x3x224x224_opsetv_17_simplified.onnx"),
        require_fixture("imagenet_calib_500.npz"),
    )

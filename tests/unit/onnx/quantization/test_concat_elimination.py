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

from _test_utils.onnx.lib_test_models import build_concat_with_constant_input_model

from modelopt.onnx.quantization.qdq_graph import get_concat_eliminated_tensors


def test_concat_elimination_groups_activation_inputs():
    model = build_concat_with_constant_input_model(constant_input=False)
    groups = get_concat_eliminated_tensors(model, ["concat", "matmul", "relu", "relu_cls"])
    assert groups["concat_out"] == {"cls", "relu_out", "concat_out"}
    assert groups["cls"] == groups["concat_out"]


def test_concat_elimination_skips_constant_inputs():
    """A constant Concat input has no calibration range, so it must not join the scale group."""
    model = build_concat_with_constant_input_model(constant_input=True)
    groups = get_concat_eliminated_tensors(model, ["concat", "matmul", "relu"])
    assert "cls" not in groups
    assert groups["concat_out"] == {"relu_out", "concat_out"}
    assert all("cls" not in group for group in groups.values())

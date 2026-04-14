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

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from modelopt.onnx.quantization.graph_utils import find_nodes_from_convs_to_exclude


def _make_conv_graph(output_channels, input_channels, kernel_shape=(3, 3), name="Conv_0"):
    """Build a minimal graph with a single Conv node."""
    spatial = [32, 32]
    inp = gs.Variable(name="input", dtype=np.float32, shape=[1, input_channels, *spatial])
    out = gs.Variable(name="output", dtype=np.float32)

    weight_shape = (output_channels, input_channels, *kernel_shape)
    weight = gs.Constant(name="weight", values=np.ones(weight_shape, dtype=np.float32))

    conv = gs.Node(
        name=name,
        op="Conv",
        inputs=[inp, weight],
        outputs=[out],
        attrs={"kernel_shape": list(kernel_shape)},
    )

    return gs.Graph(nodes=[conv], inputs=[inp], outputs=[out], opset=13)


@pytest.mark.parametrize(
    ("oc", "ic", "expected_excluded"),
    [
        (16, 64, True),
        (64, 16, True),
        (8, 8, True),
        (16, 16, True),
        (17, 64, False),
        (64, 17, False),
        (17, 17, False),
        (32, 32, False),
        (64, 64, False),
    ],
)
def test_fp8_small_channel_conv_exclusion(oc, ic, expected_excluded):
    """FP8 mode should exclude Conv nodes with OC or IC <= 16."""
    graph = _make_conv_graph(output_channels=oc, input_channels=ic)
    excluded = find_nodes_from_convs_to_exclude(graph, quantize_mode="fp8")
    if expected_excluded:
        assert "Conv_0" in excluded
    else:
        assert "Conv_0" not in excluded


def test_fp8_small_channel_exclusion_does_not_affect_int8():
    """The small-channel FP8 exclusion should not apply in int8 mode."""
    # OC=8 would be excluded in FP8 (see oc=8, ic=8 case above), but not in int8.
    graph = _make_conv_graph(output_channels=8, input_channels=64, kernel_shape=(3, 3))
    excluded = find_nodes_from_convs_to_exclude(graph, quantize_mode="int8")
    assert "Conv_0" not in excluded


@pytest.mark.parametrize(
    ("oc", "ic"),
    [
        (15, 64),
        (64, 15),
        (1, 1),
    ],
)
def test_fp8_channels_below_16_excluded_by_general_check(oc, ic):
    """Channels strictly < 16 are excluded by the general channel check, not the FP8 check."""
    graph = _make_conv_graph(output_channels=oc, input_channels=ic, kernel_shape=(3, 3))
    excluded = find_nodes_from_convs_to_exclude(graph, quantize_mode="fp8")
    assert "Conv_0" in excluded

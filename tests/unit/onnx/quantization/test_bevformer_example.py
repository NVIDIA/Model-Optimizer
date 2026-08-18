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

import numpy as np

from examples.onnx_ptq.bevformer.prepare_calibration import build_inputs


class DataContainer:
    def __init__(self, data):
        self.data = data


class NumpyTensor:
    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array


def make_data(scene_token, can_bus):
    metadata = {
        "scene_token": scene_token,
        "can_bus": np.array(can_bus, dtype=np.float64),
        "lidar2img": [np.eye(4, dtype=np.float64) for _ in range(6)],
    }
    image = np.ones((1, 6, 3, 2, 2), dtype=np.float32)
    return {
        "img": [DataContainer([NumpyTensor(image)])],
        "img_metas": [DataContainer([[metadata]])],
    }


def test_temporal_inputs_reset_and_update_between_scenes():
    prev_bev = np.zeros((4, 1, 2), dtype=np.float32)
    previous_frame = {"scene_token": None, "position": 0, "angle": 0}
    first_can_bus = [1, 2, 3, *range(4, 18), 18]

    first = build_inputs(make_data("scene-a", first_can_bus), prev_bev, previous_frame)

    assert first["use_prev_bev"].tolist() == [0.0]
    np.testing.assert_array_equal(first["can_bus"][:3], np.zeros(3))
    assert first["can_bus"][-1] == 0
    np.testing.assert_array_equal(previous_frame["position"], first_can_bus[:3])
    assert previous_frame["angle"] == first_can_bus[-1]

    second_can_bus = copy.copy(first_can_bus)
    second_can_bus[:3] = [3, 5, 7]
    second_can_bus[-1] = 21
    second = build_inputs(make_data("scene-a", second_can_bus), prev_bev, previous_frame)

    assert second["use_prev_bev"].tolist() == [1.0]
    np.testing.assert_array_equal(second["can_bus"][:3], [2, 3, 4])
    assert second["can_bus"][-1] == 3

    third = build_inputs(make_data("scene-b", second_can_bus), prev_bev, previous_frame)

    assert third["use_prev_bev"].tolist() == [0.0]
    np.testing.assert_array_equal(third["can_bus"][:3], np.zeros(3))
    assert third["can_bus"][-1] == 0

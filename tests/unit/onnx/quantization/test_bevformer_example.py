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
import pytest

from examples.onnx_ptq.bevformer import quantize as quantize_example
from examples.onnx_ptq.bevformer.prepare_calibration import build_inputs, prepare_batches


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


class FakeSession:
    def __init__(self):
        self.calls = 0

    def run(self, output_names, inputs):
        assert output_names == ["bev_embed"]
        self.calls += 1
        return [np.full_like(inputs["prev_bev"], self.calls)]


def test_temporal_inputs_reset_and_update_between_scenes():
    prev_bev = np.ones((4, 1, 2), dtype=np.float32)
    previous_frame = {"scene_token": None, "position": 0, "angle": 0}
    first_can_bus = [1, 2, 3, *range(4, 18), 18]

    first = build_inputs(make_data("scene-a", first_can_bus), prev_bev, previous_frame)

    assert first["use_prev_bev"].tolist() == [0.0]
    np.testing.assert_array_equal(first["prev_bev"], np.zeros_like(prev_bev))
    np.testing.assert_array_equal(first["can_bus"][:3], np.zeros(3))
    assert first["can_bus"][-1] == 0
    np.testing.assert_array_equal(previous_frame["position"], first_can_bus[:3])
    assert previous_frame["angle"] == first_can_bus[-1]

    second_can_bus = copy.copy(first_can_bus)
    second_can_bus[:3] = [3, 5, 7]
    second_can_bus[-1] = 21
    second = build_inputs(make_data("scene-a", second_can_bus), prev_bev, previous_frame)

    assert second["use_prev_bev"].tolist() == [1.0]
    np.testing.assert_array_equal(second["prev_bev"], prev_bev)
    np.testing.assert_array_equal(second["can_bus"][:3], [2, 3, 4])
    assert second["can_bus"][-1] == 3

    third = build_inputs(make_data("scene-b", second_can_bus), prev_bev, previous_frame)

    assert third["use_prev_bev"].tolist() == [0.0]
    np.testing.assert_array_equal(third["prev_bev"], np.zeros_like(prev_bev))
    np.testing.assert_array_equal(third["can_bus"][:3], np.zeros(3))
    assert third["can_bus"][-1] == 0


def test_prepare_batches_propagates_prev_bev_and_resets_between_scenes(tmp_path):
    can_bus = [1, 2, 3, *range(4, 18), 18]
    loader = [
        make_data("scene-a", can_bus),
        make_data("scene-a", can_bus),
        make_data("scene-b", can_bus),
    ]

    saved = prepare_batches(
        loader, FakeSession(), ["bev_embed"], (4, 1, 2), tmp_path, num_samples=3
    )

    assert saved == 3
    with np.load(tmp_path / "batch_0000.npz") as first:
        np.testing.assert_array_equal(first["prev_bev"], np.zeros((4, 1, 2)))
    with np.load(tmp_path / "batch_0001.npz") as second:
        np.testing.assert_array_equal(second["prev_bev"], np.ones((4, 1, 2)))
    with np.load(tmp_path / "batch_0002.npz") as third:
        np.testing.assert_array_equal(third["prev_bev"], np.zeros((4, 1, 2)))


def test_prepare_batches_removes_partial_output_on_short_loader(tmp_path):
    can_bus = [1, 2, 3, *range(4, 18), 18]

    with pytest.raises(RuntimeError, match="Prepared 1 of 2 requested samples"):
        prepare_batches(
            [make_data("scene-a", can_bus)],
            FakeSession(),
            ["bev_embed"],
            (4, 1, 2),
            tmp_path,
            num_samples=2,
        )

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("extra_args", "mode", "calibration_method"),
    [
        ([], "int8", "entropy"),
        (["--quantization-mode", "fp8"], "fp8", "max"),
    ],
)
def test_quantize_defaults(tmp_path, monkeypatch, extra_args, mode, calibration_method):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.touch()
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    plugin_path = tmp_path / "plugin.so"
    plugin_path.touch()
    reader = object()
    reader_args = []
    quantize_args = []

    def make_reader(*args, **kwargs):
        reader_args.append((args, kwargs))
        return reader

    monkeypatch.setattr(quantize_example, "NpzCalibrationReader", make_reader)
    monkeypatch.setattr(quantize_example, "quantize", lambda **kwargs: quantize_args.append(kwargs))

    quantize_example.main(
        [
            "--onnx",
            str(onnx_path),
            "--calibration-dir",
            str(calibration_dir),
            "--trt-plugins",
            str(plugin_path),
            *extra_args,
        ]
    )

    assert reader_args == [((calibration_dir, onnx_path), {"max_batches": 600})]
    assert len(quantize_args) == 1
    assert quantize_args[0]["quantize_mode"] == mode
    assert quantize_args[0]["calibration_data_reader"] is reader
    assert quantize_args[0]["calibration_method"] == calibration_method
    assert quantize_args[0]["output_path"] == str(tmp_path / f"model.{mode}.onnx")

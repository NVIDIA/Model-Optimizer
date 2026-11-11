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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.onnx_ptq.bevformer import prepare_calibration
from examples.onnx_ptq.bevformer import quantize as quantize_example


def make_data(scene_token, position=(1, 2, 3), angle=18):
    can_bus = np.zeros(18, dtype=np.float64)
    can_bus[:3], can_bus[-1] = position, angle
    metadata = {
        "scene_token": scene_token,
        "can_bus": can_bus,
        "lidar2img": [np.eye(4, dtype=np.float64) for _ in range(6)],
    }
    image = SimpleNamespace(numpy=lambda: np.ones((1, 6, 3, 2, 2), dtype=np.float32))
    return {
        "img": [SimpleNamespace(data=[image])],
        "img_metas": [SimpleNamespace(data=[[metadata]])],
    }


def run_batches(monkeypatch, loader, output_dir, num_samples):
    batches = []
    writer = SimpleNamespace(count=0)

    def write(inputs):
        batches.append(inputs)
        writer.count += 1

    writer.write = write
    monkeypatch.setattr(prepare_calibration, "NpzCalibrationWriter", lambda *_: writer)
    monkeypatch.setattr(
        prepare_calibration,
        "run_feedback",
        lambda _runner, _stream, inputs, _name: np.full_like(inputs["prev_bev"], len(batches) + 1),
    )
    runner = SimpleNamespace(resolve_name=lambda name: name)
    saved = prepare_calibration.prepare_batches(
        loader, runner, "model.onnx", (4, 1, 2), output_dir, num_samples, object()
    )
    return saved, batches


def test_prepare_batches_propagates_and_resets_temporal_state(tmp_path, monkeypatch):
    output_dir = tmp_path / "calibration"
    loader = [
        make_data("scene-a"),
        make_data("scene-a", (3, 5, 7), 21),
        make_data("scene-b", (3, 5, 7), 21),
        make_data("scene-b", (4, 6, 8), 22),
    ]

    saved, batches = run_batches(monkeypatch, loader, output_dir, 3)
    assert saved == 3
    assert output_dir.is_dir()
    assert [
        (
            batch["use_prev_bev"].item(),
            batch["prev_bev"][0, 0, 0],
            batch["can_bus"][:3].tolist(),
            batch["can_bus"][-1],
        )
        for batch in batches
    ] == [(0, 0, [0, 0, 0], 0), (1, 1, [2, 3, 4], 3), (0, 0, [0, 0, 0], 0)]


def test_prepare_batches_removes_short_loader_output(tmp_path, monkeypatch):
    output_dir = tmp_path / "calibration"

    with pytest.raises(RuntimeError, match="Prepared 1 of 2 requested samples"):
        run_batches(monkeypatch, [make_data("scene-a")], output_dir, 2)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("extra_args", "mode", "calibration_method"),
    [([], "int8", "entropy"), (["--quantization-mode", "fp8"], "fp8", "max")],
)
def test_quantize_defaults_preserve_source(
    tmp_path, monkeypatch, extra_args, mode, calibration_method
):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"source")
    calibration_dir, plugin = tmp_path / "calibration", tmp_path / "plugin.so"
    calibration_dir.mkdir()
    plugin.touch()
    reader = object()
    call = {}

    monkeypatch.setattr(quantize_example, "NpzCalibrationReader", lambda _path: reader)
    monkeypatch.setattr(quantize_example, "quantize", lambda **kwargs: call.update(kwargs))
    quantize_example.main(
        [
            f"--onnx={onnx_path}",
            f"--calibration-dir={calibration_dir}",
            "--trt-plugins",
            str(plugin),
            *extra_args,
        ]
    )

    temporary_onnx = Path(call.pop("onnx_path"))
    assert onnx_path.read_bytes() == b"source"
    assert temporary_onnx.parent == tmp_path and not temporary_onnx.exists()
    expected = {
        "quantize_mode": mode,
        "calibration_data_reader": reader,
        "calibration_method": calibration_method,
        "calibration_eps": ["trt", "cuda:0", "cpu"],
        "op_types_to_exclude": ["MatMul"],
        "disable_mha_qdq": True,
        "trt_plugins": [str(plugin)],
        "high_precision_dtype": "fp16",
        "output_path": str(tmp_path / f"model.{mode}.onnx"),
    }
    assert call == expected


def test_quantize_rejects_source_as_output(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.touch()
    calibration_dir, plugin = tmp_path / "calibration", tmp_path / "plugin.so"
    calibration_dir.mkdir()
    plugin.touch()

    with pytest.raises(ValueError, match="Output path must differ"):
        quantize_example.main(
            [
                f"--onnx={onnx_path}",
                f"--calibration-dir={calibration_dir}",
                "--trt-plugins",
                str(plugin),
                f"--output={onnx_path}",
            ]
        )

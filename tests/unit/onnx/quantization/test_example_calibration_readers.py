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

import io
import zipfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from examples.onnx_ptq.quantization_utils import (
    FileCalibrationReader,
    NpyCalibrationReader,
    NpzCalibrationReader,
)


class PathCalibrationReader(FileCalibrationReader):
    def load(self, batch_path):
        return batch_path.name


def make_model(tmp_path, element_type=TensorProto.FLOAT):
    model_path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "calibration_reader_test",
        [helper.make_tensor_value_info("input", element_type, [1, 2])],
        [helper.make_tensor_value_info("output", element_type, [1, 2])],
    )
    onnx.save(helper.make_model(graph), model_path)
    return model_path


def write_npz(path, array, truncate=False):
    payload = io.BytesIO()
    np.save(payload, array)
    data = payload.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("input.npy", data[:-1] if truncate else data)


def test_file_reader_sorts_and_caps_batches(tmp_path):
    for name in ("batch_0003.npz", "batch_0001.npz", "batch_0002.npz"):
        (tmp_path / name).touch()

    reader = PathCalibrationReader(tmp_path, "*.npz", max_batches=2)

    assert [path.name for path in reader.batch_paths] == ["batch_0001.npz", "batch_0002.npz"]
    assert [reader.get_next(), reader.get_next(), reader.get_next()] == [
        "batch_0001.npz",
        "batch_0002.npz",
        None,
    ]


def test_readers_load_valid_batches(tmp_path):
    model_path = make_model(tmp_path)
    array = np.ones((1, 2), dtype=np.float32)
    npy_dir = tmp_path / "npy"
    npy_dir.mkdir()
    np.save(npy_dir / "batch_0000.npy", array)
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    write_npz(npz_dir / "batch_0000.npz", array)

    np.testing.assert_array_equal(
        NpyCalibrationReader(npy_dir, model_path, "input").get_first()["input"], array
    )
    np.testing.assert_array_equal(
        NpzCalibrationReader(npz_dir, model_path).get_first()["input"], array
    )


@pytest.mark.parametrize(
    ("array", "error"),
    [
        (np.ones((2, 1), dtype=np.float32), "shape"),
        (np.ones((1, 2), dtype=np.float64), "dtype"),
    ],
)
def test_npy_reader_rejects_wrong_metadata(tmp_path, array, error):
    model_path = make_model(tmp_path)
    calibration_dir = tmp_path / "npy"
    calibration_dir.mkdir()
    np.save(calibration_dir / "batch_0000.npy", array)

    with pytest.raises(ValueError, match=error):
        NpyCalibrationReader(calibration_dir, model_path, "input").get_first()


def test_npy_reader_rejects_truncated_payload_before_loading(tmp_path, monkeypatch):
    model_path = make_model(tmp_path)
    calibration_dir = tmp_path / "npy"
    calibration_dir.mkdir()
    batch_path = calibration_dir / "batch_0000.npy"
    np.save(batch_path, np.ones((1, 2), dtype=np.float32))
    batch_path.write_bytes(batch_path.read_bytes()[:-1])
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("loaded invalid NPY"))

    with pytest.raises(ValueError, match="payload size"):
        NpyCalibrationReader(calibration_dir, model_path, "input").get_first()


@pytest.mark.parametrize(
    ("array", "error"),
    [
        (np.ones((2, 1), dtype=np.float32), "shape"),
        (np.ones((1, 2), dtype=np.float64), "dtype"),
    ],
)
def test_npz_reader_rejects_wrong_metadata(tmp_path, array, error):
    model_path = make_model(tmp_path)
    calibration_dir = tmp_path / "npz"
    calibration_dir.mkdir()
    write_npz(calibration_dir / "batch_0000.npz", array)

    with pytest.raises(ValueError, match=error):
        NpzCalibrationReader(calibration_dir, model_path).get_first()


def test_npz_reader_rejects_truncated_payload_before_loading(tmp_path, monkeypatch):
    model_path = make_model(tmp_path)
    calibration_dir = tmp_path / "npz"
    calibration_dir.mkdir()
    write_npz(
        calibration_dir / "batch_0000.npz",
        np.ones((1, 2), dtype=np.float32),
        truncate=True,
    )
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("loaded invalid NPZ"))

    with pytest.raises(ValueError, match="payload size"):
        NpzCalibrationReader(calibration_dir, model_path).get_first()


def test_npz_reader_requires_safe_cast_allowlist(tmp_path):
    model_path = make_model(tmp_path, TensorProto.DOUBLE)
    calibration_dir = tmp_path / "npz"
    calibration_dir.mkdir()
    write_npz(calibration_dir / "batch_0000.npz", np.ones((1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="dtype"):
        NpzCalibrationReader(calibration_dir, model_path).get_first()

    batch = NpzCalibrationReader(
        calibration_dir, model_path, safe_cast_inputs=("input",)
    ).get_first()
    assert batch["input"].dtype == np.float64

    model_path = make_model(tmp_path)
    write_npz(calibration_dir / "batch_0000.npz", np.ones((1, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="dtype"):
        NpzCalibrationReader(calibration_dir, model_path, safe_cast_inputs=("input",)).get_first()

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
import os
import sys
import zipfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from modelopt.onnx.quantization.calib_utils import (
    NpyCalibrationReader,
    NpzCalibrationReader,
    create_directory_calibration_reader,
)


def make_model(
    tmp_path,
    input_specs=(("input", TensorProto.FLOAT, (1, 2)),),
    initializer_input=False,
):
    model_path = tmp_path / "model.onnx"
    inputs = [helper.make_tensor_value_info(*spec) for spec in input_specs]
    output = helper.make_tensor_value_info("output", input_specs[0][1], input_specs[0][2])
    initializers = []
    if initializer_input:
        weight = helper.make_tensor("weight", TensorProto.FLOAT, (1, 2), (1.0, 1.0))
        inputs.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, (1, 2)))
        initializers.append(weight)
        node = helper.make_node("Add", [input_specs[0][0], "weight"], ["output"])
    elif len(input_specs) == 1:
        node = helper.make_node("Identity", [input_specs[0][0]], ["output"])
    else:
        node = helper.make_node("Add", [input_specs[0][0], input_specs[1][0]], ["output"])
    graph = helper.make_graph(
        [node], "calibration_reader_test", inputs, [output], initializer=initializers
    )
    onnx.save(helper.make_model(graph), model_path)
    return model_path


def write_npz(path, arrays, truncate=None):
    if isinstance(arrays, np.ndarray):
        arrays = {"input": arrays}
    with zipfile.ZipFile(path, "w") as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.save(payload, array)
            data = payload.getvalue()
            archive.writestr(f"{name}.npy", data[:-1] if name == truncate else data)


def run_cli(monkeypatch, *args):
    import modelopt.onnx.quantization.__main__ as quantization_cli

    captured = {}

    def fake_quantize(onnx_path, **kwargs):
        captured["onnx_path"] = onnx_path
        captured.update(kwargs)

    monkeypatch.setattr(quantization_cli, "quantize", fake_quantize)
    monkeypatch.setattr(sys, "argv", ["modelopt.onnx.quantization", *map(str, args)])
    quantization_cli.main()
    return captured


def test_reader_sorts_and_caps_batches(tmp_path):
    model_path = make_model(tmp_path)
    for index in (3, 1, 2):
        write_npz(
            tmp_path / f"batch_{index:04d}.npz",
            np.full((1, 2), index, dtype=np.float32),
        )

    reader = NpzCalibrationReader(tmp_path, model_path, max_batches=2)

    assert [path.name for path in reader.batch_paths] == ["batch_0001.npz", "batch_0002.npz"]
    assert reader.get_next()["input"][0, 0] == 1
    assert reader.get_next()["input"][0, 0] == 2
    assert reader.get_next() is None


def test_readers_load_valid_batches(tmp_path, monkeypatch):
    model_path = make_model(tmp_path)
    array = np.ones((1, 2), dtype=np.float32)
    np.save(tmp_path / "batch_0000.npy", array)
    write_npz(tmp_path / "batch_0000.npz", array)
    monkeypatch.delattr(os, "O_NONBLOCK", raising=False)

    np.testing.assert_array_equal(
        NpyCalibrationReader(tmp_path, model_path, "input").get_first()["input"], array
    )
    np.testing.assert_array_equal(
        NpzCalibrationReader(tmp_path, model_path).get_first()["input"], array
    )


def test_npz_reader_loads_standard_multi_input_archive(tmp_path):
    model_path = make_model(
        tmp_path,
        (
            ("input", TensorProto.FLOAT, (1, 2)),
            ("other", TensorProto.FLOAT, (1, 2)),
        ),
    )
    arrays = {
        "input": np.ones((1, 2), dtype=np.float32),
        "other": np.zeros((1, 2), dtype=np.float32),
    }
    np.savez(tmp_path / "batch_0000.npz", **arrays)

    batch = NpzCalibrationReader(tmp_path, model_path).get_first()

    assert batch.keys() == arrays.keys()
    for name in arrays:
        np.testing.assert_array_equal(batch[name], arrays[name])


@pytest.mark.parametrize(
    ("array", "error"),
    [
        (np.ones((2, 1), dtype=np.float32), "shape"),
        (np.ones((1, 2), dtype=np.float64), "dtype"),
    ],
)
def test_npy_reader_rejects_wrong_metadata(tmp_path, array, error):
    model_path = make_model(tmp_path)
    np.save(tmp_path / "batch_0000.npy", array)

    with pytest.raises(ValueError, match=error):
        NpyCalibrationReader(tmp_path, model_path, "input").get_first()


def test_npy_reader_rejects_truncated_payload_before_loading(tmp_path, monkeypatch):
    model_path = make_model(tmp_path)
    batch_path = tmp_path / "batch_0000.npy"
    np.save(batch_path, np.ones((1, 2), dtype=np.float32))
    batch_path.write_bytes(batch_path.read_bytes()[:-1])
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("loaded invalid NPY"))

    with pytest.raises(ValueError, match="payload size"):
        NpyCalibrationReader(tmp_path, model_path, "input").get_first()


@pytest.mark.parametrize(
    ("array", "error"),
    [
        (np.ones((2, 1), dtype=np.float32), "shape"),
        (np.ones((1, 2), dtype=np.float64), "dtype"),
    ],
)
def test_npz_reader_rejects_wrong_metadata(tmp_path, array, error):
    model_path = make_model(tmp_path)
    write_npz(tmp_path / "batch_0000.npz", array)

    with pytest.raises(ValueError, match=error):
        NpzCalibrationReader(tmp_path, model_path).get_first()


def test_npz_reader_rejects_truncated_payload_before_loading(tmp_path, monkeypatch):
    model_path = make_model(tmp_path)
    write_npz(
        tmp_path / "batch_0000.npz",
        np.ones((1, 2), dtype=np.float32),
        truncate="input",
    )
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("loaded invalid NPZ"))

    with pytest.raises(ValueError, match="payload size"):
        NpzCalibrationReader(tmp_path, model_path).get_first()


def test_npz_reader_requires_safe_cast_allowlist(tmp_path):
    model_path = make_model(tmp_path, (("input", TensorProto.DOUBLE, (1, 2)),))
    batch_path = tmp_path / "batch_0000.npz"
    write_npz(batch_path, np.ones((1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="dtype"):
        NpzCalibrationReader(tmp_path, model_path).get_first()

    batch = NpzCalibrationReader(tmp_path, model_path, safe_cast_inputs=("input",)).get_first()
    assert batch["input"].dtype == np.float64

    model_path = make_model(tmp_path)
    write_npz(batch_path, np.ones((1, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="dtype"):
        NpzCalibrationReader(tmp_path, model_path, safe_cast_inputs=("input",)).get_first()


def test_directory_reader_rejects_empty_and_mixed_formats(tmp_path):
    model_path = make_model(tmp_path)

    with pytest.raises(ValueError, match="No NPY or NPZ"):
        create_directory_calibration_reader(tmp_path, model_path)

    np.save(tmp_path / "batch.npy", np.ones((1, 2), dtype=np.float32))
    write_npz(tmp_path / "batch.npz", np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="mixes NPY and NPZ"):
        create_directory_calibration_reader(tmp_path, model_path)


def test_directory_reader_rejects_invalid_batch_cap(tmp_path):
    model_path = make_model(tmp_path)
    np.save(tmp_path / "batch.npy", np.ones((1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="max_batches must be positive"):
        create_directory_calibration_reader(tmp_path, model_path, max_batches=0)


def test_npy_reader_infers_only_external_input(tmp_path):
    model_path = make_model(tmp_path, initializer_input=True)
    array = np.ones((1, 2), dtype=np.float32)
    np.save(tmp_path / "batch.npy", array)

    reader = create_directory_calibration_reader(tmp_path, model_path)

    np.testing.assert_array_equal(reader.get_first()["input"], array)


def test_npy_reader_rejects_multi_input_model(tmp_path):
    model_path = make_model(
        tmp_path,
        (
            ("input", TensorProto.FLOAT, (1, 2)),
            ("other", TensorProto.FLOAT, (1, 2)),
        ),
    )
    np.save(tmp_path / "batch.npy", np.ones((1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="exactly one input"):
        create_directory_calibration_reader(tmp_path, model_path)


def test_directory_reader_uses_calibration_shapes(tmp_path):
    model_path = make_model(tmp_path, (("input", TensorProto.FLOAT, ("batch", 2)),))
    np.save(tmp_path / "batch.npy", np.ones((3, 2), dtype=np.float32))

    reader = create_directory_calibration_reader(
        tmp_path, model_path, calibration_shapes="input:3x2"
    )
    assert reader.get_first()["input"].shape == (3, 2)

    with pytest.raises(ValueError, match="shape"):
        create_directory_calibration_reader(
            tmp_path, model_path, calibration_shapes="input:4x2"
        ).get_first()


def test_cli_streams_directory_calibration_batches(tmp_path, monkeypatch):
    model_path = make_model(tmp_path)
    for index in range(3):
        np.save(
            tmp_path / f"batch_{index:04d}.npy",
            np.full((1, 2), index, dtype=np.float32),
        )
    captured = run_cli(
        monkeypatch,
        "--onnx_path",
        model_path,
        "--calibration_data_path",
        tmp_path,
        "--max_calibration_batches",
        2,
    )

    reader = captured["calibration_data_reader"]
    assert captured["onnx_path"] == str(model_path)
    assert captured["calibration_data"] is None
    assert isinstance(reader, NpyCalibrationReader)
    assert [path.name for path in reader.batch_paths] == ["batch_0000.npy", "batch_0001.npy"]


def test_cli_preserves_single_file_calibration(tmp_path, monkeypatch):
    model_path = make_model(tmp_path)
    calibration_path = tmp_path / "calibration.npy"
    calibration_data = np.ones((2, 2), dtype=np.float32)
    np.save(calibration_path, calibration_data)
    captured = run_cli(
        monkeypatch,
        "--onnx_path",
        model_path,
        "--calibration_data_path",
        calibration_path,
    )

    np.testing.assert_array_equal(captured["calibration_data"], calibration_data)
    assert captured["calibration_data_reader"] is None

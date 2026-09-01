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
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from examples.onnx_ptq import quantize_vovnet
from examples.onnx_ptq.quantization_utils import (
    NpzCalibrationReader,
    NpzCalibrationWriter,
    find_vovnet_nodes_to_exclude,
)


def make_calibration_model(tmp_path, image_shape=(1, 2)):
    model_path = tmp_path / "model.onnx"
    inputs = [
        helper.make_tensor_value_info("image", TensorProto.FLOAT, image_shape),
        helper.make_tensor_value_info("index", TensorProto.INT64, (1,)),
    ]
    outputs = [
        helper.make_tensor_value_info("image_out", TensorProto.FLOAT, image_shape),
        helper.make_tensor_value_info("index_out", TensorProto.INT64, (1,)),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["image"], ["image_out"]),
            helper.make_node("Identity", ["index"], ["index_out"]),
        ],
        "calibration",
        inputs,
        outputs,
    )
    onnx.save(helper.make_model(graph), model_path)
    return model_path


def test_npz_calibration_round_trip_and_rewind(tmp_path):
    model_path = make_calibration_model(tmp_path)
    writer = NpzCalibrationWriter(tmp_path / "batches", model_path)
    writer.write(
        {
            "image": np.array([[1, 2]], dtype=np.float64),
            "index": np.array([3], dtype=np.int32),
        }
    )
    writer.write(
        {
            "image": np.array([[4, 5]], dtype=np.float64),
            "index": np.array([6], dtype=np.int32),
        }
    )

    assert writer.count == 2
    assert [path.name for path in sorted((tmp_path / "batches").glob("*.npz"))] == [
        "batch_0000.npz",
        "batch_0001.npz",
    ]
    reader = NpzCalibrationReader(tmp_path / "batches")
    first = reader.get_first()
    assert first["image"].dtype == np.float32
    assert first["index"].dtype == np.int64
    np.testing.assert_array_equal(reader.get_next()["image"], [[1, 2]])
    np.testing.assert_array_equal(reader.get_next()["image"], [[4, 5]])
    assert reader.get_next() is None
    reader.rewind()
    np.testing.assert_array_equal(reader.get_next()["image"], [[1, 2]])

    with pytest.raises(FileExistsError, match="already contains"):
        NpzCalibrationWriter(tmp_path / "batches", model_path)


@pytest.mark.parametrize(
    "values",
    [
        {"image": np.ones((1, 2))},
        {
            "image": np.ones((1, 2)),
            "index": np.ones((1,)),
            "unexpected": np.ones((1,)),
        },
    ],
)
def test_npz_writer_rejects_wrong_input_names(tmp_path, values):
    writer = NpzCalibrationWriter(tmp_path / "batches", make_calibration_model(tmp_path))

    with pytest.raises(ValueError, match="Calibration input mismatch"):
        writer.write(values)


@pytest.mark.parametrize(
    "image_shape",
    [(2,), (1, 3)],
    ids=("wrong-rank", "wrong-static-dimension"),
)
def test_npz_writer_rejects_wrong_input_shape(tmp_path, image_shape):
    writer = NpzCalibrationWriter(tmp_path / "batches", make_calibration_model(tmp_path))

    with pytest.raises(ValueError, match="Calibration input 'image' has shape"):
        writer.write(
            {
                "image": np.ones(image_shape),
                "index": np.ones((1,)),
            }
        )


def test_npz_writer_accepts_dynamic_input_shape(tmp_path):
    writer = NpzCalibrationWriter(
        tmp_path / "batches", make_calibration_model(tmp_path, image_shape=("batch", 2))
    )

    writer.write(
        {
            "image": np.ones((3, 2)),
            "index": np.ones((1,)),
        }
    )

    assert NpzCalibrationReader(tmp_path / "batches").get_first()["image"].shape == (3, 2)


def test_find_vovnet_nodes_to_exclude(tmp_path):
    model_path = tmp_path / "vovnet.onnx"
    nodes = [
        helper.make_node("Identity", ["branch_out"], ["tail_out"], name="tail"),
        helper.make_node("Identity", ["osa_out"], ["osa_tail_out"], name="osa_tail"),
        helper.make_node("Identity", ["input"], ["osa_out"], name="backbone.OSA4_5"),
        helper.make_node("Identity", ["lateral_out"], ["branch_out"], name="branch"),
        helper.make_node("Identity", ["input"], ["lateral_out"], name="neck.lateral_convs.0"),
    ]
    graph = helper.make_graph(
        nodes,
        "vovnet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1,))],
        [
            helper.make_tensor_value_info("tail_out", TensorProto.FLOAT, (1,)),
            helper.make_tensor_value_info("osa_tail_out", TensorProto.FLOAT, (1,)),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    assert find_vovnet_nodes_to_exclude(model_path) == [
        r"^backbone\.OSA4_5$",
        r"^branch$",
        r"^tail$",
    ]


def test_quantize_vovnet_preserves_source_model(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"])],
        "external_data",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1,))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1,))],
        [numpy_helper.from_array(np.ones(1, dtype=np.float32), name="weight")],
    )
    onnx.save_model(
        helper.make_model(graph),
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    weights_path = tmp_path / "weights.bin"
    source_bytes = model_path.read_bytes()
    weight_bytes = weights_path.read_bytes()
    temporary_paths = []

    def fake_quantize(**kwargs):
        temporary_path = Path(kwargs["onnx_path"])
        onnx.load(temporary_path, load_external_data=True)
        temporary_paths.append(temporary_path)
        temporary_path.write_bytes(b"mutated")
        Path(kwargs["output_path"]).write_bytes(b"quantized")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        quantize_vovnet,
        "parse_args",
        lambda: SimpleNamespace(
            onnx_path=str(model_path),
            calibration_dir=tmp_path,
            precision="int8",
            output="quantized.onnx",
        ),
    )
    monkeypatch.setattr(quantize_vovnet, "find_vovnet_nodes_to_exclude", lambda _: [])
    monkeypatch.setattr(quantize_vovnet, "NpzCalibrationReader", lambda _: object())
    monkeypatch.setattr(quantize_vovnet, "quantize", fake_quantize)

    quantize_vovnet.main()

    assert model_path.read_bytes() == source_bytes
    assert weights_path.read_bytes() == weight_bytes
    assert not temporary_paths[0].exists()
    assert (tmp_path / "quantized.onnx").read_bytes() == b"quantized"

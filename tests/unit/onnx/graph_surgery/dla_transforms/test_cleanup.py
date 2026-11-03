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

"""End-to-end cleanup checks for public names and shared layout views."""

import tracemalloc
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms.cleanup import deduplicate_initializers
from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.stages import cleanup
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible


@pytest.mark.parametrize(
    ("shared", "preserve_incoming"), [(False, False), (True, False), (False, True)]
)
def test_cleanup_preserves_public_names_and_shared_reshape_consumers(shared, preserve_incoming):
    nodes = [
        helper.make_node("Identity", ["x"], ["incoming"]),
        helper.make_node("Reshape", ["incoming", "first"], ["view"]),
        helper.make_node("Reshape", ["view", "last"], ["restored"]),
        helper.make_node("Identity", ["restored"], ["y"]),
    ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])]
    if shared:
        nodes.append(helper.make_node("Abs", ["view"], ["branch"]))
        outputs.append(helper.make_tensor_value_info("branch", TensorProto.FLOAT, [3, 2]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "cleanup",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            outputs,
            [
                numpy_helper.from_array(np.asarray([3, 2], np.int64), "first"),
                numpy_helper.from_array(np.asarray([2, 3], np.int64), "last"),
                numpy_helper.from_array(np.asarray([99], np.int64), "unused"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    working = ir.serde.deserialize_model(model)
    identities = {next(iter(working.graph))} if preserve_incoming else None
    context = DlaGraph(working)
    initializers, inputs = context.initializers, context.inputs
    count = cleanup(context, preserve_identities=identities)
    result = ir.serde.serialize_model(working)
    onnx.checker.check_model(result, full_check=True)
    assert result.graph.input[0].name == "x"
    assert [output.name for output in result.graph.output] == [output.name for output in outputs]
    assert "unused" not in {value.name for value in result.graph.initializer}
    assert "unused" not in initializers
    assert inputs[0].name == "x"
    assert count == (0 if shared else 1) + (0 if preserve_incoming else 1)
    assert sum(node.op_type == "Identity" for node in result.graph.node) == (
        2 if preserve_incoming else 1
    )
    assert sum(node.op_type == "Reshape" for node in result.graph.node) == (2 if shared else 0)
    feeds = {"x": np.arange(-3, 3, dtype=np.float32).reshape(2, 3)}
    for original, transformed in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(result).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(original, transformed)


@pytest.mark.parametrize("fp16", [False, True])
def test_final_cleanup_deduplicates_initializers_after_precision(fp16):
    """Merge fixed payloads after rounding while preserving interfaces and exact bits."""
    arrays = {
        "weight": np.array([1, 1], np.float32),
        "duplicate": np.array([1, 1], np.float32),
        "rounded": np.array([1.0001, 1.0001], np.float32),
        "scalar": np.array(1, np.float32),
        "vector": np.array([1], np.float32),
        "positive_zero": np.array(0, np.float32),
        "negative_zero": np.array(-0.0, np.float32),
        "nan_a": np.array(0x7FC00001, np.uint32).view(np.float32),
        "nan_b": np.array(0x7FC00002, np.uint32).view(np.float32),
    }
    nodes = [
        helper.make_node("Add", ["x", name], [f"out_{name}"], name=f"add_{name}")
        for name in (*arrays, "overridable")
    ]
    nodes.append(helper.make_node("Identity", ["integer_bits"], ["integers"]))
    outputs = [
        helper.make_tensor_value_info(f"out_{name}", TensorProto.FLOAT, [2])
        for name in (*arrays, "overridable")
    ]
    outputs.extend(
        [
            helper.make_tensor_value_info("integers", TensorProto.INT32, [2]),
            helper.make_tensor_value_info("exposed", TensorProto.FLOAT16, [2]),
        ]
    )
    initializers = [numpy_helper.from_array(array, name) for name, array in arrays.items()]
    initializers.extend(
        [
            numpy_helper.from_array(np.array([1, 1], np.float32), "overridable"),
            numpy_helper.from_array(np.array([1, 1], np.float16), "exposed"),
            numpy_helper.from_array(np.array([0x3F800000, 0x3F800000], np.int32), "integer_bits"),
        ]
    )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "deduplication",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
                helper.make_tensor_value_info("overridable", TensorProto.FLOAT, [2]),
            ],
            outputs,
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    saved = model.SerializeToString()
    result = _transform_make_dla_compatible(model, force_fp16_activations=fp16, verbose=False)
    assert model.SerializeToString() == saved
    add_inputs = {
        node.name.removeprefix("add_"): node.input[1]
        for node in result.graph.node
        if node.op_type == "Add"
    }
    assert add_inputs["weight"] == add_inputs["duplicate"]
    assert (add_inputs["weight"] == add_inputs["rounded"]) == fp16
    assert len({add_inputs[name] for name in ("weight", "scalar", "vector")}) == 3
    assert add_inputs["positive_zero"] != add_inputs["negative_zero"]
    assert (add_inputs["nan_a"] == add_inputs["nan_b"]) == fp16
    kept = {value.name: value for value in result.graph.initializer}
    assert kept["integer_bits"].data_type == TensorProto.INT32
    assert {"exposed", "overridable"} <= kept.keys()
    assert add_inputs["weight"] not in {"overridable", "exposed", "integer_bits"}
    assert list(result.graph.input) == list(model.graph.input)
    assert list(result.graph.output) == list(model.graph.output)
    feeds = {"x": np.array([0, 0.5], np.float32), "overridable": np.array([2, 3], np.float32)}
    for expected, actual in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(result).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "kind",
    [
        "noncontiguous",
        "strings",
        "hash_collision",
        "int4",
        "uint4",
        "int4_proto",
        "uint4_proto",
        "int4_int32",
        "uint4_external",
        "int4_collision",
    ],
)
def test_initializer_deduplication_preserves_storage_contents(kind, monkeypatch, tmp_path):
    """Reuse logical contents across storage forms without fully unpacking INT4 weights."""
    packed = kind.startswith(("int4", "uint4"))
    from_proto = kind.endswith(("_proto", "_int32"))
    measure_memory = kind in {"int4", "uint4"}
    if kind in {"noncontiguous", "hash_collision"}:
        source = np.arange(6, dtype=np.float32).reshape(2, 3).T
        tensors = [ir.tensor(source), ir.tensor(source.copy()), ir.tensor(source + 1)]
        assert not tensors[0].numpy().flags.c_contiguous
    elif packed:
        dtype = ir.DataType.UINT4 if kind.startswith("uint4") else ir.DataType.INT4
        source = np.arange(4 * 1024 * 1024 + 1 if measure_memory else 65537, dtype=np.uint8)
        source &= 0x0F
        storage = source[::2].copy()
        storage[:-1] |= source[1::2] << 4
        padding = storage.copy()
        padding[-1] |= 0xA0
        different = storage.copy()
        different[-1] ^= 1
        tensors = [
            ir.PackedTensor(storage, dtype, shape=source.shape),
            ir.PackedTensor(padding, dtype, shape=source.shape),
            ir.tensor((source | 0xB0).view(dtype.numpy())),
            ir.PackedTensor(different, dtype, shape=source.shape),
        ]
        if kind.endswith("_external"):
            for i in (0, 1, 3):
                filename = f"constant_{i}.bin"
                (tmp_path / filename).write_bytes(b"pad" + tensors[i].numpy_packed().tobytes())
                tensors[i] = ir.ExternalTensor(
                    filename,
                    offset=3,
                    length=storage.nbytes,
                    dtype=dtype,
                    shape=ir.Shape(source.shape),
                    name=f"constant_{i}",
                    base_dir=tmp_path,
                )
    else:
        tensors = [
            ir.StringTensor([b"a", b"b\x00"], shape=ir.Shape([2])),
            ir.StringTensor([b"a", b"b\x00"], shape=ir.Shape([2])),
            ir.StringTensor([b"a\x00", b"b"], shape=ir.Shape([2])),
        ]
    values = [
        ir.Value(
            name=f"constant_{i}",
            const_value=tensor,
            shape=tensor.shape,
            type=ir.TensorType(tensor.dtype),
        )
        for i, tensor in enumerate(tensors)
    ]
    nodes = [
        ir.Node(
            "",
            "Identity",
            [value],
            outputs=[ir.Value(name=f"output_{i}", shape=value.shape, type=value.type)],
        )
        for i, value in enumerate(values)
    ]
    working = ir.Model(
        ir.Graph(
            [],
            [node.outputs[0] for node in nodes],
            nodes=nodes,
            initializers=values,
            opset_imports={"": 21},
            name="storage_deduplication",
        ),
        ir_version=10,
    )
    if from_proto:
        serialized = ir.serde.serialize_model(working)
        assert serialized.graph.initializer[0].raw_data != serialized.graph.initializer[1].raw_data
        if kind.endswith("_int32"):
            for tensor in serialized.graph.initializer:
                tensor.int32_data.extend(tensor.raw_data)
                tensor.ClearField("raw_data")
        working = ir.serde.deserialize_model(serialized)
        nodes = list(working.graph)
    with monkeypatch.context() as patch:
        if kind == "hash_collision" or kind.endswith("_collision"):
            patch.setattr(
                "modelopt.onnx.graph_surgery.dla_transforms.cleanup.hashlib.sha256",
                lambda: SimpleNamespace(update=lambda data: None, digest=lambda: b"collision"),
            )
        if measure_memory:
            tracemalloc.start()
        try:
            assert deduplicate_initializers(working) == (2 if packed else 1)
            if measure_memory:
                _, peak = tracemalloc.get_traced_memory()
                assert peak < 2 * 1024 * 1024
        finally:
            if measure_memory:
                tracemalloc.stop()
    assert nodes[0].inputs[0] is nodes[1].inputs[0]
    if packed:
        assert nodes[0].inputs[0] is nodes[2].inputs[0]
    assert nodes[0].inputs[0] is not nodes[-1].inputs[0]
    result = ir.serde.serialize_model(working)
    if kind.endswith("_external"):
        onnx.external_data_helper.load_external_data_for_model(result, str(tmp_path))
    onnx.checker.check_model(result, full_check=True)
    if kind == "strings":
        assert list(result.graph.initializer[0].string_data) == [b"a", b"b\x00"]
        assert list(result.graph.initializer[1].string_data) == [b"a\x00", b"b"]

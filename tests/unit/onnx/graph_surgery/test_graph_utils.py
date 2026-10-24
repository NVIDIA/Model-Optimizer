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

"""Shared graph queries, shape resolution, and initializer contracts."""

import copy
import io
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.utils.graph_utils import (
    GraphCache,
    _shape_before_boundary_promotion,
    add_initializer,
    batch_replace_nodes,
    calculate_clip_range,
    check_to_apply_transpose,
    infer_shapes,
    insert_nodes_at_position,
    parse_qdq_quantized_dtype_list,
    resolve_reshape_shape,
    save_onnx_model,
)


@pytest.mark.parametrize(
    ("source_dtype", "requested_dtype"),
    [
        (np.float32, None),
        (np.float32, TensorProto.FLOAT16),
        (np.float64, TensorProto.FLOAT),
        (np.int64, TensorProto.INT32),
    ],
)
def test_add_initializer_honors_requested_dtype(source_dtype, requested_dtype):
    """Store and reuse initializers using the requested dtype and matching metadata."""
    dtype = TensorProto.FLOAT16 if requested_dtype is None else requested_dtype
    data = np.array([1, 2], dtype=source_dtype)
    graph = helper.make_graph([], "initializer_dtype", [], [])
    options = {} if requested_dtype is None else {"dtype": requested_dtype}
    assert add_initializer(graph, "weight", data, **options) == "weight"
    assert graph.initializer[0].data_type == dtype
    assert graph.value_info[0].type.tensor_type.elem_type == dtype
    np.testing.assert_array_equal(
        numpy_helper.to_array(graph.initializer[0]),
        data.astype(helper.tensor_dtype_to_np_dtype(dtype)),
    )
    before = graph.SerializeToString()
    assert add_initializer(graph, "weight", data, **options) == "weight"
    assert graph.SerializeToString() == before


def test_add_initializer_rejects_conflicting_requested_dtype():
    """Reject initializer reuse when its type conflicts with the requested dtype."""
    data = np.array([1], dtype=np.float32)
    graph = helper.make_graph([], "dtype_collision", [], [], [numpy_helper.from_array(data, "w")])
    before = graph.SerializeToString()
    with pytest.raises(ValueError, match="Conflicting generated initializer"):
        add_initializer(graph, "w", data, TensorProto.FLOAT16, with_value_info=False)
    assert graph.SerializeToString() == before


def test_add_initializer_preserves_bfloat16_raw_bits():
    """Preserve BFLOAT16 raw bits when adding and reusing an initializer."""
    data = np.array([0x3F80, 0x8000, 0x7FC1], dtype=np.uint16).view(np.int16)
    graph = helper.make_graph([], "bfloat16_bits", [], [])
    add_initializer(graph, "w", data, TensorProto.BFLOAT16)
    assert graph.initializer[0].data_type == TensorProto.BFLOAT16
    assert graph.initializer[0].raw_data == data.tobytes()
    before = graph.SerializeToString()
    add_initializer(graph, "w", data, TensorProto.BFLOAT16)
    assert graph.SerializeToString() == before


@pytest.mark.parametrize(
    ("values", "dtype", "onnx_dtype"),
    [([3], np.int64, TensorProto.INT64), ([2], np.int32, TensorProto.INT32)],
)
def test_generated_initializer_collision_leaves_graph_unchanged(values, dtype, onnx_dtype):
    """Reject conflicting initializer values or types without mutating the graph."""
    graph = helper.make_graph([], "collision", [], [])
    graph.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"))
    before = graph.SerializeToString()
    with pytest.raises(ValueError, match="Conflicting generated initializer"):
        add_initializer(
            graph, "axes", np.asarray(values, dtype=dtype), onnx_dtype, with_value_info=False
        )
    assert graph.SerializeToString() == before


def test_original_boundary_shape_survives_promotion():
    """Recover original input shapes after the generated boundary Unsqueeze."""
    graph = helper.make_graph(
        [helper.make_node("Unsqueeze", ["x", "axes"], ["x_unsqueeze_input"])],
        "boundary",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("x_unsqueeze_input", TensorProto.FLOAT, [1, 2, 3, 4])],
        [numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")],
    )
    cache = GraphCache(graph)
    assert _shape_before_boundary_promotion(cache, "x_unsqueeze_input") == [2, 3, 4]
    assert cache.get_shape("x_unsqueeze_input") == [1, 2, 3, 4]


@pytest.mark.parametrize("target", [[-1, -1], [4, 2], [-1, 4], [0, 0, 0]])
def test_invalid_reshape_does_not_change_requested_shape(target):
    """Leave the requested shape array unchanged when reshape validation fails."""
    values = np.array(target, dtype=np.int64)
    with pytest.raises(ValueError):
        resolve_reshape_shape(values, [2, 3])
    np.testing.assert_array_equal(values, target)


def test_symbolic_shape_is_not_reported_as_a_concrete_prefix():
    """Report symbolic dimensions as unknown rather than a concrete partial shape."""
    graph = helper.make_graph(
        [], "symbolic", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, "N", 4])], []
    )
    assert GraphCache(graph).get_shape("x") is None


def test_cache_rebuild_refreshes_initializer_values():
    """Refresh cached values, shapes, and consumer positions after graph edits."""
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "cache",
        [helper.make_tensor_value_info("x", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [2])],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), "a")],
    )
    cache = GraphCache(graph)
    np.testing.assert_array_equal(cache.get_init_array("a"), [1])
    graph.initializer[0].CopyFrom(numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "a"))
    del graph.node[:]
    graph.node.extend(
        [
            helper.make_node("Add", ["x", "a"], ["middle"]),
            helper.make_node("Add", ["middle", "x"], ["y"]),
        ]
    )
    cache.rebuild()
    np.testing.assert_array_equal(cache.get_init_array("a"), [2, 3])
    assert cache.get_shape("a") == [2]
    assert cache.get_dtype("a") == TensorProto.INT64
    assert cache.get_producer("y") is graph.node[1]
    assert cache.get_consumers("x") == [(graph.node[0], 0), (graph.node[1], 1)]
    assert cache.nodes_by_op("Identity") == []
    assert cache.nodes_by_op("Add") == list(graph.node)
    onnx.checker.check_model(helper.make_model(graph))


@pytest.mark.parametrize(("target", "source"), [([-1, 0], [0, 3]), ([-2, -3], [2, 3])])
def test_invalid_reshape_dimensions_fail_clearly(target, source):
    """Reject empty source dimensions and invalid negative reshape dimensions."""
    with pytest.raises(ValueError):
        resolve_reshape_shape(np.array(target), source)


def test_generated_initializer_collision_preserves_signed_zero():
    """Reject initializer reuse when zero values have different sign bits."""
    graph = helper.make_graph(
        [],
        "signed_zero",
        [],
        [],
        [numpy_helper.from_array(np.array([-0.0], dtype=np.float32), "a")],
    )
    with pytest.raises(ValueError, match="Conflicting generated initializer"):
        add_initializer(
            graph,
            "a",
            np.asarray([0.0], dtype=np.float32),
            TensorProto.FLOAT,
            with_value_info=False,
        )


@pytest.mark.parametrize("replacement_count", [0, 1, 2])
def test_node_edits_preserve_topology_and_boundary_values(replacement_count):
    """Keep graph edits topologically valid and preserve boundary values."""
    original = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["middle"]),
                helper.make_node("Identity", ["middle"], ["y"]),
            ],
            "topology",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    candidate = copy.deepcopy(original)
    final_output = "middle" if replacement_count else "restored"
    nodes = [
        helper.make_node("Neg", ["x"], ["negative"]),
        helper.make_node("Neg", ["negative"], [final_output]),
    ]
    existing = candidate.graph.node[0]
    if replacement_count:
        replacements = {0: ([existing], nodes)}
        if replacement_count == 2:
            replacements[1] = (
                [candidate.graph.node[1]],
                [
                    helper.make_node("Neg", ["middle"], ["second_negative"]),
                    helper.make_node("Neg", ["second_negative"], ["y"]),
                ],
            )
        batch_replace_nodes(candidate.graph, replacements)
    else:
        existing.input[0] = final_output
        insert_nodes_at_position(candidate.graph, nodes, existing)
    onnx.checker.check_model(candidate)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    results = [
        ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        ).run(None, feeds)[0]
        for model in (original, candidate)
    ]
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize(("target", "expected"), [([0, -1], [2, 12]), ([6, 4], [6, 4])])
def test_reshape_resolution_preserves_elements_and_requested_shape(target, expected):
    """Resolve copied and inferred dimensions without changing the requested target."""
    requested = np.array(target, dtype=np.int64)
    resolved = resolve_reshape_shape(requested, [2, 3, 4])
    assert resolved == expected
    np.testing.assert_array_equal(requested, target)


def test_generated_names_do_not_shadow_existing_values_or_other_reservations():
    """Reserve fresh names across graph values, node names, and earlier reservations."""
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "x_1"], ["x_3"], name="x_2"),
            helper.make_node("Identity", ["x_3"], ["x_4"]),
        ],
        "names",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("x_4", TensorProto.FLOAT, [2])],
        [numpy_helper.from_array(np.ones(2, dtype=np.float32), "x_1")],
    )
    cache = GraphCache(graph)
    first, second = cache.unique_name("x"), cache.unique_name("x")
    assert (first, second) == ("x_5", "x_6")
    assert cache.unique_name("adapter") == "adapter"
    tail = graph.node[-1]
    insert_nodes_at_position(graph, [helper.make_node("Identity", ["x_3"], [first])], tail)
    tail.input[0] = first
    onnx.checker.check_model(helper.make_model(graph))


def test_shape_inference_refreshes_intermediate_and_output_metadata_in_place():
    """Propagate an input shape edit through real symbolic inference without replacing the model."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "weight"], ["hidden"]),
                helper.make_node("Relu", ["hidden"], ["y"]),
            ],
            "inference",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, None])],
            [numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), "weight")],
            value_info=[helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [None, None])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    cache = GraphCache(model.graph)
    assert cache.update_shape("x", [4, 2])
    nodes_before = [node.SerializeToString() for node in model.graph.node]
    weight_before = model.graph.initializer[0].SerializeToString()
    assert infer_shapes(model) is model
    cache.rebuild()
    assert cache.get_shape("hidden") == [4, 3]
    assert cache.get_shape("y") == [4, 3]
    assert cache.get_dtype("y") == TensorProto.FLOAT
    assert [node.SerializeToString() for node in model.graph.node] == nodes_before
    assert model.graph.initializer[0].SerializeToString() == weight_before
    onnx.checker.check_model(model)


@pytest.mark.parametrize(
    ("operation", "input_dtype", "zero_point", "constant_scale", "expected"),
    [
        ("DequantizeLinear", TensorProto.INT8, [-128, 127], False, ([0, -127.5], [63.75, 0])),
        ("DequantizeLinear", TensorProto.INT8, None, True, ([-32, -64], [31.75, 63.5])),
        ("QuantizeLinear", TensorProto.FLOAT, None, False, ([0, 0], [63.75, 127.5])),
    ],
)
def test_quantization_clip_bounds_respect_signedness_and_optional_zero_point(
    operation, input_dtype, zero_point, constant_scale, expected
):
    """Compute per-channel bounds without integer overflow or implicit signedness changes."""
    scale = numpy_helper.from_array(np.array([0.25, 0.5], dtype=np.float32), "scale")
    nodes = [helper.make_node("Constant", [], ["scale"], value=scale)] if constant_scale else []
    initializers = [] if constant_scale else [scale]
    inputs = ["x", "scale"]
    if zero_point is not None:
        initializers.append(numpy_helper.from_array(np.array(zero_point, dtype=np.int8), "zero"))
        inputs.append("zero")
    nodes.append(helper.make_node(operation, inputs, ["y"], axis=1))
    output_dtype = TensorProto.FLOAT if operation == "DequantizeLinear" else TensorProto.UINT8
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "clip_bounds",
            [helper.make_tensor_value_info("x", input_dtype, [1, 2])],
            [helper.make_tensor_value_info("y", output_dtype, [1, 2])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    before = model.SerializeToString()
    lower, upper = calculate_clip_range(model.graph.node[-1], model)
    np.testing.assert_array_equal(lower, np.array(expected[0], dtype=np.float32))
    np.testing.assert_array_equal(upper, np.array(expected[1], dtype=np.float32))
    assert model.SerializeToString() == before
    onnx.checker.check_model(model)


@pytest.mark.parametrize(
    ("weight_shape", "quantized", "eligible"),
    [([3, 2], False, True), ([1, 1, 3, 2], True, True), ([2, 1, 3, 2], True, False)],
)
def test_matmul_weight_layout_eligibility_preserves_nonunit_batch_dimensions(
    weight_shape, quantized, eligible
):
    """Allow supported static weights while refusing layouts with nonunit batch dimensions."""
    dtype = np.int8 if quantized else np.float32
    initializers = [numpy_helper.from_array(np.ones(weight_shape, dtype=dtype), "weight")]
    nodes = []
    if quantized:
        initializers.append(numpy_helper.from_array(np.array(0.5, dtype=np.float32), "scale"))
        nodes.append(helper.make_node("DequantizeLinear", ["weight", "scale"], ["dequantized"]))
    nodes.append(helper.make_node("MatMul", ["x", "dequantized" if quantized else "weight"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "weight_layout",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None] * len(weight_shape))],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    before = model.SerializeToString()
    assert check_to_apply_transpose(model.graph.node[-1], model, {"x": [2, 3]}) is eligible
    assert model.SerializeToString() == before
    onnx.checker.check_model(model)


def test_qdq_dtype_filters_normalize_aliases_and_reject_unknown_types():
    """Normalize equivalent filter spellings and reject typos instead of selecting a fallback dtype."""
    assert parse_qdq_quantized_dtype_list(
        [TensorProto.INT16, " int16 ", str(TensorProto.UINT16), "u16"]
    ) == frozenset({TensorProto.INT16, TensorProto.UINT16})
    with pytest.raises(ValueError):
        parse_qdq_quantized_dtype_list(["int16", "unit16"])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int8])
def test_repeated_external_save_keeps_sidecar_bounded(tmp_path, monkeypatch, dtype):
    """Round-trip adjacent payloads, including a Windows text-mode EOF byte."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "model.onnx"
    location = "weights/data.bin" if dtype == np.float32 else "model.onnx_data"
    values = np.arange(2048, dtype=np.float32).astype(dtype)
    for offset in [0, 1]:
        first, second = values + offset, np.full_like(values, 39 + offset)
        first.view(np.uint8)[-1] = 0x1A
        model = helper.make_model(
            helper.make_graph(
                [],
                "weights",
                [],
                [
                    helper.make_tensor_value_info(
                        name, helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), [2048]
                    )
                    for name in ("w", "other")
                ],
                [numpy_helper.from_array(first, "w"), numpy_helper.from_array(second, "other")],
            )
        )
        save_onnx_model(model, str(path), external_data_name=location, verbose=False)
        assert all(
            tensor.data_location == TensorProto.EXTERNAL and not tensor.HasField("raw_data")
            for tensor in model.graph.initializer
        )
        assert (tmp_path / location).stat().st_size == 2 * values.nbytes
        saved = onnx.load(path)
        onnx.checker.check_model(saved)
        for tensor, expected in zip(saved.graph.initializer, (first, second), strict=True):
            assert numpy_helper.to_array(tensor).tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "location", ["../weights.bin", "absolute", "model.onnx", "", "directory", "model_alias"]
)
def test_invalid_external_location_preserves_model_and_existing_output(tmp_path, location):
    """Reject invalid sidecar paths before changing caller tensors or existing files."""
    model = helper.make_model(
        helper.make_graph(
            [], "save", [], [], [numpy_helper.from_array(np.ones(2048, np.float32), "w")]
        )
    )
    before = model.SerializeToString()
    path = tmp_path / "model.onnx"
    path.write_bytes(b"unchanged")
    if location == "absolute":
        location = str(tmp_path / "weights.bin")
    elif location == "directory":
        (tmp_path / location).mkdir()
        (tmp_path / location / "existing.bin").write_bytes(b"existing weights")
    elif location == "model_alias":
        try:
            (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"Directory symlinks are unavailable: {error}")
        location = "alias/model.onnx"
    with pytest.raises(ValueError):
        save_onnx_model(model, str(path), external_data_name=location, verbose=False)
    assert model.SerializeToString() == before
    assert path.read_bytes() == b"unchanged"
    if location == "directory":
        assert (tmp_path / location / "existing.bin").read_bytes() == b"existing weights"


@pytest.mark.parametrize(
    ("link_kind", "contained"),
    [
        ("directory", False),
        ("directory", True),
        ("file", False),
        ("file", True),
        ("dangling", True),
        ("hardlink", True),
    ],
)
def test_external_save_checks_sidecar_links(tmp_path, link_kind, contained):
    """Allow contained directory links and reject file links without changing existing data."""
    output = tmp_path / "output"
    output.mkdir()
    target = output / "target" if contained else tmp_path / "outside"
    target.mkdir()
    sidecar = target / "data.bin"
    if link_kind != "dangling":
        sidecar.write_bytes(b"existing weights")
    destination = output / "weights" / "data.bin"
    try:
        if link_kind == "directory":
            destination.parent.symlink_to(target, target_is_directory=True)
        else:
            destination.parent.mkdir()
            if link_kind == "hardlink":
                destination.hardlink_to(sidecar)
            else:
                destination.symlink_to(sidecar)
    except OSError as error:
        pytest.skip(f"{link_kind} links are unavailable: {error}")
    path = output / "model.onnx"
    path.write_bytes(b"existing model")
    values = np.ones(2048, np.float32)
    model = helper.make_model(
        helper.make_graph([], "save", [], [], [numpy_helper.from_array(values, "w")])
    )
    before = model.SerializeToString()
    if link_kind == "directory" and contained:
        save_onnx_model(model, str(path), external_data_name="weights/data.bin", verbose=False)
        assert sidecar.read_bytes() == values.tobytes()
        saved = onnx.load(path, load_external_data=False)
        assert saved.graph.initializer[0].data_location == TensorProto.EXTERNAL
    else:
        with pytest.raises(ValueError):
            save_onnx_model(model, str(path), external_data_name="weights/data.bin", verbose=False)
        assert model.SerializeToString() == before
        assert path.read_bytes() == b"existing model"
        if link_kind == "dangling":
            assert destination.is_symlink()
            assert not sidecar.exists()
        else:
            assert sidecar.read_bytes() == b"existing weights"
            assert destination.read_bytes() == b"existing weights"


@pytest.mark.skipif(os.name != "nt", reason="Win32 filename aliases and alternate data streams")
@pytest.mark.parametrize("existing_output", [False, True])
@pytest.mark.parametrize(
    "location", ["MODEL.onnx", "model.onnx.", "model.onnx ", "model.onnx:weights"]
)
def test_external_save_rejects_windows_model_aliases(tmp_path, existing_output, location):
    """Reject Win32 aliases without losing caller payloads or the existing model."""
    path = tmp_path / "model.onnx"
    if existing_output:
        path.write_bytes(b"original model")
    model = helper.make_model(
        helper.make_graph(
            [], "alias", [], [], [numpy_helper.from_array(np.ones(2048, np.float32), "w")]
        )
    )
    before = model.SerializeToString()
    with pytest.raises((OSError, ValueError)):
        save_onnx_model(model, str(path), external_data_name=location, verbose=False)
    assert model.SerializeToString() == before
    if existing_output:
        assert path.read_bytes() == b"original model"
    else:
        assert not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing_output else [])


@pytest.mark.parametrize("existing_sidecar", [False, True])
@pytest.mark.parametrize("failure_stage", ["weights", "model", "short_read"])
@pytest.mark.parametrize("external_metadata", [False, True])
def test_external_save_rolls_back_failed_replacement(
    tmp_path, monkeypatch, existing_sidecar, failure_stage, external_metadata
):
    """Keep the previous model and weights usable when either replacement fails."""
    path = tmp_path / "model.onnx"
    sidecar = tmp_path / "weights" / "data.bin"
    values = np.ones(2048, np.float32)
    original = helper.make_model(
        helper.make_graph(
            [],
            "save",
            [],
            [helper.make_tensor_value_info("w", TensorProto.FLOAT, [2048])],
            [
                numpy_helper.from_array(values, "w"),
                numpy_helper.from_array(values + 2, "other"),
            ],
        )
    )
    save_onnx_model(
        original,
        str(path),
        use_external_data=existing_sidecar,
        external_data_name="weights/data.bin",
        verbose=False,
    )
    original_bytes = path.read_bytes()
    original_weights = sidecar.read_bytes() if existing_sidecar else None
    candidate = onnx.load(path)
    candidate.graph.initializer[0].CopyFrom(numpy_helper.from_array(values + 1, "w"))
    candidate.graph.initializer[1].data_location = TensorProto.DEFAULT
    if external_metadata:
        onnx.external_data_helper.set_external_data(
            candidate.graph.initializer[0],
            "original.weights",
            offset=17,
            length=values.nbytes,
            checksum="original-checksum",
            basepath="original-directory",
        )
    candidate_bytes = candidate.SerializeToString()
    failure_path = sidecar if failure_stage == "weights" else path
    short_read = failure_stage == "short_read"
    failure = PermissionError("Replacement denied")
    replace = os.replace
    failed = False

    def fail_selected_replacement(source, destination):
        """Fail one selected replacement to verify tensor and file rollback."""
        nonlocal failed
        if not failed and os.fspath(destination) == str(failure_path):
            failed = True
            if short_read:
                with sidecar.open("r+b") as stream:
                    stream.truncate(values.nbytes - 1)
            raise failure
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_selected_replacement)
    with pytest.raises(OSError) as error:
        save_onnx_model(candidate, str(path), external_data_name="weights/data.bin", verbose=False)
    if short_read:
        assert not candidate.graph.initializer[0].HasField("raw_data")
        assert error.value.__context__ is failure
    else:
        assert error.value is failure
        assert candidate.SerializeToString() == candidate_bytes
    assert path.read_bytes() == original_bytes
    if existing_sidecar:
        assert sidecar.read_bytes() == original_weights
    else:
        assert not sidecar.exists()
    restored = onnx.load(path)
    onnx.checker.check_model(restored)
    assert numpy_helper.to_array(restored.graph.initializer[0]).tobytes() == values.tobytes()
    expected_files = {path, sidecar} if existing_sidecar else {path}
    assert {entry for entry in tmp_path.rglob("*") if entry.is_file()} == expected_files
    if short_read:
        return
    save_onnx_model(candidate, str(path), external_data_name="weights/data.bin", verbose=False)
    assert candidate.graph.initializer[0].data_location == TensorProto.EXTERNAL
    assert not candidate.graph.initializer[0].HasField("raw_data")
    retried = onnx.load(path)
    onnx.checker.check_model(retried)
    assert numpy_helper.to_array(retried.graph.initializer[0]).tobytes() == (values + 1).tobytes()
    assert numpy_helper.to_array(retried.graph.initializer[1]).tobytes() == (values + 2).tobytes()


@pytest.mark.parametrize("existing_output", [False, True])
@pytest.mark.parametrize("use_external_data", [False, True])
def test_save_rejects_unloaded_external_tensors_before_changes(
    tmp_path, existing_output, use_external_data
):
    """Reject partially loaded models before creating directories or changing existing outputs."""
    values = np.ones(2048, np.float32)
    source = tmp_path / "source" / "model.onnx"
    model = helper.make_model(
        helper.make_graph(
            [],
            "save",
            [],
            [helper.make_tensor_value_info("first", TensorProto.FLOAT, [2048])],
            [numpy_helper.from_array(values, name) for name in ("first", "missing")],
        )
    )
    save_onnx_model(model, str(source), verbose=False)
    unloaded = onnx.load(source, load_external_data=False)
    unloaded.graph.initializer[0].CopyFrom(numpy_helper.from_array(values, "first"))
    before = unloaded.SerializeToString()
    output = tmp_path / "output"
    path = output / "model.onnx"
    if existing_output:
        output.mkdir()
        path.write_bytes(b"existing model")
        (output / "model.onnx_data").write_bytes(b"existing weights")
    files_before = {entry: entry.read_bytes() for entry in tmp_path.rglob("*") if entry.is_file()}
    with pytest.raises(ValueError):
        save_onnx_model(unloaded, str(path), use_external_data=use_external_data, verbose=False)
    assert unloaded.SerializeToString() == before
    assert output.exists() == existing_output
    assert {
        entry: entry.read_bytes() for entry in tmp_path.rglob("*") if entry.is_file()
    } == files_before


@pytest.mark.parametrize("storage", ["initializer", "attribute", "subgraph"])
@pytest.mark.parametrize("write_failure", [False, True])
def test_inline_save_preserves_loaded_external_tensors(tmp_path, storage, write_failure):
    """Inline loaded external payloads without changing caller state or existing sidecars."""
    values = np.arange(2048, dtype=np.float32)
    values.view(np.uint8)[-1] = 0x1A
    weight = numpy_helper.from_array(values, "weight")
    onnx.external_data_helper.set_external_data(
        weight, "original.weights", offset=7, length=values.nbytes, checksum="original-checksum"
    )
    value = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2048])
    if storage == "initializer":
        graph = helper.make_graph(
            [],
            "inline",
            [],
            [helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2048])],
            [weight],
        )
    elif storage == "attribute":
        graph = helper.make_graph(
            [helper.make_node("Constant", [], ["y"], value=weight)], "inline", [], [value]
        )
    else:
        branch = helper.make_graph(
            [helper.make_node("Constant", [], ["y"], value=weight)], "branch", [], [value]
        )
        graph = helper.make_graph(
            [helper.make_node("If", ["condition"], ["y"], then_branch=branch, else_branch=branch)],
            "inline",
            [],
            [value],
            [numpy_helper.from_array(np.asarray(True), "condition")],
        )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    before = model.SerializeToString()
    sidecar = tmp_path / "original.weights"
    sidecar.write_bytes(b"original sidecar")
    path = tmp_path / "inline.onnx"
    if write_failure:
        path.mkdir()
        with pytest.raises(OSError):
            save_onnx_model(model, str(path), use_external_data=False, verbose=False)
    else:
        save_onnx_model(model, str(path), use_external_data=False, verbose=False)
        saved = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(saved)
        weights = [
            tensor
            for tensor in onnx.external_data_helper._get_all_tensors(saved)
            if tensor.name == "weight"
        ]
        assert len(weights) == (2 if storage == "subgraph" else 1)
        for tensor in weights:
            assert tensor.data_location != TensorProto.EXTERNAL
            assert not tensor.external_data
            assert numpy_helper.to_array(tensor).tobytes() == values.tobytes()
    assert model.SerializeToString() == before
    assert sidecar.read_bytes() == b"original sidecar"
    assert {entry for entry in tmp_path.iterdir() if entry.is_file()} == (
        {sidecar} if write_failure else {sidecar, path}
    )


@pytest.mark.parametrize("external_metadata", [False, True])
@pytest.mark.parametrize("failure_stage", ["write", "replace"])
def test_inline_save_failure_preserves_existing_model(
    tmp_path, monkeypatch, external_metadata, failure_stage
):
    """Preserve existing files and caller tensors after partial writes or replacement errors."""
    path = tmp_path / "model.onnx"
    values = np.arange(2048, dtype=np.float32)
    model = helper.make_model(
        helper.make_graph(
            [],
            "inline_failure",
            [],
            [helper.make_tensor_value_info("w", TensorProto.FLOAT, [2048])],
            [numpy_helper.from_array(values, "w")],
        )
    )
    onnx.save(model, path)
    original = path.read_bytes()
    model.graph.initializer[0].CopyFrom(numpy_helper.from_array(values + 1, "w"))
    sidecar = tmp_path / "original.weights"
    sidecar.write_bytes(b"existing sidecar")
    if external_metadata:
        onnx.external_data_helper.set_external_data(
            model.graph.initializer[0], "original.weights", offset=7, length=values.nbytes
        )
    before = model.SerializeToString()
    failure = OSError("Injected inline save failure")
    write_bytes = onnx._save_bytes

    def fail_write(content, destination):
        """Use the real writer to leave partial serialized bytes before failing."""
        write_bytes(content[: len(content) // 2], destination)
        raise failure

    def fail_replace(source, destination):
        """Inject a replacement failure after the staged model was saved."""
        raise failure

    with monkeypatch.context() as patch:
        if failure_stage == "write":
            patch.setattr(onnx, "_save_bytes", fail_write)
        else:
            patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError) as error:
            save_onnx_model(model, str(path), use_external_data=False, verbose=False)
        assert error.value is failure
    assert path.read_bytes() == original
    assert model.SerializeToString() == before
    assert sidecar.read_bytes() == b"existing sidecar"
    assert {entry.name for entry in tmp_path.iterdir()} == {"model.onnx", "original.weights"}
    onnx.checker.check_model(onnx.load(path))
    save_onnx_model(model, str(path), use_external_data=False, verbose=False)
    saved = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(saved)
    assert numpy_helper.to_array(saved.graph.initializer[0]).tobytes() == (values + 1).tobytes()
    assert not saved.graph.initializer[0].external_data
    assert model.SerializeToString() == before
    assert sidecar.read_bytes() == b"existing sidecar"


@pytest.mark.parametrize("external_metadata", [False, True])
@pytest.mark.parametrize("failure_stage", ["write", "close"])
def test_external_save_preserves_payloads_when_buffered_writes_fail(
    tmp_path, monkeypatch, external_metadata, failure_stage
):
    """Keep loaded payloads and prior files intact when a buffered sidecar write fails."""
    path = tmp_path / "model.onnx"
    sidecar = tmp_path / "weights" / "data.bin"
    values = np.arange(8, dtype=np.float32)
    model = helper.make_model(
        helper.make_graph(
            [],
            "buffered_failure",
            [],
            [helper.make_tensor_value_info("w", TensorProto.FLOAT, [8])],
            [
                numpy_helper.from_array(values, "w"),
                numpy_helper.from_array(values + 1, "other"),
            ],
        )
    )
    save_onnx_model(
        model, str(path), external_data_name="weights/data.bin", size_threshold=0, verbose=False
    )
    original, original_weights = path.read_bytes(), sidecar.read_bytes()
    candidate = onnx.load(path)
    candidate.graph.initializer[0].CopyFrom(numpy_helper.from_array(values + 10, "w"))
    if external_metadata:
        onnx.external_data_helper.set_external_data(
            candidate.graph.initializer[0], "original.weights", offset=7, length=values.nbytes
        )
    before = candidate.SerializeToString()
    failure = OSError("Injected buffered sidecar failure")
    open_path = Path.open

    class FailingFile(io.FileIO):
        """Leave a real partial sidecar when the buffered writer flushes."""

        def write(self, data):
            """Write a single byte before reporting an injected I/O error."""
            super().write(data[:1])
            raise failure

    def open_sidecar(file_path, mode="r", *args, **kwargs):
        """Fail a real buffer flush during the second write or during close."""
        if mode == "wb" and file_path.name == "data.bin":
            buffer_size = 48 if failure_stage == "write" else 128
            return io.BufferedWriter(FailingFile(file_path, "w"), buffer_size=buffer_size)
        return open_path(file_path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_sidecar)
        with pytest.raises(OSError) as error:
            save_onnx_model(
                candidate,
                str(path),
                external_data_name="weights/data.bin",
                size_threshold=0,
                verbose=False,
            )
    assert candidate.SerializeToString() == before
    assert error.value is failure
    assert path.read_bytes() == original
    assert sidecar.read_bytes() == original_weights
    assert {entry for entry in tmp_path.rglob("*") if entry.is_file()} == {path, sidecar}
    assert {entry for entry in tmp_path.rglob("*") if entry.is_dir()} == {sidecar.parent}
    onnx.checker.check_model(onnx.load(path))
    save_onnx_model(
        candidate, str(path), external_data_name="weights/data.bin", size_threshold=0, verbose=False
    )
    saved = onnx.load(path)
    onnx.checker.check_model(saved)
    for tensor, expected in zip(saved.graph.initializer, (values + 10, values + 1), strict=True):
        assert numpy_helper.to_array(tensor).tobytes() == expected.tobytes()

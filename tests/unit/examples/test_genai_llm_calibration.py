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

"""Offline regressions for the standalone ONNX LLM quantization example."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from datasets import Dataset
from huggingface_hub import constants as hub_constants
from onnx import TensorProto, helper, numpy_helper
from safetensors import SafetensorError
from safetensors.torch import save_file
from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/windows/onnx_ptq/genai_llm"
_SPEC = importlib.util.spec_from_file_location("llm_calibration", _EXAMPLE / "llm_calibration.py")
calibration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibration)


@pytest.mark.parametrize("mode", ["file", "module"])
def test_quantize_example_import_modes(monkeypatch, mode):
    monkeypatch.delitem(sys.modules, "llm_calibration", raising=False)
    if mode == "file":
        monkeypatch.syspath_prepend(str(_EXAMPLE))
        spec = importlib.util.spec_from_file_location("quantize_example", _EXAMPLE / "quantize.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        name = "examples.windows.onnx_ptq.genai_llm.quantize"
        monkeypatch.delitem(sys.modules, name, raising=False)
        module = importlib.import_module(name)
    assert callable(module.main)


def _signature(tmp_path, hybrid):
    inputs = [
        helper.make_tensor_value_info(
            "inputs_embeds" if hybrid else "input_ids",
            TensorProto.FLOAT16 if hybrid else TensorProto.INT64,
            ["batch_size", "sequence_length", 4] if hybrid else ["batch_size", "sequence_length"],
        ),
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch_size", "sequence_length"]
        ),
        helper.make_tensor_value_info(
            "position_ids",
            TensorProto.INT64,
            [3, "batch_size", "sequence_length"] if hybrid else ["batch_size", "sequence_length"],
        ),
    ]
    for suffix, dims in (
        (("conv_state", ["batch_size", 4, 3]), ("recurrent_state", ["batch_size", 2, 4, 4]))
        if hybrid
        else (
            ("key", ["batch_size", 2, "past_sequence_length", 4]),
            ("value", ["batch_size", 2, "past_sequence_length", 4]),
        )
    ):
        inputs.append(
            helper.make_tensor_value_info("past_key_values.0." + suffix, TensorProto.FLOAT16, dims)
        )
    # Initializers can also occur in graph.input in older exports; they are not feeds.
    inputs.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, [1]))
    model = helper.make_model(
        helper.make_graph(
            [],
            "signature",
            inputs,
            [],
            [numpy_helper.from_array(np.ones(1, dtype=np.float32), "weight")],
        )
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return calibration.read_signature(path)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_signature_calibration_and_profiles(tmp_path, hybrid, batch_size):
    signature = _signature(tmp_path, hybrid)
    assert calibration.needs_signature_inputs(signature) == hybrid
    ids = np.array([[1, 2, 0], [2, 1, 2]])[:batch_size]
    mask = np.array([[1, 1, 0], [1, 1, 1]])[:batch_size]
    embeddings = torch.arange(16, dtype=torch.float16).reshape(4, 4)
    feed = calibration.calibration_feed(signature, ids, mask, embeddings if hybrid else None)
    assert set(feed) == {name for name, _, _ in signature}
    np.testing.assert_array_equal(
        feed["inputs_embeds"] if hybrid else feed["input_ids"], embeddings[ids] if hybrid else ids
    )
    positions = np.array([[0, 1, 1], [0, 1, 2]])[:batch_size]
    np.testing.assert_array_equal(
        feed["position_ids"],
        np.broadcast_to(positions, (3, batch_size, 3)) if hybrid else positions,
    )
    if hybrid:
        assert feed["past_key_values.0.conv_state"].shape == (batch_size, 4, 3)
        assert feed["past_key_values.0.recurrent_state"].shape == (batch_size, 2, 4, 4)
    else:
        assert feed["past_key_values.0.key"].shape == (batch_size, 2, 0, 4)
    profiles = calibration.ep_profiles(signature, ["NvTensorRtRtx", "trt", "cpu"], batch_size)
    assert profiles[2] == {}
    for profile, prefix in zip(profiles, ["nv", "trt"]):
        shapes = {
            kind: {
                name: tuple(map(int, dims.split("x")))
                for name, dims in (
                    part.split(":")
                    for part in profile[f"{prefix}_profile_{kind}_shapes"].split(",")
                )
            }
            for kind in ["min", "opt", "max"]
        }
        assert set(shapes["min"]) == set(feed)
        for name, value in feed.items():
            assert all(
                low <= actual <= high
                for low, actual, high in zip(shapes["min"][name], value.shape, shapes["max"][name])
            )


def test_hybrid_token_input_and_unknown_input(tmp_path):
    signature = _signature(tmp_path, True)
    signature[0] = ("input_ids", ["batch_size", "sequence_length"], "tensor(int64)")
    feed = calibration.calibration_feed(signature, [[1, 2]], [[1, 1]])
    assert feed["input_ids"].shape == (1, 2)
    with pytest.raises(ValueError):
        calibration.calibration_feed([*signature, ("unknown", [1], "tensor(float)")], [[1]], [[1]])
    with pytest.raises(ValueError):
        calibration.calibration_feed(
            [
                ("input_ids", ["unknown", "sequence_length"], "tensor(int64)"),
                ("past_key_values.0.key", ["unknown"], "tensor(float)"),
            ],
            [[1]],
            [[1]],
        )


@pytest.mark.parametrize("past_dimension", [None, 0])
@pytest.mark.parametrize("cache_kind", ["key", "conv_state"])
def test_signature_distinguishes_unspecified_and_zero_dimensions(
    tmp_path, past_dimension, cache_kind
):
    name = f"past_key_values.0.{cache_kind}"
    model = helper.make_model(
        helper.make_graph(
            [],
            "signature",
            [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 2, past_dimension, 4])],
            [],
        )
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    signature = calibration.read_signature(path)
    assert signature[0][1][2] == past_dimension
    assert calibration.needs_signature_inputs(signature) == (cache_kind == "conv_state")
    if past_dimension is None:
        with pytest.raises(ValueError, match=name):
            calibration.calibration_feed(signature, [[1]], [[1]])
        with pytest.raises(ValueError, match=name):
            calibration.ep_profiles(signature, ["trt"])
    else:
        assert calibration.calibration_feed(signature, [[1]], [[1]])[name].shape == (1, 2, 0, 4)


def test_nested_decoder_config_preserves_plain_config():
    plain = SimpleNamespace(hidden_size=4)
    assert calibration.text_config_of(plain) is plain
    assert calibration.text_config_of(SimpleNamespace(text_config=plain)) is plain
    assert calibration.text_config_of(SimpleNamespace(text_config=None)).text_config is None


def test_local_hybrid_config_through_calibration_inputs(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "vocab_size": 3,
                },
            }
        )
    )
    Qwen3_5Tokenizer(vocab={"<|endoftext|>": 0, "h": 1, "i": 2}, merges=[]).save_pretrained(
        tmp_path
    )
    example = importlib.import_module("examples.windows.onnx_ptq.genai_llm.quantize")
    monkeypatch.setattr(
        example, "load_dataset", lambda *_a, **_k: Dataset.from_dict({"article": ["hi", "h"]})
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    feeds = example.get_calib_inputs(
        "cnn", str(tmp_path), str(tmp_path), 2, 2, 4, "cpu", False, False, False, 2, True, False
    )
    assert len(feeds) == 1
    assert feeds[0]["input_ids"].shape == (2, 2)
    assert feeds[0]["position_ids"].shape == (2, 2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("hub", [False, True])
def test_embedding_checkpoint_calibration_offline(tmp_path, monkeypatch, dtype, indexed, hub):
    model = tmp_path / "checkpoint"
    if hub:
        model = tmp_path / "models--test--model" / "snapshots" / ("a" * 40)
        refs = model.parents[1] / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text("a" * 40)
    model.mkdir(parents=True)
    values = torch.tensor([[1.0001, 2.0002], [3.0003, 4.0004], [5.0005, 6.0006]], dtype=dtype)
    shard = "model-00001-of-00002.safetensors" if indexed else "model.safetensors"
    save_file({"model.embed_tokens.weight": values}, model / shard)
    if indexed:
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": shard}})
        )
    elif hub:
        missing = model.parents[1] / ".no_exist" / ("a" * 40)
        missing.mkdir(parents=True)
        (missing / "model.safetensors.index.json").touch()
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    table = calibration.embed_table("test/model" if hub else str(model), tmp_path)
    assert table.dtype == dtype
    # The tensor remains usable after safe_open closes, without a converted disk cache.
    for target, numpy_dtype in (("tensor(float16)", np.float16), ("tensor(float)", np.float32)):
        feed = calibration.calibration_feed(
            [("inputs_embeds", [1, 2, 2], target)], [[2, 0]], [[1, 1]], table
        )
        np.testing.assert_array_equal(
            feed["inputs_embeds"],
            values[torch.tensor([[2, 0]])].float().numpy().astype(numpy_dtype),
        )
    assert not list(tmp_path.rglob("*.npy"))


@pytest.mark.parametrize("bad_value", [float("inf"), float("nan"), 1e10])
def test_embedding_selected_values_validate_model_dtype(tmp_path, bad_value):
    values = torch.tensor([[1.0001], [bad_value]], dtype=torch.float32)
    save_file({"model.embed_tokens.weight": values}, tmp_path / "model.safetensors")
    table = calibration.embed_table(str(tmp_path))
    signature = [("inputs_embeds", [1, 1, 1], "tensor(float16)")]
    assert calibration.calibration_feed(signature, [[0]], [[1]], table)["inputs_embeds"].shape == (
        1,
        1,
        1,
    )
    with pytest.raises(ValueError):
        calibration.calibration_feed(signature, [[1]], [[1]], table)
    if bad_value == 1e10:
        signature[0] = ("inputs_embeds", [1, 1, 1], "tensor(float)")
        assert (
            calibration.calibration_feed(signature, [[1]], [[1]], table)["inputs_embeds"].item()
            == values[1].item()
        )


@pytest.mark.parametrize(
    "filename", ["../weights.safetensors", "sub/weights.safetensors", "sub\\weights.safetensors"]
)
def test_embedding_rejects_index_path_traversal(tmp_path, filename):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": filename}})
    )
    with pytest.raises(ValueError):
        calibration.embed_table(str(tmp_path))


@pytest.mark.parametrize(
    "case",
    ["missing_file", "invalid_file", "missing_embedding", "multiple_embeddings", "shape", "dtype"],
)
def test_embedding_rejects_invalid_checkpoint(tmp_path, case):
    path = tmp_path / "model.safetensors"
    tensors = {"model.embed_tokens.weight": torch.ones(2, 4)}
    if case == "missing_embedding":
        tensors = {"other.weight": torch.ones(2, 4)}
    elif case == "multiple_embeddings":
        tensors["other.embed_tokens.weight"] = torch.ones(2, 4)
    elif case == "shape":
        tensors["model.embed_tokens.weight"] = torch.ones(4)
    elif case == "dtype":
        tensors["model.embed_tokens.weight"] = torch.ones(2, 4, dtype=torch.int32)
    save_file(tensors, path)
    if case == "missing_file":
        path.unlink()
    if case == "invalid_file":
        path.write_bytes(path.read_bytes()[:-1])
    error = (
        FileNotFoundError
        if case == "missing_file"
        else SafetensorError
        if case == "invalid_file"
        else ValueError
    )
    with pytest.raises(error):
        calibration.embed_table(str(tmp_path))


def test_calibration_shared_zero_states_remain_zero_in_ort():
    signature = [
        ("past_key_values.0.conv_state", ["batch_size", 4, 3], "tensor(float)"),
        ("past_key_values.1.conv_state", ["batch_size", 4, 3], "tensor(float)"),
        ("past_key_values.0.recurrent_state", ["batch_size", 2, 4, 4], "tensor(float16)"),
        ("past_key_values.2.conv_state", ["batch_size", 4, 3], "tensor(float16)"),
    ]
    cache = {}
    feeds = [
        calibration.calibration_feed(signature, [[1, 2]], [[1, 1]], zero_cache=cache)
        for _ in range(2)
    ]
    first, second = feeds
    assert first[signature[0][0]] is first[signature[1][0]] is second[signature[0][0]]
    assert first[signature[0][0]] is not first[signature[2][0]]
    assert first[signature[0][0]] is not first[signature[3][0]]
    graph_inputs = [
        helper.make_tensor_value_info(
            n,
            TensorProto.FLOAT if t == "tensor(float)" else TensorProto.FLOAT16,
            list(first[n].shape),
        )
        for n, _, t in signature
    ]
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", [signature[0][0], signature[1][0]], ["y"])],
            "states",
            graph_inputs,
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for feed in feeds:
        np.testing.assert_array_equal(session.run(None, feed)[0], np.zeros((1, 4, 3)))
    assert all(not value.any() for value in cache.values())
    different_batch = calibration.calibration_feed(
        signature, [[1], [2]], [[1], [1]], zero_cache=cache
    )
    assert different_batch[signature[0][0]].shape == (2, 4, 3)


def test_external_weights_keep_shape_constants_loadable(tmp_path):
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
    weight = numpy_helper.from_array(np.ones((2, 1024), dtype=np.float32), "weight")
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["x", "weight"], ["sum"]),
                helper.make_node("ReduceMean", ["sum", "axes"], ["y"], keepdims=0),
            ],
            "external",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 1024])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            [axes, weight],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    path = tmp_path / "quantized.onnx"
    calibration.save_quantized_model(model, str(path))
    saved = onnx.load(path, load_external_data=False)
    assert saved.graph.initializer[0].data_location != TensorProto.EXTERNAL
    assert saved.graph.initializer[1].data_location == TensorProto.EXTERNAL
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(
        session.run(None, {"x": np.ones((2, 1024), dtype=np.float32)})[0], [2, 2]
    )

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
import os
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from datasets import Dataset
from onnx import TensorProto, helper, numpy_helper
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
    embeddings = np.arange(16, dtype=np.float16).reshape(4, 4)
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


def _checkpoint(directory, dtype="F16", indexed=False, invalid=False, offset=0, values=None):
    directory.mkdir(exist_ok=True)
    if values is None:
        values = np.arange(24, dtype=np.float32).reshape(6, 4)
    values = values + offset
    raw = (
        (values.view(np.uint32) >> 16).astype("<u2").tobytes()
        if dtype == "BF16"
        else values.astype("<f2" if dtype == "F16" else "<f4").tobytes()
    )
    name = "model.embed_tokens.weight"
    metadata = {
        name: {
            "dtype": dtype,
            "shape": list(values.shape),
            "data_offsets": [0, len(raw) + int(invalid)],
        }
    }
    header = json.dumps(metadata).encode()
    payload = struct.pack("<Q", len(header)) + header + raw
    shard = "model-00001-of-00002.safetensors" if indexed else "model.safetensors"
    (directory / shard).write_bytes(payload)
    if indexed:
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: shard}})
        )
    return values, payload


@pytest.mark.parametrize("dtype", ["F16", "F32", "BF16"])
@pytest.mark.parametrize("indexed", [False, True])
def test_embedding_tensor_from_local_checkpoint_and_cache(tmp_path, dtype, indexed):
    model = tmp_path / "checkpoint"
    values, _ = _checkpoint(model, dtype, indexed)
    cache = tmp_path / "cache" / "embedding.npy"
    got = calibration.embed_table(str(model), cache)
    assert isinstance(got, np.memmap)
    np.testing.assert_array_equal(got, values.astype(np.float16))
    cached = calibration.embed_table(str(model), cache)
    assert cached.filename == got.filename
    with pytest.raises(ValueError):
        calibration.embed_table(str(model), cache, max_output_bytes=values.size * 2 - 1)
    shard = next(model.glob("*.safetensors"))
    stat = shard.stat()
    updated, _ = _checkpoint(model, dtype, indexed, offset=2)
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    refreshed = calibration.embed_table(str(model), cache)
    np.testing.assert_array_equal(refreshed, updated.astype(np.float16))
    np.testing.assert_array_equal(got, values.astype(np.float16))
    assert refreshed.filename != got.filename


def test_invalid_tensor_leaves_no_cache(tmp_path):
    model = tmp_path / "checkpoint"
    _checkpoint(model, invalid=True)
    cache = tmp_path / "cache" / "embedding.npy"
    with pytest.raises(ValueError):
        calibration.embed_table(str(model), cache)
    assert not cache.exists()


@pytest.mark.parametrize("corruption", ["shape", "dtype", "nonfinite", "truncated"])
def test_invalid_embedding_cache_is_rebuilt(tmp_path, corruption):
    model = tmp_path / "checkpoint"
    values, _ = _checkpoint(model)
    cache = tmp_path / "cache" / "embedding.npy"
    table = calibration.embed_table(str(model), cache)
    cached_path = Path(table.filename)
    del table
    if corruption == "truncated":
        cached_path.write_bytes(cached_path.read_bytes()[:-2])
    else:
        corrupt = values.astype(np.float32 if corruption == "dtype" else np.float16)
        if corruption == "shape":
            corrupt = corrupt[:1]
        elif corruption == "nonfinite":
            corrupt[0, 0] = np.nan
        np.save(cached_path, corrupt)
    recovered = calibration.embed_table(str(model), cache)
    assert recovered.dtype == np.float16
    np.testing.assert_array_equal(recovered, values.astype(np.float16))


@pytest.mark.parametrize("destination", ["missing", "valid", "invalid"])
def test_embedding_cache_publication_failure(tmp_path, monkeypatch, destination):
    model = tmp_path / "checkpoint"
    values, _ = _checkpoint(model)
    cache = tmp_path / "cache" / "embedding.npy"

    def blocked_replace(source, target):
        if destination != "missing":
            contents = values.astype(np.float16)
            if destination == "invalid":
                contents[0, 0] = np.nan
            np.save(target, contents)
        raise PermissionError("Cache destination is locked")

    monkeypatch.setattr(calibration.os, "replace", blocked_replace)
    if destination == "valid":
        recovered = calibration.embed_table(str(model), cache)
        np.testing.assert_array_equal(recovered, values.astype(np.float16))
    else:
        with pytest.raises(PermissionError):
            calibration.embed_table(str(model), cache)
    assert len(list(cache.parent.iterdir())) == (destination != "missing")


@pytest.mark.parametrize("oversized", [False, True])
def test_hub_embedding_is_downloaded_in_bounded_ranges(tmp_path, monkeypatch, oversized):
    values = np.ones((4096, 1024), dtype="<f4")
    metadata = {
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [2**30, 1024] if oversized else list(values.shape),
            "data_offsets": [0, 2**42 if oversized else values.nbytes],
        }
    }
    header = json.dumps(metadata).encode()
    payload = struct.pack("<Q", len(header)) + header + values.tobytes()
    ranges = []
    revision = "a" * 40
    payloads = {
        revision: payload,
        "b" * 40: struct.pack("<Q", len(header)) + header + (values + 1).tobytes(),
    }

    class Response:
        status = 206

        def __init__(self, first, last, content):
            self.first, self.last = first, last
            self.content = content
            self.headers = {"Content-Range": f"bytes {first}-{last}/{len(content)}"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, limit):
            return self.content[self.first : min(self.last + 1, self.first + limit)]

    def open_request(request, timeout):
        resolved_revision = request.full_url.split("/resolve/")[1].split("/")[0]
        assert resolved_revision in payloads
        if request.full_url.endswith(".index.json"):
            raise urllib.error.HTTPError(request.full_url, 404, "Not found", {}, None)
        assert request.get_header("Authorization") == "Bearer test-token"
        first, last = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        ranges.append((first, last))
        return Response(first, last, payloads[resolved_revision])

    monkeypatch.setattr(
        calibration.urllib.request, "build_opener", lambda *_: SimpleNamespace(open=open_request)
    )
    monkeypatch.setattr(calibration, "get_token", lambda: "test-token")
    monkeypatch.setattr(
        calibration,
        "HfApi",
        lambda: SimpleNamespace(
            endpoint="https://huggingface.co",
            model_info=lambda *_a, **_k: SimpleNamespace(sha=revision),
        ),
        raising=False,
    )
    if oversized:

        def reject_allocation(*_args, **_kwargs):
            pytest.fail("Oversized checkpoint must be rejected before cache allocation")

        monkeypatch.setattr(calibration.np.lib.format, "open_memmap", reject_allocation)
        with pytest.raises(ValueError, match="exceeds the configured maximum cache size"):
            calibration.embed_table("test/model", tmp_path / "cache" / "embedding.npy")
        assert not (tmp_path / "cache").exists()
        return
    table = calibration.embed_table("test/model", tmp_path / "embedding.npy")
    np.testing.assert_array_equal(table, values)
    assert max(last - first + 1 for first, last in ranges) <= 8 * 1024 * 1024
    assert sum(last - first + 1 for first, last in ranges) == len(payload)
    ranges.clear()
    cached = calibration.embed_table("test/model", tmp_path / "embedding.npy")
    assert cached.filename == table.filename
    assert sum(last - first + 1 for first, last in ranges) < len(payload)
    revision = "b" * 40
    refreshed = calibration.embed_table("test/model", tmp_path / "embedding.npy")
    np.testing.assert_array_equal(refreshed, values + 1)
    np.testing.assert_array_equal(table, values)


def test_nonfinite_embedding_cleans_partial_download(tmp_path):
    model = tmp_path / "checkpoint"
    _checkpoint(model)
    shard = model / "model.safetensors"
    payload = bytearray(shard.read_bytes())
    payload[-2:] = np.array([np.inf], dtype="<f2").tobytes()
    shard.write_bytes(payload)
    cache = tmp_path / "cache" / "embedding.npy"
    with pytest.raises(ValueError):
        calibration.embed_table(str(model), cache)
    assert not list(cache.parent.iterdir())


def test_checkpoint_change_during_conversion_cleans_partial_cache(tmp_path, monkeypatch):
    model = tmp_path / "checkpoint"
    _checkpoint(model, dtype="F32", values=np.ones((4096, 1024), dtype=np.float32))
    original = np.frombuffer
    changed = False

    def change_later_chunk(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            with (model / "model.safetensors").open("r+b") as stream:
                stream.seek(-4, 2)
                stream.write(np.array([2], dtype="<f4").tobytes())
        return original(*args, **kwargs)

    monkeypatch.setattr(calibration.np, "frombuffer", change_later_chunk)
    cache = tmp_path / "cache" / "embedding.npy"
    with pytest.raises(ValueError):
        calibration.embed_table(str(model), cache)
    assert not list(cache.parent.iterdir())


def test_range_reader_rejects_ignored_ranges_and_strips_cross_host_auth():
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    opener = SimpleNamespace(open=lambda *_args, **_kwargs: Response())
    with pytest.raises(ValueError):
        calibration._read_range(opener, "https://huggingface.co/model", 0, 8, None)
    request = urllib.request.Request(
        "https://huggingface.co/model", headers={"Authorization": "Bearer test"}
    )
    redirected = calibration._HubRedirectHandler().redirect_request(
        request, None, 302, "", {}, "https://cdn.example/model"
    )
    assert not redirected.has_header("Authorization")
    with pytest.raises(ValueError):
        calibration._HubRedirectHandler().redirect_request(
            request, None, 302, "", {}, "http://cdn.example/model"
        )


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

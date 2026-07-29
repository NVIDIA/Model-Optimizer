# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for offload-aware unified HF export helpers (CPU-only, no GPU required)."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open

try:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import set_module_tensor_to_device
except ImportError:
    pytest.skip("accelerate not available", allow_module_level=True)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import _postprocess_single_tensor
from modelopt.torch.export.unified_export_hf import (
    _export_quantized_weight,
    _has_accelerate_offload,
    _StreamingShardWriter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_offloaded_linear(dim: int = 16):
    """Return a Linear with a CPU-offload AlignDevicesHook attached and params on meta."""
    linear = nn.Linear(dim, dim, bias=False)
    weights_map = {"weight": linear.weight.data.clone().cpu()}
    hook = AlignDevicesHook(execution_device="cpu", offload=True, weights_map=weights_map)
    add_hook_to_module(linear, hook)
    set_module_tensor_to_device(linear, "weight", "meta")
    return linear, weights_map


# ---------------------------------------------------------------------------
# _has_accelerate_offload
# ---------------------------------------------------------------------------


def test_has_accelerate_offload_true():
    linear, _ = _make_offloaded_linear()
    assert _has_accelerate_offload(linear) is True


def test_has_accelerate_offload_false_no_hooks():
    linear = nn.Linear(16, 16)
    assert _has_accelerate_offload(linear) is False


def test_has_accelerate_offload_false_non_offload_hook():
    """A hook with offload=False should not be detected as offloaded."""
    linear = nn.Linear(16, 16)
    hook = AlignDevicesHook(execution_device="cpu", offload=False)
    add_hook_to_module(linear, hook)
    assert _has_accelerate_offload(linear) is False


def test_has_accelerate_offload_detects_nested_module():
    """Offload hook on a child module should be detected when scanning the parent."""

    class _Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.child(x)

    parent = _Parent()
    weights_map = {"weight": parent.child.weight.data.clone().cpu()}
    hook = AlignDevicesHook(execution_device="cpu", offload=True, weights_map=weights_map)
    add_hook_to_module(parent.child, hook)
    set_module_tensor_to_device(parent.child, "weight", "meta")

    assert _has_accelerate_offload(parent) is True


# ---------------------------------------------------------------------------
# _export_quantized_weight meta guard
# ---------------------------------------------------------------------------


def test_meta_guard_raises_on_meta_weight():
    """_export_quantized_weight must raise RuntimeError when weight is a meta tensor."""
    linear = nn.Linear(16, 16, bias=False)

    mtq.quantize(linear, mtq.FP8_DEFAULT_CFG, lambda m: m(torch.randn(1, 16)))

    # Manually set weight to meta to simulate what happens after hooks are removed.
    linear.weight = nn.Parameter(torch.empty(16, 16, device="meta"))

    with pytest.raises(RuntimeError, match="meta tensor"):
        _export_quantized_weight(linear, torch.float32)


def test_meta_guard_not_raised_for_real_weight():
    """No RuntimeError when weight is a real (non-meta) tensor."""
    linear = nn.Linear(32, 32, bias=False)
    mtq.quantize(linear, mtq.FP8_DEFAULT_CFG, lambda m: m(torch.randn(1, 32)))
    # Should not raise
    _export_quantized_weight(linear, torch.float32)


# ---------------------------------------------------------------------------
# _StreamingShardWriter
# ---------------------------------------------------------------------------


def test_streaming_shard_writer_single_shard():
    """Small tensors that fit in one shard produce model.safetensors without an index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = _StreamingShardWriter(tmpdir, max_shard_size=10 * 1024**3)
        writer.add("a", torch.ones(4, 4))
        writer.add("b", torch.zeros(2, 2))
        weight_map = writer.finalize()

        single = Path(tmpdir) / "model.safetensors"
        index = Path(tmpdir) / "model.safetensors.index.json"
        assert single.exists(), "model.safetensors not written"
        assert not index.exists(), "index file must not exist for single-shard export"
        assert set(weight_map.values()) == {"model.safetensors"}
        assert set(weight_map.keys()) == {"a", "b"}


def test_streaming_shard_writer_multi_shard():
    """Tensors exceeding max_shard_size produce multiple shards and an index file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # One float32 4x4 tensor = 64 bytes; set limit to 64 so each tensor goes to a new shard
        writer = _StreamingShardWriter(tmpdir, max_shard_size=64)
        writer.add("x", torch.ones(4, 4))
        writer.add("y", torch.ones(4, 4))
        weight_map = writer.finalize()

        index_path = Path(tmpdir) / "model.safetensors.index.json"
        assert index_path.exists(), "model.safetensors.index.json not written"
        assert weight_map["x"] != weight_map["y"], "keys must be in different shards"

        with open(index_path) as f:
            index = json.load(f)
        assert "weight_map" in index
        assert "metadata" in index
        assert index["metadata"]["total_size"] > 0


def test_multi_shard_files_exist_after_finalize():
    """All numbered shard files referenced in the index must exist on disk after finalize().

    Regression guard: an earlier code path called model.save_pretrained(state_dict={}) after
    finalize(), triggering transformers' stale-shard cleanup loop which matched and deleted
    every model-NNNNN-of-NNNNN.safetensors file because filename_to_tensors was empty.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = _StreamingShardWriter(tmpdir, max_shard_size=64)
        writer.add("x", torch.ones(4, 4))
        writer.add("y", torch.ones(4, 4))
        writer.finalize()

        index_path = Path(tmpdir) / "model.safetensors.index.json"
        assert index_path.exists()
        with open(index_path) as f:
            index = json.load(f)

        for key, shard_name in index["weight_map"].items():
            shard_path = Path(tmpdir) / shard_name
            assert shard_path.exists(), (
                f"Shard '{shard_name}' (for key '{key}') missing from disk after finalize()"
            )
            assert shard_path.stat().st_size > 0, f"Shard file {shard_name} is empty"


def test_streaming_shard_writer_tensors_readable():
    """Tensors written by the shard writer can be read back correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t = torch.randn(8, 8)
        writer = _StreamingShardWriter(tmpdir, max_shard_size=10 * 1024**3)
        writer.add("weight", t)
        weight_map = writer.finalize()

        shard_file = Path(tmpdir) / weight_map["weight"]
        with safe_open(str(shard_file), framework="pt") as f:
            recovered = f.get_tensor("weight")
        assert torch.allclose(recovered, t), "recovered tensor does not match original"


def test_streaming_shard_writer_drops_tied_alias():
    """Two keys sharing storage must not both reach save_file, which rejects aliases.

    The name-based _tied_weights_keys filter misses ties that transformers does not
    declare (e.g. tie_word_embeddings=False but shared storage), so the writer needs
    its own guard.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        shared = torch.ones(4, 4)
        writer = _StreamingShardWriter(tmpdir, max_shard_size=10 * 1024**3)
        writer.add("embed_tokens.weight", shared)
        writer.add("lm_head.weight", shared)
        weight_map = writer.finalize()

        assert set(weight_map) == {"embed_tokens.weight"}, (
            "tied alias should be dropped, keeping only the first key"
        )


def test_streaming_shard_writer_copies_aliased_view():
    """A distinct view onto shared storage must be copied, not dropped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        view = base.view(16)  # same data_ptr, different shape
        writer = _StreamingShardWriter(tmpdir, max_shard_size=10 * 1024**3)
        writer.add("base", base)
        writer.add("view", view)
        weight_map = writer.finalize()

        assert set(weight_map) == {"base", "view"}, "aliased view must be kept, not dropped"
        shard_file = Path(tmpdir) / weight_map["view"]
        with safe_open(str(shard_file), framework="pt") as f:
            assert torch.equal(f.get_tensor("view"), view)
            assert torch.equal(f.get_tensor("base"), base)


# ---------------------------------------------------------------------------
# _postprocess_single_tensor
# ---------------------------------------------------------------------------


def test_postprocess_passthrough_normal_key():
    """Non-quantizer weights pass through unchanged."""
    key, val = _postprocess_single_tensor(
        "model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4), 448.0, None
    )
    assert key == "model.layers.0.self_attn.q_proj.weight"
    assert val is not None
    assert val.shape == (4, 4)


def test_postprocess_amax_dropped():
    """weight_quantizer._amax matches skip_keys but has no replacement — dropped."""
    key, val = _postprocess_single_tensor(
        "model.layers.0.weight_quantizer._amax", torch.tensor(1.0), 448.0, None
    )
    assert key is None
    assert val is None


def test_postprocess_output_quantizer_dropped():
    """output_quantizer keys are always dropped."""
    key, val = _postprocess_single_tensor(
        "model.layers.0.output_quantizer._amax", torch.tensor(0.5), 448.0, None
    )
    assert key is None


def test_postprocess_kv_scale_renamed_and_divided():
    """k_bmm_quantizer._amax is renamed to k_proj.k_scale and divided by maxbound."""
    from modelopt.torch.export.model_config import KV_CACHE_FP8

    key, val = _postprocess_single_tensor(
        "model.layers.0.self_attn.k_bmm_quantizer._amax",
        torch.tensor(224.0),
        448.0,
        KV_CACHE_FP8,
    )
    assert key == "model.layers.0.self_attn.k_proj.k_scale"
    assert abs(val.item() - 0.5) < 1e-5


def test_postprocess_scale_squeezed():
    """3D scale tensors with shape[0]==1 are squeezed."""
    t = torch.ones(1, 4, 4)
    key, val = _postprocess_single_tensor("model.weight_scale", t, 448.0, None)
    assert key == "model.weight_scale"
    assert val.shape == (4, 4), f"expected (4, 4), got {val.shape}"


def test_postprocess_real_quant_param_dropped():
    """Keys matching RealQuantLinear scale tensors are dropped."""
    from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear

    for q_key in RealQuantLinear.list_of_scale_tensors:
        full_key = f"model.layers.0.weight_quantizer.{q_key}"
        key, val = _postprocess_single_tensor(full_key, torch.tensor(1.0), 448.0, None)
        assert key is None, f"expected None for real quant key '{full_key}'"


def test_postprocess_vision_model_summary_idxs_dropped():
    """The vision model summary_idxs parameter is always skipped."""
    key, val = _postprocess_single_tensor(
        "vision_model.radio_model.summary_idxs", torch.tensor([0, 1]), 448.0, None
    )
    assert key is None


# ---------------------------------------------------------------------------
# data_ptr identity under offload
#
# Both guards below exist because ``data_ptr()`` only identifies a tensor while
# that tensor is resident. Getting this wrong silently exported wrong weights:
# meta tensors all report 0, and freed addresses are recycled by the allocator.
# ---------------------------------------------------------------------------


class _FakeAmaxQuantizer(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.register_buffer("_amax", torch.tensor(value))
        self.is_enabled = True

    @property
    def amax(self):
        return self._amax

    @amax.setter
    def amax(self, v):
        self._amax = v


class _MetaLinearWithInputQuantizer(nn.Module):
    def __init__(self, amax: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 4, device="meta"))
        self.input_quantizer = _FakeAmaxQuantizer(amax)


def test_sync_tied_input_amax_skips_offloaded_modules():
    """Untied modules whose weights are offloaded must not be merged together.

    Every meta tensor reports ``data_ptr() == 0``, so without the residency guard
    these two unrelated Linears land in one group and both get amax 9.0.
    """
    from modelopt.torch.export.quant_utils import sync_tied_input_amax

    model = nn.Module()
    model._tied_weights_keys = {"b.weight": "a.weight"}
    model.a = _MetaLinearWithInputQuantizer(1.0)
    model.b = _MetaLinearWithInputQuantizer(9.0)

    with pytest.warns(UserWarning, match="offloaded weights"):
        merged = sync_tied_input_amax(model)

    assert merged == 0
    assert model.a.input_quantizer._amax.item() == 1.0
    assert model.b.input_quantizer._amax.item() == 9.0

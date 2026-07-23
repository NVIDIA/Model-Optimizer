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
    _process_quantized_modules_offloaded,
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
# _process_quantized_modules_offloaded — non-decoder materialization
# ---------------------------------------------------------------------------


def test_non_decoder_offloaded_tensors_are_collected():
    """Non-decoder modules with disk-offload hooks must have no meta tensors in the result.

    Reproduces the NemotronH 550B crash: embed_tokens (and norm, lm_head) are
    disk-offloaded and return meta from model.state_dict().  After
    revert_weight_conversion_quant_aware renames them to hub-original names, transformers'
    remove_tied_weights_from_state_dict tries to look them up in the model by that name
    and crashes.  Fix: _process_quantized_modules_offloaded materialises non-decoder
    offloaded modules directly so the returned state dict contains no meta tensors.

    The decoder layer here is NOT disk-offloaded (all weights GPU-resident) so the
    decoder-layer loop exercises the null-context path and we focus on the non-decoder
    collection pass that was previously missing.
    """

    class _TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.proj(x)

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.layers = nn.ModuleList([_TinyLayer()])

        def forward(self, x):
            return self.layers[0](self.embed(x))

    model = _TinyModel()

    # Install a CPU-offload hook on embed ONLY (non-decoder module).
    # The decoder layer is left GPU-resident so enable_weight_access_and_writeback
    # returns a no-op nullcontext and the decoder-layer state_dict() returns real tensors.
    embed_val = model.embed.weight.data.clone().cpu()
    embed_weights_map = {"weight": embed_val}
    embed_hook = AlignDevicesHook(
        execution_device="cpu", offload=True, weights_map=embed_weights_map
    )
    add_hook_to_module(model.embed, embed_hook)
    set_module_tensor_to_device(model.embed, "weight", "meta")

    from unittest.mock import patch

    with patch(
        "modelopt.torch.quantization.utils.layerwise_calib"
        ".LayerActivationCollector.get_decoder_layers",
        return_value=list(model.layers),
    ):
        result = _process_quantized_modules_offloaded(model, torch.float32)

    assert "embed.weight" in result, "embed.weight missing from state dict"
    emb = result["embed.weight"]
    assert not emb.is_meta, "embed.weight must not be meta in exported state dict"
    assert emb.shape == (16, 8)

    assert "layers.0.proj.weight" in result
    assert not result["layers.0.proj.weight"].is_meta

    for key, val in result.items():
        if isinstance(val, torch.Tensor):
            assert not val.is_meta, f"meta tensor found for key '{key}'"


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


# ---------------------------------------------------------------------------
# _postprocess_single_tensor
# ---------------------------------------------------------------------------


def test_postprocess_passthrough_normal_key():
    """Non-quantizer weights pass through unchanged."""
    key, val = _postprocess_single_tensor("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4), 448.0, None)
    assert key == "model.layers.0.self_attn.q_proj.weight"
    assert val is not None
    assert val.shape == (4, 4)


def test_postprocess_amax_dropped():
    """weight_quantizer._amax matches skip_keys but has no replacement — dropped."""
    key, val = _postprocess_single_tensor("model.layers.0.weight_quantizer._amax", torch.tensor(1.0), 448.0, None)
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

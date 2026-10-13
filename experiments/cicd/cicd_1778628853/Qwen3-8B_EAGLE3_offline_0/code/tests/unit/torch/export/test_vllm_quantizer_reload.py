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

import pytest
import torch

from modelopt.torch.export.plugins.vllm_fakequant_hf import (
    infer_quantizer_prefix_remap,
    merge_amax_tensors_for_group,
)


def _map_backbone_to_model(sd: dict) -> dict:
    """Test mapper: rename top-level ``backbone.`` to ``model.`` (typical HF vs vLLM)."""
    out = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            out["model." + k[len("backbone.") :]] = v
        else:
            out[k] = v
    return out


def test_infer_prefix_remap_simple_root_rename():
    """``infer_quantizer_prefix_remap`` infers one HF root → vLLM root from ``*.weight`` probes."""
    q = {
        "backbone.layers.0.mlp.gate_proj.input_quantizer": {},
        "backbone.layers.1.self_attn.q_proj.weight_quantizer": {},
    }
    rem = infer_quantizer_prefix_remap(q, _map_backbone_to_model)
    assert rem == {"backbone": "model"}


def test_infer_prefix_remap_multiple_probes_same_root_agree():
    """Regression: every quantizer key under the same HF root must agree on the mapped vLLM root."""
    q = {
        "backbone.a.w.input_quantizer": {},
        "backbone.b.w.weight_quantizer": {},
    }
    rem = infer_quantizer_prefix_remap(q, _map_backbone_to_model)
    assert rem == {"backbone": "model"}


def test_infer_prefix_remap_raises_on_inconsistent_root():
    """If ``map_fun`` maps the same HF root to different vLLM roots, raise with a clear error."""

    def bad_map(sd: dict) -> dict:
        out = {}
        for k, v in sd.items():
            if "layers.0" in k:
                out[k.replace("backbone.", "model.")] = v
            elif "head" in k:
                out[k.replace("backbone.", "encoder.")] = v
            else:
                out[k] = v
        return out

    q = {
        "backbone.layers.0.mlp.gate_proj.input_quantizer": {},
        "backbone.head.proj.input_quantizer": {},
    }
    with pytest.raises(ValueError, match="Inconsistent HF→vLLM prefix remap"):
        infer_quantizer_prefix_remap(q, bad_map)


def test_infer_prefix_remap_identity_empty():
    """When keys already match the mapper output, the inferred remap is empty (no rename)."""
    q = {"model.layers.0.foo.input_quantizer": {}}
    rem = infer_quantizer_prefix_remap(q, lambda d: dict(d))
    assert rem == {}


def test_infer_prefix_remap_probe_failure_skipped():
    """A probe that raises does not block remap if another key under the same root succeeds."""

    def map_drop_layers0(sd: dict) -> dict:
        out = {}
        for k, v in sd.items():
            if "layers.0" in k:
                raise RuntimeError("simulate missing layer")
            if k.startswith("backbone."):
                out["model." + k[len("backbone.") :]] = v
            else:
                out[k] = v
        return out

    q = {
        "backbone.layers.0.mlp.gate_proj.input_quantizer": {},
        "backbone.layers.1.mlp.gate_proj.input_quantizer": {},
    }
    rem = infer_quantizer_prefix_remap(q, map_drop_layers0)
    assert rem == {"backbone": "model"}


def test_infer_prefix_remap_no_quantizer_segment_still_probes_weight_path():
    """Short paths (e.g. ``embed.weight_quantizer``) still build a ``.weight`` probe path."""
    q = {"backbone.embed.weight_quantizer": {}}
    rem = infer_quantizer_prefix_remap(q, _map_backbone_to_model)
    assert rem == {"backbone": "model"}


def test_infer_prefix_remap_complex_mapper_not_one_root_raises_or_wrong():
    """Same HF root ``x`` mapping to different first components (``va.*`` vs ``vb.*``) must error."""

    def split_map(sd: dict) -> dict:
        k = next(iter(sd))
        v = sd[k]
        if "branch_a" in k:
            return {"va." + k[2:]: v}  # x.branch_a... -> va.branch_a...
        return {"vb." + k[2:]: v}

    q = {
        "x.branch_a.mlp.w.input_quantizer": {},
        "x.branch_b.mlp.w.input_quantizer": {},
    }
    with pytest.raises(ValueError, match="Inconsistent HF→vLLM prefix remap"):
        infer_quantizer_prefix_remap(q, split_map)


def test_merge_amax_same_shape_elementwise_max():
    """``merge_amax_tensors_for_group``: identical shapes → element-wise max (stack then amax)."""
    a = torch.tensor([1.0, 4.0, 2.0])
    b = torch.tensor([2.0, 3.0, 5.0])
    out = merge_amax_tensors_for_group([a, b])
    assert torch.allclose(out, torch.tensor([2.0, 4.0, 5.0]))


def test_merge_amax_different_1d_lengths_uses_cat():
    """``merge_amax_tensors_for_group``: mismatched 1-D lengths (e.g. GQA q/k/v) → ``cat`` on dim 0."""
    q = torch.tensor([1.0, 2.0, 3.0])  # e.g. 3 heads
    k = torch.tensor([0.5, 0.5])  # 2 KV heads
    v = torch.tensor([0.5, 0.5])
    out = merge_amax_tensors_for_group([q, k, v])
    assert out.shape == (7,)
    assert torch.allclose(out, torch.cat([q, k, v]))


def test_merge_amax_incompatible_shapes_scalar_fallback():
    """``merge_amax_tensors_for_group``: when ``cat`` fails, fall back to a scalar global max."""
    a = torch.ones(2, 3)
    b = torch.ones(2, 2)  # cannot cat along dim=0 with matching trailing dims
    out = merge_amax_tensors_for_group([a, b])
    assert out.shape == ()
    assert out.item() == 1.0

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

"""Tests for embedding pruning, width masking, and physical-width execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_model_descriptor import (
    Qwen3P5VLModelDescriptor,
)
from modelopt.torch.puzzletron.plugins.automodel.hooks.embedding import HiddenWidthSiteScorer
from modelopt.torch.puzzletron.plugins.automodel.reduction import MeshGroups
from modelopt.torch.puzzletron.pruning.embedding_pruning import (
    EmbeddingPruningSpec,
    PackedMinitronImportance,
    TensorAxisRule,
)
from modelopt.torch.puzzletron.pruning.runtime_hidden_width import (
    _mask_last_dim,
    hidden_width_layer_context,
    hidden_width_module_context,
)


def _spec() -> EmbeddingPruningSpec:
    return EmbeddingPruningSpec(
        hidden_size=4,
        legal_widths=(4, 2),
        alignment=2,
        tensor_rules=(
            TensorAxisRule(r"^embed\.weight$", (1,), "token embedding"),
            TensorAxisRule(r"^lm_head\.weight$", (1,), "language head"),
            TensorAxisRule(r"^norm\.weight$", (0,), "normalization"),
            TensorAxisRule(r"^layer\.qkv\.weight$", (1,), "attention input"),
            TensorAxisRule(r"^layer\.o\.weight$", (0,), "attention output"),
            TensorAxisRule(r"^layer\.residual\.weight$", (0, 1), "hidden-to-hidden"),
            TensorAxisRule(r"^projector\.weight$", (0,), "VLM projector output"),
            TensorAxisRule(r"^mtp\.weight$", (0, 1), "MTP residual map"),
        ),
        exempt_patterns=(r"^visual\.",),
        tie_groups=(("embed.weight", "lm_head.weight"),),
        config_paths=(("hidden_size",), ("text_config", "hidden_size")),
    )


def _state_dict() -> dict[str, torch.Tensor]:
    tied = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    return {
        "embed.weight": tied,
        "lm_head.weight": tied,
        "norm.weight": torch.arange(4, dtype=torch.float32),
        "layer.qkv.weight": torch.arange(20, dtype=torch.float32).reshape(5, 4),
        "layer.o.weight": torch.arange(24, dtype=torch.float32).reshape(4, 6),
        "layer.residual.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "projector.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "mtp.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "visual.block.weight": torch.ones(4, 4),
    }


def test_permutation_covers_both_residual_axes_and_preserves_ties():
    state = _state_dict()
    order = torch.tensor([2, 0, 3, 1])

    permuted = _spec().permute_state_dict(state, order)

    expected_residual = state["layer.residual.weight"].index_select(0, order).index_select(1, order)
    assert torch.equal(permuted["layer.residual.weight"], expected_residual)
    assert torch.equal(
        permuted["layer.qkv.weight"], state["layer.qkv.weight"].index_select(1, order)
    )
    assert torch.equal(
        permuted["projector.weight"], state["projector.weight"].index_select(0, order)
    )
    assert permuted["embed.weight"] is permuted["lm_head.weight"]
    assert permuted["visual.block.weight"] is state["visual.block.weight"]


def test_slice_state_dict_updates_every_hidden_axis_and_parameter_count_handles_ties():
    sliced = _spec().slice_state_dict(_state_dict(), 2)

    assert sliced["embed.weight"].shape == (3, 2)
    assert sliced["layer.qkv.weight"].shape == (5, 2)
    assert sliced["layer.o.weight"].shape == (2, 6)
    assert sliced["layer.residual.weight"].shape == (2, 2)
    assert sliced["projector.weight"].shape == (2, 2)
    assert sliced["mtp.weight"].shape == (2, 2)
    assert sliced["embed.weight"] is sliced["lm_head.weight"]
    expected = sum(value.numel() for key, value in sliced.items() if key != "lm_head.weight")
    assert _spec().parameter_count(sliced) == expected


def test_group_preserving_permutation_supports_block_quantized_hidden_inputs():
    spec = EmbeddingPruningSpec(
        hidden_size=8,
        legal_widths=(8, 4),
        alignment=4,
        permutation_group_size=4,
        tensor_rules=(
            TensorAxisRule(r"^embed$", (0,), "regular residual"),
            TensorAxisRule(
                r"^packed$",
                (),
                "four-channel quantization blocks",
                grouped_axes=((1, 4),),
            ),
        ),
    )
    # Group 1 is more important than group 0. Channels within each four-wide
    # quantization block retain their original order.
    order = spec.order_from_scores(torch.tensor([1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0, 5.0]))
    state = {
        "embed": torch.arange(8),
        "packed": torch.tensor([[[10], [20]]]),
    }

    assert torch.equal(order, torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]))
    permuted = spec.permute_state_dict(state, order)
    assert torch.equal(permuted["embed"], torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]))
    assert torch.equal(permuted["packed"].reshape(-1), torch.tensor([20, 10]))
    sliced = spec.slice_state_dict(permuted, 4)
    assert sliced["embed"].shape == (4,)
    assert sliced["packed"].shape == (1, 1, 1)


def test_audit_rejects_unhandled_hidden_sensitive_tensor_but_allows_vit_exemption():
    state = _state_dict()
    state["unknown.weight"] = torch.ones(7, 4)

    with pytest.raises(ValueError, match="unknown.weight"):
        _spec().audit_state_dict(state)


def test_packed_minitron_metric_matches_per_sample_mean_l2_then_site_sum():
    scorer = PackedMinitronImportance(hidden_size=2)
    activations_a = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [2.0, 8.0], [4.0, 12.0], [6.0, 16.0]]])
    activations_b = activations_a * 0.5
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    scorer.update("attention_norm", activations_a, cu_seqlens=cu)
    scorer.update("ffn_norm", activations_b, cu_seqlens=cu)
    scores = scorer.scores()

    means_a = torch.tensor([[2.0, 3.0], [4.0, 12.0]])
    expected = means_a.square().sum(0).sqrt()
    expected += (means_a * 0.5).square().sum(0).sqrt()
    assert torch.allclose(scores, expected)
    assert scorer.sample_count == 4
    assert scorer.site_names == ("attention_norm", "ffn_norm")


def test_qwen_vlm_embedding_spec_covers_language_projector_mtp_and_exempts_vit():
    config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=1024),
        vision_config=SimpleNamespace(out_hidden_size=1024),
    )
    spec = Qwen3P5VLModelDescriptor.embedding_pruning_spec(
        config,
        widths=(1024, 768),
        alignment=128,
    )
    state = {
        "model.language_model.embed_tokens.weight": torch.zeros(32, 1024),
        "lm_head.weight": torch.zeros(32, 1024),
        "model.language_model.norm.weight": torch.zeros(1024),
        "model.language_model.layers.0.input_layernorm.weight": torch.zeros(1024),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.zeros(1024),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.zeros(3584, 1024),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.zeros(1024, 3584),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2048, 1024),
        # This input happens to equal hidden size but is an independent head axis.
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.zeros(1024, 1024),
        "model.language_model.layers.1.linear_attn.in_proj_qkv.weight": torch.zeros(3072, 1024),
        "model.language_model.layers.1.linear_attn.out_proj.weight": torch.zeros(1024, 2048),
        "model.visual.merger.linear_fc2.weight": torch.zeros(1024, 512),
        "model.visual.merger.linear_fc2.bias": torch.zeros(1024),
        "model.visual.blocks.0.attn.qkv.weight": torch.zeros(3072, 1024),
        "mtp.fc.weight": torch.zeros(1024, 2048),
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(1024),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(1024),
        "mtp.layers.0.mlp.up_proj.weight": torch.zeros(3584, 1024),
        "mtp.layers.0.mlp.down_proj.weight": torch.zeros(1024, 3584),
    }

    audit = spec.audit_state_dict(state)
    sliced = spec.slice_state_dict(state, 768)
    sliced_config = spec.update_config_object(config, 768)

    assert "model.visual.blocks.0.attn.qkv.weight" in audit["exempt"]
    assert sliced["model.language_model.layers.0.self_attn.o_proj.weight"].shape == (
        768,
        1024,
    )
    assert sliced_config.text_config.hidden_size == 768
    assert sliced_config.vision_config.out_hidden_size == 768
    assert sliced["model.visual.merger.linear_fc2.weight"].shape == (768, 512)
    assert sliced["mtp.fc.weight"].shape == (768, 1536)


def test_hidden_width_site_scorer_consumes_pp_microbatch_sequence_ids():
    module = torch.nn.Identity()
    scorer = HiddenWidthSiteScorer(
        module,
        MeshGroups(),
        hidden_size=2,
        name="model.layers.0.input_layernorm",
    )
    scorer.set_batch_metadata(
        sequence_ids=torch.tensor([[0, 0, -1], [1, 1, 1]]),
        num_samples=2,
    )

    scorer(module, (), torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]]))
    scorer(module, (), torch.tensor([[[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]]))
    result = scorer.finalize()

    expected_means = torch.tensor([[2.0, 4.0], [4.0, 6.0]])
    torch.testing.assert_close(result["score"], expected_means.square().sum(0).sqrt())
    assert result["sample_count"] == 2


def test_hidden_width_site_scorer_restores_zero_cp_peer_from_exact_checkpoint():
    module = torch.nn.LayerNorm(2)
    scorer = HiddenWidthSiteScorer(
        module,
        MeshGroups(),
        hidden_size=2,
        name="model.layers.0.input_layernorm",
    )
    scorer.load_checkpoint_state({"_squared_sum": None, "_sample_count": 0})

    result = scorer.finalize()

    torch.testing.assert_close(result["score"], torch.zeros(2))
    assert result["sample_count"] == 0


def test_nested_width_envelope_matches_physical_prefix_and_zeros_inactive_gradients():
    class ToyRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 1.5, 2.0, 2.5]))
            self.variance_epsilon = 1.0e-6

        def forward(self, x):
            return (
                x
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.variance_epsilon).to(
                    x.dtype
                )
                * self.weight
            )

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = ToyRMSNorm()
            self.residual = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return x + self.residual(self.input_layernorm(x))

    spec = EmbeddingPruningSpec(
        hidden_size=4,
        legal_widths=(4, 2),
        alignment=2,
        tensor_rules=(
            TensorAxisRule(r"^layer\.input_layernorm\.weight$", (0,), "norm"),
            TensorAxisRule(r"^layer\.residual\.weight$", (0, 1), "residual"),
        ),
    )
    layer = ToyLayer()
    x = torch.tensor([[1.0, 2.0, 9.0, 10.0]], requires_grad=True)
    with hidden_width_layer_context(
        layer,
        canonical_layer_name="layer",
        spec=spec,
        width=2,
    ):
        actual = layer(x)

    x2 = x.detach()[..., :2]
    norm2 = x2 * torch.rsqrt(x2.square().mean(-1, keepdim=True) + 1.0e-6)
    norm2 = norm2 * layer.input_layernorm.weight[:2].detach()
    expected = x2 + torch.nn.functional.linear(norm2, layer.residual.weight[:2, :2].detach())
    torch.testing.assert_close(actual[..., :2], expected)
    assert torch.count_nonzero(actual[..., 2:]) == 0

    actual[..., :2].square().sum().backward()
    assert torch.count_nonzero(layer.residual.weight.grad[2:, :]) == 0
    assert torch.count_nonzero(layer.residual.weight.grad[:, 2:]) == 0
    assert torch.count_nonzero(layer.input_layernorm.weight.grad[2:]) == 0


def test_active_prefix_uses_native_rmsnorm_semantics_before_bfloat16_rounding():
    class OffsetRMSNorm(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.linspace(0.0, 0.3, width, dtype=torch.bfloat16))
            self.eps = 3.0e-5

        def forward(self, x):
            normalized = x.float() * torch.rsqrt(
                x.float().square().mean(-1, keepdim=True) + self.eps
            )
            return normalized.to(x.dtype) * (1 + self.weight)

    spec = EmbeddingPruningSpec(
        hidden_size=4,
        legal_widths=(4, 2),
        alignment=2,
        tensor_rules=(TensorAxisRule(r"^norm\.weight$", (0,), "normalization"),),
    )
    envelope = OffsetRMSNorm(4)
    physical = OffsetRMSNorm(2)
    physical.weight.data.copy_(envelope.weight[:2])
    x = torch.tensor([[0.1015625, 0.205078125, 4.0, -3.0]], dtype=torch.bfloat16)

    with hidden_width_module_context(
        envelope,
        canonical_module_name="norm",
        spec=spec,
        width=2,
        mask_boundary_input=True,
    ):
        actual = envelope(x)
    expected = physical(x[..., :2])

    assert torch.equal(actual[..., :2], expected)
    assert torch.count_nonzero(actual[..., 2:]) == 0


def test_active_prefix_layernorm_matches_physical_module_and_gradients():
    spec = EmbeddingPruningSpec(
        hidden_size=4,
        legal_widths=(4, 2),
        alignment=2,
        tensor_rules=(TensorAxisRule(r"^norm\.weight$", (0,), "normalization"),),
    )
    envelope = torch.nn.LayerNorm(4, eps=7.0e-4, dtype=torch.float64)
    physical = torch.nn.LayerNorm(2, eps=envelope.eps, dtype=torch.float64)
    physical.weight.data.copy_(envelope.weight[:2])
    physical.bias.data.copy_(envelope.bias[:2])
    x = torch.tensor([[1.0, 3.0, 8.0, -4.0]], dtype=torch.float64, requires_grad=True)

    with hidden_width_module_context(
        envelope,
        canonical_module_name="norm",
        spec=spec,
        width=2,
        mask_boundary_input=True,
    ):
        actual = envelope(x)
    physical_x = x.detach()[..., :2].clone().requires_grad_(True)
    expected = physical(physical_x)
    torch.testing.assert_close(actual[..., :2], expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad[..., :2], physical_x.grad)
    torch.testing.assert_close(envelope.weight.grad[:2], physical.weight.grad)
    torch.testing.assert_close(envelope.bias.grad[:2], physical.bias.grad)
    assert torch.count_nonzero(x.grad[..., 2:]) == 0
    assert torch.count_nonzero(envelope.weight.grad[2:]) == 0
    assert torch.count_nonzero(envelope.bias.grad[2:]) == 0


def _distributed_hidden_width_mask_job(rank: int, size: int) -> None:
    assert size == 4
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
    full = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    sharded = distribute_tensor(full, mesh, [Shard(1), Shard(2)])

    masked = _mask_last_dim(sharded, width=2, hidden_size=4)

    assert isinstance(masked, DTensor)
    expected = full * torch.tensor([1.0, 1.0, 0.0, 0.0])
    torch.testing.assert_close(masked.full_tensor(), expected)


def test_hidden_width_mask_supports_cp_tp_dtensor_activations() -> None:
    spawn_multiprocess_job(size=4, job=_distributed_hidden_width_mask_job, backend="gloo")

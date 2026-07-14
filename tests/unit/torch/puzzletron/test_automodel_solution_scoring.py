# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CPU tests for the AutoModel replace-1-block scoring building blocks (no distributed)."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    MLAConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.pruning.runtime_candidate import apply_runtime_candidate
from modelopt.torch.puzzletron.plugins.automodel.solution_metrics import (
    aggregate_solution_scores,
    retain_teacher_channels,
    score_batch,
)
from modelopt.torch.puzzletron.plugins.automodel.solution_launch import (
    _source_hidden_channel_indices,
    _solution_prune_target,
    _solution_hidden_width,
    _validate_parent_equivalence,
)
from modelopt.torch.puzzletron.stages.diagnostics import _annotate_solution_selections
from modelopt.torch.puzzletron.plugins.automodel.solution_recipe import (
    _layer_runtime_fingerprint,
    _masked_native_gate_forward,
)
from modelopt.torch.puzzletron.plugins.automodel.solution_recipe import ReplaceBlockScoringRecipe
from modelopt.torch.puzzletron.plugins.automodel.teacher_cache import TeacherTargetCache


def test_baseline_only_scoring_does_not_require_candidate_solutions(tmp_path):
    from modelopt.torch.puzzletron.plugins.automodel import solution_launch

    solutions, pending_ids = solution_launch._load_solution_work(
        {"baseline_only": True},
        tmp_path,
    )

    assert solutions == []
    assert pending_ids == []


def test_rpc_executor_infers_descriptor_when_config_has_no_override(monkeypatch, tmp_path):
    from modelopt.torch.puzzletron.distributed_eval import automodel_executor

    expected = object()
    calls = []
    monkeypatch.setattr(
        automodel_executor,
        "resolve_descriptor_from_pretrained",
        lambda path, **kwargs: calls.append((path, kwargs))
        or SimpleNamespace(name="test_descriptor", descriptor=expected),
    )

    descriptor = automodel_executor._resolve_executor_descriptor(
        {"model": {"trust_remote_code": True}}, tmp_path
    )

    assert descriptor is expected
    assert calls == [(str(tmp_path), {"trust_remote_code": True})]


def test_runtime_fingerprint_ignores_distributed_compute_dtype_transition():
    class FSDPLinear(FSDPModule, nn.Linear):
        pass

    # Composable FSDP mutates an existing module's class in place; directly
    # constructing an FSDPModule subclass intentionally returns the original
    # class from FSDPModule.__new__.
    layer = nn.Linear(4, 4, bias=False)
    layer.__class__ = FSDPLinear
    assert isinstance(layer, FSDPModule)
    baseline = _layer_runtime_fingerprint(layer)

    layer.weight.data = layer.weight.data.to(torch.bfloat16)

    assert _layer_runtime_fingerprint(layer) == baseline


def test_teacher_cache_roundtrip():
    cache = TeacherTargetCache()
    cache.set_lm_head_weight(torch.randn(20, 8))
    cache.append_hidden(torch.randn(2, 4, 8))
    cache.append_hidden(torch.randn(2, 4, 8))
    cache.seal()

    assert len(cache) == 2
    assert cache.lm_head().shape == (20, 8)
    assert cache.hidden(1).shape == (2, 4, 8)
    # Sealed cache rejects further appends.
    with pytest.raises(AssertionError):
        cache.append_hidden(torch.randn(2, 4, 8))


def test_solution_capture_concatenates_pipeline_microbatches_in_schedule_order():
    recipe = ReplaceBlockScoringRecipe.__new__(ReplaceBlockScoringRecipe)
    recipe._capture_enabled = True
    recipe._captured_hidden = None
    first = torch.randn(1, 4, 8)
    second = torch.randn(1, 4, 8)

    recipe._capture_hook(nn.Identity(), (), first)
    recipe._capture_hook(nn.Identity(), (), second)

    torch.testing.assert_close(recipe._captured_hidden, torch.cat((first, second), dim=0))


def test_solution_metrics_trim_pp_padding_rows():
    recipe = ReplaceBlockScoringRecipe.__new__(ReplaceBlockScoringRecipe)
    recipe._last_unpadded_batch_size = 3
    hidden = torch.randn(4, 2, 8)
    targets = torch.arange(8).reshape(4, 2)
    masks = {
        "ce_mask": torch.ones(4, 2, dtype=torch.bool),
        "kd_mask": torch.ones(4, 2, dtype=torch.bool),
        "hidden_mask": torch.ones(4, 2, dtype=torch.bool),
    }

    hidden, targets, masks = recipe._trim_pp_padding(hidden, targets, masks)

    assert hidden.shape[0] == targets.shape[0] == 3
    assert all(mask.shape[0] == 3 for mask in masks.values())


def test_parent_equivalence_does_not_gate_raw_hidden_metrics_after_basis_permutation(tmp_path):
    teacher_path = tmp_path / "teacher.json"
    parent_path = tmp_path / "parent.json"
    teacher_path.write_text(
        json.dumps(
            {
                "lm_loss": {"avg": 2.0},
                "kl_div": {"avg": 0.0},
                "top_1_logit_agreement": {"avg": 1.0},
            }
        )
    )
    parent_path.write_text(
        json.dumps(
            {
                "lm_loss": {"avg": 2.0005},
                "kl_div": {"avg": 0.001},
                "top_1_logit_agreement": {"avg": 0.98},
                "cosine_embedding_loss_hidden_states": {"avg": 1.0},
                "normalized_mse_loss_hidden_states": {"avg": 2.0},
                "mse_loss_hidden_states": {"avg": 30.0},
                "mae_loss_hidden_states": {"avg": 4.0},
            }
        )
    )

    summary = _validate_parent_equivalence(
        teacher_result_path=teacher_path,
        parent_result_path=parent_path,
        tolerances={},
        hidden_basis_permuted=True,
    )

    assert summary["passed"] is True
    assert summary["checks"]["top_1_logit_agreement"]["passed"] is True
    assert summary["checks"]["cosine_embedding_loss_hidden_states"]["gated"] is False
    assert summary["checks"]["cosine_embedding_loss_hidden_states"]["reason"] == "basis_permuted"


def test_parent_equivalence_still_gates_hidden_metrics_without_basis_permutation(tmp_path):
    teacher_path = tmp_path / "teacher.json"
    parent_path = tmp_path / "parent.json"
    teacher_path.write_text(json.dumps({"lm_loss": {"avg": 2.0}}))
    parent_path.write_text(
        json.dumps(
            {
                "lm_loss": {"avg": 2.0},
                "kl_div": {"avg": 0.0},
                "top_1_logit_agreement": {"avg": 1.0},
                "cosine_embedding_loss_hidden_states": {"avg": 1.0},
                "normalized_mse_loss_hidden_states": {"avg": 2.0},
                "mse_loss_hidden_states": {"avg": 30.0},
                "mae_loss_hidden_states": {"avg": 4.0},
            }
        )
    )

    with pytest.raises(RuntimeError, match="cosine_embedding_loss_hidden_states"):
        _validate_parent_equivalence(
            teacher_result_path=teacher_path,
            parent_result_path=parent_path,
            tolerances={},
            hidden_basis_permuted=False,
        )


def test_native_gate_can_mask_ranked_nonprefix_expert_ids():
    gate = nn.Module()
    gate.weight = nn.Parameter(torch.eye(4))
    gate.bias = nn.Parameter(torch.zeros(4))
    gate.score_func = "softmax"
    gate.softmax_before_topk = False
    gate.topk = 2
    gate.norm_topk_prob = True
    gate.route_scale = 1.0
    gate.gate_precision = None
    gate.forward = _masked_native_gate_forward(
        gate,
        target_num_experts=2,
        target_top_k=2,
        kept_expert_indices=(1, 3),
    )

    _, indices, _ = gate.forward(torch.tensor([[10.0, 1.0, 9.0, 2.0]]))

    assert set(indices.flatten().tolist()) == {1, 3}


def test_solution_prune_target_carries_ranked_expert_ids():
    teacher = BlockConfig(
        subblock_configs=(MoEConfig(num_experts=4, expert_intermediate_size=8, top_k=2),)
    )
    child = BlockConfig(
        subblock_configs=(MoEConfig(num_experts=2, expert_intermediate_size=8, top_k=2),)
    )

    target = _solution_prune_target(
        [
            {
                "parent_layer_indices": [0],
                "child_block_configs": [child],
                "diagnostic": {"kept_experts": [3, 1]},
            }
        ],
        [teacher],
        num_q_heads=2,
        head_dim=4,
    )

    assert target["expert_keep_ids"] == (3, 1)


def test_diagnostic_annotation_records_ranked_expert_ids(tmp_path):
    teacher = BlockConfig(
        subblock_configs=(MoEConfig(num_experts=4, expert_intermediate_size=8, top_k=2),)
    )
    child = BlockConfig(
        subblock_configs=(MoEConfig(num_experts=2, expert_intermediate_size=8, top_k=2),)
    )
    (tmp_path / "sorted_permutations.json").write_text(
        json.dumps({"moe.experts.0": [3, 1, 0, 2]})
    )
    diagnostic = {
        "axis": "moe_experts",
        "field": "num_experts",
        "subblock_kind": "moe",
        "layer_idx": 0,
        "target_value": 2,
    }
    solutions = [
        {
            "block_configs": [child],
            "diagnostic": dict(diagnostic),
            "single_sequence_replacement": {"diagnostic": dict(diagnostic)},
        }
    ]

    _annotate_solution_selections(
        solutions=solutions,
        teacher_block_configs=[teacher],
        sorted_teacher_dir=tmp_path,
    )

    assert solutions[0]["diagnostic"]["kept_experts"] == [3, 1]
    assert solutions[0]["diagnostic"]["removed_experts"] == [0, 2]


def test_score_batch_identical_model_is_perfect():
    # Candidate == teacher: consistency is perfect, while label accuracy is independent.
    torch.manual_seed(0)
    b, t, d, v = 2, 5, 8, 30
    hidden = torch.randn(b, t, d)
    w = torch.randn(v, d)
    predicted = (hidden @ w.t()).argmax(dim=-1)
    targets = (predicted + 1) % v

    out = score_batch(hidden, w, hidden.clone(), w.clone(), targets, chunk_size=7)

    assert max(abs(x) for x in out["kl_div"]) < 1e-4
    assert all(acc == 1.0 for acc in out["token_accuracy_top_1_consistency"])
    assert all(acc == 0.0 for acc in out["token_accuracy_top_1"])
    assert "lm_loss" in out and "cosine_embedding_loss_hidden_states" in out
    assert len(out["lm_loss"]) == b  # one value per sample


def test_aggregate_solution_scores_no_distributed():
    # token_group=None (single process): avg over the concatenated per-batch per-sample lists.
    per_batch = [
        {"lm_loss": [1.0, 3.0], "kl_div": [0.1, 0.3]},
        {"lm_loss": [5.0], "kl_div": [0.5]},
    ]
    reduced = aggregate_solution_scores(per_batch, token_group=None)
    assert reduced["lm_loss"]["per_sample"] == [1.0, 3.0, 5.0]
    assert reduced["lm_loss"]["avg"] == pytest.approx(3.0)
    assert reduced["kl_div"]["avg"] == pytest.approx(0.3)


def test_retain_teacher_channels_projects_full_teacher_to_student_width():
    student_hidden = torch.randn(2, 5, 6)
    student_head = torch.randn(32, 6)
    teacher_hidden = torch.randn(2, 5, 8)
    teacher_head = torch.randn(32, 8)

    retained_hidden, retained_head = retain_teacher_channels(
        student_hidden,
        student_head,
        teacher_hidden,
        teacher_head,
    )

    torch.testing.assert_close(retained_hidden, teacher_hidden[..., :6])
    torch.testing.assert_close(retained_head, teacher_head[:, :6])


def test_retain_teacher_channels_uses_sorted_basis_indices():
    student_hidden = torch.randn(2, 5, 4)
    student_head = torch.randn(32, 4)
    teacher_hidden = torch.randn(2, 5, 8)
    teacher_head = torch.randn(32, 8)
    order = torch.tensor([7, 2, 5, 1])

    retained_hidden, retained_head = retain_teacher_channels(
        student_hidden,
        student_head,
        teacher_hidden,
        teacher_head,
        channel_indices=order,
    )

    torch.testing.assert_close(retained_hidden, teacher_hidden.index_select(-1, order))
    torch.testing.assert_close(retained_head, teacher_head.index_select(-1, order))


def test_source_hidden_channel_indices_reads_sorted_permutation(tmp_path):
    (tmp_path / "sorted_permutations.json").write_text(
        json.dumps({"embedding.hidden_order": [7, 2, 5, 1, 6, 0, 4, 3]})
    )

    assert _source_hidden_channel_indices(tmp_path, 4, 8) == (7, 2, 5, 1)
    assert _source_hidden_channel_indices(tmp_path / "unsorted", 4, 8) == (0, 1, 2, 3)


def test_aggregate_exposes_explicit_raw_replacement_loss():
    per_batch = [
        {
            "_cp_hidden_dot": [0.0],
            "_cp_hidden_candidate_sq": [1.0],
            "_cp_hidden_teacher_sq": [1.0],
            "_cp_hidden_diff_sq": [4.0],
            "_cp_hidden_target_eps_sq": [2.0],
            "_cp_hidden_abs_diff": [2.0],
            "_cp_hidden_count": [2.0],
        }
    ]
    reduced = aggregate_solution_scores(per_batch)
    assert reduced["raw_replacement_loss"]["avg"] == pytest.approx(2.0)
    assert reduced["raw_replacement_loss"] == reduced["mse_loss_hidden_states"]


def test_solution_hidden_width_supports_direct_and_wrapped_solutions():
    assert _solution_hidden_width({"hidden_width": 768}) == 768
    assert _solution_hidden_width({"puzzle_solution": {"hidden_width": 1024}}) == 1024
    assert _solution_hidden_width({}) is None


def test_solution_metrics_apply_separate_supervision_and_hidden_masks():
    torch.manual_seed(4)
    hidden = torch.randn(1, 4, 6)
    teacher = hidden.clone()
    teacher[:, 1:] += 100.0
    head = torch.randn(20, 6)
    labels = torch.tensor([[3, 4, 5, 6]])
    supervised = torch.tensor([[True, False, False, False]])

    out = score_batch(
        hidden,
        head,
        teacher,
        head,
        labels,
        ce_mask=supervised,
        kd_mask=supervised,
        hidden_mask=supervised,
        chunk_size=7,
    )

    expected_ce = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden[:, :1], head).reshape(1, -1),
        labels[:, :1].reshape(-1),
    )
    assert out["lm_loss"][0] == pytest.approx(float(expected_ce), rel=1e-5)
    assert out["mse_loss_hidden_states"][0] == pytest.approx(0.0)


class _RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, dtype=torch.float64))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon) * self.weight


class _MLAProjection(nn.Module):
    def __init__(self, hidden: int, q_rank: int, kv_rank: int, rope_dim: int):
        super().__init__()
        self.kv_lora_rank = kv_rank
        self.q_a_proj = nn.Linear(hidden, q_rank, bias=False, dtype=torch.float64)
        self.q_a_layernorm = _RMSNorm(q_rank)
        self.q_b_proj = nn.Linear(q_rank, 7, bias=False, dtype=torch.float64)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, kv_rank + rope_dim, bias=False, dtype=torch.float64
        )
        self.kv_a_layernorm = _RMSNorm(kv_rank)
        self.kv_b_proj = nn.Linear(kv_rank, 9, bias=False, dtype=torch.float64)

    def forward(self, x):
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        compressed = self.kv_a_proj_with_mqa(x)
        kv, rope = compressed.split((self.kv_lora_rank, compressed.shape[-1] - self.kv_lora_rank), -1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv))
        return q, kv, rope


class _MLALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _MLAProjection(hidden=6, q_rank=4, kv_rank=3, rope_dim=2)


def test_runtime_mla_prefix_slice_matches_physical_projection_and_restores() -> None:
    torch.manual_seed(9)
    layer = _MLALayer()
    x = torch.randn(5, 6, dtype=torch.float64)
    teacher = BlockConfig(
        subblock_configs=(MLAConfig(q_lora_rank=4, kv_lora_rank=3),)
    )
    child = BlockConfig(
        subblock_configs=(MLAConfig(q_lora_rank=2, kv_lora_rank=1),)
    )
    original = layer.self_attn(x)

    handle = apply_runtime_candidate(layer, teacher, child)
    actual = layer.self_attn(x)
    handle.remove()

    mla = layer.self_attn
    q_a = x @ mla.q_a_proj.weight[:2].t()
    q_norm = q_a * torch.rsqrt(q_a.square().mean(-1, keepdim=True) + 1e-6)
    q = (q_norm * mla.q_a_layernorm.weight[:2]) @ mla.q_b_proj.weight[:, :2].t()
    compressed = x @ mla.kv_a_proj_with_mqa.weight.t()
    kv_a = compressed[:, :1]
    kv_norm = kv_a * torch.rsqrt(kv_a.square().mean(-1, keepdim=True) + 1e-6)
    kv = (kv_norm * mla.kv_a_layernorm.weight[:1]) @ mla.kv_b_proj.weight[:, :1].t()
    rope = compressed[:, 3:]

    for got, expected in zip(actual, (q, kv, rope)):
        torch.testing.assert_close(got, expected)
    for got, expected in zip(layer.self_attn(x), original):
        torch.testing.assert_close(got, expected)


class _MLAHeadsProjection(nn.Module):
    def __init__(self, num_heads: int, v_head_dim: int, hidden: int):
        super().__init__()
        self.o_proj = nn.Linear(
            num_heads * v_head_dim,
            hidden,
            bias=False,
            dtype=torch.float64,
        )

    def forward(self, head_outputs):
        return self.o_proj(head_outputs)


class _MLAHeadsLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _MLAHeadsProjection(num_heads=4, v_head_dim=3, hidden=5)


def test_runtime_mla_head_slice_masks_sorted_o_proj_prefix_and_restores() -> None:
    torch.manual_seed(10)
    layer = _MLAHeadsLayer()
    head_outputs = torch.randn(7, 12, dtype=torch.float64)
    teacher = BlockConfig(
        subblock_configs=(MLAConfig(num_heads=4, q_lora_rank=8, kv_lora_rank=6),)
    )
    child = BlockConfig(
        subblock_configs=(MLAConfig(num_heads=2, q_lora_rank=8, kv_lora_rank=6),)
    )
    original = layer.self_attn(head_outputs)

    handle = apply_runtime_candidate(layer, teacher, child)
    actual = layer.self_attn(head_outputs)
    handle.remove()

    expected = torch.nn.functional.linear(
        head_outputs[:, :6],
        layer.self_attn.o_proj.weight[:, :6],
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(layer.self_attn(head_outputs), original)


@pytest.mark.parametrize(("target", "expected"), [(256, 256), ("full", None)])
def test_runtime_attention_window_change_is_exact_and_reversible(target, expected) -> None:
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.sliding_window = 512

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    layer = Layer()
    teacher = BlockConfig(
        subblock_configs=(AttentionConfig(sliding_window_size=512),)
    )
    child = BlockConfig(
        subblock_configs=(AttentionConfig(sliding_window_size=target),)
    )

    handle = apply_runtime_candidate(layer, teacher, child)
    assert layer.self_attn.sliding_window == expected
    handle.remove()

    assert layer.self_attn.sliding_window == 512

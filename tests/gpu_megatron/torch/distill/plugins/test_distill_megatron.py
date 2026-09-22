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

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import run_mcore_inference_with_dummy_input
from _test_utils.torch.misc import set_seed

import modelopt.torch.distill as mtd
from modelopt.torch.distill.plugins.megatron import (
    DistillationConfig,
    LogitsAndIntermediatesLossBalancer,
    LogitsKLLoss,
    TopKLogitsKLLoss,
    _mtp_excluded_from_quantization,
    adjust_distillation_model_for_mcore,
    setup_distillation_config,
)
from modelopt.torch.quantization.nn import TensorQuantizer

SEED = 1234


def _test_logits_kl_loss(rank, size):
    """Test basic LogitsKLLoss with simple forward/backward pass."""
    set_seed(SEED)

    num_layers = 2
    hidden_size = 8
    num_attention_heads = 4
    num_query_groups = 2
    ffn_hidden_size = 8
    max_sequence_length = 8
    vocab_size = 32
    batch_size = 2

    # Create teacher model (slightly larger)
    teacher_model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="squared_relu",
    ).cuda()

    # Create student model (same size for simplicity)
    student_model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=False,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="squared_relu",
    ).cuda()

    # Setup distillation config
    distill_cfg = setup_distillation_config(
        config_or_path=None,
        student_cfg=student_model.config,
        teacher_cfg=teacher_model.config,
    )

    # Convert to distillation model
    kd_config = {
        "teacher_model": teacher_model,
        "criterion": distill_cfg.criterion,
        "loss_balancer": distill_cfg.loss_balancer,
    }
    distillation_model = mtd.convert(student_model, mode=[("kd_loss", kd_config)])

    # Apply Megatron-specific adjustments
    adjust_distillation_model_for_mcore(distillation_model, distill_cfg)

    # Forward pass with dummy input
    distillation_model.train()
    run_mcore_inference_with_dummy_input(distillation_model, batch_size, hidden_size)

    # Forward and backward pass to verify gradients
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    labels = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    position_ids = (
        torch.arange(max_sequence_length, dtype=torch.long)
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .cuda()
    )
    attention_mask = torch.tril(
        torch.ones((batch_size, 1, max_sequence_length, max_sequence_length), dtype=torch.bool)
    ).cuda()

    student_loss = distillation_model(prompt_tokens, position_ids, attention_mask, labels=labels)

    # Compute distillation loss
    loss = distillation_model.compute_kd_loss(
        student_loss=student_loss, loss_reduction_fn=lambda x: x[0].mean()
    )
    assert isinstance(loss, dict), "Loss should be a dictionary"
    assert "kd_loss" in loss, "Should contain kd_loss key"

    # Backward pass
    loss["kd_loss"].backward()


def _test_topk_logits_kl_loss(kd_kwargs, rank, size):
    """Test TopKLogitsKLLoss with simple forward/backward pass."""
    set_seed(SEED)

    num_layers = 2
    hidden_size = 8
    num_attention_heads = 4
    num_query_groups = 2
    ffn_hidden_size = 8
    max_sequence_length = 8
    vocab_size = 128
    batch_size = 2

    # Create teacher model
    teacher_model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="squared_relu",
    ).cuda()

    # Create student model
    student_model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=False,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="squared_relu",
    ).cuda()

    # Setup distillation config with TopKLogitsKLLoss via logit_kl_topk argument
    distill_cfg = setup_distillation_config(
        config_or_path=DistillationConfig(**kd_kwargs),
        student_cfg=student_model.config,
        teacher_cfg=teacher_model.config,
    )

    # Convert to distillation model
    kd_config = {
        "teacher_model": teacher_model,
        "criterion": distill_cfg.criterion,
        "loss_balancer": distill_cfg.loss_balancer,
    }
    distillation_model = mtd.convert(student_model, mode=[("kd_loss", kd_config)])

    # Apply Megatron-specific adjustments
    adjust_distillation_model_for_mcore(distillation_model, distill_cfg)

    # Forward pass with dummy input
    distillation_model.train()
    run_mcore_inference_with_dummy_input(distillation_model, batch_size, hidden_size)

    # Forward and backward pass to verify gradients
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    labels = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    position_ids = (
        torch.arange(max_sequence_length, dtype=torch.long)
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .cuda()
    )
    attention_mask = torch.tril(
        torch.ones((batch_size, 1, max_sequence_length, max_sequence_length), dtype=torch.bool)
    ).cuda()

    student_loss = distillation_model(prompt_tokens, position_ids, attention_mask, labels=labels)

    # Compute distillation loss
    loss = distillation_model.compute_kd_loss(
        student_loss=student_loss, loss_reduction_fn=lambda x: x[0].mean()
    )
    assert isinstance(loss, dict), "Loss should be a dictionary"
    assert "kd_loss" in loss, "Should contain kd_loss key"

    # All TP ranks operate on the same global Top-K, so the loss must be identical across ranks.
    gathered = [torch.empty_like(loss["kd_loss"]) for _ in range(size)]
    torch.distributed.all_gather(gathered, loss["kd_loss"].detach())
    for other in gathered[1:]:
        assert torch.allclose(gathered[0], other), "Top-K KD loss differs across TP ranks"

    # Backward pass
    loss["kd_loss"].backward()


def _test_skip_lm_loss_with_mtp(rank, size):
    """Test that skip_lm_loss only zeroes the main LM head and not MTP heads."""
    set_seed(SEED)

    num_layers = 2
    hidden_size = 8
    num_attention_heads = 4
    num_query_groups = 2
    ffn_hidden_size = 8
    max_sequence_length = 8
    vocab_size = 32
    batch_size = 2
    mtp_num_layers = 1

    teacher_model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="squared_relu",
        mtp_num_layers=mtp_num_layers,
    ).cuda()

    student_model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=False,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="squared_relu",
        mtp_num_layers=mtp_num_layers,
    ).cuda()

    distill_cfg = setup_distillation_config(
        config_or_path=DistillationConfig(kd_loss_alpha=1.0),  # skips LM loss
        student_cfg=student_model.config,
        teacher_cfg=teacher_model.config,
    )
    kd_config = {
        "teacher_model": teacher_model,
        "criterion": distill_cfg.criterion,
        "loss_balancer": distill_cfg.loss_balancer,
    }
    distillation_model = mtd.convert(student_model, mode=[("kd_loss", kd_config)])
    adjust_distillation_model_for_mcore(distillation_model, distill_cfg)

    # Intercept each call to compute_language_model_loss and record return values.
    recorded_losses = []
    original_patched = distillation_model.compute_language_model_loss

    def _recording_loss(labels, logits):
        loss = original_patched(labels, logits)
        recorded_losses.append(loss)
        return loss

    distillation_model.compute_language_model_loss = _recording_loss

    distillation_model.train()
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    labels = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    position_ids = (
        torch.arange(max_sequence_length, dtype=torch.long)
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .cuda()
    )
    attention_mask = torch.tril(
        torch.ones((batch_size, 1, max_sequence_length, max_sequence_length), dtype=torch.bool)
    ).cuda()

    distillation_model(prompt_tokens, position_ids, attention_mask, labels=labels)

    # Expect mtp_num_layers + 1 total calls: first mtp_num_layers are MTP heads,
    # the last one is the main LM head.
    assert len(recorded_losses) == mtp_num_layers + 1, (
        f"Expected {mtp_num_layers + 1} loss calls, got {len(recorded_losses)}"
    )
    for i, loss in enumerate(recorded_losses[:-1]):
        assert loss.any(), f"MTP head {i} loss should be non-zero with skip_lm_loss=True"
    assert not recorded_losses[-1].any(), "Main LM head loss should be zero with skip_lm_loss=True"


def test_logits_kl_loss(dist_workers):
    """Test LogitsKLLoss with TP parallelism."""
    dist_workers.run(_test_logits_kl_loss)


@pytest.mark.parametrize(
    ("top_p", "top_p_min_k"),
    [(None, 1), (0.9, 1), (0.9, 3)],
)
def test_topk_logits_kl_loss(dist_workers, top_p, top_p_min_k, top_k: int = 5):
    """Test TopKLogitsKLLoss with TP parallelism."""
    kd_kwargs = {
        "logit_kl_topk": top_k,
        "logit_kl_top_p": top_p,
        "logit_kl_top_p_min_k": top_p_min_k,
    }
    dist_workers.run(partial(_test_topk_logits_kl_loss, kd_kwargs))


def _make_loss_inputs(seq=4, batch=3, vocab=16):
    torch.manual_seed(SEED)
    student = torch.randn(seq, batch, vocab, requires_grad=True)
    teacher = torch.randn(seq, batch, vocab) * 3  # peaky teacher so top-P actually truncates
    return student, teacher


def test_topk_logits_kl_loss_numerics_full_vocab_matches_dense():
    """With K = vocab and ghost token, Top-K KL equals the dense full-vocab KL (residual ~0)."""
    cfg = SimpleNamespace(tensor_model_parallel_size=1)
    student, teacher = _make_loss_inputs()
    dense = LogitsKLLoss(cfg)(student, teacher)[0]
    topk = TopKLogitsKLLoss(cfg, top_k=student.size(-1))(student, teacher)[0]
    assert torch.allclose(dense, topk, atol=1e-6)


def test_topk_logits_kl_loss_numerics_ghost_token_reference():
    """Top-K + ghost token matches a hand-written reference on the K+1 bucketed distributions."""
    cfg = SimpleNamespace(tensor_model_parallel_size=1)
    student, teacher = _make_loss_inputs()
    k = 4
    loss = TopKLogitsKLLoss(cfg, top_k=k)(student, teacher)[0]

    q_full = F.log_softmax(teacher, dim=-1)
    p_full = F.log_softmax(student, dim=-1)
    _, idx = torch.topk(teacher, k, dim=-1)
    q_k, p_k = q_full.gather(-1, idx), p_full.gather(-1, idx)
    q_rest = torch.log1p(-q_k.exp().sum(-1, keepdim=True))
    p_rest = torch.log1p(-p_k.exp().sum(-1, keepdim=True))
    q = torch.cat([q_k, q_rest], -1)
    p = torch.cat([p_k, p_rest], -1)
    ref = (q.exp() * (q - p)).sum(-1).transpose(0, 1)
    assert torch.allclose(loss, ref, atol=1e-6)
    # Sanity: total mass within the K+1 buckets is 1 for both distributions.
    assert torch.allclose(q.exp().sum(-1), torch.ones_like(q[..., 0]), atol=1e-5)
    assert torch.allclose(p.exp().sum(-1), torch.ones_like(p[..., 0]), atol=1e-5)


@pytest.mark.parametrize("temperature", [0.5, 2.0, 3.7])
def test_logits_kl_losses_temperature_scaling(temperature):
    """Dense and Top-K losses match a plain ``log_softmax(x / T)`` reference at T != 1."""
    cfg = SimpleNamespace(tensor_model_parallel_size=1)
    student, teacher = _make_loss_inputs()
    q = F.log_softmax(teacher / temperature, dim=-1)
    p = F.log_softmax(student / temperature, dim=-1)

    dense = LogitsKLLoss(cfg, temperature=temperature)(student, teacher)[0]
    ref_dense = (q.exp() * (q - p)).sum(-1).transpose(0, 1)
    assert torch.allclose(dense, ref_dense, atol=1e-5)

    k = 4
    topk = TopKLogitsKLLoss(cfg, temperature=temperature, top_k=k)(student, teacher)[0]
    _, idx = torch.topk(teacher, k, dim=-1)
    q_k, p_k = q.gather(-1, idx), p.gather(-1, idx)
    q_rest = torch.log1p(-q_k.exp().sum(-1, keepdim=True))
    p_rest = torch.log1p(-p_k.exp().sum(-1, keepdim=True))
    qq = torch.cat([q_k, q_rest], -1)
    pp = torch.cat([p_k, p_rest], -1)
    ref_topk = (qq.exp() * (qq - pp)).sum(-1).transpose(0, 1)
    assert torch.allclose(topk, ref_topk, atol=1e-6)


def test_topk_logits_kl_loss_top_p_masks_tail():
    """Top-P moves out-of-nucleus mass into the ghost token and honors the min_k floor."""
    cfg = SimpleNamespace(tensor_model_parallel_size=1)
    student, teacher = _make_loss_inputs()
    k = 8
    q_full = F.log_softmax(teacher, dim=-1)
    q_k, idx = torch.topk(q_full, k, dim=-1)
    p_k = F.log_softmax(student, dim=-1).gather(-1, idx)

    def reference(keep):
        partial = (keep * q_k.exp() * (q_k - p_k)).sum(-1)
        q_rest = torch.log1p(-(q_k.exp() * keep).sum(-1))
        p_rest = torch.log1p(-(p_k.exp() * keep).sum(-1))
        return (partial + q_rest.exp() * (q_rest - p_rest)).transpose(0, 1)

    probs = q_k.exp()
    keep = (probs.cumsum(-1) - probs) < 0.5
    assert not keep.all(), "test inputs should produce some truncation"
    loss = TopKLogitsKLLoss(cfg, top_k=k, top_p=0.5)(student, teacher)[0]
    assert torch.allclose(loss, reference(keep), atol=1e-6)
    assert loss.shape == (student.size(1), student.size(0))

    # min_k floor forces at least min_k entries even when the nucleus is tiny.
    min_k = 3
    loss_min = TopKLogitsKLLoss(cfg, top_k=k, top_p=1e-6, top_p_min_k=min_k)(student, teacher)[0]
    assert torch.allclose(loss_min, reference(torch.arange(k) < min_k), atol=1e-6)

    loss.sum().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_distillation_config_top_p_validation():
    with pytest.raises(AssertionError):
        DistillationConfig(logit_kl_top_p=0.9)  # requires logit_kl_topk
    with pytest.raises(AssertionError):
        DistillationConfig(logit_kl_topk=8, logit_kl_top_p=1.5)
    with pytest.raises(AssertionError):
        DistillationConfig(logit_kl_topk=8, logit_kl_top_p=0.9, logit_kl_top_p_min_k=0)
    DistillationConfig(logit_kl_topk=8, logit_kl_top_p=1.0, logit_kl_top_p_min_k=2)


def test_skip_lm_loss_with_mtp(dist_workers):
    """Test that skip_lm_loss only zeroes the main LM head, not MTP heads."""
    dist_workers.run(_test_skip_lm_loss_with_mtp)


def test_mtp_excluded_from_quantization():
    """MTP loss is skipped only when the model is quantized and MTP is left out of it."""

    def _quantizer(enabled: bool) -> TensorQuantizer:
        quantizer = TensorQuantizer()
        if not enabled:
            quantizer.disable()
        return quantizer

    def _model(*, with_mtp: bool, body_quant: bool, mtp_quant: bool) -> nn.Module:
        model = nn.Module()
        model.decoder = nn.Module()
        if body_quant:
            model.decoder.weight_quantizer = _quantizer(True)
        if with_mtp:
            model.mtp = nn.Module()
            model.mtp.weight_quantizer = _quantizer(mtp_quant)
        return model

    # Plain distillation (e.g. pruning recovery) still trains the MTP head.
    assert not _mtp_excluded_from_quantization(
        _model(with_mtp=True, body_quant=False, mtp_quant=False)
    )
    # QAD with MTP excluded from the recipe: no quantization error to recover there.
    assert _mtp_excluded_from_quantization(_model(with_mtp=True, body_quant=True, mtp_quant=False))
    # QAD with a quantized MTP head: keep its loss.
    assert not _mtp_excluded_from_quantization(
        _model(with_mtp=True, body_quant=True, mtp_quant=True)
    )
    # No MTP at all.
    assert not _mtp_excluded_from_quantization(
        _model(with_mtp=False, body_quant=True, mtp_quant=False)
    )


def test_loss_balancer_convex_combination():
    """Total loss is (1 - alpha) * lm + alpha * (logits + rescaled intermediate)."""
    lm = torch.tensor(2.0)
    logits = torch.tensor(0.5)
    inter = torch.tensor(4.0)  # rescaled to logits magnitude -> contributes 0.5
    key = mtd.loss_balancers.STUDENT_LOSS_KEY

    out = LogitsAndIntermediatesLossBalancer(kd_loss_alpha=0.25)(
        {key: lm, "LogitsKLLoss_0": logits, "HiddenStateCosineLoss_0": inter}
    )
    assert torch.allclose(out["kd_loss"], torch.tensor(0.75 * 2.0 + 0.25 * (0.5 + 0.5)))
    assert torch.allclose(out["logits_loss"], logits)

    # alpha=1 ignores the LM loss entirely; skip_original_loss does the same regardless of alpha.
    out = LogitsAndIntermediatesLossBalancer(kd_loss_alpha=1.0)({key: lm, "LogitsKLLoss_0": logits})
    assert torch.allclose(out["kd_loss"], logits)
    out = LogitsAndIntermediatesLossBalancer(kd_loss_alpha=0.0, skip_original_loss=True)(
        {key: lm, "LogitsKLLoss_0": logits}
    )
    assert torch.allclose(out["kd_loss"], logits)

    with pytest.raises(AssertionError):
        LogitsAndIntermediatesLossBalancer(kd_loss_alpha=1.5)
    with pytest.raises(AssertionError):
        DistillationConfig(kd_loss_alpha=-0.1)


def test_distillation_config_deprecations():
    """skip_lm_loss is derived from kd_loss_alpha; legacy fields warn with FutureWarning."""
    assert DistillationConfig(kd_loss_alpha=1.0).skip_lm_loss is True
    assert DistillationConfig(kd_loss_alpha=0.9).skip_lm_loss is False

    # Explicit skip_lm_loss=True is translated to kd_loss_alpha=1.0 rather than overridden.
    with pytest.warns(FutureWarning, match="translating skip_lm_loss=True"):
        cfg = DistillationConfig(kd_loss_alpha=0.9, skip_lm_loss=True)
    assert cfg.kd_loss_alpha == 1.0 and cfg.skip_lm_loss is True

    # skip_lm_loss=False with alpha=1.0 is a conflict; alpha wins.
    with pytest.warns(FutureWarning, match="conflicts with kd_loss_alpha=1.0"):
        cfg = DistillationConfig(kd_loss_alpha=1.0, skip_lm_loss=False)
    assert cfg.skip_lm_loss is True

    # Consistent but deprecated usage still warns.
    with pytest.warns(FutureWarning, match="skip_lm_loss is deprecated"):
        cfg = DistillationConfig(kd_loss_alpha=0.9, skip_lm_loss=False)
    assert cfg.skip_lm_loss is False

    with pytest.warns(FutureWarning, match="kd_loss_scale is deprecated"):
        cfg = DistillationConfig(kd_loss_scale=2.0)
    assert cfg.kd_loss_alpha == 0.9

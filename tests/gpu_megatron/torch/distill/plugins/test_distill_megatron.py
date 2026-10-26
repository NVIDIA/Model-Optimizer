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

import dataclasses
import warnings
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import run_mcore_inference_with_dummy_input
from _test_utils.torch.misc import set_seed
from megatron.core import parallel_state

import modelopt.torch.distill as mtd
from modelopt.torch.distill.plugins.megatron import (
    DistillationConfig,
    LogitsAndIntermediatesLossBalancer,
    LogitsKLLoss,
    TopKLogitsKLLoss,
    TopLogitsKLLoss,
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
        # Reduce the per-token LM loss to a scalar, as Megatron's loss function does in training.
        student_loss=student_loss.mean(),
        loss_reduction_fn=lambda x: x[0].mean(),
    )
    assert isinstance(loss, dict), "Loss should be a dictionary"
    assert "kd_loss" in loss, "Should contain kd_loss key"

    # Backward pass
    loss["kd_loss"].backward()


def _test_topk_logits_kl_loss(kd_kwargs, rank, size):
    """Test TopLogitsKLLoss with simple forward/backward pass."""
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

    # Setup distillation config with TopLogitsKLLoss via logit_kl_topk argument
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
        # Reduce the per-token LM loss to a scalar, as Megatron's loss function does in training.
        student_loss=student_loss.mean(),
        loss_reduction_fn=lambda x: x[0].mean(),
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


def _test_skip_lm_loss_with_mtp(quantized, skip_lm_loss, freeze_mtp, rank, size):
    """Test that skip_lm_loss only zeroes the main LM head, unless MTP is left out of quantization.

    ``freeze_mtp`` zeroes the MTP loss and freezes the MTP head regardless.
    """
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
    if quantized:
        # A quantized body with the MTP head left out of quantization.
        student_model.decoder.weight_quantizer = TensorQuantizer()

    distill_cfg = setup_distillation_config(
        # kd_loss_alpha == 1.0 skips the LM loss; any alpha < 1 keeps it.
        config_or_path=DistillationConfig(
            kd_loss_alpha=1.0 if skip_lm_loss else 0.5, freeze_mtp=freeze_mtp
        ),
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

    # An untrained MTP head is frozen so DDP doesn't wait for its grads.
    mtp_skipped = freeze_mtp or (quantized and skip_lm_loss)
    with distillation_model.hide_teacher_model():
        mtp_params = [p for n, p in distillation_model.named_parameters() if "mtp" in n.split(".")]
    assert mtp_params
    assert all(p.requires_grad != mtp_skipped for p in mtp_params)

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
        assert loss.any() != mtp_skipped, f"MTP head {i} loss should be zero iff MTP is skipped"
    assert recorded_losses[-1].any() != skip_lm_loss, "Main LM head loss is zero iff skip_lm_loss"


def _test_freeze_mtp_with_quantized_mtp(freeze_mtp, kd_loss_alpha, rank, size):
    """With the MTP head quantized too, only ``freeze_mtp`` keeps it out of a training step."""
    set_seed(SEED)

    vocab_size = 32
    batch_size = 2
    max_sequence_length = 8
    model_kwargs = {
        "tensor_model_parallel_size": size,
        "pipeline_model_parallel_size": 1,
        "num_layers": 2,
        "hidden_size": 8,
        "num_attention_heads": 4,
        "num_query_groups": 2,
        "ffn_hidden_size": 8,
        "max_sequence_length": max_sequence_length,
        "vocab_size": vocab_size,
        "activation_func": "squared_relu",
        "mtp_num_layers": 1,
    }
    teacher_model = get_mcore_gpt_model(initialize_megatron=True, **model_kwargs).cuda()
    student_model = get_mcore_gpt_model(initialize_megatron=False, **model_kwargs).cuda()
    # A recipe that quantizes the MTP head as well as the body.
    student_model.decoder.weight_quantizer = TensorQuantizer()
    student_model.mtp.weight_quantizer = TensorQuantizer()
    assert not _mtp_excluded_from_quantization(student_model)

    distill_cfg = setup_distillation_config(
        config_or_path=DistillationConfig(kd_loss_alpha=kd_loss_alpha, freeze_mtp=freeze_mtp),
        student_cfg=student_model.config,
        teacher_cfg=teacher_model.config,
    )
    kd_config = {
        "teacher_model": teacher_model,
        "criterion": distill_cfg.criterion,
        "loss_balancer": distill_cfg.loss_balancer,
    }
    distillation_model = mtd.convert(student_model, mode=[("kd_loss", kd_config)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adjust_distillation_model_for_mcore(distillation_model, distill_cfg)
    if rank == 0:
        frozen_msg = [w for w in caught if "MTP head is frozen" in str(w.message)]
        assert bool(frozen_msg) == freeze_mtp

    with distillation_model.hide_teacher_model():
        named_params = dict(distillation_model.named_parameters())
    mtp_params = {n: p for n, p in named_params.items() if "mtp" in n.split(".")}
    body_params = {n: p for n, p in named_params.items() if n.startswith("decoder.")}
    assert mtp_params and body_params
    mtp_before = {n: p.detach().clone() for n, p in mtp_params.items()}
    optimizer = torch.optim.AdamW(
        [p for p in named_params.values() if p.requires_grad], lr=1e-2, weight_decay=0.1
    )

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
    student_loss = distillation_model(prompt_tokens, position_ids, attention_mask, labels=labels)
    loss = distillation_model.compute_kd_loss(
        student_loss=student_loss.mean(), loss_reduction_fn=lambda x: x[0].mean()
    )
    loss["kd_loss"].backward()
    optimizer.step()

    # The main LM loss follows kd_loss_alpha either way.
    assert student_loss.any() == (kd_loss_alpha < 1)
    expected = (1 - kd_loss_alpha) * student_loss.mean() + kd_loss_alpha * loss["logits_loss"]
    assert torch.allclose(loss["kd_loss"], expected)
    # The rest of the student keeps training.
    assert any(p.grad is not None and p.grad.any() for p in body_params.values())
    # The MTP head trains unless frozen; frozen, it gets no gradient and the step leaves it as is.
    mtp_trained = any(p.grad is not None and p.grad.any() for p in mtp_params.values())
    assert mtp_trained != freeze_mtp
    if freeze_mtp:
        assert all(torch.equal(p, mtp_before[n]) for n, p in mtp_params.items())


def test_logits_kl_loss(dist_workers):
    """Test LogitsKLLoss with TP parallelism."""
    dist_workers.run(_test_logits_kl_loss)


@pytest.mark.parametrize(
    ("top_p", "top_p_min_k"),
    [(None, 1), (0.9, 1), (0.9, 3)],
)
def test_topk_logits_kl_loss(dist_workers, top_p, top_p_min_k, top_k: int = 5):
    """Test TopLogitsKLLoss with TP parallelism."""
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
    topk = TopLogitsKLLoss(cfg, top_k=student.size(-1))(student, teacher)[0]
    assert torch.allclose(dense, topk, atol=1e-6)


def _test_topk_logits_kl_loss_numerics_ghost_token_reference(
    top_p, reverse, temperature, rank, size
):
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=size)
    cfg = SimpleNamespace(tensor_model_parallel_size=size)
    device = torch.device("cuda", rank)
    # Equal vocab shards (torch.chunk is uneven when size does not divide the vocab)
    student, teacher = _make_loss_inputs(vocab=16 * size)
    student = student.detach().to(device).requires_grad_()
    teacher = teacher.to(device)
    student_shard = student.chunk(size, dim=-1)[rank].detach().contiguous().requires_grad_()
    teacher_shard = teacher.chunk(size, dim=-1)[rank].contiguous()
    k = 4
    loss = TopLogitsKLLoss(cfg, top_k=k, top_p=top_p, reverse=reverse, temperature=temperature)(
        student_shard, teacher_shard
    )[0]

    q_full = F.log_softmax(teacher / temperature, dim=-1)
    p_full = F.log_softmax(student / temperature, dim=-1)
    _, idx = torch.topk(teacher, k, dim=-1)
    q_k, p_k = q_full.gather(-1, idx), p_full.gather(-1, idx)
    keep = torch.ones_like(q_k, dtype=torch.bool)
    if top_p is not None:
        q_probs = q_k.exp()
        keep = (q_probs.cumsum(-1) - q_probs) < top_p
        assert not keep.all(), "test inputs should produce some truncation"
    q_rest = torch.log1p(-(q_k.exp() * keep).sum(-1, keepdim=True))
    p_rest = torch.log1p(-(p_k.exp() * keep).sum(-1, keepdim=True))
    q = torch.cat([q_k, q_rest], -1)
    p = torch.cat([p_k, p_rest], -1)
    keep = torch.cat([keep, torch.ones_like(keep[..., :1])], -1)
    # Sanity: total mass within the K+1 buckets is 1 for both distributions.
    assert torch.allclose((q.exp() * keep).sum(-1), torch.ones_like(q[..., 0]), atol=1e-5)
    assert torch.allclose((p.exp() * keep).sum(-1), torch.ones_like(p[..., 0]), atol=1e-5)
    if reverse:
        p, q = q, p
    ref = (keep * q.exp() * (q - p)).sum(-1).transpose(0, 1)
    torch.testing.assert_close(loss, ref, atol=1e-6, rtol=1e-5)
    loss.mean().backward()
    ref.mean().backward()
    torch.testing.assert_close(
        student_shard.grad, student.grad.chunk(size, dim=-1)[rank], atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize(
    ("top_p", "reverse", "temperature"),
    [(None, False, 1.0), (0.5, False, 2.0), (0.5, True, 1.0)],
)
def test_topk_logits_kl_loss_numerics_ghost_token_reference(
    dist_workers_size_2, top_p, reverse, temperature
):
    """TP loss and gradients match a full-vocab reference on the K+1 bucketed distributions."""
    dist_workers_size_2.run(
        partial(
            _test_topk_logits_kl_loss_numerics_ghost_token_reference, top_p, reverse, temperature
        )
    )


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
    topk = TopLogitsKLLoss(cfg, temperature=temperature, top_k=k)(student, teacher)[0]
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
    loss = TopLogitsKLLoss(cfg, top_k=k, top_p=0.5)(student, teacher)[0]
    assert torch.allclose(loss, reference(keep), atol=1e-6)
    assert loss.shape == (student.size(1), student.size(0))

    # min_k floor forces at least min_k entries even when the nucleus is tiny.
    min_k = 3
    loss_min = TopLogitsKLLoss(cfg, top_k=k, top_p=1e-6, top_p_min_k=min_k)(student, teacher)[0]
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


@pytest.mark.parametrize(
    ("quantized", "skip_lm_loss", "freeze_mtp"),
    [
        (False, True, False),
        (True, True, False),
        (True, False, False),
        (False, True, True),
        (False, False, True),
    ],
)
def test_skip_lm_loss_with_mtp(dist_workers, quantized, skip_lm_loss, freeze_mtp):
    """Test that skip_lm_loss only zeroes the main LM head, unless MTP is left out of quantization."""
    dist_workers.run(partial(_test_skip_lm_loss_with_mtp, quantized, skip_lm_loss, freeze_mtp))


@pytest.mark.parametrize(("freeze_mtp", "kd_loss_alpha"), [(True, 1.0), (True, 0.5), (False, 1.0)])
def test_freeze_mtp_with_quantized_mtp(dist_workers, freeze_mtp, kd_loss_alpha):
    """With the MTP head quantized too, only ``freeze_mtp`` keeps it out of a training step."""
    dist_workers.run(partial(_test_freeze_mtp_with_quantized_mtp, freeze_mtp, kd_loss_alpha))


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

    # Zero intermediate loss contributes nothing (and no division by zero).
    out = LogitsAndIntermediatesLossBalancer(kd_loss_alpha=1.0)(
        {key: lm, "LogitsKLLoss_0": logits, "HiddenStateCosineLoss_0": torch.tensor(0.0)}
    )
    assert torch.allclose(out["kd_loss"], logits)
    assert torch.isfinite(out["intermediate_loss"])

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


def test_distillation_config_removed_fields():
    """The removed fields raise, and a config can be rebuilt from itself."""
    cfg = DistillationConfig()
    assert cfg.kd_loss_alpha == 1.0 and cfg.skip_lm_loss  # pure KD by default
    assert not DistillationConfig(kd_loss_alpha=0.9).skip_lm_loss
    assert {"skip_lm_loss", "kd_loss_scale"}.isdisjoint(f.name for f in dataclasses.fields(cfg))

    for removed in ({"skip_lm_loss": True}, {"skip_lm_loss": False}, {"kd_loss_scale": 2.0}):
        with pytest.raises(ValueError, match="have been removed"):
            DistillationConfig(**removed)

    # skip_lm_loss is a derived property, not a field, so round-trips do not trip the check.
    cfg = dataclasses.replace(DistillationConfig(kd_loss_alpha=0.9), logit_kl_topk=4)
    assert cfg.kd_loss_alpha == 0.9 and cfg.logit_kl_topk == 4
    assert DistillationConfig(**dataclasses.asdict(cfg)).kd_loss_alpha == 0.9


def test_topk_logits_kl_loss_deprecated_alias():
    """The old class name still works and warns."""
    cfg = SimpleNamespace(tensor_model_parallel_size=1)
    with pytest.warns(FutureWarning, match="use TopLogitsKLLoss instead"):
        loss_fn = TopKLogitsKLLoss(cfg, top_k=4)
    assert isinstance(loss_fn, TopLogitsKLLoss)

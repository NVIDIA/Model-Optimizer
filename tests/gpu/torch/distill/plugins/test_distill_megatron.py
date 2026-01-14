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

import torch
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron(apex_or_te_required=True)

from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import run_mcore_inference_with_dummy_input
from _test_utils.torch.misc import set_seed

import modelopt.torch.distill as mtd
from modelopt.torch.distill.plugins.megatron import (
    DistillationConfig,
    adjust_distillation_model_for_mcore,
    setup_distillation_config,
)

SEED = 1234


def _test_logits_kl_loss(rank, size):
    """Test basic LogitsKLLoss with simple forward/backward pass."""
    channel_divisor = 4

    num_layers = 2
    hidden_size = channel_divisor * 2
    num_attention_heads = 4
    num_query_groups = 2
    ffn_hidden_size = channel_divisor * 2
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


def _test_topk_logits_kl_loss(top_k, rank, size):
    """Test TopKLogitsKLLoss with simple forward/backward pass."""
    channel_divisor = 4

    num_layers = 2
    hidden_size = channel_divisor * 2
    num_attention_heads = 4
    num_query_groups = 2
    ffn_hidden_size = channel_divisor * 2
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
        config_or_path=DistillationConfig(logit_kl_topk=top_k),
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


def test_logits_kl_loss():
    """Test LogitsKLLoss with TP parallelism."""
    set_seed(SEED)
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=_test_logits_kl_loss,
        backend="nccl",
    )


def test_topk_logits_kl_loss(top_k: int = 5):
    """Test TopKLogitsKLLoss with TP parallelism."""
    set_seed(SEED)
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_topk_logits_kl_loss, top_k),
        backend="nccl",
    )

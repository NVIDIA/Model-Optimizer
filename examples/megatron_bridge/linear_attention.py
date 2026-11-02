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

"""Linear-attention hooks for the standard Megatron Bridge training loop."""

from contextlib import ExitStack, contextmanager

from linear_attention_batch import mask_prefix_labels, phase_lengths
from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.training.gpt_step import forward_step_modelopt
from megatron.bridge.training.post_training.checkpointing import (
    has_modelopt_state,
    load_modelopt_state,
)
from megatron.bridge.training.state import GlobalState
from megatron.core.ssm import gated_delta_net
from megatron.core.utils import unwrap_model

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import (
    get_linear_attention_layers,
    linear_attention_training_phase,
)
from modelopt.torch.quantization.utils import is_quantized

__all__ = ["linear_attention_training"]


@contextmanager
def linear_attention_training(
    config,
    quant_config,
    prefill_tokens=None,
    teacher_provider=None,
    *,
    train_all_weights=True,
    teacher_serving_arithmetic=False,
    recompute_gdn=False,
):
    """Install quantization hooks and yield a phase-aware ModelOpt forward step."""
    if config.model.context_parallel_size != 1:
        raise ValueError("Linear-attention QAT/QAD currently requires CP=1")
    if not quant_config.linear_attention:
        raise ValueError("A linear-attention quantization policy is required")
    operand_qat = any(entry.cfg.backend == "gdn_prefill" for entry in quant_config.linear_attention)
    data_lengths = prefill_tokens is None or isinstance(config.dataset, FinetuningDatasetConfig)
    if data_lengths and config.model.pipeline_model_parallel_size > 1:
        raise ValueError(
            "Data-derived state-QAT phases currently require PP=1; fixed splits support PP"
        )
    if prefill_tokens is not None and not 0 <= prefill_tokens < config.dataset.seq_length:
        raise ValueError("prefill-tokens must leave at least one suffix label")
    fixed_phase = config.dataset.seq_length if operand_qat else prefill_tokens
    resume = config.checkpoint.load and has_modelopt_state(config.checkpoint.load)
    with ExitStack() as phases:

        def prepare_student(models):
            # Restore the saved policy before quantization and before the QAD conversion hook.
            if resume:
                load_modelopt_state(models, config.checkpoint.load)
            layer_types = (gated_delta_net.GatedDeltaNet,)
            if hasattr(gated_delta_net, "KimiDeltaAttention"):
                layer_types += (gated_delta_net.KimiDeltaAttention,)
            for student in unwrap_model(models):
                layers = [m for m in student.modules() if isinstance(m, layer_types)]
                if not layers:
                    raise ValueError(
                        "Each local model chunk must contain Megatron GatedDeltaNet or "
                        "KimiDeltaAttention; choose a pipeline layout with linear attention on every stage"
                    )
                student.requires_grad_(train_all_weights)
                if not train_all_weights:
                    for layer in layers:
                        layer.requires_grad_(True)
                if not is_quantized(student):
                    mtq.quantize(student, quant_config)
                if not get_linear_attention_layers(student):
                    raise ValueError(
                        "The student checkpoint has no linear-attention training policy; "
                        "initialize state/operand quantization before resuming QAD"
                    )
                if recompute_gdn:
                    for layer in layers:
                        layer.recompute_gdn = True
                if not data_lengths:
                    phases.enter_context(
                        linear_attention_training_phase(
                            student, [fixed_phase] * config.train.micro_batch_size
                        )
                    )
            return models

        def masked_batch(data_iterator):
            batch = dict(next(data_iterator))
            mask_prefix_labels(batch, [prefill_tokens] * batch["loss_mask"].shape[0])
            yield batch

        def forward_step(state: GlobalState, data_iterator, model, return_schedule_plan=False):
            if data_lengths:
                # Bridge consumes and inspects the data iterator on every TP rank before
                # broadcasting tensor fields. Supplying None on non-source ranks crashes.
                batch = dict(next(data_iterator))
                prefix, valid, boundaries = phase_lengths(batch, prefill_tokens)
                if prefill_tokens is not None:
                    mask_prefix_labels(batch, prefix, boundaries)
                with linear_attention_training_phase(
                    unwrap_model(model),
                    valid if operand_qat else prefix,
                    sequence_lengths=valid,
                    cu_seqlens=boundaries,
                ):
                    return forward_step_modelopt(state, iter([batch]), model, return_schedule_plan)
            # Consume lazily: middle pipeline stages may not need any batch tensors.
            return forward_step_modelopt(
                state, masked_batch(data_iterator), model, return_schedule_plan
            )

        config.model.register_pre_wrap_hook(prepare_student, prepend=True)
        if teacher_provider is not None and teacher_serving_arithmetic:

            def prepare_teacher(models):
                teacher_config = quant_config.model_dump()
                teacher_config["quant_cfg"] = [{"quantizer_name": "*", "enable": False}]
                for teacher in unwrap_model(models):
                    mtq.quantize(teacher, teacher_config)
                    teacher.requires_grad_(False).eval()
                    if not data_lengths:
                        phases.enter_context(
                            linear_attention_training_phase(
                                teacher, [fixed_phase] * config.train.micro_batch_size
                            )
                        )
                return models

            teacher_provider.register_pre_wrap_hook(prepare_teacher)
        yield forward_step

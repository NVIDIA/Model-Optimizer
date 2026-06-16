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

"""ModelOpt plugin to train HuggingFace models with knowledge distillation.

Only logit-level distillation is supported. For intermediate-layer distillation
or Megatron models, use ``mtd.convert()`` directly.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import field

import torch
import torch.nn as nn
from transformers.trainer_pt_utils import find_batch_size

from modelopt.torch.distill.losses import LogitsDistillationLoss
from modelopt.torch.opt.plugins import ModelOptHFTrainer
from modelopt.torch.opt.plugins.transformers import (
    _LIGER_KERNEL_IMPORT_ERROR,
    ModelOptHFArguments,
    _forward_redirect,
    is_liger_available,
)
from modelopt.torch.utils import print_rank_0

__all__ = [
    "IGNORE_INDEX",
    "DistillArgsWithTeacherModel",
    "DistillArguments",
    "KDTrainer",
]

IGNORE_INDEX = nn.CrossEntropyLoss().ignore_index

_SUPPORTED_CRITERIA = {"logits_loss"}


class DistillArguments(ModelOptHFArguments):
    """Distillation arguments for knowledge distillation training."""

    distill: bool = field(
        default=False,
        metadata={"help": "Enable training with knowledge distillation."},
    )
    teacher_model: str | None = field(
        default=None,
        metadata={"help": "The name or path of the teacher model."},
    )
    criterion: str = field(
        default="logits_loss",
        metadata={
            "help": "Distillation loss criterion. Currently only 'logits_loss' is supported."
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": (
                "Softmax temperature for softening logits in KD loss. "
                "Used by both standard and Liger KD loss."
            )
        },
    )
    liger_jsd_beta: float = field(
        default=0.0,
        metadata={
            "help": (
                "JSD beta coefficient in [0, 1]. 0=forward KL, 1=reverse KL. "
                "Only used when --use_liger_kernel is enabled."
            )
        },
    )


class DistillArgsWithTeacherModel(DistillArguments):
    """Runtime distillation arguments with a pre-loaded teacher model."""

    teacher_model: nn.Module | None = field(
        default=None,
        metadata={"help": "Pre-loaded teacher model."},
    )


class KDTrainer(ModelOptHFTrainer):
    """Distillation trainer for HuggingFace models.

    Supports logit-level knowledge distillation only. The teacher model is stored
    separately on the trainer and forwarded explicitly during loss computation.
    No ``mtd.convert()`` or ``DistillationModel`` wrapping is used.
    """

    def __init__(
        self,
        *args,
        distill_args: DistillArgsWithTeacherModel | dict | None = None,
        **kwargs,
    ):
        """Initialize the trainer.

        Args:
            distill_args: Runtime distillation config with a pre-loaded teacher model.
        """
        super().__init__(*args, **kwargs)
        if self.is_fsdp_enabled and not self.accelerator.is_fsdp2:
            raise ValueError("FSDP1 is not supported for distillation. Use FSDP2 instead.")

        if distill_args is None:
            raise ValueError("`distill_args` is required for distillation.")
        if isinstance(distill_args, dict):
            distill_args = DistillArgsWithTeacherModel(**distill_args)

        if distill_args.criterion not in _SUPPORTED_CRITERIA:
            raise ValueError(
                f"Unsupported criterion: {distill_args.criterion!r}. "
                f"Supported: {_SUPPORTED_CRITERIA}"
            )

        teacher = distill_args.teacher_model
        if teacher is None:
            raise ValueError("`distill_args.teacher_model` is required.")
        if not isinstance(teacher, nn.Module):
            raise TypeError(
                "`distill_args.teacher_model` must be a pre-loaded nn.Module. "
                "Load the teacher in the training script before constructing KDTrainer."
            )

        self._teacher_model = teacher
        self._teacher_model.requires_grad_(False)
        self._kd_criterion = LogitsDistillationLoss(
            temperature=distill_args.temperature, reduction="none"
        )
        self._teacher_prepared = False
        self._eval_ce_loss_totals = None

        if self.use_liger_kernel:
            self._liger_temperature = distill_args.temperature
            self._liger_jsd_beta = distill_args.liger_jsd_beta

    def _setup_liger_fused_loss(self):
        """Set student fused-loss path and require Liger KD dependencies."""
        if not is_liger_available():
            raise ImportError(_LIGER_KERNEL_IMPORT_ERROR)
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "lm_head"):
            self.use_liger_kernel = False
            return
        self.compute_loss_func = self._liger_loss_func

    def _ensure_teacher_prepared(self):
        """Prepare teacher model via accelerator (handles FSDP2, DeepSpeed, DDP)."""
        if self._teacher_prepared:
            return
        self._teacher_prepared = True
        self._teacher_model = self._prepare_model(self._teacher_model)
        print_rank_0("Teacher model prepared for distillation.")

    def _get_unwrapped_teacher(self):
        """Unwrap teacher model (removes FSDP/DDP/DeepSpeed wrapper)."""
        return self.accelerator.unwrap_model(self._teacher_model)

    @contextmanager
    def _ds_gather(self, params):
        """Gather DS ZeRO-3 partitioned params; no-op if DeepSpeed disabled.

        The teacher is loaded under an active ``zero.Init`` but not wrapped in a
        DeepSpeedEngine, so its params have no per-module gather hooks and need an
        explicit gather around any forward use.
        """
        if self.is_deepspeed_enabled:
            import deepspeed

            with deepspeed.zero.GatheredParameters(list(params), modifier_rank=None):
                yield
        else:
            yield

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Train and evaluate on KD loss, with eval CE tracked as a metric."""
        self._ensure_teacher_prepared()
        kd_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        labels = inputs.get("labels")
        is_training = model.training

        if is_training:
            student_context = self._liger_identity_lm_head if self.use_liger_kernel else nullcontext
            with student_context():
                outputs = model(**kd_inputs)
        else:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        kd_loss = self._compute_kd_loss(outputs, labels, kd_inputs, **kwargs)
        if is_training:
            loss = kd_loss
        else:
            batch_size = find_batch_size(inputs)
            self._record_eval_ce_loss(loss, batch_size)
            loss = kd_loss

        return (loss, outputs) if return_outputs else loss

    def _compute_kd_loss(self, outputs, labels, inputs, **kwargs):
        """Run teacher forward and compute KD loss.

        The student forward has already run. When Liger is enabled, teacher
        forward also needs the identity-lm_head context so
        both student and teacher outputs are hidden states for fused KD.
        """
        lm_head_context = (
            self._teacher_liger_identity_lm_head if self.use_liger_kernel else nullcontext
        )
        with lm_head_context():
            teacher_outputs = self._compute_teacher_outputs(inputs)
        self._last_teacher_outputs = teacher_outputs

        if self.use_liger_kernel:
            return self._liger_kd_loss(outputs, labels, **kwargs)
        return self._standard_kd_loss(outputs, labels, **kwargs)

    def _compute_teacher_outputs(self, inputs):
        with torch.no_grad(), self._ds_gather(self._teacher_model.parameters()):
            self._teacher_model.eval()
            return self._teacher_model(**inputs)

    def _standard_kd_loss(self, outputs, labels, **kwargs):
        """KD loss with causal shift and ignore-index masking."""
        # Match causal LM CE: logits at position t are scored against label t+1.
        student_logits = outputs.logits[..., :-1, :].float()
        teacher_logits = self._last_teacher_outputs.logits[..., :-1, :].float()
        self._last_teacher_outputs = None
        per_token_loss = self._kd_criterion(student_logits, teacher_logits)
        if labels is None:
            return per_token_loss.mean()
        shift_labels = labels[..., 1:]
        mask = shift_labels != IGNORE_INDEX
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
        return loss

    @contextmanager
    def _teacher_liger_identity_lm_head(self):
        """Patch teacher lm_head to identity for fused KD."""
        teacher = self._get_unwrapped_teacher()
        teacher_lm_head = self._get_lm_head(teacher)
        teacher_orig = teacher_lm_head.forward
        teacher_lm_head.forward = lambda x: x
        try:
            yield
        finally:
            teacher_lm_head.forward = teacher_orig

    def _liger_kd_loss(self, outputs, labels, **kwargs):
        """Fused lm_head + JSD for KD."""
        from liger_kernel.transformers import LigerFusedLinearJSD

        model = self.accelerator.unwrap_model(self.model)
        teacher = self._get_unwrapped_teacher()

        student_lm_head = self._get_lm_head(model)
        teacher_lm_head = self._get_lm_head(teacher)

        student_hs = outputs.logits.to(student_lm_head.weight.dtype)  # RMSNorm may upcast to fp32
        teacher_hs = self._last_teacher_outputs.logits.to(teacher_lm_head.weight.dtype)
        self._last_teacher_outputs = None

        # Causal LM shift
        student_hs = student_hs[..., :-1, :].contiguous().view(-1, student_hs.size(-1))
        teacher_hs = teacher_hs[..., :-1, :].contiguous().view(-1, teacher_hs.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)

        jsd = LigerFusedLinearJSD(
            jsd_beta=self._liger_jsd_beta,
            ignore_index=IGNORE_INDEX,
            temperature=self._liger_temperature,
        )

        def _compute():
            return self._teacher_liger_enabled(
                lambda: jsd(
                    student_hs,
                    student_lm_head.weight,
                    teacher_hs,
                    teacher_lm_head.weight,
                    shift_labels,
                ),
                teacher_lm_head,
            )

        return super()._sharded_liger_compute(_compute)

    def _teacher_liger_enabled(self, fn, teacher_lm_head):
        if self.is_fsdp_enabled:
            return _forward_redirect(self._teacher_model, fn)
        if self.is_deepspeed_enabled:
            # Teacher is not in the DS engine; gather its lm_head explicitly.
            with self._ds_gather([teacher_lm_head.weight]):
                return fn()
        return fn()

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Add CE loss as a secondary evaluation metric."""
        self._eval_ce_loss_totals = None
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self._eval_ce_loss_totals is not None:
            output.metrics[f"{metric_key_prefix}_ce_loss"] = self._get_eval_ce_loss()
        return output

    def _record_eval_ce_loss(self, loss, batch_size):
        count = loss.new_tensor(float(batch_size or 1))
        totals = torch.stack([loss.detach() * count, count])
        self._eval_ce_loss_totals = (
            totals
            if self._eval_ce_loss_totals is None
            else self._eval_ce_loss_totals + totals.to(self._eval_ce_loss_totals.device)
        )

    def _get_eval_ce_loss(self):
        totals = self.accelerator.gather_for_metrics(self._eval_ce_loss_totals)
        totals = totals.reshape(-1, 2).sum(dim=0)
        return (totals[0] / totals[1].clamp(min=1)).item()

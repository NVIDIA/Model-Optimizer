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

from contextlib import contextmanager
from dataclasses import field

import torch
import torch.nn as nn
import transformers

from modelopt.torch.distill.losses import LogitsDistillationLoss
from modelopt.torch.opt.plugins import ModelOptHFTrainer
from modelopt.torch.opt.plugins.transformers import ModelOptHFArguments, _forward_redirect
from modelopt.torch.utils import print_rank_0

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
    """DistillArguments that accepts a pre-loaded nn.Module as teacher_model."""

    teacher_model: nn.Module | str | None = field(
        default=None,
        metadata={"help": "Pre-loaded teacher model or path/name string."},
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
        distill_args: DistillArguments | dict | None = None,
        **kwargs,
    ):
        """Initialize the trainer.

        Args:
            distill_args: Distillation config — either a :class:`DistillArguments` dataclass
                (CLI-parsed, ``teacher_model`` is a string path auto-loaded) or a ``dict``
                (programmatic, ``teacher_model`` can be an ``nn.Module``).
        """
        super().__init__(*args, **kwargs)
        if self.is_fsdp_enabled and not self.accelerator.is_fsdp2:
            raise ValueError("FSDP1 is not supported for distillation. Use FSDP2 instead.")

        assert distill_args is not None, "`distill_args` is required for distillation."

        # Normalize dict → DistillArgsWithTeacherModel (defaults come from dataclass)
        if isinstance(distill_args, dict):
            distill_args = DistillArgsWithTeacherModel(**distill_args)

        if distill_args.criterion not in _SUPPORTED_CRITERIA:
            raise ValueError(
                f"Unsupported criterion: {distill_args.criterion!r}. "
                f"Supported: {_SUPPORTED_CRITERIA}"
            )

        # Resolve teacher: nn.Module directly or string path → auto-load
        teacher = distill_args.teacher_model
        assert teacher is not None, "`distill_args.teacher_model` is required."
        if isinstance(teacher, str):
            teacher = transformers.AutoModelForCausalLM.from_pretrained(
                teacher, torch_dtype=torch.bfloat16
            )

        self._teacher_model = teacher
        self._teacher_model.requires_grad_(False)
        self._kd_criterion = LogitsDistillationLoss(
            temperature=distill_args.temperature, reduction="none"
        )
        self._teacher_prepared = False
        self.compute_loss_func = self.compute_kd_loss_func

        if self.use_liger_kernel:
            self._liger_temperature = distill_args.temperature
            self._liger_jsd_beta = distill_args.liger_jsd_beta

    def _ensure_teacher_prepared(self):
        """Prepare teacher model via accelerator (handles FSDP2, DeepSpeed, DDP)."""
        if self._teacher_prepared:
            return
        self._teacher_prepared = True
        self._teacher_model = self._prepare_model(self._teacher_model)
        print_rank_0("Teacher model prepared for distillation.")

        if self.use_liger_kernel:
            model = self.accelerator.unwrap_model(self.model)
            teacher = self._get_unwrapped_teacher()
            if not hasattr(model, "lm_head") or not hasattr(teacher, "lm_head"):
                self.use_liger_kernel = False

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

    def compute_loss(self, model, inputs, **kwargs):
        """Store teacher inputs before delegating to parent (which handles liger ctx)."""
        self._ensure_teacher_prepared()
        self._teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        return super().compute_loss(model, inputs, **kwargs)

    def compute_kd_loss_func(self, outputs, labels, **kwargs):
        """Run teacher forward and compute KD loss.

        Teacher forward runs here so it is inside the liger identity-lm_head
        context when liger is enabled (ModelOptHFTrainer wraps compute_loss).
        """
        with torch.no_grad(), self._ds_gather(self._teacher_model.parameters()):
            self._teacher_model.eval()
            teacher_outputs = self._teacher_model(**self._teacher_inputs)
        self._teacher_inputs = None
        self._last_teacher_outputs = teacher_outputs

        if self.use_liger_kernel:
            return self._liger_kd_loss(outputs, labels, **kwargs)
        return self._standard_kd_loss(outputs, labels, **kwargs)

    def _standard_kd_loss(self, outputs, labels, **kwargs):
        """KD loss with causal shift and ignore-index masking."""
        # Causal LM shift (match _liger_kd_loss semantics)
        student_logits = outputs.logits[..., :-1, :].contiguous().float()
        teacher_logits = self._last_teacher_outputs.logits[..., :-1, :].contiguous().float()
        per_token_loss = self._kd_criterion(student_logits, teacher_logits)
        if labels is None:
            return per_token_loss.sum()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != IGNORE_INDEX
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
        self._last_teacher_outputs = None
        return loss

    @contextmanager
    def _liger_identity_lm_head(self):
        """Patch both student+teacher lm_heads to identity."""
        model = self.accelerator.unwrap_model(self.model)
        teacher = self._get_unwrapped_teacher()
        student_lm_head = self._get_lm_head(model)
        teacher_lm_head = self._get_lm_head(teacher)
        student_orig = student_lm_head.forward
        teacher_orig = teacher_lm_head.forward
        student_lm_head.forward = lambda x: x
        teacher_lm_head.forward = lambda x: x
        try:
            yield
        finally:
            student_lm_head.forward = student_orig
            teacher_lm_head.forward = teacher_orig

    def _sharded_liger_compute(self, fn):
        """Delegate student lm_head gather to parent; add teacher lm_head gather on top."""
        return super()._sharded_liger_compute(self._apply_teacher_gather(fn))

    def _apply_teacher_gather(self, fn):
        """Wrap fn so the teacher's lm_head params are gathered when it runs."""
        if self.is_fsdp_enabled:
            teacher = self._teacher_model
            return lambda: _forward_redirect(teacher, fn)
        if self.is_deepspeed_enabled:
            # Teacher is not in the DS engine; gather its lm_head explicitly.
            teacher_lm_head = self._get_lm_head(self._get_unwrapped_teacher())

            def _wrapped():
                with self._ds_gather([teacher_lm_head.weight]):
                    return fn()

            return _wrapped
        return fn

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
            return jsd(
                student_hs,
                student_lm_head.weight,
                teacher_hs,
                teacher_lm_head.weight,
                shift_labels,
            )

        return self._sharded_liger_compute(_compute)

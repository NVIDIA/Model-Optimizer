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

"""ModelOpt plugin to train HuggingFace models with knowledge distillation."""

from contextlib import contextmanager

import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

import modelopt.torch.distill as mtd
from modelopt.torch.opt.plugins import ModelOptHFTrainer
from modelopt.torch.opt.plugins.transformers import _fsdp_forward_redirect
from modelopt.torch.utils import print_rank_0

IGNORE_INDEX = nn.CrossEntropyLoss().ignore_index


class KDTrainer(ModelOptHFTrainer):
    """Distillation trainer for HuggingFace models."""

    def __init__(self, *args, distill_config=None, **kwargs):
        """Initialize the trainer."""
        super().__init__(*args, **kwargs)
        if self.is_fsdp_enabled and not self.accelerator.is_fsdp2:
            raise ValueError("FSDP1 is not supported for distillation. Use FSDP2 instead.")

        assert distill_config is not None, "`distill_config` is required for distillation."
        self.distill_config = distill_config
        self._convert_to_distillation_model()
        if self.use_liger_kernel:
            self._setup_liger_fused_loss()
        else:
            self.compute_loss_func = self.compute_kd_loss

    def _get_lm_head(self, model):
        """Resolve student lm_head. Overrides base to handle distillation wrapper."""
        return model.lm_head

    def _get_teacher_lm_head(self, model):
        """Resolve teacher lm_head at call time."""
        return self._get_lm_head(model._teacher_model)

    def _setup_liger_fused_loss(self):
        """Set up fused JSD for KD.

        No-op when called from ModelOptHFTrainer.__init__ (teacher not yet created).
        Re-called from KDTrainer.__init__ after _convert_to_distillation_model().
        """
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "_teacher_model"):
            return
        teacher = model._teacher_model
        if not hasattr(model, "lm_head") or not hasattr(teacher, "lm_head"):
            self.use_liger_kernel = False
            self.compute_loss_func = self.compute_kd_loss
            return

        loss_fn = next(iter(model._layers_to_loss.values()))
        self._liger_temperature = getattr(loss_fn, "_temperature", 1.0)
        self.compute_loss_func = self._liger_loss_func

    @contextmanager
    def _liger_identity_lm_head(self):
        """Patch both student+teacher lm_heads to identity."""
        model = self.accelerator.unwrap_model(self.model)
        student_lm_head = self._get_lm_head(model)
        teacher_lm_head = self._get_teacher_lm_head(model)
        student_orig = student_lm_head.forward
        teacher_orig = teacher_lm_head.forward
        student_lm_head.forward = lambda x: x
        teacher_lm_head.forward = lambda x: x
        try:
            yield
        finally:
            student_lm_head.forward = student_orig
            teacher_lm_head.forward = teacher_orig

    def _liger_loss_func(self, outputs, labels, **kwargs):
        """Fused lm_head + JSD for KD."""
        from liger_kernel.transformers import LigerFusedLinearJSD

        model = self.accelerator.unwrap_model(self.model)

        (student_layer, teacher_layer), _ = next(iter(model._layers_to_loss.items()))
        student_hs = student_layer._intermediate_output.logits
        teacher_hs = teacher_layer._intermediate_output.logits
        student_layer._intermediate_output = None
        teacher_layer._intermediate_output = None

        student_lm_head = self._get_lm_head(model)
        teacher_lm_head = self._get_teacher_lm_head(model)

        # Causal LM shift
        student_hs = student_hs[..., :-1, :].contiguous().view(-1, student_hs.size(-1))
        teacher_hs = teacher_hs[..., :-1, :].contiguous().view(-1, teacher_hs.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)

        jsd = LigerFusedLinearJSD(jsd_beta=0.0, temperature=self._liger_temperature)

        def _compute():
            return jsd(
                student_hs,
                student_lm_head.weight,
                teacher_hs,
                teacher_lm_head.weight,
                shift_labels,
            )

        if self.is_fsdp_enabled:
            return _fsdp_forward_redirect(self.model, _compute)
        return _compute()

    def compute_kd_loss(self, outputs, labels, **kwargs):
        """KD loss with ignore-index masking."""
        mask = (labels != IGNORE_INDEX).float()

        def loss_reduction_fn(loss):
            return (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)

        return self.model.compute_kd_loss(loss_reduction_fn=loss_reduction_fn)

    def _convert_to_distillation_model(self):
        """Convert the model to a distillation model."""
        mtd.convert(self.model, mode=[("kd_loss", self.distill_config)])
        print_rank_0("Distillation model created.")

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
        *args,
        **kwargs,
    ):
        """Dumps model and ModelOpt states to disk.

        Note: Will save pretrained model in safetensors format if called manually, otherwise will
            save in training checkpointformat (when called internally by transformers Trainer).

        Args:
            output_dir: The directory to save the model and ModelOpt states.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        model = self.accelerator.unwrap_model(self.model)
        with model.hide_teacher_model(), model.hide_loss_modules(enable=not _internal_call):
            if _internal_call:
                return super().save_model(output_dir, _internal_call, *args, **kwargs)

            extra_kwargs = {}
            if self.is_fsdp_enabled:
                extra_kwargs["save_function"] = self.accelerator.save
                extra_kwargs["state_dict"] = self.accelerator.get_state_dict(self.model)
                self.accelerator.wait_for_everyone()  # needed to prevent hang somehow

            model.save_pretrained(
                output_dir,
                is_main_process=self.accelerator.is_main_process,
                **extra_kwargs,
            )
            self.processing_class.save_pretrained(output_dir)


class LMLogitsLoss(mtd.LogitsDistillationLoss):
    """Per-token KL-div logits loss for causal LM knowledge distillation.

    Returns unreduced per-token losses ``(B*S,)`` so that a ``loss_reduction_fn``
    (e.g. with ignore-index masking) can be applied in ``compute_kd_loss()``.
    """

    def __init__(self, temperature: float = 1.0):
        """Constructor.

        Args:
            temperature: Softmax temperature for softening logits.
        """
        super().__init__(temperature=temperature, reduction="none")

    def forward(self, out_student: CausalLMOutputWithPast, out_teacher: CausalLMOutputWithPast):
        """Forward pass returning per-token KD losses.

        Args:
            out_student: The student model output.
            out_teacher: The teacher model output.

        Returns:
            Per-token KL-div losses of shape ``(B*S,)``.
        """
        per_element = super().forward(out_student.logits, out_teacher.logits)  # (B*S, V)
        return per_element.sum(dim=-1)  # (B*S,)

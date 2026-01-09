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

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
from modelopt.torch.opt.plugins import ModelOptHFTrainer


class KDTrainer(ModelOptHFTrainer):
    """Distillation trainer for HuggingFace models."""

    def compute_loss(self, model, inputs, *args, **kwargs):
        """Compute loss for distillation.

        Change the training loss to distillation loss and keep the original validation loss.

        Args:
            model: The model to compute loss for.
            inputs: The inputs to the model.
        """
        if not model.training:
            _compute_loss_func = self.compute_loss_func
            self.compute_loss_func = None

        loss = super().compute_loss(model, inputs, *args, **kwargs)
        if not model.training:
            self.compute_loss_func = _compute_loss_func

        return loss

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
        export_student: bool = False,
        *args,
        **kwargs,
    ):
        """Dumps model and ModelOpt states to disk.

        Args:
            output_dir: The directory to save the model and ModelOpt states.
            export_student: Whether to export the student model.

        """
        if output_dir is None:
            output_dir = self.args.output_dir
        model = self.accelerator.unwrap_model(self.model)
        if not _internal_call and self.is_fsdp_enabled:
            with model.hide_teacher_model(enable=export_student):
                state_dict = self.accelerator.get_state_dict(self.model)
            modelopt_state = mto.modelopt_state(model)
            if export_student:
                # Need to wait, otherwise FSDP weights may be deleted before rank 0 can gather them
                self.accelerator.wait_for_everyone()
                model = model.export()

            if self.accelerator.is_main_process:
                model.save_pretrained(
                    output_dir,
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                    state_dict=state_dict,
                )
                self.processing_class.save_pretrained(output_dir)
                torch.save(modelopt_state, f"{output_dir}/modelopt_state.pth")
        else:
            model = model.export() if export_student else model
            super().save_model(output_dir, _internal_call, *args, **kwargs)

    def train(self, *args, **kwargs):
        """Train the model."""
        self.compute_loss_func = lambda *args, **kwargs: self.model.compute_kd_loss()
        return super().train(*args, **kwargs)


class LMLogitsLoss(mtd.LogitsDistillationLoss):
    """Logits loss for knowledge distillation."""

    def forward(self, out_student: CausalLMOutputWithPast, out_teacher: CausalLMOutputWithPast):
        """Forward pass for logits distillation loss.

        Args:
            out_student: The student model output.
            out_teacher: The teacher model output.
        """
        return super().forward(out_student.logits, out_teacher.logits)

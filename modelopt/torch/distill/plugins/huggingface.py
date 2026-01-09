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

from transformers.modeling_outputs import CausalLMOutputWithPast

import modelopt.torch.distill as mtd
from modelopt.torch.opt.plugins import ModelOptHFTrainer
from modelopt.torch.utils import print_rank_0


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

    def _convert_to_distillation_model(self):
        """Convert the model to a distillation model."""
        mtd.convert(self.model, mode=[("kd_loss", self.distill_config)])
        print_rank_0("Distillation model created.")

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

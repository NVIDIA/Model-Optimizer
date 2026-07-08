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
"""Callbacks used by the Megatron-Bridge distillation example."""

import contextlib
from dataclasses import replace
from pathlib import Path

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.data.loaders import cyclic_iter, get_train_valid_test_num_samples
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.eval import evaluate
from megatron.bridge.training.gpt_step import forward_step_modelopt
from megatron.bridge.utils.common_utils import print_rank_last
from megatron.core import parallel_state
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.utils import unwrap_model
from transformers import AutoConfig

import modelopt.torch.distill as mtd
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import print_rank_0


class _HFValidationExportCallback(Callback):
    """Export the live student to Hugging Face format at selected validation stages."""

    def __init__(
        self,
        export_dir: str,
        student_hf_model: str,
        student_hf_path: str,
        trust_remote_code: bool,
        export_interval: int,
    ) -> None:
        self.export_dir = Path(export_dir)
        self.student_hf_path = student_hf_path
        self.trust_remote_code = trust_remote_code
        self.export_interval = export_interval
        self._last_exported_iteration: int | None = None
        self.bridge = AutoBridge.from_hf_pretrained(
            student_hf_model, trust_remote_code=trust_remote_code
        )

    def on_eval_end(self, context) -> None:
        """Export the student at the iteration that was just validated."""
        iteration = context.state.train_state.step
        train_iters = context.state.cfg.train.train_iters
        if iteration % self.export_interval != 0 and iteration != train_iters:
            return
        # The final iteration can be validated both on its regular interval and after training.
        # Avoid exporting and overwriting the same Hugging Face checkpoint twice.
        if iteration == self._last_exported_iteration:
            return
        output_path = self.export_dir / f"iter_{iteration:07d}"
        print_rank_0(f"Exporting validation checkpoint {iteration} to {output_path}")

        # DistillationModel is the student with teacher and KD-loss modules attached. Hide the
        # auxiliary modules temporarily so the Hugging Face export contains only student weights.
        with contextlib.ExitStack() as stack:
            for model_chunk in unwrap_model(context.model):
                if isinstance(model_chunk, mtd.DistillationModel):
                    stack.enter_context(model_chunk.hide_teacher_model())
                    stack.enter_context(model_chunk.hide_loss_modules())
            self.bridge.save_hf_pretrained(
                context.model,
                output_path,
                show_progress=True,
                strict=True,
            )

        if dist.rank() == 0:
            # Preserve the student architecture from student_hf_path, including heterogeneous
            # layer changes; AutoConfig supports both local paths and Hugging Face model IDs.
            AutoConfig.from_pretrained(
                self.student_hf_path, trust_remote_code=self.trust_remote_code
            ).save_pretrained(output_path)
        torch.distributed.barrier()
        self._last_exported_iteration = iteration


# TODO: Replace this callback once Megatron-Bridge can evaluate and report multiple validation
# datasets separately.
class _TargetValidationCallback(Callback):
    """Evaluate a fixed target blend after the normal training-blend validation."""

    def __init__(self, target_blend) -> None:
        self.target_blend = target_blend
        self._data_iterator: RerunDataIterator | None = None

    def _build_data_iterator(self, context) -> RerunDataIterator:
        config = context.state.cfg
        validation_config = getattr(config, "validation", config.train)
        eval_global_batch_size = (
            getattr(validation_config, "eval_global_batch_size", None)
            or config.train.global_batch_size
        )
        eval_micro_batch_size = (
            getattr(validation_config, "eval_micro_batch_size", None)
            or config.train.micro_batch_size
        )
        target_config = replace(
            config.dataset,
            blend=None,
            blend_per_split=[None, self.target_blend, None],
            split=None,
        )
        target_config.finalize()
        target_samples = get_train_valid_test_num_samples(config)[1]
        _, target_dataset, _ = pretrain_train_valid_test_datasets_provider(
            [0, target_samples, 0],
            target_config,
        )
        target_dataloader = build_pretraining_data_loader(
            target_dataset,
            consumed_samples=0,
            dataloader_type="cyclic",
            micro_batch_size=eval_micro_batch_size,
            num_workers=target_config.num_workers,
            data_sharding=target_config.data_sharding,
            collate_fn=(
                target_dataset.collate_fn if hasattr(target_dataset, "collate_fn") else None
            ),
            pin_memory=target_config.pin_memory,
            persistent_workers=target_config.persistent_workers,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            global_batch_size=eval_global_batch_size,
        )
        return RerunDataIterator(iter(cyclic_iter(target_dataloader)))

    def on_eval_end(self, context) -> None:
        """Run target validation after the normal validation."""
        iteration = context.state.train_state.step
        if self._data_iterator is None:
            self._data_iterator = self._build_data_iterator(context)

        total_loss_dict, _, timelimit = evaluate(
            context.state,
            forward_step_modelopt,
            self._data_iterator,
            context.model,
            None,
            context.state.cfg,
        )
        # Do not report an incomplete time-limited evaluation. Pipeline ranks that do not
        # compute the loss receive an empty dictionary and have nothing to report.
        if timelimit or not total_loss_dict:
            return
        metrics = {
            f"target {key} validation": value.item() for key, value in total_loss_dict.items()
        }
        print_rank_last(
            f"target validation loss at iteration {iteration} | "
            + " | ".join(f"{key}: {value:.6E}" for key, value in metrics.items())
        )
        if context.state.tensorboard_logger:
            for key, value in metrics.items():
                context.state.tensorboard_logger.add_scalar(key, value, iteration)

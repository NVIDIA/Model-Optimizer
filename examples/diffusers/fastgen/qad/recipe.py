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

"""FastGen quantization-aware distillation recipe.

QAD is deliberately separate from DMD2: one frozen Diffusers teacher and one
quantized student see the same noisy latent, timestep, and conditioning, and
ModelOpt's standard ``kd_loss`` API supplies output and optional representation
MSE losses.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import wandb
import yaml
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer

try:
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
    from nemo_automodel.components.training.utils import (
        clip_grad_norm,
        prepare_after_first_microbatch,
        prepare_for_final_backward,
        prepare_for_grad_accumulation,
    )
    from nemo_automodel.recipes.base_recipe import (
        _find_latest_checkpoint,
        _resolve_restore_from_to_ckpt_dir,
    )
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe, is_main_process
except ImportError as exc:
    raise ImportError(
        "The FastGen QAD example requires nemo_automodel. Install dependencies with:\n"
        "    pip install -r examples/diffusers/fastgen/requirements.txt"
    ) from exc

from fastgen_checkpoint import make_optimizer_partial_load_tolerant

from .artifacts import StudentSettings, patch_student_build
from .modeling import build_distillation_controller, clear_captured_outputs
from .pipeline import QADPipeline


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


class QADDiffusionRecipe(TrainDiffusionRecipe):
    """AutoModel diffusion recipe with a ModelOpt KD controller."""

    def __init__(self, cfg) -> None:
        # AutoModel's dotted CLI setter updates live ConfigNodes but not the
        # raw_config later written beside checkpoints. Materialize the resolved
        # runtime values so QAD paths, scope, teacher, and losses are reproducible.
        if hasattr(cfg, "to_yaml_dict"):
            cfg.__dict__["_raw_config"] = cfg.to_yaml_dict(
                resolve_env=False,
                redact_sensitive=True,
                use_orig_values=False,
            )
        super().__init__(cfg)

    def setup(self) -> None:
        settings, loss_config = self._resolve_qad_config()
        self.__dict__["_qad_resume_signature"] = self._resume_signature(
            settings,
            loss_config,
        )

        if self.cfg.get("peft", None) is not None:
            raise ValueError(
                "Do not set AutoModel's top-level peft block for QAD. SVDQuant's "
                "modelopt_svdquant HF PEFT topology comes from the student bundle."
            )
        if str(self.cfg.get("model.mode", "finetune")).lower() != "finetune":
            raise ValueError("QAD supports model.mode=finetune only.")
        if self.cfg.get("ddp", None) is not None:
            raise ValueError("QAD currently supports AutoModel FSDP2, not DDP.")

        # Diffusers' ModelMixin must be patched before from_pretrained so both
        # regular NVFP4 and SVDQuant bundles rebuild their ModelOpt topology and
        # load component-local modelopt_state.pth before AutoModel applies FSDP.
        mto.enable_huggingface_checkpointing()

        with patch_student_build(settings) as build_state:
            super().setup()

        # Diffusers loads ModelMixin objects in eval mode. QAD owns the student
        # train/eval boundary because the controller delegate intentionally does
        # not register the live FSDP module as a child.
        self.model.train()

        # Parent checkpoint restore established the exact next-step RNG state.
        # Teacher construction/sharding is transient setup and must not perturb
        # the first fresh or resumed training sample.
        training_rng_state = self.rng.state_dict()
        try:
            parallel_scheme = build_state.parallel_scheme
            if parallel_scheme is None:
                raise RuntimeError("QAD failed to capture the student's parallel scheme.")
            teacher = self._load_frozen_teacher(
                loss_config["teacher_model_name_or_path"],
                parallel_scheme,
            )
            controller, loss_names = build_distillation_controller(
                student=self.model,
                teacher=teacher,
                output_weight=loss_config["output_weight"],
                task_weight=loss_config["task_weight"],
                layer_pairs=loss_config["layer_pairs"],
            )
            if any(True for _ in controller.parameters()):
                raise RuntimeError(
                    "The QAD controller must remain parameter-free; optimizer/checkpoint "
                    "ownership belongs exclusively to self.model."
                )

            # BaseRecipe tracks nn.Module assignments. Bypass it for the frozen teacher
            # and transient controller so checkpoint selection cannot mistake either for
            # the student.
            object.__setattr__(self, "_qad_teacher", teacher)
            object.__setattr__(self, "_qad_controller", controller)
            object.__setattr__(
                self,
                "_qad_pipeline",
                QADPipeline(self.flow_matching_pipeline, controller, loss_names),
            )
            object.__setattr__(self, "_qad_student_settings", settings)
            object.__setattr__(self, "_qad_loss_config", loss_config)

            tracked = self.__dict__.get("__state_tracked", set())
            forbidden = {"_qad_teacher", "_qad_controller", "_qad_pipeline"} & set(tracked)
            if forbidden:
                raise RuntimeError(
                    f"QAD transient objects were accidentally state-tracked: {forbidden}"
                )
            self._validate_state_ownership()
        finally:
            self.rng.load_state_dict(training_rng_state)

        if is_main_process():
            logging.info(
                "[QAD] initialized: teacher=%s student=%s mode=%s train_scope=%s "
                "task_weight=%g output_weight=%g layer_pairs=%d",
                loss_config["teacher_model_name_or_path"],
                settings.model_name_or_path,
                settings.mode,
                settings.train_scope,
                loss_config["task_weight"],
                loss_config["output_weight"],
                len(loss_config["layer_pairs"]),
            )
            logging.info("[QAD] student quantizer summary:")
            mtq.print_quant_summary(self.model)

    def _resolve_qad_config(self) -> tuple[StudentSettings, dict[str, Any]]:
        qad = _as_dict(self.cfg.get("qad", None))
        if not qad:
            raise ValueError("Missing required qad configuration block.")

        student_cfg = _as_dict(qad.get("student"))
        secondary_artifact_fields = sorted(
            field
            for field in ("quant_state_path", "modelopt_state_path")
            if student_cfg.get(field) is not None
        )
        if secondary_artifact_fields:
            raise ValueError(
                "QAD accepts one complete Diffusers student bundle through "
                "model.pretrained_model_name_or_path; remove unsupported secondary "
                "artifact field(s): "
                + ", ".join(f"qad.student.{field}" for field in secondary_artifact_fields)
            )
        model_name_or_path = self.cfg.get("model.pretrained_model_name_or_path", None)
        if not model_name_or_path:
            raise ValueError(
                "model.pretrained_model_name_or_path is required and is the canonical "
                "student source recorded in checkpoints."
            )
        duplicate_student_path = student_cfg.get("model_name_or_path")
        if duplicate_student_path is not None and str(duplicate_student_path) != str(
            model_name_or_path
        ):
            raise ValueError(
                "qad.student.model_name_or_path conflicts with the canonical "
                "model.pretrained_model_name_or_path. Remove the duplicate QAD field."
            )
        mode = str(student_cfg.get("mode", "nvfp4")).lower()
        # Accept the early design spelling while emitting one canonical name.
        if mode == "svdquant_nvfp4":
            mode = "nvfp4_svdquant"
        settings = StudentSettings(
            mode=mode,
            model_name_or_path=str(model_name_or_path),
            train_scope=str(student_cfg.get("train_scope", "all")).lower(),
        )
        settings.validate()

        teacher_model_name_or_path = qad.get("teacher_model_name_or_path")
        if not teacher_model_name_or_path:
            raise ValueError("qad.teacher_model_name_or_path is required.")

        output_cfg = _as_dict(qad.get("output_loss"))
        if str(output_cfg.get("type", "mse")).lower() != "mse":
            raise ValueError("QAD currently supports only output_loss.type=mse.")
        output_weight = float(output_cfg.get("weight", 1.0))

        task_cfg = _as_dict(qad.get("task_loss"))
        task_weight = float(task_cfg.get("weight", 0.0))

        layerwise_cfg = _as_dict(qad.get("layerwise"))
        layer_pairs = layerwise_cfg.get("pairs", []) if layerwise_cfg.get("enabled", False) else []
        layer_pairs = [_as_dict(pair) for pair in layer_pairs]

        all_weights = [output_weight, task_weight] + [
            float(pair.get("weight", 1.0)) for pair in layer_pairs
        ]
        if any(not math.isfinite(weight) or weight < 0.0 for weight in all_weights):
            raise ValueError("QAD loss weights must be finite and non-negative.")
        if not any(weight > 0.0 for weight in all_weights):
            raise ValueError("At least one QAD loss weight must be positive.")
        for index, pair in enumerate(layer_pairs):
            if not pair.get("student_layer"):
                raise ValueError(f"qad.layerwise.pairs[{index}].student_layer is required.")

        return settings, {
            "teacher_model_name_or_path": str(teacher_model_name_or_path),
            "output_weight": output_weight,
            "task_weight": task_weight,
            "layer_pairs": layer_pairs,
        }

    @staticmethod
    def _resume_signature(
        settings: StudentSettings,
        loss_config: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "student_source": settings.model_name_or_path,
            "student_mode": settings.mode,
            "train_scope": settings.train_scope,
            "teacher_source": loss_config["teacher_model_name_or_path"],
            "output_weight": float(loss_config["output_weight"]),
            "task_weight": float(loss_config["task_weight"]),
            "layer_pairs": tuple(
                (
                    str(pair["student_layer"]),
                    str(pair.get("teacher_layer", pair["student_layer"])),
                    str(pair.get("selector", "hidden_states")),
                    float(pair.get("weight", 1.0)),
                )
                for pair in loss_config["layer_pairs"]
            ),
        }

    @classmethod
    def _resume_signature_from_saved_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        model_cfg = _as_dict(config.get("model"))
        qad_cfg = _as_dict(config.get("qad"))
        student_cfg = _as_dict(qad_cfg.get("student"))
        secondary_artifact_fields = sorted(
            field
            for field in ("quant_state_path", "modelopt_state_path")
            if student_cfg.get(field) is not None
        )
        if secondary_artifact_fields:
            raise RuntimeError(
                "The saved QAD checkpoint uses unsupported secondary student artifact "
                "field(s): "
                + ", ".join(f"qad.student.{field}" for field in secondary_artifact_fields)
            )
        output_cfg = _as_dict(qad_cfg.get("output_loss"))
        task_cfg = _as_dict(qad_cfg.get("task_loss"))
        layerwise_cfg = _as_dict(qad_cfg.get("layerwise"))

        mode = str(student_cfg.get("mode", "nvfp4")).lower()
        if mode == "svdquant_nvfp4":
            mode = "nvfp4_svdquant"
        raw_pairs = layerwise_cfg.get("pairs", []) if layerwise_cfg.get("enabled", False) else []
        loss_config = {
            "teacher_model_name_or_path": str(qad_cfg.get("teacher_model_name_or_path", "")),
            "output_weight": float(output_cfg.get("weight", 1.0)),
            "task_weight": float(task_cfg.get("weight", 0.0)),
            "layer_pairs": [_as_dict(pair) for pair in raw_pairs],
        }
        settings = StudentSettings(
            mode=mode,
            model_name_or_path=str(model_cfg.get("pretrained_model_name_or_path", "")),
            train_scope=str(student_cfg.get("train_scope", "all")).lower(),
        )
        return cls._resume_signature(settings, loss_config)

    def _resolved_checkpoint_dir(self, restore_from: str | None) -> str | None:
        if not self.checkpointer.config.enabled:
            return None
        if restore_from:
            resolved = _resolve_restore_from_to_ckpt_dir(
                self.checkpointer.config.checkpoint_dir,
                restore_from,
            )
        else:
            resolved = _find_latest_checkpoint(self.checkpointer.config.checkpoint_dir)
        if resolved is None:
            return None
        return os.fspath(resolved)

    def _validate_qad_checkpoint_signature(self, checkpoint_dir: str) -> None:
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.isfile(config_path):
            raise RuntimeError(
                "QAD cannot safely restore optimizer shards from a checkpoint without "
                f"config.yaml: {checkpoint_dir}"
            )
        with open(config_path) as config_file:
            saved_config = yaml.safe_load(config_file) or {}
        saved_signature = self._resume_signature_from_saved_config(saved_config)
        current_signature = self.__dict__["_qad_resume_signature"]
        if saved_signature != current_signature:
            changed = [
                key
                for key in current_signature
                if saved_signature.get(key) != current_signature[key]
            ]
            raise RuntimeError(
                "QAD resume contract changed for "
                + ", ".join(changed)
                + ". Use the same student bundle, quantization mode, train scope, "
                "teacher, and loss configuration as the saved run."
            )

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Validate QAD topology before enabling FSDP2 partial-shard optimizer load."""
        checkpoint_dir = self._resolved_checkpoint_dir(restore_from)
        if checkpoint_dir is not None and os.path.isdir(checkpoint_dir):
            self._validate_qad_checkpoint_signature(checkpoint_dir)
            make_optimizer_partial_load_tolerant(self.checkpointer)
        super().load_checkpoint(restore_from)

    def _rebuild_dataloader_for_resume(self, global_step: int) -> None:
        """Rebuild the loader and deterministically skip to the restored data position."""
        epoch_len = int(getattr(self.step_scheduler, "epoch_len", 0) or 0)
        grad_acc = int(getattr(self.step_scheduler, "grad_acc_steps", 1) or 1)
        if epoch_len <= 0 or self.sampler is None or global_step <= 0:
            return

        current_epoch = global_step // epoch_len
        skip_batches = (global_step % epoch_len) * grad_acc
        old_dataloader = self.dataloader
        dataloader_kwargs = {
            "collate_fn": getattr(old_dataloader, "collate_fn", None),
            "num_workers": int(getattr(old_dataloader, "num_workers", 0) or 0),
            "pin_memory": bool(getattr(old_dataloader, "pin_memory", False)),
        }
        if dataloader_kwargs["num_workers"] > 0:
            dataloader_kwargs["prefetch_factor"] = getattr(
                old_dataloader,
                "prefetch_factor",
                2,
            )
            dataloader_kwargs["persistent_workers"] = bool(
                getattr(old_dataloader, "persistent_workers", False)
            )

        # Keep the parent's existing tracked state key while replacing the
        # StatefulDataLoader object whose restored cursor is known to stick.
        self.__dict__["dataloader"] = StatefulDataLoader(
            old_dataloader.dataset,
            batch_sampler=self.sampler,
            **dataloader_kwargs,
        )
        self.step_scheduler.epoch = current_epoch
        self.sampler.set_epoch(current_epoch)
        self.sampler._batches_to_skip = skip_batches
        if is_main_process():
            logging.info(
                "[QAD][resume] rebuilt dataloader at epoch=%d skip_batches=%d "
                "(global_step=%d epoch_len=%d grad_acc=%d)",
                current_epoch,
                skip_batches,
                global_step,
                epoch_len,
                grad_acc,
            )

    def _load_frozen_teacher(
        self,
        model_name_or_path: str,
        parallel_scheme: dict[str, dict[str, Any]],
    ) -> nn.Module:
        pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=self.bf16,
            device=self.device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=False,
            low_cpu_mem_usage=True,
        )
        teacher = pipe.transformer
        if mto.ModeloptStateManager.is_converted(teacher):
            raise RuntimeError(
                "QAD teacher must be a plain BF16 Diffusers checkpoint without ModelOpt modes."
            )
        if any(isinstance(module, TensorQuantizer) for module in teacher.modules()):
            raise RuntimeError("QAD teacher must be an unquantized BF16 Diffusers checkpoint.")
        teacher.eval()
        teacher.requires_grad_(False)
        return teacher

    def _validate_state_ownership(self) -> None:
        optimizer_parameters = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        student_parameters = {
            id(parameter) for parameter in self.model.parameters() if parameter.requires_grad
        }
        teacher_parameters = {id(parameter) for parameter in self._qad_teacher.parameters()}
        if optimizer_parameters != student_parameters:
            raise RuntimeError("QAD optimizer does not exactly own the trainable student state.")
        if optimizer_parameters & teacher_parameters:
            raise RuntimeError("Frozen teacher parameters leaked into the student optimizer.")
        if any(parameter.requires_grad for parameter in self._qad_teacher.parameters()):
            raise RuntimeError("QAD teacher must be completely frozen.")

    def run_train_validation_loop(self) -> None:
        """Run a conventional optimizer loop using the QAD objective."""
        self.model.train()
        logging.info(
            "[QAD] starting training: global_batch_size=%s local_batch_size=%s dp_size=%s",
            self.global_batch_size,
            self.local_batch_size,
            self.dp_size,
        )
        global_step = int(self.step_scheduler.step)
        self._rebuild_dataloader_for_resume(global_step)

        try:
            for epoch in self.step_scheduler.epochs:
                if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                    self.sampler.set_epoch(epoch)

                tqdm_initial = int(getattr(self.sampler, "_batches_to_skip", 0) or 0)
                if is_main_process():
                    from tqdm import tqdm

                    self.step_scheduler.dataloader = tqdm(
                        self.dataloader,
                        desc=f"Epoch {epoch + 1}/{self.num_epochs} (global step {global_step})",
                        initial=tqdm_initial,
                    )
                else:
                    self.step_scheduler.dataloader = self.dataloader

                epoch_loss = 0.0
                num_steps = 0
                for batch_group in self.step_scheduler:
                    # StepScheduler increments only after control returns to its
                    # generator, so refresh at the top of every yielded group.
                    global_step = int(self.step_scheduler.step)
                    self.optimizer.zero_grad(set_to_none=True)
                    prepare_for_grad_accumulation([self.model], pp_enabled=False)
                    num_microbatches = len(batch_group)
                    micro_metrics: list[dict[str, torch.Tensor]] = []

                    for microbatch_index, micro_batch in enumerate(batch_group):
                        if microbatch_index == num_microbatches - 1:
                            prepare_for_final_backward([self.model], pp_enabled=False)
                        try:
                            total_loss, metrics = self._qad_pipeline.step(
                                batch=micro_batch,
                                device=self.device,
                                dtype=self.bf16,
                                global_step=global_step,
                                check_loss=self.check_loss,
                            )
                            (total_loss / num_microbatches).backward()
                            micro_metrics.append(metrics)
                        finally:
                            # Full-block NO_REENTRANT checkpoint wrappers avoid hook
                            # repopulation during recompute; this final cleanup is also
                            # safe when activation checkpointing is disabled.
                            self._qad_pipeline.clear()

                        if microbatch_index == 0:
                            prepare_after_first_microbatch()

                    self._validate_first_step_gradients(global_step)
                    grad_norm = clip_grad_norm(
                        self.clip_grad_max_norm,
                        [self.model],
                        foreach=self.grad_clip_foreach,
                    )
                    grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler[0].step(1)

                    reduced_metrics = {
                        name: float(
                            torch.stack([metrics[name] for metrics in micro_metrics]).mean().item()
                        )
                        for name in micro_metrics[0]
                    }
                    group_loss = reduced_metrics["total_loss"]
                    epoch_loss += group_loss
                    num_steps += 1

                    if self.log_every and global_step % self.log_every == 0 and is_main_process():
                        log_dict = {
                            "train_loss": group_loss,
                            "train_avg_loss": epoch_loss / num_steps,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "grad_norm": grad_norm,
                            "epoch": epoch,
                            "global_step": global_step,
                            **{f"qad/{name}": value for name, value in reduced_metrics.items()},
                        }
                        if wandb.run is not None:
                            wandb.log(log_dict, step=global_step)
                        component_text = " ".join(
                            f"{name}={value:.6f}" for name, value in reduced_metrics.items()
                        )
                        logging.info(
                            "[QAD][TRAIN] step=%d epoch=%d %s lr=%.3e grad_norm=%.3f",
                            global_step,
                            epoch,
                            component_text,
                            self.optimizer.param_groups[0]["lr"],
                            grad_norm,
                        )
                        if hasattr(self.step_scheduler.dataloader, "set_postfix"):
                            self.step_scheduler.dataloader.set_postfix(
                                {
                                    "loss": f"{group_loss:.4f}",
                                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                    "gn": f"{grad_norm:.2f}",
                                }
                            )

                    if self.step_scheduler.is_ckpt_step:
                        self.save_checkpoint(epoch, global_step, epoch_loss / num_steps)

                if num_steps == 0:
                    logging.info(
                        "[QAD] epoch %d skipped (already completed in previous run)", epoch + 1
                    )
                    continue
                logging.info(
                    "[QAD] epoch %d complete: avg_loss=%.6f",
                    epoch + 1,
                    epoch_loss / num_steps,
                )

            if is_main_process() and wandb.run is not None:
                wandb.finish()
            logging.info("[QAD] training complete at step %d", global_step)
        finally:
            self._release_distillation_controller()

    def _validate_first_step_gradients(self, global_step: int) -> None:
        if global_step != 0:
            return
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not any(parameter.grad is not None for parameter in trainable):
            raise RuntimeError("QAD produced no gradients for any trainable student parameter.")
        if self._qad_student_settings.train_scope == "lora_only":
            missing = [
                name
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if missing:
                raise RuntimeError(
                    "SVDQuant lora_only parameters missing gradients on the first step: "
                    + ", ".join(missing[:5])
                )

    def _release_distillation_controller(self) -> None:
        controller = getattr(self, "_qad_controller", None)
        if controller is None or not hasattr(controller, "_layers_to_loss"):
            return
        layer_pairs = tuple(controller._layers_to_loss)
        clear_captured_outputs(controller)
        mtd.export(controller)
        for student_layer, teacher_layer in layer_pairs:
            for layer in (student_layer, teacher_layer):
                if hasattr(layer, "_intermediate_output"):
                    delattr(layer, "_intermediate_output")

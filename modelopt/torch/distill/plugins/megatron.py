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

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Distillation loss function(s)."""

import logging
import re
from abc import ABCMeta
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING

import torch
import torch.distributed.nn as dist_nn
import torch.nn as nn
import torch.nn.functional as F
import yaml
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.core.transformer import MegatronModule, TransformerLayer
from megatron.core.utils import get_model_config
from torch import Tensor
from torch.nn.modules.loss import _Loss

import modelopt.torch.distill as mtd
from modelopt.torch.distill.config import Criterion

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.transformer import TransformerConfig


logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Knowledge-Distillation config.

    Args:
        intermediate_layer_pairs: List of tuples of intermediate layer names.
        logit_layers: Tuple of logit layer names.
        skip_lm_loss: Whether to skip computing the standard language model loss (default: ``True``).
        kd_loss_scale: Relative scaling factor for the distillation loss if ``skip_lm_loss`` is ``False``.
        logit_kl_temperature: Temperature for the logit KL-divergence loss.
        logit_kl_topk: If not None, use TopKLogitsKLLoss instead of LogitsKLLoss with this top-k value.
    """

    intermediate_layer_pairs: list[tuple[str, ...]] = field(default_factory=list)
    logit_layers: tuple[str, str] = ("output_layer", "output_layer")
    skip_lm_loss: bool = True
    kd_loss_scale: float = 1.0
    logit_kl_temperature: float = 1.0
    logit_kl_topk: int | None = None
    criterion: Criterion | None = None
    loss_balancer: mtd.DistillationLossBalancer | None = None

    def __post_init__(self):
        assert len(self.logit_layers) == 2, f"{self.logit_layers=}"
        assert all(len(pair) in (2, 3) for pair in self.intermediate_layer_pairs), (
            f"{self.intermediate_layer_pairs=}"
        )
        assert self.kd_loss_scale > 0, f"{self.kd_loss_scale=}"
        assert self.logit_kl_temperature > 0, f"{self.logit_kl_temperature=}"

    @staticmethod
    def parse_intermediate_entry(entry: tuple[str, ...]) -> tuple[str, str, Callable]:
        """Parse an intermediate entry into a student layer, teacher layer, and loss function."""
        if len(entry) == 3:
            student_layer, teacher_layer, loss_fn_name = entry
            if loss_fn_name == "cosine":
                loss_fn = HiddenStateCosineLoss
            elif loss_fn_name == "mse":
                loss_fn = MSELoss
            else:
                raise ValueError(f"Unknown intermediate loss function: {loss_fn_name}")
        else:
            student_layer, teacher_layer = entry
            loss_fn = HiddenStateCosineLoss  # default to cosine loss
        return student_layer, teacher_layer, loss_fn


def setup_distillation_config(
    config_or_path: str | DistillationConfig | None,
    student_cfg: "TransformerConfig",
    teacher_cfg: "TransformerConfig",
) -> DistillationConfig:
    """Setup and/or finalize the distillation config.

    Either reads the distillation yaml config file from a path, fills in an
    existing DistillationConfig, or creates a default one.

    Args:
        config_or_path: Path to user-defined distillation settings yaml file, or the incomplete config itself.
            If `None`, uses default logits-only distillation mode for GPT models.
        student_cfg: Model config for student model.
        teacher_cfg: Model config for teacher model.

    WARNING: Assumes intermediate hidden sizes are always that found in the model config's ``hidden_size`` attribute.
    """
    if config_or_path is None:
        logger.warning("Distillation config not provided. Using default.")
        cfg = DistillationConfig()
    elif isinstance(config_or_path, DistillationConfig):
        cfg = config_or_path
    else:
        with open(config_or_path) as f:
            cfg = yaml.safe_load(f)
        cfg = DistillationConfig(**cfg)

    if cfg.criterion is None:
        criterion = {}
        if parallel_state.is_pipeline_last_stage():
            # Use TopKLogitsKLLoss if logit_kl_topk is specified, otherwise use LogitsKLLoss
            if cfg.logit_kl_topk is not None:
                criterion[tuple(cfg.logit_layers)] = TopKLogitsKLLoss(
                    student_cfg, temperature=cfg.logit_kl_temperature, top_k=cfg.logit_kl_topk
                )
            else:
                criterion[tuple(cfg.logit_layers)] = LogitsKLLoss(
                    student_cfg, temperature=cfg.logit_kl_temperature
                )
            # NOTE: Projection layer shared among intermediate layer pairs.
            projection_layer = ProjectionLayer(student_cfg, teacher_cfg)

            for entry in cfg.intermediate_layer_pairs:
                student_layer, teacher_layer, loss_fn = cfg.parse_intermediate_entry(entry)
                if parallel_state.get_tensor_and_context_parallel_rank() == 0:
                    logger.info(
                        "Distillation: Adding intermediate loss between"
                        f" `{student_layer}` of student (hidden size {student_cfg.hidden_size}) and"
                        f" `{teacher_layer}` of teacher (hidden size {teacher_cfg.hidden_size})."
                    )
                student_layer = _adjust_layer_index_for_pp(student_layer, student_cfg)
                teacher_layer = _adjust_layer_index_for_pp(teacher_layer, teacher_cfg)
                criterion[(student_layer, teacher_layer)] = loss_fn(
                    student_cfg, projection_layer=projection_layer
                )
        cfg.criterion = criterion

    if cfg.loss_balancer is None:
        cfg.loss_balancer = LogitsAndIntermediatesLossBalancer(
            kd_loss_scale=cfg.kd_loss_scale, skip_original_loss=cfg.skip_lm_loss
        )

    return cfg


def _adjust_layer_index_for_pp(submodule_name, model_cfg):
    """Adjust any sequence-based layer indices found in a submodule name for Pipeline Parallelism."""
    match = re.search(r"(?<=\.)\d+(?=\.)", submodule_name)
    if not match:
        return submodule_name

    offset = TransformerLayer._get_layer_offset(model_cfg)
    new_layer_idx = int(match.group(0)) - offset
    if new_layer_idx < 0:
        raise ValueError(f"Layer {submodule_name} does not fall on final PP rank.")

    new_submodule_name = submodule_name.replace(match.group(0), str(new_layer_idx))
    if parallel_state.get_tensor_and_context_parallel_rank() == 0:
        logger.info(
            f'Distillation: Renamed layer "{submodule_name}" on final PP rank to "{new_submodule_name}"'
        )
    return new_submodule_name


########################################################


class BaseLoss(_Loss, metaclass=ABCMeta):
    """Abstract base class for Megatron distillation losses."""

    def __init__(
        self, model_config: "TransformerConfig", projection_layer: nn.Module | None = None
    ):
        """Constructor.

        Args:
            model_config: MCore transformer config.
            projection_layer: Module which projects student activations to teacher's hidden dim.
        """
        super().__init__()
        self._config = model_config
        self._projection = projection_layer

    def pre_forward(self, predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Performs projection of student tensor to match teacher's size if necessary."""
        if isinstance(predictions, tuple):
            # `ColumnParallelLinear` returns bias too
            predictions, targets = predictions[0], targets[0]

        if self._projection is not None:
            predictions = self._projection(predictions)
        targets = targets.detach()

        return predictions, targets

    def post_forward(
        self, loss: Tensor, tp_reduce: bool = False, is_sequence_parallel: bool = False
    ) -> Tensor:
        """Reshapes tensor from [s, b] to [b, s] for upcoming loss masking."""
        loss = loss.transpose(0, 1).contiguous()
        return (loss, tp_reduce, is_sequence_parallel)


class MSELoss(BaseLoss):
    """Calculates MSE loss between two tensors without reducing the sequence dim."""

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            MSE loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        loss = F.mse_loss(predictions, targets, reduction="none")
        loss = loss.mean(dim=-1)

        return self.post_forward(loss, is_sequence_parallel=self._config.sequence_parallel)


class HiddenStateCosineLoss(BaseLoss):
    """Calculates Cosine loss between two tensors without reducing the sequence dim.

    The tensors are assumed to be intermediate activations, with full hidden dimension size.
    We recommend only applying this loss to LayerNorm outputs, which have full hidden dim even when TP is used.
    """

    def __init__(
        self, model_config: "TransformerConfig", projection_layer: nn.Module | None = None
    ):
        """Constructor.

        Args:
            model_config: MCore transformer config.
            projection_layer: Module which projects student activations to teacher's hidden dim.
        """
        super().__init__(model_config, projection_layer=projection_layer)

        if self._config.tensor_model_parallel_size > 1:
            logger.warning(
                "``HiddenStateCosineLoss`` only works with tensors with full hidden dim. Ensure the "
                "tensor inputs meet this requirement. We recommend only applying this loss to LayerNorm outputs, "
                "which have full hidden dim even when TP is used."
            )

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            Cosine loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        loss = F.cosine_embedding_loss(
            predictions.view(-1, predictions.size(-1)),
            targets.view(-1, targets.size(-1)),
            targets.new_ones(1),
            reduction="none",
        )
        loss = loss.view(*predictions.shape[:2])

        # NOTE: Tensor sequence length is still split among TP ranks.
        return self.post_forward(loss, is_sequence_parallel=self._config.sequence_parallel)


class LogitsKLLoss(BaseLoss):
    """Calculates KL-Divergence loss between two logits tensors without reducing the sequence dim."""

    def __init__(
        self, model_config: "TransformerConfig", temperature: float = 1.0, reverse: bool = False
    ):
        """Constructor.

        Args:
            model_config: MCore transformer config.
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
        """
        super().__init__(model_config)
        self._temperature = temperature
        self._reverse = reverse

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            KLD loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        # Division by temp should happen prior to finding max for both student and teacher.
        output_teacher = targets.float() / self._temperature
        output_student = predictions.float() / self._temperature

        # Compute local softmax, and the reweight to compute global softmax.
        if self._config.tensor_model_parallel_size > 1:
            tp_group = parallel_state.get_tensor_model_parallel_group()

            # Subtract maximum value along vocab dimension across all GPUs (for stability)
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1, keepdim=True)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=tp_group,
            )
            output_teacher -= teacher_logits_max

            student_logits_max, _ = torch.max(output_student, dim=-1, keepdim=True)
            torch.distributed.all_reduce(
                student_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=tp_group,
            )
            output_student -= student_logits_max.detach()

            # Compute global softmax denominators
            # We can't use standard all_reduce function here since the computation
            # that follows it isn't identical across TP ranks.
            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1, keepdim=True)
            denom_teacher = dist_nn.functional.all_reduce(denom_teacher, group=tp_group)

            denom_student = torch.sum(torch.exp(output_student), dim=-1, keepdim=True)
            denom_student = dist_nn.functional.all_reduce(denom_student, group=tp_group)

            # Compute log probabilities (log softmax)
            teacher_log_prob = output_teacher - torch.log(denom_teacher)
            student_log_prob = output_student - torch.log(denom_student)

            # KL divergence
            p, q = student_log_prob, teacher_log_prob
        else:
            # Compute log probabilities
            p, q = F.log_softmax(output_student, dim=-1), F.log_softmax(output_teacher, dim=-1)

        # KL divergence
        if self._reverse:
            p, q = q, p
        loss = torch.sum(F.kl_div(p, q, reduction="none", log_target=True), dim=-1)

        return self.post_forward(loss, tp_reduce=True)


class TopKLogitsKLLoss(LogitsKLLoss):
    """Calculates KL-Divergence loss restricted to the Teacher's Top-K vocabulary entries.

    Calculates using the global Top-K entries without gathering full logits.
    NOTE: Will gather Top-K logits per rank, so mind the value of K for memory and communication.
    """

    def __init__(
        self,
        model_config: "TransformerConfig",
        temperature: float = 1.0,
        reverse: bool = False,
        top_k: int = 1024,
    ):
        """Constructor.

        Args:
            model_config: MCore transformer config.
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
            top_k: The number of top vocabulary entries to keep from the teacher's distribution.
        """
        super().__init__(model_config, temperature, reverse)
        self.top_k = top_k

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            Top-K KLD loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        tp_size = self._config.tensor_model_parallel_size
        assert self.top_k <= targets.size(-1) * tp_size, (
            f"top_k ({self.top_k}) is larger than total vocab size ({targets.size(-1) * tp_size})"
        )

        # Divide by temperature first
        output_teacher = targets.float() / self._temperature
        output_student = predictions.float() / self._temperature

        # Extract local Top-K
        # We take K from each rank and then find the global Top-K of all those.
        local_top_k = min(self.top_k, targets.size(-1))
        top_teacher_vals, top_idx = torch.topk(output_teacher, local_top_k, dim=-1)
        top_student_vals = torch.gather(output_student, dim=-1, index=top_idx)

        if tp_size > 1:
            tp_group = parallel_state.get_tensor_model_parallel_group()

            # Gather all candidates into shape [N_rows, local_k * tp_size]
            # Use all_gather from torch.distributed.nn.functional to preserve gradients
            all_teacher_vals = dist_nn.functional.all_gather(
                top_teacher_vals.contiguous(), group=tp_group
            )
            all_student_vals = dist_nn.functional.all_gather(
                top_student_vals.contiguous(), group=tp_group
            )
            all_teacher_vals = torch.cat(all_teacher_vals, dim=-1)
            all_student_vals = torch.cat(all_student_vals, dim=-1)

            # Pick the true Top-K based on Teacher values
            global_top_vals, global_top_idx = torch.topk(all_teacher_vals, self.top_k, dim=-1)

            final_teacher_logits = global_top_vals
            final_student_logits = torch.gather(all_student_vals, dim=-1, index=global_top_idx)
        else:
            final_teacher_logits = top_teacher_vals
            final_student_logits = top_student_vals

        # Standard (dense) Softmax + KL
        p = F.log_softmax(final_student_logits, dim=-1)
        q = F.log_softmax(final_teacher_logits, dim=-1)

        # KL divergence
        if self._reverse:
            p, q = q, p
        loss = torch.sum(F.kl_div(p, q, reduction="none", log_target=True), dim=-1)

        # No need to reduce since all ranks compute same global Top-K
        return self.post_forward(loss, tp_reduce=False)


class LogitsAndIntermediatesLossBalancer(mtd.DistillationLossBalancer):
    """LossBalancer implementation for Logit and Intermediate losses.

    Dynamically weighs distillation and original losses to balance during training.
    """

    def __init__(self, kd_loss_scale: float = 1.0, skip_original_loss: bool = False):
        """Constructor.

        Args:
            kd_loss_scale: Multiply distillation losses by this before weighing.
                (Not used when `skip_original_loss` is True.)
            skip_original_loss: Used to signal whether the original loss should be used, regardless
                of whether it was passed into ``mtd.DistillationModel.compute_kd_loss()`` or not.
        """
        super().__init__()
        self._kd_loss_scale = kd_loss_scale
        self._skip_original_loss = skip_original_loss

    def forward(self, loss_dict: dict[str, Tensor]) -> Tensor:
        """Forward function.

        Args:
            loss_dict: All individual scalar losses, passed in during ``mtd.DistillationModel.compute_kd_loss()``

        Returns:
            Aggregate total scalar loss.
        """
        original_loss = loss_dict.pop(mtd.loss_balancers.STUDENT_LOSS_KEY)
        for _key in loss_dict:
            if "Logits" in _key:  # class name
                logits_key = _key  # should only be one
        logits_loss = loss_dict.pop(logits_key)
        intermediate_loss = sum(loss_dict.values()) / max(len(loss_dict), 1)

        if intermediate_loss > 0:
            dynamic_scale = logits_loss.detach() / intermediate_loss.detach()
            intermediate_loss_scaled = intermediate_loss * dynamic_scale
        else:
            intermediate_loss = logits_loss.new_tensor(intermediate_loss)
            intermediate_loss_scaled = intermediate_loss

        if self._skip_original_loss:
            total_loss = logits_loss + intermediate_loss_scaled
        else:
            kd_loss = logits_loss + intermediate_loss_scaled
            if kd_loss > 0 and original_loss > 0:  # zero when one CP rank has only context tokens
                kd_loss *= original_loss.detach() / kd_loss.detach()
            total_loss = original_loss + kd_loss * self._kd_loss_scale

        out_dict = {
            "kd_loss": total_loss,
            "logits_loss": logits_loss,
            "intermediate_loss": intermediate_loss,
        }
        return out_dict


class ProjectionLayer(MegatronModule):
    """Module to project student layer activations to teacher's size."""

    def __init__(self, student_config: "TransformerConfig", teacher_config: "TransformerConfig"):
        """Constructor.

        Args:
            student_config: Student's MCore transformer config.
            teacher_config: Teacher's MCore transformer config.
        """
        super().__init__(config=student_config)
        if student_config.hidden_size == teacher_config.hidden_size:
            self._fit = nn.Identity()
        else:
            self._fit = nn.Linear(student_config.hidden_size, teacher_config.hidden_size)
            self.apply(self._init_weights)
            # Attribute below needed to reduce gradients during backward properly.
            setattr(self._fit.weight, "sequence_parallel", self.config.sequence_parallel)
            setattr(self._fit.bias, "sequence_parallel", self.config.sequence_parallel)

    def forward(self, student_tensor: Tensor):
        """Forward function.

        Args:
            student_tensor: Tensor to be fit to teacher size.
        """
        return self._fit(student_tensor)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            self.config.init_method(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()


########################################################


def adjust_distillation_model_for_mcore(
    model: mtd.DistillationModel, distill_cfg: DistillationConfig
):
    """Extra modifications to ``mtd.DistillationModel`` required for Megatron-Core."""

    # Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> "ShardedStateDict":
        with self.hide_teacher_model():
            return type(self).sharded_state_dict(self, *args, **kwargs)

    model.sharded_state_dict = MethodType(_sharded_state_dict, model)

    # Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_student_lm_loss(self, labels, logits) -> Tensor:
        if distill_cfg.skip_lm_loss and self.training:
            return torch.zeros_like(labels, dtype=logits.dtype)
        return type(self).compute_language_model_loss(self, labels, logits)

    model.compute_language_model_loss = MethodType(_compute_student_lm_loss, model)

    # Skip `lm_loss` always for teacher.
    def _compute_teacher_lm_loss(self, labels, logits) -> Tensor:
        return torch.zeros_like(labels, dtype=logits.dtype)

    model.teacher_model.compute_language_model_loss = MethodType(
        _compute_teacher_lm_loss, model.teacher_model
    )

    # HACK: Pipeline-parallel Distillation requires splitting input tensor into student and teacher parts.
    def _set_student_input_tensor_shape(self, shapes: list[tuple[int]]):
        self._tensor_split_idx = shapes[0][-1]

    def _set_input_tensor(self, input_tensors: list[Tensor]):
        teacher_inputs = [
            t[..., self._tensor_split_idx :] if t is not None else t for t in input_tensors
        ]
        student_inputs = [
            t[..., : self._tensor_split_idx] if t is not None else t for t in input_tensors
        ]
        type(self).set_input_tensor(self.teacher_model, teacher_inputs)
        type(self).set_input_tensor(self, student_inputs)

    model.set_student_input_tensor_shape = MethodType(_set_student_input_tensor_shape, model)
    model.set_input_tensor = MethodType(_set_input_tensor, model)

    # HACK: Concatenate output tensors when PP>1 so they can be passed between ranks.
    def _forward(self, *args, **kwargs):
        with torch.no_grad():
            self._teacher_model.eval()
            teacher_output = self._teacher_model(*args, **kwargs)
        with self.only_student_forward():
            student_output = type(self).forward(self, *args, **kwargs)

        if not parallel_state.is_pipeline_last_stage():
            return torch.cat([student_output, teacher_output], dim=-1)
        else:
            return student_output

    model.forward = MethodType(_forward, model)


def get_tensor_shapes_adjust_fn_for_distillation(
    model: torch.nn.Module | list[torch.nn.Module], **kwargs
) -> Callable | None:
    """Return the function to adjust tensor shapes for Distillation in Megatron-Core's forward pass.

    Currently only used during non-interleaved pipelining for Distillation.
    Concatenates sizes of student and teacher output tensors for inter-process communication.
    """
    if (
        parallel_state.get_pipeline_model_parallel_world_size() == 1
        or parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        return None
    # Unwrap
    if isinstance(model, list):
        model = model[0]
    while hasattr(model, "module"):
        model = model.module
    if not isinstance(model, mtd.DistillationModel):
        return None

    def adjust_tensor_shapes(
        recv_tensor_shapes: list[tuple[int, ...]], send_tensor_shapes: list[tuple[int, ...]]
    ):
        teacher_config = get_model_config(model.teacher_model)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()

        teacher_recv_tensor_shapes = get_tensor_shapes(
            config=teacher_config, tp_group=tp_group, cp_group=cp_group, **kwargs
        )
        teacher_send_tensor_shapes = get_tensor_shapes(
            config=teacher_config, tp_group=tp_group, cp_group=cp_group, **kwargs
        )
        model.set_student_input_tensor_shape(recv_tensor_shapes)

        for i, shape in enumerate(recv_tensor_shapes):
            shape = list(shape)
            shape[-1] += teacher_recv_tensor_shapes[0][-1]  # type: ignore[index]
            recv_tensor_shapes[i] = tuple(shape)
        for i, shape in enumerate(send_tensor_shapes):
            shape = list(shape)
            shape[-1] += teacher_send_tensor_shapes[0][-1]  # type: ignore[index]
            send_tensor_shapes[i] = tuple(shape)

        return recv_tensor_shapes, send_tensor_shapes

    return adjust_tensor_shapes

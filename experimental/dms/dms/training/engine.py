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

"""DMS training engine: model configuration, distillation, noise, trainer state, and combined model."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from dms.core import DMSTrainingStateAux
from dms.logging import get_logger

logger = get_logger("Engine")


# =============================================================================
# Model configuration and loading
# =============================================================================

_DMS_PROJ_ALPHA_PATTERN = "dms_proj_alpha"
_UNFROZEN_DUMMY_PATTERN = "_unfrozen_dummy_param"


@dataclass
class ModelArguments:
    """Arguments for specifying a model to load."""

    model_name_or_path: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models."
        },
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Data type for model weights (e.g., 'bfloat16', 'float16', 'float32')."},
    )
    forward_fn_kwargs: dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Additional arguments for the forward function."}
    )


@dataclass
class DistillationModelArguments:
    """Arguments for student-teacher distillation setup."""

    student: ModelArguments = field(default_factory=ModelArguments)
    teacher: ModelArguments = field(default_factory=ModelArguments)


def _is_trainable_param(name: str) -> bool:
    """Check if a parameter should have gradients enabled for DMS training."""
    return _DMS_PROJ_ALPHA_PATTERN in name or _UNFROZEN_DUMMY_PATTERN in name


def _configure_gradients(model: PreTrainedModel) -> tuple[list[str], list[str]]:
    """Configure which parameters require gradients for DMS training.

    Only DMS projection alpha parameters and unfrozen dummy parameters are trainable.

    Returns:
        Tuple of (disabled_grad_names, enabled_grad_names).
    """
    disabled_grad: list[str] = []
    enabled_grad: list[str] = []

    for name, param in model.named_parameters():
        if _is_trainable_param(name):
            param.requires_grad = True
            enabled_grad.append(name)
        else:
            param.requires_grad = False
            disabled_grad.append(name)

    return disabled_grad, enabled_grad


def get_student_model(
    model_args: DistillationModelArguments,
    model_constructor: type[PreTrainedModel],
    zero_out_proj_alpha: bool,
    dms_kwargs: dict[str, Any] | None,
) -> PreTrainedModel:
    """Load and configure a student model for DMS distillation training.

    Args:
        model_args: Distillation model arguments containing student config.
        model_constructor: Model class with from_pretrained method.
        zero_out_proj_alpha: If True, zero out DMS projection alpha parameters.
        dms_kwargs: Additional keyword arguments for model construction.

    Returns:
        Configured student model in training mode.
    """
    if dms_kwargs is None:
        dms_kwargs = {}

    logger.info(
        f"Loading student model from {model_args.student.model_name_or_path} "
        f"with dtype {model_args.student.dtype} and dms kwargs {json.dumps(dms_kwargs, indent=4)}"
    )

    model = model_constructor.from_pretrained(
        model_args.student.model_name_or_path,
        dtype=model_args.student.dtype,
        **dms_kwargs,
    )

    if zero_out_proj_alpha:
        for name, param in model.named_parameters():
            if _DMS_PROJ_ALPHA_PATTERN in name and f"{_DMS_PROJ_ALPHA_PATTERN}_norm" not in name:
                logger.info(f"Zeroing out {name}")
                param.data.zero_()

    disabled_grad, enabled_grad = _configure_gradients(model)
    logger.info(f"Disabled gradients for: {disabled_grad}")
    logger.info(f"Enabled gradients for: {enabled_grad}")

    model.train()
    return model


def get_teacher_model(
    model_args: DistillationModelArguments,
    model_constructor: type[PreTrainedModel],
) -> PreTrainedModel:
    """Load a teacher model for DMS distillation training.

    Args:
        model_args: Distillation model arguments containing teacher config.
        model_constructor: Model class with from_pretrained method.

    Returns:
        Teacher model in evaluation mode.
    """
    logger.info(
        f"Loading teacher model from {model_args.teacher.model_name_or_path} with dtype {model_args.teacher.dtype}"
    )

    model = model_constructor.from_pretrained(
        model_args.teacher.model_name_or_path,
        dtype=model_args.teacher.dtype,
    )
    model.eval()
    return model


def get_tokenizer(model_args: ModelArguments) -> PreTrainedTokenizer:
    """Load a tokenizer for the specified model.

    Args:
        model_args: Model arguments containing the model path.

    Returns:
        Loaded tokenizer instance.
    """
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    return AutoTokenizer.from_pretrained(model_args.model_name_or_path)


# =============================================================================
# DMS noise generation and schedule
# =============================================================================


def get_gumbel_dist(dtype, device):
    """Get the Gumbel distribution for DMS."""
    return torch.distributions.gumbel.Gumbel(
        loc=torch.tensor(0.0, dtype=dtype, device=device),
        scale=torch.tensor(1.0, dtype=dtype, device=device),
        validate_args=None,
    )


def str_to_seed(text: str):
    """Convert a string to a seed."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32 - 1)


def get_dms_noise(dist, device: torch.device, dms_state: DMSTrainingStateAux):
    """Get the Gumbel noise for DMS.

    Uses current process index, gradient accumulation step and current step
    to seed the random number generator. The shape of the noise is the same as the shape of the KV cache.
    """
    with torch.random.fork_rng(devices=[device], enabled=True):
        seed = str_to_seed(
            f"{dms_state.process_index}_{dms_state.grad_acc_step}_{dms_state.current_step}"
        )
        torch.manual_seed(seed)
        a, b = [dist.sample(sample_shape=dms_state.kv_cache_shape).bfloat16() for _ in range(2)]
        noise = a - b

    return noise


def dms_schedule(
    step: int,
    training_args: TrainingArguments,
    dms_initial_cr,
    dms_final_cr,
    dms_final_step: int | None = None,
):
    """Given the current training step, compute the DMS schedule.

    Returns the target fraction of DMS key-value pairs to evict and the compression ratio.
    """
    if dms_final_step is not None:
        max_steps = dms_final_step
    else:
        max_steps = training_args.max_steps

    progress = min(step / max_steps, 1.0)

    cr = dms_initial_cr + (dms_final_cr - dms_initial_cr) * progress

    frac = 1 / cr

    target = 1 - frac  # what fraction of gates to close

    return target, cr


# =============================================================================
# Trainer state (replaces module-level globals)
# =============================================================================


class DMSTrainerState:
    """Encapsulates training state shared between the trainer and the combined model.

    Usage:
        state = DMSTrainerState()
        combined_model = CombinedModel(..., trainer_state=state)
        trainer = ModifiedTrainer(trainer_state=state, ...)
        state.set_trainer(trainer)
    """

    def __init__(self):
        """Initialize with no trainer (set later after trainer construction)."""
        self.trainer: Trainer | None = None
        self.logs: dict[str, float] = {}
        self.grad_acc_step: int = 0

    def set_trainer(self, trainer: Trainer):
        """Set the HF trainer after construction."""
        self.trainer = trainer

    @property
    def step(self) -> int:
        """Get the current global training step."""
        assert self.trainer is not None, "Trainer not set. Call set_trainer() first."
        return self.trainer.state.global_step

    @property
    def process_index(self) -> int:
        """Get the process index in distributed training."""
        assert self.trainer is not None, "Trainer not set. Call set_trainer() first."
        return self.trainer.args.process_index

    def reset_grad_acc_step(self):
        """Update the gradient accumulation step."""
        self.grad_acc_step = 0

    def increment_grad_acc_step(self):
        """Increment the gradient accumulation step."""
        self.grad_acc_step += 1

    def update_logs(self, logs: dict[str, float]):
        """Update logs and track gradient accumulation step."""
        self.logs = dict(**logs)


class DMSGradAccCallback(TrainerCallback):
    """Callback to track gradient accumulation steps for DMS noise seeding."""

    def __init__(self, trainer_state: DMSTrainerState):
        """Initialize with a reference to the shared trainer state."""
        self.trainer_state = trainer_state

    def on_step_begin(self, *args, **kwargs):
        """Reset the gradient accumulation step on new step."""
        self.trainer_state.reset_grad_acc_step()

    def on_substep_end(self, *args, **kwargs):
        """Increment the gradient accumulation step on substep end."""
        self.trainer_state.increment_grad_acc_step()


class ModifiedTrainer(Trainer):
    """Modified Trainer class that gathers DMS logs across distributed processes."""

    def __init__(self, trainer_state: DMSTrainerState, **kwargs):
        """Initialize with a reference to the shared trainer state."""
        super().__init__(**kwargs)
        self.trainer_state = trainer_state
        self.add_callback(DMSGradAccCallback(trainer_state))

    def log(self, logs: dict[str, float], start_time: float | None = None):
        """Log training metrics with gathered global DMS logs."""
        custom_logs = dict(**self.trainer_state.logs)
        names = list(custom_logs.keys())
        values = [custom_logs[key] for key in names]

        if dist.is_initialized():
            values = torch.tensor(values, dtype=torch.float32, device=torch.cuda.current_device())
            dist.all_reduce(values, op=dist.ReduceOp.AVG)

            values = values.tolist()

        for key, value in zip(names, values):
            logs["gl_" + key] = value

        super().log(logs=logs, start_time=start_time)


# =============================================================================
# Distillation loss computation
# =============================================================================


@torch.compile()
def distillation_loss(
    student_raw_logits: torch.Tensor,  # bfloat16 of shape batch, seq, vocab
    teacher_raw_logits: torch.Tensor,  # bfloat16 of shape batch, seq, vocab
    loss_mask: torch.Tensor,  # boolean of shape batch, seq
    vocab_chunk: int,
):
    """Compute KL divergence distillation loss between student and teacher logits (forward KL)."""
    assert student_raw_logits.ndim == 3, (
        f"student_raw_logits.ndim: {student_raw_logits.ndim} != 3 (batch, seq, vocab)"
    )
    assert teacher_raw_logits.shape == student_raw_logits.shape, (
        f"teacher_raw_logits.shape: {teacher_raw_logits.shape} != student_raw_logits.shape: {student_raw_logits.shape}"
    )
    assert loss_mask.shape == student_raw_logits.shape[:2], (
        f"loss_mask.shape: {loss_mask.shape} != student_raw_logits.shape[:2]: {student_raw_logits.shape[:2]}"
    )
    assert loss_mask.dtype == torch.bool, f"loss_mask.dtype: {loss_mask.dtype} != torch.bool"

    # log Denominator of size batch, seq
    s_lse = torch.logsumexp(student_raw_logits.float(), dim=-1)  # batch, seq
    t_lse = torch.logsumexp(teacher_raw_logits.float(), dim=-1)

    # per-token KL
    token_kl = torch.zeros_like(s_lse, dtype=torch.float32)

    vocab_size = student_raw_logits.shape[-1]
    for start in range(0, vocab_size, vocab_chunk):
        end = min(start + vocab_chunk, vocab_size)

        # batch, seq, vchunk
        s_chunk = student_raw_logits[..., start:end]
        t_chunk = teacher_raw_logits[..., start:end]

        s_logp = s_chunk.float() - s_lse[:, :, None]
        t_logp = t_chunk.float() - t_lse[:, :, None]

        # Forward KL: KLD(Student, Teacher)
        token_kl = token_kl + (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)

    denom = loss_mask.sum().clamp_min(1)
    token_kl = token_kl.masked_fill(~loss_mask, 0.0)
    return token_kl.sum() / denom


@torch.compile()
def calc_lm_loss(
    student_raw_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_mask: torch.Tensor,
):
    """Compute language modeling cross-entropy loss for the student model."""
    assert student_raw_logits.ndim == 3, (
        f"student_raw_logits.ndim: {student_raw_logits.ndim} != 3 (batch, seq, vocab)"
    )
    assert input_ids.ndim == 2, f"input_ids.ndim: {input_ids.ndim} != 2 (batch, seq)"
    assert attention_mask.ndim == 2, f"attention_mask.ndim: {attention_mask.ndim} != 2 (batch, seq)"
    assert attention_mask.dtype == torch.bool, (
        f"attention_mask.dtype: {attention_mask.dtype} != torch.bool"
    )
    assert eos_mask.ndim == 2, f"eos_mask.ndim: {eos_mask.ndim} != 2 (batch, seq)"
    assert eos_mask.dtype == torch.bool, f"eos_mask.dtype: {eos_mask.dtype} != torch.bool"
    student_raw_logits = student_raw_logits[:, :-1, :]
    s_lse = torch.logsumexp(student_raw_logits, dim=-1)

    target_ids = input_ids[:, 1:]
    source_raw_logits = student_raw_logits.gather(dim=-1, index=target_ids[:, :, None])[:, :, 0]

    assert source_raw_logits.shape == s_lse.shape
    source_logp = source_raw_logits - s_lse

    # first do not predict from masked
    # second do not predict masked
    neg_loss_mask = torch.logical_or(~attention_mask[:, :-1], ~attention_mask[:, 1:])
    # do not predict from eos
    neg_loss_mask = torch.logical_or(neg_loss_mask, ~eos_mask[:, :-1])
    source_logp = source_logp.masked_fill(neg_loss_mask, 0.0)

    lm_loss = -source_logp.sum() / (~neg_loss_mask).sum().clamp_min(1)

    return lm_loss


# =============================================================================
# Distillation forward pass
# =============================================================================


def distillation_forward(
    student_model: PreTrainedModel,
    teacher_model: PreTrainedModel | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dms_schedule: Callable[[int], tuple[float, float]],
    process_index: int,
    tokenizer: PreTrainedTokenizer,
    student_is_teacher: bool,
    current_step: int,
    grad_acc_step: int,
    process_vocab_using_chunk: int,
    forward_fn_kwargs_student: dict[str, Any],
    forward_fn_kwargs_teacher: dict[str, Any],
    **kwargs: Any,
):
    """Run the distillation forward pass with student and teacher models.

    Args:
        student_model: the student model
        teacher_model: the teacher model, None if student is teacher
        input_ids: input tokens for the models
        attention_mask: boolean attention mask, true -> not masked, false -> masked
        dms_schedule: fraction_of_kv_pairs_to_evict, 1/(1-fraction_of_kv_pairs_to_evict) = dms_schedule(current_step)
        process_index: used for seeding of DMS noise
        tokenizer: the tokenizer used for detecting eos tokens
        student_is_teacher: true -> student is teacher, false -> student is student
        current_step: the current step of the training loop (same across gradient accumulation steps)
        grad_acc_step: the current gradient accumulation step (from 0 to gradient_accumulation_steps - 1)
        process_vocab_using_chunk: the chunk size for processing the vocabulary in distillation loss calculation
        forward_fn_kwargs_student: additional arguments for the student forward function
        forward_fn_kwargs_teacher: additional arguments for the teacher forward function
        **kwargs: additional arguments for the student and teacher models.

    Returns:
        dict with loss (distillation + dms) and other metrics
    """
    # here hf style mask
    # true -> not masked
    eos_mask = input_ids != tokenizer.eos_token_id
    # no prediction from attn masked and no prediction from eos
    assert attention_mask.dtype == torch.bool, (
        f"attention_mask.dtype: {attention_mask.dtype} != torch.bool"
    )
    distill_loss_mask = torch.logical_and(attention_mask, eos_mask)

    dms_target_frac, dms_target_cr = dms_schedule(current_step)

    dms_state = DMSTrainingStateAux(
        target_frac_to_close=dms_target_frac,
        current_step=current_step,
        grad_acc_step=grad_acc_step,
        process_index=process_index,
        noise=None,
        right_padding_size=0,
        kv_cache_shape=(
            student_model.config.num_hidden_layers,
            input_ids.shape[0],
            student_model.config.num_key_value_heads,
            input_ids.shape[1],
        ),
        dms_teacher_mode=False,
    )
    dist_obj = get_gumbel_dist(dtype=torch.bfloat16, device=input_ids.device)
    dms_state.noise = get_dms_noise(dist=dist_obj, device=input_ids.device, dms_state=dms_state)

    with torch.no_grad():
        dms_state_teacher = DMSTrainingStateAux(
            target_frac_to_close=None,
            current_step=current_step,
            grad_acc_step=grad_acc_step,
            process_index=process_index,
            noise=None,
            right_padding_size=0,
            kv_cache_shape=dms_state.kv_cache_shape,
            dms_teacher_mode=True,
        )
        if student_is_teacher:
            assert teacher_model is None, "teacher_model is not None when student is teacher"
            teacher_output = student_model(
                input_ids,
                attention_mask,
                dms_state=dms_state_teacher,
                **forward_fn_kwargs_teacher,
                **kwargs,
            )
        else:
            assert teacher_model is not None, "teacher_model is None when student is not teacher"
            teacher_output = teacher_model(
                input_ids, attention_mask, **forward_fn_kwargs_teacher, **kwargs
            )
        teacher_logits = teacher_output.logits
        assert teacher_logits.ndim == 3, (
            f"teacher_logits.ndim: {teacher_logits.ndim} != 3 (batch, seq, vocab)"
        )

    student_output = student_model(
        input_ids, attention_mask, dms_state=dms_state, **forward_fn_kwargs_student, **kwargs
    )
    dms_loss = student_output.dms_loss

    student_logits = student_output.logits
    assert student_logits.ndim == 3, (
        f"student_logits.ndim: {student_logits.ndim} != 3 (batch, seq, vocab)"
    )
    assert input_ids.ndim == 2, f"input_ids.ndim: {input_ids.ndim} != 2 (batch, seq)"
    distil_loss = torch.utils.checkpoint.checkpoint(
        distillation_loss,
        student_logits,
        teacher_logits,
        distill_loss_mask,
        process_vocab_using_chunk,
        use_reentrant=False,
    )
    with torch.no_grad():
        lm_loss_detach = calc_lm_loss(student_logits.detach(), input_ids, attention_mask, eos_mask)

    loss = distil_loss + dms_loss

    dms_closed_frac_detach = student_output.dms_frac_closed.detach()

    result = {
        "loss": loss,
        "dms_loss": dms_loss.detach(),
        "distil_loss": distil_loss.detach(),
        "dms_target_frac": torch.tensor(dms_target_frac, dtype=torch.float32, device=loss.device),
        "dms_closed_frac": dms_closed_frac_detach,
        "dms_target_cr": torch.tensor(dms_target_cr, dtype=torch.float32, device=loss.device),
        "dms_cr": 1 / torch.clamp(1.0 - dms_closed_frac_detach, min=1e-6),
        "input_tokens": torch.tensor(input_ids.shape[1], dtype=torch.int32, device=loss.device),
        "detached_lm_loss": lm_loss_detach,
        "positions_for_loss_calculation": distill_loss_mask.sum(),
        "positions_without_loss_calculation": (~distill_loss_mask).sum(),
        "eos_tokens": (~eos_mask).to(torch.int32).sum(),
        "masked_tokens": (~attention_mask).to(torch.int32).sum(),
    }

    return result


# =============================================================================
# Combined student-teacher model
# =============================================================================


class CombinedModel(torch.nn.Module):
    """Combined student-teacher model wrapper for distillation training."""

    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        trainer_state: DMSTrainerState,
        dms_schedule: Callable[[int], float],
        forward_fn: Callable,
        student_is_teacher: bool,
        tokenizer: PreTrainedTokenizer,
        process_vocab_using_chunk: int,
        forward_fn_kwargs_student: dict[str, Any] = {},
        forward_fn_kwargs_teacher: dict[str, Any] = {},
    ):
        """Initialize the combined model for distillation.

        Args:
            student_model: the student model
            teacher_model: the teacher model
            trainer_state: shared trainer state object (replaces global callbacks)
            dms_schedule: a function that given current step returns the DMS schedule
                (target fraction of tokens to evict and compression ratio)
            forward_fn: a function that performs the forward pass
            student_is_teacher: whether the student is the teacher
            tokenizer: the tokenizer
            process_vocab_using_chunk: the chunk size for processing the vocabulary
            forward_fn_kwargs_student: additional arguments for the student forward function
            forward_fn_kwargs_teacher: additional arguments for the teacher forward function
        """
        super().__init__()
        self.student_is_teacher = student_is_teacher
        self.student_model = student_model
        if self.student_is_teacher:
            self.teacher_model = None
        else:
            self.teacher_model = teacher_model
            self._freeze_teacher_model()
        self.trainer_state = trainer_state
        self.dms_schedule = dms_schedule
        self.forward_fn = forward_fn

        self.tokenizer = tokenizer
        self.process_vocab_using_chunk = process_vocab_using_chunk
        self.forward_fn_kwargs_student = forward_fn_kwargs_student
        self.forward_fn_kwargs_teacher = forward_fn_kwargs_teacher

    def _freeze_teacher_model(self):
        assert self.teacher_model is not None, "teacher_model is None"
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any):
        """Run the forward pass with student and teacher models."""
        process_index = self.trainer_state.process_index
        current_step = self.trainer_state.step
        grad_acc_step = self.trainer_state.grad_acc_step
        result = self.forward_fn(
            student_model=self.student_model,
            teacher_model=self.teacher_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dms_schedule=self.dms_schedule,
            process_index=process_index,
            student_is_teacher=self.student_is_teacher,
            tokenizer=self.tokenizer,
            current_step=current_step,
            grad_acc_step=grad_acc_step,
            process_vocab_using_chunk=self.process_vocab_using_chunk,
            forward_fn_kwargs_student=self.forward_fn_kwargs_student,
            forward_fn_kwargs_teacher=self.forward_fn_kwargs_teacher,
            **kwargs,
        )

        self.trainer_state.update_logs(
            {key: value.detach().clone().cpu().item() for key, value in result.items()}
        )

        return result

    def get_parameters_to_optimize(self):
        """Get the trainable parameters from the student model."""
        params_to_optimize = [
            param for param in self.student_model.parameters() if param.requires_grad
        ]
        return tuple(params_to_optimize)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the student model."""
        self.student_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

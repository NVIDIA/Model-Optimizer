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

"""ModelOpt plugin for transformers Trainer."""

import contextlib
import gc
import json
import os
import types
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from transformers import TrainerCallback
from transformers.training_args import ParallelMode

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.huggingface import KDTrainer
from modelopt.torch.opt.conversion import restore_from_modelopt_state
from modelopt.torch.opt.plugins import ModelOptHFTrainer
from modelopt.torch.utils import print_rank_0

from ..config import QuantizeConfig
from ..nn import TensorQuantizer
from ..nn.modules.tensor_quantizer import NVFP4StaticAdaRoundQuantizer, StaticBlockScaleQuantizer
from ..utils import (
    calibrate_with_adapters,
    disable_lora_quantizers_in_config,
    get_quantizer_state_dict,
    is_quantized,
    quantizer_attr_names,
    set_quantizer_state_dict,
    weight_attr_names,
)

# TODO: Enable documentation rendering for this class


@dataclass
class QuantizationArguments:
    """Quantization arguments for quantization aware training.

    This classes is intended to be used with ModelOpt's QAT/QAD trainers for HuggingFace models.
    This class can also be used to parse the quantization arguments
    from the command line to the taining script.
    """

    quant_cfg: str | None = field(
        default=None,
        metadata={
            "help": (
                "Specify the quantization format for PTQ/QAT. if specified, PTQ/QAT will be enabled"
                " with the specified quantization format"
            ),
        },
    )
    calib_size: int = field(
        default=512,
        metadata={
            "help": (
                "Specify the calibration size for quantization. The calibration dataset is used to"
                " setup the quantization scale parameters for PTQ/QAT."
            )
        },
    )
    compress: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compress the model weights after quantization for QLoRA. "
                "This is useful for reducing the model size."
            )
        },
    )


@dataclass
class AdaRoundTrainingArguments:
    """Training-time arguments for AdaRound dist_loss annealing.

    These are separated from :class:`QuantizationArguments` because they are
    specific to the AdaRound training procedure and not general QAT knobs.
    """

    beta_start: float = field(
        default=20.0,
        metadata={"help": "Initial beta for dist_loss annealing (high = permissive)."},
    )
    beta_end: float = field(
        default=2.0,
        metadata={"help": "Final beta for dist_loss annealing (low = forces binary)."},
    )
    dist_loss_coeff: float = field(
        default=0.01,
        metadata={"help": "Lambda multiplier for dist_loss regularization."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sigmoid temperature for rounding logits."},
    )
    freeze_weights: bool = field(
        default=True,
        metadata={"help": "Freeze parent module weights when adaround is active."},
    )


@dataclass
class QuantErrorTrainingArguments:
    """Training-time arguments for quantization error regularization.

    After each optimizer step, computes MSE between quantized and original weights
    and applies a manual gradient step to push weights toward quantized grid points.
    The coefficient is linearly annealed from ``qerr_coeff_start`` to ``qerr_coeff_stop``.
    """

    qerr_coeff_start: float = field(
        default=0.0,
        metadata={
            "help": "Initial quantization error coefficient (0 = monitor only, no gradient)."
        },
    )
    qerr_coeff_stop: float = field(
        default=0.0,
        metadata={"help": "Final quantization error coefficient (0 = monitor only, no gradient)."},
    )
    qerr_reduction: str = field(
        default="sum",
        metadata={"help": "Reduction for quantization error: 'mean' or 'sum'. Default 'sum'."},
    )


class QuantizationArgumentsWithConfig(QuantizationArguments):
    """Quantization arguments for quantization aware training with config.

    This class is intended to be used with ModelOpt's QAT/QAD trainers for HuggingFace models,
    however, it cannot be used for command line parsing.
    """

    quant_cfg: str | QuantizeConfig | None = field(
        default=None,
        metadata={
            "help": (
                "Specify the quantization format for PTQ/QAT. if specified, PTQ/QAT will be enabled"
                " with the specified quantization format"
            ),
        },
    )


def _patch_fsdp2_post_backward():
    """Patch FSDP2 ``post_backward`` to handle mixed-precision gradient dtypes.

    FSDP2 with bf16 mixed precision upcasts bf16 parameters to fp32 for optimizer
    precision, while gradients are reduced in bf16. In PyTorch >= 2.6, assigning a
    bf16 gradient to a fp32 parameter raises a ``RuntimeError`` due to the
    ``grad_dtype`` check, and the fused Adam optimizer also rejects mixed dtypes.

    This patch wraps ``FSDPParamGroup.post_backward`` to:
    1. Set ``grad_dtype=None`` on sharded params before reduction (allowing bf16 assignment).
    2. Cast gradients to match parameter dtype after reduction (so the optimizer sees matching dtypes).

    .. note::
        This is a workaround. The proper fix should come from PyTorch's FSDP2
        ``foreach_reduce`` (which should cast gradients to match the parameter dtype)
        or from accelerate (which should set ``grad_dtype`` when it upcasts params).
        Remove this once the upstream fix is available.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    except ImportError:
        return

    if hasattr(FSDPParamGroup, "_modelopt_original_post_backward"):
        return  # Already patched

    FSDPParamGroup._modelopt_original_post_backward = FSDPParamGroup.post_backward

    @torch.no_grad()
    def _patched_post_backward(self):
        # Allow bf16 gradients to be assigned to fp32 parameters
        for fsdp_param in self.fsdp_params:
            with contextlib.suppress(AttributeError):
                fsdp_param.sharded_param.grad_dtype = None

        self._modelopt_original_post_backward()

        # Cast gradients to parameter dtype so the optimizer sees matching dtypes
        for fsdp_param in self.fsdp_params:
            sp = fsdp_param.sharded_param
            if sp.grad is not None and sp.grad.dtype != sp.dtype:
                sp.grad = sp.grad.to(sp.dtype)

    FSDPParamGroup.post_backward = _patched_post_backward


class QATTrainer(ModelOptHFTrainer):
    """A drop-in replacement of HuggingFace's Trainer for quantization aware training with ModelOpt.

    Args:
        quant_args: General quantization arguments (quant_cfg, calib_size).
        adaround_args: AdaRound-specific training arguments (beta annealing, dist_loss_coeff).
            When ``None``, the AdaRound auxiliary callback is a no-op.
        qerr_args: Quantization error regularization arguments (coefficient annealing).
            When ``None``, the quantization error callback is a no-op.
    """

    def __init__(
        self,
        *args,
        quant_args: QuantizationArgumentsWithConfig | QuantizationArguments | None = None,
        adaround_args: AdaRoundTrainingArguments | None = None,
        qerr_args: QuantErrorTrainingArguments | None = None,
        **kwargs,
    ):
        """Initialize the trainer with modelopt states."""
        super().__init__(*args, **kwargs)

        if adaround_args is not None and qerr_args is not None:
            raise ValueError("adaround_args and qerr_args are mutually exclusive.")

        self.quant_args = quant_args
        self.adaround_args = adaround_args
        self.qerr_args = qerr_args
        quant_cfg = None
        if quant_args is not None and getattr(quant_args, "quant_cfg", None):
            quant_cfg = (
                getattr(mtq, quant_args.quant_cfg)
                if isinstance(quant_args.quant_cfg, str)
                else quant_args.quant_cfg
            )
        self.quant_cfg = quant_cfg

        # Add lora adapter before quantizing the model
        if getattr(self.args, "lora_config", None) is not None and not hasattr(
            self.model, "peft_config"
        ):
            # TODO: use get_peft_model here instead of add_adapter
            self.model.add_adapter(self.args.lora_config)
            print_rank_0("Lora adapter added.")

        if hasattr(self.model, "peft_config") and self.quant_cfg is not None:
            target_modules = (
                self.args.lora_config.target_modules if hasattr(self.args, "lora_config") else []
            )
            disable_lora_quantizers_in_config(self.quant_cfg, target_modules)

        if self.args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            raise ValueError(
                "QATTrainer does not support torch.nn.DataParallel (multiple GPUs in a single"
                " process). Use DistributedDataParallel via `torchrun` or `accelerate launch`"
                " instead."
            )

        self._patch_accelerate_for_fsdp2_fix()

        self._modelopt_state_path = os.path.join(self.args.output_dir, "modelopt_state_train.pth")
        if os.path.exists(self._modelopt_state_path) and not is_quantized(self.model):
            self._restore_modelopt_state_with_weights()
        elif is_quantized(self.model):
            self._save_modelopt_state_with_weights()

        if self.quant_cfg is not None and not is_quantized(self.model):
            algorithm = self.quant_cfg.get("algorithm")
            method = (
                algorithm
                if isinstance(algorithm, str)
                else (algorithm.get("method") if isinstance(algorithm, dict) else None)
            )
            if method is not None and method != "max":
                raise ValueError(
                    f"Only 'max' calibration is supported for training-time quantization. "
                    f"Got algorithm='{method}'. For other algorithms, please quantize the "
                    f"model first using mtq.quantize() and pass the quantized model."
                )

        self._original_dtype = getattr(
            getattr(self.model, "config", None), "dtype", None
        ) or getattr(getattr(self.model, "config", None), "torch_dtype", None)

        if self._is_adaround():
            self._setup_adaround()
        elif self.qerr_args is not None:
            self._setup_qerr()

    def _setup_qerr(self):
        """Set up quantization error regularization: register aux callback."""
        self._qerr_pending_metrics = {}
        self.add_callback(_QuantErrorAuxCallback(trainer=self))

    def _is_adaround(self):
        """Return True if adaround_args is provided and model contains AdaRound quantizers."""
        return self.adaround_args is not None and any(
            isinstance(m, NVFP4StaticAdaRoundQuantizer) and m._adaround_enabled
            for _, m in self.model.named_modules()
        )

    def _setup_adaround(self):
        """Set up AdaRound: register aux callback and freeze parent weights."""
        self._adaround_pending_metrics = {}
        self.add_callback(_AdaRoundAuxCallback(trainer=self))
        self._freeze_adaround_weights()

    def log(self, logs, *args, **kwargs):
        """Override to inject auxiliary metrics before standard logging."""
        if hasattr(self, "_adaround_pending_metrics") and self._adaround_pending_metrics:
            logs.update(self._adaround_pending_metrics)
            self._adaround_pending_metrics = {}
        if hasattr(self, "_qerr_pending_metrics") and self._qerr_pending_metrics:
            logs.update(self._qerr_pending_metrics)
            self._qerr_pending_metrics = {}
        return super().log(logs, *args, **kwargs)

    def _freeze_adaround_weights(self):
        """Freeze parent module weights when adaround is active."""
        if self.adaround_args is None or not self.adaround_args.freeze_weights:
            return

        frozen_count = 0
        for _name, module in self.model.named_modules():
            for weight_name in weight_attr_names(module):
                wq_name = quantizer_attr_names(weight_name).weight_quantizer
                quantizer = getattr(module, wq_name, None)
                if not isinstance(quantizer, NVFP4StaticAdaRoundQuantizer):
                    continue
                if not quantizer._adaround_enabled:
                    continue
                for pname, param in module.named_parameters(recurse=False):
                    if pname == weight_name:
                        param.requires_grad_(False)
                        frozen_count += 1
        if frozen_count:
            print_rank_0(f"AdaRound: froze {frozen_count} weight parameter(s).")

    def _save_modelopt_state_with_weights(self):
        """Save the modelopt weights for fsdp2 models."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        modelopt_state = mto.modelopt_state(self.model)
        modelopt_state["modelopt_state_weights"] = get_quantizer_state_dict(self.model)

        if self.args.should_save:
            torch.save(modelopt_state, self._modelopt_state_path)

        print_rank_0(f"Saved modelopt state to {self._modelopt_state_path}")

    def _restore_modelopt_state_with_weights(self):
        # Security NOTE: weights_only=False is used here on ModelOpt-generated state_dict, not on untrusted user input
        modelopt_state = torch.load(self._modelopt_state_path, weights_only=False)
        modelopt_weights = modelopt_state.pop("modelopt_state_weights", None)
        restore_from_modelopt_state(self.model, modelopt_state)
        if modelopt_weights is not None:
            set_quantizer_state_dict(self.model, modelopt_weights)
        print_rank_0("Restored modelopt state with weights.")

    def _quantize_model(self):
        """Quantize the model. Restore the quantization state if it exists."""
        dataset = self.train_dataset if self.train_dataset is not None else self.eval_dataset
        assert dataset is not None, "Calibration requires either eval or train dataset."
        num_samples = min(self.quant_args.calib_size, len(dataset))  # type: ignore [union-attr]
        dataset = torch.utils.data.Subset(dataset, list(range(num_samples)))
        data_loader = self.get_eval_dataloader(dataset)

        def forward_loop(model):
            for batch in tqdm(data_loader, desc="Calibrating", disable=not self.args.should_save):
                batch = self._prepare_inputs(batch)
                # Important: We should forward pass using the unwrapped model
                # mtq.quantize will unwrap the model & pass to the forward_loop
                self.model(**batch)

        # TODO: Remove calibrate_with_adapters - this should not be needed
        with calibrate_with_adapters(self.model, self.args):
            print_rank_0("Quantizing the model...")
            mtq.quantize(self.model, self.quant_cfg, forward_loop)  # type: ignore [arg-type]

        # Save modelopt state
        self._save_modelopt_state_with_weights()

        if getattr(self.quant_args, "compress", False):
            print_rank_0("Compressing model after calibration")
            mtq.compress(self.model)

        # Force garbage collection to free up memory
        gc.collect()

        torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
            mtq.print_quant_summary(self.model)

    def training_step(self, *args, **kwargs):
        """Training step."""
        if self.quant_cfg is not None and not is_quantized(self.model):
            self._quantize_model()
        return super().training_step(*args, **kwargs)

    def prediction_step(self, *args, **kwargs):
        """Prediction step."""
        if self.quant_cfg is not None and not is_quantized(self.model):
            self._quantize_model()
        return super().prediction_step(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Evaluate the model."""
        if self.args.do_eval and not self.args.do_train and self.accelerator.is_fsdp2:
            # [Not related to ModelOpt] HF does not support eval only for FSDP2.
            # This is a hack to make it work
            dummy_optimizer = torch.optim.SGD([next(self.model.parameters())], lr=0.0)
            self.model, _ = self.accelerator.prepare(self.model, dummy_optimizer)
        return super().evaluate(*args, **kwargs)

    def train(self, *args, **kwargs):
        """Train the model."""
        outputs = super().train(*args, **kwargs)
        print_rank_0(
            "Training completed. Please save the final model using `Trainer.save_model()` to preserve ModelOpt states."
        )
        return outputs

    def save_model(self, *args, **kwargs):
        """Save the quantized model."""
        if (
            (not self.is_in_train)
            and self.is_fsdp_enabled
            and self.accelerator.state.fsdp_plugin.state_dict_type != "FULL_STATE_DICT"
        ):
            print_rank_0("Setting state_dict_type to FULL_STATE_DICT for final checkpoint save.")
            original_type = self.accelerator.state.fsdp_plugin.state_dict_type
            self.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            outputs = super().save_model(*args, **kwargs)
            self.accelerator.wait_for_everyone()
            if mto.ModeloptStateManager.is_converted(self.accelerator.unwrap_model(self.model)):
                print_rank_0(
                    "Model saved. To restore, call mto.enable_huggingface_checkpointing() first before loading the "
                    "model. See https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.opt.plugins.huggingface.html#modelopt.torch.opt.plugins.huggingface.enable_huggingface_checkpointing"
                )
            self.accelerator.state.fsdp_plugin.set_state_dict_type(original_type)
        else:
            outputs = super().save_model(*args, **kwargs)
        if (not self.is_in_train) and self.args.should_save:
            out_dir = args[0]
            # FSDP may upcast parameter dtype to float32 during mixed-precision training,
            # we convert it back to original dtype by updating `torch-dtype` in `config.json`
            self._update_config_json_dtype(out_dir, str(self._original_dtype).split(".")[1])
        return outputs

    def _load_best_model(self, *args, **kwargs):
        """Load the best model for final evaluation."""
        is_lora = getattr(self.args, "lora", None)
        if is_lora and not self.is_fsdp_enabled:
            # Custom logic for loading best model with LoRA
            # TODO: Remove once we migrate to using get_peft_model()
            # This custom logic only loads best adapters. Ensure base model is frozen
            assert all(
                not param.requires_grad
                for name, param in self.model.base_model.named_parameters()
                if "base_layer" in name
            ), "Some base_layer parameters are not frozen"

            adapter_name = self.model.active_adapters()[0]
            self.model.delete_adapter(adapter_name)
            self.model.load_adapter(self.state.best_model_checkpoint, adapter_name)
        else:
            super()._load_best_model(*args, **kwargs)

    def _update_config_json_dtype(self, output_dir: str, dtype_str: str | None) -> None:
        """Rewrite <output_dir>/config.json 'dtype' (preferred) or 'torch_dtype' to dtype_str."""
        cfg_path = os.path.join(output_dir, "config.json")
        if not os.path.isfile(cfg_path):
            print_rank_0(f"[warn] config.json not found under {output_dir}; skip dtype rewrite.")
            return
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = json.load(f)
            # Prefer 'dtype', else fall back to 'torch_dtype'
            key_to_update = (
                "dtype" if "dtype" in data else ("torch_dtype" if "torch_dtype" in data else None)
            )
            if key_to_update is None:
                print_rank_0(
                    "[warn] Neither 'dtype' nor 'torch_dtype' present in config.json; skip dtype rewrite."
                )
                return
            if data.get(key_to_update) != dtype_str:
                data[key_to_update] = dtype_str
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print_rank_0(f'Updated config.json: {key_to_update} -> "{dtype_str}"')
        except Exception as e:
            print_rank_0(f"[warn] Failed to update dtype in config.json: {e}")

    def _patch_accelerate_for_fsdp2_fix(self):
        """Fixes for accelerate prepare.

        Accelerate fsdp2 prepare assumes that all parameters and buffers are sharded. This assumption
        is causing issues with quantized models since quantization modules adds buffers which are not sharded.
        This patch hides the buffers added by quantization modules from the original accelerate prepare.
        """
        _patch_fsdp2_post_backward()

        def _modelopt_prepare(self, *args, **kwargs):
            if not self.is_fsdp2:
                return self._original_prepare(*args, **kwargs)

            model = next((obj for obj in args if isinstance(obj, torch.nn.Module)), None)
            if model is None:
                return self._original_prepare(*args, **kwargs)

            tq_og_non_prsist_buffers = {}
            for tq in (m for m in model.modules() if isinstance(m, TensorQuantizer)):
                tq.to(device=self.device)
                tq_og_non_prsist_buffers[tq] = tq._non_persistent_buffers_set.copy()
                tq._non_persistent_buffers_set.update(tq._buffers.keys())

            outputs = self._original_prepare(*args, **kwargs)

            for tq in (m for m in model.modules() if isinstance(m, TensorQuantizer)):
                tq._non_persistent_buffers_set.clear()
                tq._non_persistent_buffers_set.update(tq_og_non_prsist_buffers[tq])

            return outputs

        self.accelerator._original_prepare = self.accelerator.prepare
        self.accelerator.prepare = types.MethodType(_modelopt_prepare, self.accelerator)


class QADTrainer(KDTrainer, QATTrainer):
    """A drop-in replacement of HuggingFace's Trainer for quantization aware distillation with ModelOpt.

    This class takes additional arguments for both distillation and quantization configuration.
    For details, see
    :class:`QATTrainer <QATTrainer>`
    and
    :class:`KDTrainer <modelopt.torch.distill.plugins.huggingface.KDTrainer>`.
    """

    def _quantize_model(self):
        """Quantize the model."""
        model = self.accelerator.unwrap_model(self.model)
        with model.hide_teacher_model(), model.only_student_forward():
            return super()._quantize_model()


class _AdaRoundAuxCallback(TrainerCallback):
    """Callback that updates AdaRound rounding logits via per-param dist_loss after each step."""

    def __init__(self, trainer):
        self._trainer = trainer
        self._adaround_quantizers = None  # lazy init

    def _lazy_init(self):
        model = self._trainer.accelerator.unwrap_model(self._trainer.model)
        self._adaround_quantizers = []
        for name, m in model.named_modules():
            if isinstance(m, NVFP4StaticAdaRoundQuantizer) and m._adaround_enabled:
                m._debug_name = name
                self._adaround_quantizers.append(m)

        # Map each quantizer's round_logits to its optimizer param group index
        pid_to_group = {}
        for group_idx, group in enumerate(self._trainer.optimizer.param_groups):
            for p in group["params"]:
                pid_to_group[id(p)] = group_idx
        self._multiplier = {}  # quantizer -> scalar tensor
        self._param_group_idx = {}  # quantizer -> optimizer group index
        dev = self._adaround_quantizers[0].round_logits.device
        for q in self._adaround_quantizers:
            self._multiplier[q] = torch.zeros(1, device=dev)
            self._param_group_idx[q] = pid_to_group.get(id(q.round_logits), 0)
        self._round_logits_params = [q.round_logits for q in self._adaround_quantizers]
        self._dist_loss_acc = torch.zeros(1, device=dev)

    def _update_multipliers(self):
        assert self._adaround_quantizers is not None
        ada_args = self._trainer.adaround_args
        param_groups = self._trainer.optimizer.param_groups
        for q in self._adaround_quantizers:
            lr = param_groups[self._param_group_idx[q]]["lr"]
            self._multiplier[q].copy_(torch.tensor(lr * ada_args.dist_loss_coeff))

    def on_step_end(self, args, state, control, **kwargs):
        ada_args = self._trainer.adaround_args
        if self._adaround_quantizers is None:
            self._lazy_init()
        assert self._adaround_quantizers is not None

        self._update_multipliers()
        progress = state.global_step / max(state.max_steps, 1)
        beta = ada_args.beta_start + (ada_args.beta_end - ada_args.beta_start) * progress

        self._dist_loss_acc.zero_()
        for q in self._adaround_quantizers:
            if q.round_logits.grad is not None:
                q.round_logits.grad.zero_()
            dist_loss = q.dist_loss(beta=beta)
            dist_loss.backward()
            self._dist_loss_acc += dist_loss.detach()

        total_grad_norm = torch.nn.utils.get_total_norm(self._round_logits_params)

        for q in self._adaround_quantizers:
            with torch.no_grad():
                q.round_logits.data -= self._multiplier[q] * q.round_logits.grad
            q.round_logits.grad.zero_()

        self._trainer._adaround_pending_metrics = {
            "adaround/dist_loss": self._dist_loss_acc.item(),
            "adaround/beta": beta,
            "adaround/grad_norm": total_grad_norm.item()
            if isinstance(total_grad_norm, torch.Tensor)
            else total_grad_norm,
        }
        return control


class _QuantErrorAuxCallback(TrainerCallback):
    """Callback that regularizes weights toward quantized grid points after each optimizer step.

    Computes per-weight MSE = mean((Q(w) - w)^2) and applies independent manual SGD steps.
    For StaticBlockScaleQuantizer, uses _cast_ste (pointwise grid snap without scaling).
    For other TensorQuantizers, uses the full quantizer forward.
    The coefficient is linearly annealed from qerr_coeff_start to qerr_coeff_stop.
    """

    def __init__(self, trainer):
        self._trainer = trainer
        self._weight_entries = None  # lazy init

    def _lazy_init(self):
        model = self._trainer.accelerator.unwrap_model(self._trainer.model)
        self._weight_entries = []  # list of (weight_param, quantizer)
        for _name, module in model.named_modules():
            for weight_name in weight_attr_names(module):
                wq_name = quantizer_attr_names(weight_name).weight_quantizer
                quantizer = getattr(module, wq_name, None)
                if quantizer is None or not quantizer.is_enabled:
                    continue
                weight = getattr(module, weight_name, None)
                if not isinstance(weight, torch.nn.Parameter):
                    continue
                self._weight_entries.append((weight, quantizer))

        pid_to_group = {}
        for group_idx, group in enumerate(self._trainer.optimizer.param_groups):
            for p in group["params"]:
                pid_to_group[id(p)] = group_idx
        self._param_group_idx = {}
        self._multiplier = {}
        for weight, _q in self._weight_entries:
            self._param_group_idx[id(weight)] = pid_to_group.get(id(weight), 0)
            self._multiplier[id(weight)] = torch.zeros(1, device=weight.device)

        dev = self._weight_entries[0][0].device if self._weight_entries else torch.device("cpu")
        self._mse_acc = torch.zeros(1, device=dev)

    def _compute_mse(self, weight, quantizer):
        """Compute MSE between original and quantized weight."""
        if isinstance(quantizer, StaticBlockScaleQuantizer):
            # Reshape to [num_blocks, block_size] as expected by the Triton FP4 kernel
            orig_shape = weight.shape
            if hasattr(quantizer, "_block_reshape_size"):
                w = weight.reshape(quantizer._block_reshape_size)
            else:
                w = weight
            q_weight = quantizer._cast_ste(w)
            q_weight = q_weight.reshape(orig_shape)
        else:
            q_weight = quantizer(weight)
        sq_err = (q_weight - weight) ** 2
        if self._trainer.qerr_args.qerr_reduction == "sum":
            return sq_err.sum()
        return sq_err.mean()

    def _update_multipliers(self, qerr_coeff):
        param_groups = self._trainer.optimizer.param_groups
        for weight, _q in self._weight_entries:  # type: ignore[union-attr]
            lr = param_groups[self._param_group_idx[id(weight)]]["lr"]
            self._multiplier[id(weight)].fill_(lr * qerr_coeff)

    def on_step_end(self, args, state, control, **kwargs):
        if self._weight_entries is None:
            self._lazy_init()
        if not self._weight_entries:
            return control

        qerr_args = self._trainer.qerr_args
        progress = state.global_step / max(state.max_steps, 1)
        qerr_coeff = (
            qerr_args.qerr_coeff_start
            + (qerr_args.qerr_coeff_stop - qerr_args.qerr_coeff_start) * progress
        )

        self._mse_acc.zero_()

        if qerr_coeff > 0:
            # Active regularization: compute MSE, apply gradient step
            self._update_multipliers(qerr_coeff)
            for weight, quantizer in self._weight_entries:
                if weight.grad is not None:
                    weight.grad.zero_()
                mse = self._compute_mse(weight, quantizer)
                mse.backward()
                self._mse_acc += mse.detach()
                with torch.no_grad():
                    weight.data -= self._multiplier[id(weight)] * weight.grad
                weight.grad.zero_()
        else:
            # Monitor only: compute MSE without gradient
            with torch.no_grad():
                for weight, quantizer in self._weight_entries:
                    self._mse_acc += self._compute_mse(weight, quantizer)

        reduction = self._trainer.qerr_args.qerr_reduction
        self._trainer._qerr_pending_metrics = {
            f"qerr/{reduction}": self._mse_acc.item(),
            "qerr/coeff": qerr_coeff,
        }
        return control

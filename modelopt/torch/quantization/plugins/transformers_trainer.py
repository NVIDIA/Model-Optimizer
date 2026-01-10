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

import gc
import json
import os
import types
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.huggingface import KDTrainer
from modelopt.torch.opt.conversion import restore_from_modelopt_state
from modelopt.torch.opt.plugins import ModelOptHFTrainer
from modelopt.torch.utils import print_rank_0

from ..config import QuantizeConfig
from ..nn import TensorQuantizer
from ..utils import (
    calibrate_with_adapters,
    disable_lora_quantizers_in_config,
    get_quantizer_state_dict,
    is_quantized,
    set_quantizer_state_dict,
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


def check_awq_smoothquant(quant_cfg):
    # TODO: Remove this once deepspeed for AWQ and SmoothQuant is added
    """Get the quantization type from the configuration."""
    if quant_cfg is None:
        return False
    algorithm = quant_cfg.get("algorithm", {})
    is_awq_smoothquant = False
    # Check SmoothQuant and AWQ
    if algorithm and ("smoothquant" in algorithm or "awq" in algorithm):
        is_awq_smoothquant = True

    return is_awq_smoothquant


class QATTrainer(ModelOptHFTrainer):
    """A drop-in replacement of HuggingFace's Trainer for quantization aware training with ModelOpt.

    This class takes an additional optional argument `quant_args` of type
    :class:`QuantizationArgumentsWithConfig <QuantizationArgumentsWithConfig>`
    to specify the quantization arguments.
    """

    def __init__(
        self,
        *args,
        quant_args: QuantizationArgumentsWithConfig | QuantizationArguments | None = None,
        **kwargs,
    ):
        """Initialize the trainer with modelopt states."""
        super().__init__(*args, **kwargs)

        self.quant_args = quant_args
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

        if self.is_deepspeed_enabled:
            assert not check_awq_smoothquant(self.quant_cfg), (
                f"QAT DeepSpeed does not currently support AWQ or SmoothQuant: {self.quant_cfg}"
            )

        self._patch_accelerate_for_fsdp2_fix()

        self._modelopt_state_path = os.path.join(self.args.output_dir, "modelopt_state_train.pth")
        if os.path.exists(self._modelopt_state_path):
            self._restore_modelopt_state_with_weights()
        elif is_quantized(self.model):
            self._save_modelopt_state_with_weights()

        self._original_dtype = getattr(
            getattr(self.model, "config", None), "dtype", None
        ) or getattr(getattr(self.model, "config", None), "torch_dtype", None)

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

        def _modelopt_prepare(self, *args, **kwargs):
            if not self.is_fsdp2:
                return self._original_prepare(*args, **kwargs)

            model = next((obj for obj in args if isinstance(obj, torch.nn.Module)), None)
            if model is None:
                return self._original_prepare(*args, **kwargs)

            tq_og_non_prsist_buffers = {}
            for tq in (m for m in model.modules() if isinstance(m, TensorQuantizer)):
                tq.to_empty(device=self.device)
                tq_og_non_prsist_buffers[tq] = tq._non_persistent_buffers_set.copy()
                tq._non_persistent_buffers_set.update(tq._buffers.keys())

            outputs = self._original_prepare(*args, **kwargs)

            for tq in (m for m in model.modules() if isinstance(m, TensorQuantizer)):
                tq._non_persistent_buffers_set.clear()
                tq._non_persistent_buffers_set.update(tq_og_non_prsist_buffers[tq])

            return outputs

        self.accelerator._original_prepare = self.accelerator.prepare
        self.accelerator.prepare = types.MethodType(_modelopt_prepare, self.accelerator)


class QADTrainer(QATTrainer, KDTrainer):
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

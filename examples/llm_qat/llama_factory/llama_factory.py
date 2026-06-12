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

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import transformers
import yaml
from llamafactory.extras import logging
from llamafactory.extras.misc import (
    count_parameters,
    skip_check_imports,
    try_download_model_from_other_hub,
)
from llamafactory.hparams import FinetuningArguments, ModelArguments
from llamafactory.model.adapter import init_adapter
from llamafactory.model.loader import load_config
from llamafactory.model.patcher import patch_config
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, Trainer
from wrapt import register_post_import_hook

import modelopt.torch.opt as mto
from modelopt.torch.distill.plugins.huggingface import LMLogitsLoss
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer, QATTrainer

logger = logging.get_logger(__name__)


@dataclass
class QuantizationArguments:
    quant_cfg: str | None = None
    calib_size: int = 512
    compress: bool = False


@dataclass
class DistillationArguments:
    teacher_model: str | None = None
    distill: bool = False


def _get_init_kwargs(model_args: ModelArguments) -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


# Enable ModelOpt checkpointing for HuggingFace models
mto.enable_huggingface_checkpointing()


def parse_args():
    """Parse configuration file and extract ModelOpt quantization/distillation arguments.

    Returns:
        tuple: (config, quant_args, distill_args) where:
            - config: Main configuration dict with modelopt section removed
            - quant_args: QuantizationArguments with quantization parameters
            - distill_args: DistillationArguments with distillation parameters (or None)
    """
    if len(sys.argv) < 2:
        raise SystemExit("Usage: llama_factory.py <config.yaml>")
    config_path = sys.argv[1]

    assert config_path.endswith(".yaml"), "Config file must be a YAML file"
    # Load config file
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    quant_args = None
    distill_args = None

    # Extract ModelOpt configuration if present
    modelopt_dict = config.pop("modelopt", None)
    if modelopt_dict:
        # Parse quantization configuration
        parser = transformers.HfArgumentParser((QuantizationArguments, DistillationArguments))
        quant_args, distill_args = parser.parse_dict(modelopt_dict)
        if distill_args.distill:
            assert distill_args.teacher_model is not None, (
                "Teacher model is required for distillation, "
                "specify as `teacher_model: <path_to_teacher_model>` under modelopt section in the config file."
            )

    return config, quant_args, distill_args


def patch_load_module(module):
    """Patch the load_model function in the module to restrict to causal LM models only."""

    def load_model(
        tokenizer: PreTrainedTokenizer,
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: bool = False,
        add_valuehead: bool = False,
    ) -> PreTrainedModel:
        r"""Load pretrained model."""

        init_kwargs = _get_init_kwargs(model_args)
        config = load_config(model_args)

        patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

        assert not model_args.enable_liger_kernel, "Liger kernel is currently not supported."
        assert not model_args.use_unsloth, "Unsloth is currently not supported."
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        # Check if this is a causal LM model
        if type(config) in AutoModelForCausalLM._model_mapping:
            load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(
                    config, trust_remote_code=model_args.trust_remote_code
                )
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) == "qwen2_5_omni":
                    model = model.thinker  # use part of Omni model
        else:
            raise ValueError(f"Model type {type(config)} is not supported.")

        assert model_args.mixture_of_depths != "convert", (
            "Mixture of Depth is currently not supported."
        )
        assert not add_valuehead, "Valuehead is currently not supported."

        model = init_adapter(config, model, model_args, finetuning_args, is_trainable)
        if not is_trainable:
            model.requires_grad_(False)
            for param in model.parameters():
                if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                    param.data = param.data.to(model_args.compute_dtype)

            model.eval()
        else:
            model.train()

        trainable_params, all_param = count_parameters(model)
        if is_trainable:
            param_stats = (
                f"trainable params: {trainable_params:,} || "
                f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
            )
        else:
            param_stats = f"all params: {all_param:,}"

        logger.info_rank0(param_stats)

        if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
            for name, param in model.named_parameters():
                print(
                    f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}"
                )

        return model

    # Replace the load_model function in the module
    module.load_model = load_model


def create_patch_module(quant_args=None, distill_args=None):
    """Create a patch function that modifies LLaMA-Factory's trainer with ModelOpt capabilities.

    This function uses monkey patching to inject quantization and distillation functionality
    into LLaMA-Factory's training pipeline without modifying the original code.

    Args:
        quant_args: SimpleNamespace containing quantization parameters
        distill_args: SimpleNamespace containing distillation parameters

    Returns:
        function: Patch function that modifies the trainer class
    """

    def patch_module(trainer_class):
        # Choose appropriate ModelOpt trainer based on whether distillation is enabled
        modelopt_trainer_cls: type = (
            QATTrainer if not distill_args or not distill_args.distill else QADTrainer
        )

        custom_seq2seq_trainer: type = getattr(trainer_class, "CustomSeq2SeqTrainer", None)
        assert custom_seq2seq_trainer is not None, "CustomSeq2SeqTrainer not found"

        # Create custom trainer that inherits from both LLaMA-Factory and ModelOpt trainers
        class CustomTrainer(modelopt_trainer_cls, custom_seq2seq_trainer):
            def __init__(self, *args, **kwargs):
                # Initialize parent classes
                modelopt_trainer_args = {"quant_args": quant_args}
                if distill_args and distill_args.distill:
                    teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
                        distill_args.teacher_model,
                    )
                    distill_config = {
                        "teacher_model": teacher_model,
                        "criterion": LMLogitsLoss(),
                    }
                    modelopt_trainer_args["distill_config"] = distill_config
                super().__init__(*args, **modelopt_trainer_args, **kwargs)

        # Replace the trainer class in the module
        setattr(trainer_class, "CustomSeq2SeqTrainer", CustomTrainer)

    return patch_module


def create_patch_modelcard_and_push(module):
    original_fn = module.create_modelcard_and_push

    def create_modelcard_and_push(
        trainer: Trainer,
        *args,
        **kwargs,
    ) -> None:
        original_fn(trainer, *args, **kwargs)

        # save the model in the output directory
        trainer.save_state()
        trainer.save_model(output_dir=trainer.args.output_dir)

    module.create_modelcard_and_push = create_modelcard_and_push


if __name__ == "__main__":
    # Parse configuration and extract ModelOpt arguments
    config, quant_args, distill_args = parse_args()

    # Create patch function with pre-configured arguments
    patch_trainer_module = create_patch_module(quant_args, distill_args)

    # Register post-import hook to patch LLaMA-Factory's trainer module
    # This allows us to modify the trainer without changing the original code
    if distill_args is not None:
        assert quant_args is not None, "Quantization arguments must be provided for QAD"

    register_post_import_hook(patch_load_module, "llamafactory.model.loader")
    register_post_import_hook(patch_load_module, "llamafactory.model")

    if quant_args is not None:
        register_post_import_hook(patch_trainer_module, "llamafactory.train.sft.trainer")
        register_post_import_hook(
            create_patch_modelcard_and_push, "llamafactory.train.trainer_utils"
        )

    # Import and run LLaMA-Factory's training pipeline
    from llamafactory.train.tuner import run_exp as llama_factory_run_exp

    llama_factory_run_exp(config)

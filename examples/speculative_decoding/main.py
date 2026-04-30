# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/3783d18/train.py

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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

import argparse
import os
from dataclasses import dataclass

import torch
import transformers
from eagle_utils import (
    EagleTrainerWithAccLog,
    EagleTrainingPlot,
    LoRAWarmupCallback,
    make_speculative_data_module,
    patch_ring_attention_for_ttt,
)
from rich.pretty import pprint
from transformers.trainer_utils import get_last_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.recipe import load_recipe
from modelopt.recipe.config import ModelOptDFlashRecipe, ModelOptEagleRecipe, ModelOptMedusaRecipe
from modelopt.torch.speculative.config import DFlashConfig
from modelopt.torch.speculative.utils import load_vlm_or_llm, patch_transformers5_params_loading
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.distributed import is_master

torch.manual_seed(0)
mto.enable_huggingface_checkpointing()


@dataclass
class HfTrainingArguments(transformers.TrainingArguments):
    """HF-compatible TrainingArguments with our speculative-decoding extensions.

    Used only to build the ``transformers.Trainer``-compatible object at runtime via
    ``HfTrainingArguments(**recipe.training.model_dump())``. Field set MUST stay in sync
    with :class:`modelopt.torch.speculative.plugins.hf_training_args.TrainingArguments`.
    """

    training_seq_len: int = 2048
    estimate_ar: bool = False
    ar_validate_steps: int = 1000
    answer_only_loss: bool = False
    cp_size: int = 1
    dp_shard_size: int | None = None


def _parse_cli() -> tuple[str, list[str]]:
    """Parse --config (required) from argv; return remaining args as dotlist overrides.

    Extra positional args use dotlist syntax, e.g.
    ``model.model_name_or_path=meta-llama/Llama-3.2-1B training.output_dir=ckpts/test``.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--config",
        required=True,
        help=(
            "Path to a modelopt speculative-decoding recipe YAML "
            "(speculative_eagle / speculative_dflash / speculative_medusa)."
        ),
    )
    args, overrides = p.parse_known_args()
    return args.config, overrides


def train():
    config_path, overrides = _parse_cli()
    recipe = load_recipe(config_path, overrides=overrides)

    # Pydantic-typed sections flow straight through as *_args; only TrainingArguments is
    # reconstructed as an HF dataclass so it can be handed to transformers.Trainer.
    training_args = HfTrainingArguments(**recipe.training.model_dump())

    if not recipe.data.data_path and not recipe.data.offline_data_path:
        raise ValueError(
            "Either data.data_path or data.offline_data_path must be set in the config."
        )
    if training_args.cp_size > 1:
        patch_ring_attention_for_ttt()
        # Specific patch to accelerate 1.12.0. Removable after move to 1.13.0
        training_args.parallelism_config.sp_backend = None
    if is_master():
        pprint(recipe)

    # Detect checkpoint to resume from
    last_checkpoint = (
        get_last_checkpoint(training_args.output_dir)
        if os.path.isdir(training_args.output_dir)
        else None
    )
    if last_checkpoint:
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint

    use_offline_training = recipe.data.offline_data_path is not None

    if checkpoint:
        with patch_transformers5_params_loading():
            model = load_vlm_or_llm(
                checkpoint, dtype="auto", trust_remote_code=recipe.model.trust_remote_code
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=recipe.model.trust_remote_code
        )
    else:
        # To avoid OOM for large models, we load and convert model on CPU first.
        # Model will be moved to GPU during HF trainer.init().
        model = load_vlm_or_llm(
            recipe.model.model_name_or_path,
            use_fake_base=recipe.model.use_fake_base_for_offline,
            use_offline_training=use_offline_training,
            dtype="auto",
            device_map="cpu",
            trust_remote_code=recipe.model.trust_remote_code,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            recipe.model.model_name_or_path,
            model_max_length=training_args.training_seq_len,
            trust_remote_code=recipe.model.trust_remote_code,
        )
        if isinstance(recipe, ModelOptMedusaRecipe):
            medusa_cfg: dict = recipe.medusa.model_dump()
            mtsp.convert(model, [("medusa", medusa_cfg)])
        elif isinstance(recipe, ModelOptEagleRecipe):
            eagle_cfg: dict = recipe.eagle.model_dump()
            mtsp.convert(model, [("eagle", eagle_cfg)])
            # Load draft vocab cache
            mtsp.plugins.HFEagleModel.load_draft_vocab_cache(model, recipe.data.draft_vocab_cache)
        elif isinstance(recipe, ModelOptDFlashRecipe):
            # Re-validate with tokenizer to resolve dflash_mask_token_id and enforce its presence.
            dflash_cfg: dict = DFlashConfig.model_validate(
                recipe.dflash.model_dump(), context={"tokenizer": tokenizer}
            ).model_dump()
            mtsp.convert(model, [("dflash", dflash_cfg)])
        else:
            raise ValueError(f"Unsupported speculative recipe type: {type(recipe).__name__}")

    # Move any remaining CPU buffers to CUDA so DDP (NCCL-only) can broadcast
    # them.  We iterate named_buffers and reassign via the owning module to
    # keep the module tree consistent.  Parameters are left on CPU — the HF
    # Trainer will move them during init.
    if torch.cuda.is_available():
        _target_dev = torch.device("cuda", 0)
        for name, buf in list(model.named_buffers()):
            if buf.device.type == "cpu":
                parts = name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                setattr(mod, parts[-1], buf.to(_target_dev))

    print_rank_0("Loading dataset...")
    is_dflash = isinstance(recipe, ModelOptDFlashRecipe)
    if isinstance(recipe, (ModelOptEagleRecipe, ModelOptDFlashRecipe)):
        data_module = make_speculative_data_module(
            tokenizer,
            recipe.data,
            train_len=training_args.training_seq_len,
            answer_only_loss=training_args.answer_only_loss,
            shift_labels=not is_dflash,
        )

    callbacks = [EagleTrainingPlot(training_args.ar_validate_steps, training_args.estimate_ar)]
    if (
        isinstance(recipe, ModelOptEagleRecipe)
        and eagle_cfg.get("eagle_base_lora")
        and eagle_cfg.get("eagle_base_lora_warmup_steps", 0) > 0
    ):
        callbacks.append(LoRAWarmupCallback(eagle_cfg["eagle_base_lora_warmup_steps"]))

    trainer = EagleTrainerWithAccLog(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    # Manually enable this to return loss in eval
    trainer.can_return_loss = True
    # Make sure label_smoother is None
    assert trainer.label_smoother is None, (
        "label_smoother is not supported in speculative decoding!"
    )

    print_rank_0("Start training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()

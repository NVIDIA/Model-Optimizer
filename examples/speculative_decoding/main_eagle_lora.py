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

"""Co-training of Eagle3 draft head and LoRA adapters.

The idea: by co-training a LoRA adapter alongside the Eagle3 draft head, the base
model's next-token distribution can become "easier" for the drafter to predict,
improving acceptance rate (AR) without significantly degrading model quality.

Training flow:
  1. Load base model (VLM or LLM).
  2. Inject LoRA adapters into the base model's attention layers (via HuggingFace peft).
  3. Convert model to Eagle3 with eagle_freeze_base_model=False so gradients
     flow through the LoRA-adapted base model.
  4. Freeze all base model parameters *except* LoRA weights.
  5. Train jointly: the combined loss = base_model_loss + eagle_loss ensures
     LoRA maintains language modeling quality while Eagle3 learns to draft.

At deployment time, merge the LoRA adapter into the base model weights and serve
the eagle head via vLLM speculative decoding.
"""

import json
import os
from dataclasses import dataclass, field

import torch
import transformers
from accelerate import ParallelismConfig
from eagle_utils import (
    EagleTrainerWithAccLog,
    EagleTrainingPlot,
    make_eagle_supervised_data_module,
    patch_ring_attention_for_ttt,
)
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
from transformers.trainer_utils import get_last_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.utils import load_vlm_or_llm_with_kwargs
from modelopt.torch.utils import print_rank_0

torch.manual_seed(0)
mto.enable_huggingface_checkpointing()


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    offline_data_path: str = field(default=None, metadata={"help": "Not used in co-training."})
    lazy_preprocess: bool = True
    draft_vocab_cache: str | None = field(
        default=None,
        metadata={"help": "Path to d2t.pt cache file."},
    )
    vlm_img_dir: str = field(default=None, metadata={"help": "Path to the VLM image directory."})
    vlm_processor: str = field(default=None, metadata={"help": "Path to the VLM processor."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    training_seq_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length."},
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)
    estimate_ar: bool = field(
        default=False, metadata={"help": "Whether to estimate AR during training for logging."}
    )
    ar_validate_steps: int = field(default=1000, metadata={"help": "Steps between AR validation."})
    disable_tqdm: bool = field(default=False, metadata={"help": "Disable tqdm progress bar."})
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Set to False to keep extra args for VLM."}
    )
    cp_size: int = field(default=1, metadata={"help": "Context parallelism size."})
    dp_shard_size: int = field(default=1, metadata={"help": "Data parallelism shard size."})


@dataclass
class EagleArguments:
    eagle_config: str = field(default=None, metadata={"help": "Path to eagle_config.json"})
    eagle_decoder_type: str = field(
        default="llama",
        metadata={"help": "The class of eagle decoder to use. Available options: llama, kimik2"},
    )
    mix_hidden_states: bool = field(
        default=False,
        metadata={"help": "Whether to mix hidden states from previous TTT step."},
    )


@dataclass
class LoRAArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling factor."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate."})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated list of module names to apply LoRA to."},
    )
    lora_adapter_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a pre-trained LoRA adapter to load. "
                "If not provided, LoRA adapters are initialized from scratch."
            )
        },
    )


def _inject_lora(model, lora_config: LoraConfig):
    """Inject LoRA adapters into the model.

    Must be called BEFORE eagle conversion so that LoRA is only applied to
    base model attention layers, not the eagle module.
    """
    inject_adapter_in_model(lora_config, model, adapter_name="default")

    lora_param_count = sum(
        p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    )
    total_param_count = sum(p.numel() for p in model.parameters())
    print_rank_0(
        f"LoRA injected: {lora_param_count:,} trainable LoRA params "
        f"/ {total_param_count:,} total params "
        f"({100 * lora_param_count / total_param_count:.2f}%)"
    )


def _freeze_non_trainable_params(model):
    """Freeze all params except LoRA and eagle_module."""
    trainable_count = 0
    frozen_count = 0
    for name, param in model.named_parameters():
        if "lora_" in name or "eagle_module" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print_rank_0(f"Trainable: {trainable_count:,} params | Frozen: {frozen_count:,} params")


def _load_lora_weights(model, adapter_path: str):
    """Load pre-trained LoRA weights into an already-injected LoRA model."""
    weights_file = os.path.join(adapter_path, "lora_weights.pt")
    if not os.path.isfile(weights_file):
        raise FileNotFoundError(f"LoRA weights not found at {weights_file}")

    lora_state_dict = torch.load(weights_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
    loaded = len(lora_state_dict) - len(unexpected)
    print_rank_0(f"Loaded {loaded} LoRA weight tensors from {adapter_path}")
    if unexpected:
        print_rank_0(f"Warning: {len(unexpected)} unexpected keys in LoRA checkpoint")


def _save_lora_weights(model, output_dir: str):
    """Save LoRA weights separately for easy extraction at export time."""
    lora_state_dict = get_peft_model_state_dict(model)
    lora_dir = os.path.join(output_dir, "lora_adapter")
    os.makedirs(lora_dir, exist_ok=True)
    torch.save(lora_state_dict, os.path.join(lora_dir, "lora_weights.pt"))
    print_rank_0(f"Saved {len(lora_state_dict)} LoRA weight tensors to {lora_dir}")


class EagleLoRATrainer(EagleTrainerWithAccLog):
    """Trainer that saves LoRA weights alongside eagle checkpoints."""

    def save_model(self, output_dir=None, _internal_call=False):
        """Save both the full model and LoRA weights separately."""
        super().save_model(output_dir, _internal_call)
        if output_dir is None:
            output_dir = self.args.output_dir
        _save_lora_weights(self.model, output_dir)


def train():
    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            EagleArguments,
            LoRAArguments,
        )
    )
    model_args, data_args, training_args, eagle_args, lora_args = (
        parser.parse_args_into_dataclasses()
    )
    training_args.parallelism_config = ParallelismConfig(
        cp_size=training_args.cp_size, dp_shard_size=training_args.dp_shard_size
    )
    if training_args.cp_size > 1:
        patch_ring_attention_for_ttt()
        training_args.parallelism_config.sp_backend = None
    print_rank_0(f"arguments: {model_args}, {training_args}, {eagle_args}, {lora_args}")

    # Detect checkpoint to resume from
    last_checkpoint = (
        get_last_checkpoint(training_args.output_dir)
        if os.path.isdir(training_args.output_dir)
        else None
    )
    if last_checkpoint:
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=[m.strip() for m in lora_args.lora_target_modules.split(",")],
        lora_dropout=lora_args.lora_dropout,
        bias="none",
    )

    # Always build the model the same way (fresh from the original base model),
    # regardless of whether we are starting fresh or resuming from a checkpoint.
    # The FSDP/Trainer checkpoint loader will overwrite weights when resuming.
    # This avoids key-name mismatches: the checkpoint stores LoRA-wrapped names
    # (e.g. q_proj.base_layer.weight) that a plain model cannot load.
    model_config, model = load_vlm_or_llm_with_kwargs(
        model_args.model_name_or_path,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.training_seq_len,
        trust_remote_code=True,
    )

    # Step 1: Inject LoRA into the base model BEFORE eagle conversion.
    # This ensures LoRA only targets the base model's attention layers,
    # not the eagle module (which doesn't exist yet).
    print_rank_0("Injecting LoRA adapters into base model...")
    _inject_lora(model, lora_config)

    # Step 2: Convert to Eagle3 with eagle_freeze_base_model=False.
    # This means _base_model_forward runs WITHOUT torch.no_grad(),
    # allowing gradients to flow through the LoRA-adapted base model.
    custom_config = json.load(open(eagle_args.eagle_config)) if eagle_args.eagle_config else {}

    eagle_convert_config = {
        "eagle_decoder_type": eagle_args.eagle_decoder_type,
        "eagle_freeze_base_model": False,
        "eagle_mix_hidden_states": eagle_args.mix_hidden_states,
        "eagle_architecture_config": custom_config,
    }

    print_rank_0("Converting model to Eagle3...")
    mtsp.convert(model, [("eagle", eagle_convert_config)])

    # Step 3: Freeze everything except LoRA and eagle_module.
    _freeze_non_trainable_params(model)

    # Step 4: Load draft vocab cache if needed
    if model.eagle_config.draft_vocab_size < model.eagle_config.vocab_size:
        if not os.path.isfile(data_args.draft_vocab_cache):
            raise FileNotFoundError(
                f"Draft vocab cache provided but not found: {data_args.draft_vocab_cache}"
            )
        model.eagle_module.d2t = torch.load(data_args.draft_vocab_cache)
        print_rank_0(f"Loaded draft vocab cache from {data_args.draft_vocab_cache}.")

    # Load pre-trained LoRA weights if provided (for warm-starting LoRA)
    if lora_args.lora_adapter_path:
        _load_lora_weights(model, lora_args.lora_adapter_path)

    if checkpoint:
        print_rank_0(
            f"Resuming from {checkpoint}. "
            "Model weights will be loaded by the Trainer from the FSDP/HF checkpoint."
        )

    # Log parameter summary
    eagle_params = sum(p.numel() for n, p in model.named_parameters() if "eagle_module" in n)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0(
        f"Eagle params: {eagle_params:,} | LoRA params: {lora_params:,} | "
        f"Total trainable: {trainable_params:,}"
    )

    print_rank_0("Loading dataset...")
    data_module = make_eagle_supervised_data_module(
        tokenizer, data_args, train_len=training_args.training_seq_len
    )

    trainer = EagleLoRATrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[EagleTrainingPlot(training_args.ar_validate_steps, training_args.estimate_ar)],
        **data_module,
    )

    trainer.can_return_loss = True
    assert trainer.label_smoother is None, (
        "label_smoother is not supported in speculative decoding!"
    )

    print_rank_0("Start co-training Eagle3 + LoRA...")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    print_rank_0(
        f"Training complete. Checkpoints saved to {training_args.output_dir}\n"
        f"LoRA weights saved to {training_args.output_dir}/lora_adapter/"
    )


if __name__ == "__main__":
    train()

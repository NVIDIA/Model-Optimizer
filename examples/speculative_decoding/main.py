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

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import transformers
from accelerate import ParallelismConfig
from eagle_utils import (
    EagleTrainerWithAccLog,
    EagleTrainingPlot,
    make_eagle_supervised_data_module,
    patch_ring_attention_for_ttt,
)
from transformers.trainer_utils import get_last_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.utils import (
    load_vlm_or_llm_with_kwargs,
    patch_transformers5_params_loading,
)
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
    offline_data_path: str = field(
        default=None,
        metadata={
            "help": """Path to the offline training data. Providing this flag sets
                  `eagle_offline` in the EagleConfig and enables offline training.
                  The directory should contain many `.pt` files, each containing a pre-processed
                  data sample. `data_path` should still point to the original conversations file.
                  """
        },
    )
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
        metadata={
            "help": (
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            )
        },
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)
    mode: Literal["eagle3", "medusa"] = "eagle3"
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
class MedusaArguments:
    medusa_num_heads: int | None = field(default=1)
    medusa_num_layers: int | None = field(default=1)


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


def train():
    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            MedusaArguments,
            EagleArguments,
        )
    )
    model_args, data_args, training_args, medusa_args, eagle_args = (
        parser.parse_args_into_dataclasses()
    )
    training_args.parallelism_config = ParallelismConfig(
        cp_size=training_args.cp_size, dp_shard_size=training_args.dp_shard_size
    )
    if training_args.cp_size > 1:
        patch_ring_attention_for_ttt()
        # Specific patch to accelerate 1.12.0. Removable after move to 1.13.0
        training_args.parallelism_config.sp_backend = None
    print_rank_0(f"arguments: {model_args}, {training_args}, {medusa_args}, {eagle_args}")

    # Detect checkpoint to resume from
    last_checkpoint = (
        get_last_checkpoint(training_args.output_dir)
        if os.path.isdir(training_args.output_dir)
        else None
    )
    if last_checkpoint:
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint

    use_offline_training = data_args.offline_data_path is not None

    if checkpoint:
        with patch_transformers5_params_loading():
            _, model = load_vlm_or_llm_with_kwargs(
                checkpoint, torch_dtype="auto", trust_remote_code=True
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    else:
        # To avoid OOM for large models, we load and convert model on CPU first.
        # Model will be moved to GPU during HF trainer.init().
        offline_kwargs = {"num_hidden_layers": 0} if use_offline_training else {}
        model_config, model = load_vlm_or_llm_with_kwargs(
            model_args.model_name_or_path,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
            **offline_kwargs,
        )
        if use_offline_training:
            # When doing offline training, we need to set num_hidden_layers
            # since we override it when loading the model for space savings
            model.config.num_orig_hidden_layers = model_config.num_hidden_layers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.training_seq_len,
            trust_remote_code=True,
        )
        if training_args.mode == "medusa":
            config = {
                "medusa_num_heads": medusa_args.medusa_num_heads,
                "medusa_num_layers": medusa_args.medusa_num_layers,
            }
            mtsp.convert(model, [("medusa", config)])
        elif training_args.mode == "eagle3":
            custom_config = (
                json.load(open(eagle_args.eagle_config)) if eagle_args.eagle_config else {}
            )

            config = {
                "eagle_decoder_type": eagle_args.eagle_decoder_type,
                "eagle_offline": use_offline_training,
                "eagle_mix_hidden_states": eagle_args.mix_hidden_states,
                "eagle_architecture_config": custom_config,
            }

            mtsp.convert(model, [("eagle", config)])

            # read draft vocab cache
            if model.eagle_config.draft_vocab_size < model.eagle_config.vocab_size:
                if not os.path.isfile(data_args.draft_vocab_cache):
                    raise FileNotFoundError(
                        f"Draft vocab cache provided but not found: {data_args.draft_vocab_cache}"
                    )
                model.eagle_module.d2t = torch.load(data_args.draft_vocab_cache)
                print_rank_0(f"Loaded draft vocab cache from {data_args.draft_vocab_cache}.")
        else:
            raise Exception(f"{training_args.mode} is not supported!")

    print_rank_0("Loading dataset...")
    if training_args.mode == "eagle3":
        data_module = make_eagle_supervised_data_module(
            tokenizer, data_args, train_len=training_args.training_seq_len
        )

    trainer = EagleTrainerWithAccLog(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[EagleTrainingPlot(training_args.ar_validate_steps, training_args.estimate_ar)],
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

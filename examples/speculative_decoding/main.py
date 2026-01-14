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
from eagle_utils import EagleTrainerWithAccLog, EagleTrainingPlot, make_eagle_supervised_data_module
from medusa_utils import make_medusa_supervised_data_module
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import get_last_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
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
    draft_vocab_cache_dir: str = field(
        default="draft_vocab_cache",
        metadata={"help": "Path to the d2t cache directory."},
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
    mode: Literal["eagle1", "eagle3", "medusa"] = "eagle3"
    ar_validate_steps: int = field(default=1000, metadata={"help": "Steps between AR validation."})
    disable_tqdm: bool = field(default=False, metadata={"help": "Disable tqdm progress bar."})
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Set to False to keep extra args for VLM."}
    )


@dataclass
class MedusaArguments:
    medusa_num_heads: int | None = field(default=1)
    medusa_num_layers: int | None = field(default=1)


@dataclass
class EagleArguments:
    eagle_config: str = field(default=None, metadata={"help": "Path to eagle_config.json"})
    draft_mode: str = field(
        default="eagle",
        metadata={"help": "The draft mode for eagle module."},
        choices=["eagle", "pard", "dflash"],
    )
    num_spec_tokens: int = field(
        default=3, metadata={"help": "The number of speculative tokens for eagle module."}
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
    print_rank_0(f"arguments: {model_args}, {training_args}, {medusa_args}, {eagle_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    use_offline_training = data_args.offline_data_path is not None

    if checkpoint:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype="auto", trust_remote_code=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    else:
        # To avoid OOM for large models, we load and convert model on CPU first.
        # Model will be moved to GPU during HF trainer.init().
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
        )
        if use_offline_training:
            # When doing offline training, we need to set num_hidden_layers
            # since we override it when loading the model for space savings
            model_config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True
            )
            model.config.num_orig_hidden_layers = model_config.num_hidden_layers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.training_seq_len,
            trust_remote_code=True,
        )
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{%- for message in messages %}"
                "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
                "{%- endfor %}"
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if training_args.mode == "medusa":
            config = {
                "medusa_num_heads": medusa_args.medusa_num_heads,
                "medusa_num_layers": medusa_args.medusa_num_layers,
            }
            mtsp.convert(model, [("medusa", config)])
        elif training_args.mode in ["eagle1", "eagle3"]:
            from modelopt.torch.speculative.config import EAGLE1_DEFAULT_CFG, EAGLE3_DEFAULT_CFG

            # Load default config
            config = {
                "eagle1": EAGLE1_DEFAULT_CFG,
                "eagle3": EAGLE3_DEFAULT_CFG,
            }[training_args.mode]["config"]

            # overwrite config with custom config
            if use_offline_training:
                config["eagle_offline"] = True

            if eagle_args.eagle_config:
                with open(eagle_args.eagle_config) as f:
                    custom_config = json.load(f)
                config["eagle_architecture_config"].update(custom_config)

            config.update({
                "eagle_draft_mode": eagle_args.draft_mode,
                "eagle_num_spec_tokens": eagle_args.num_spec_tokens,
            })

            # Hidden size and vocab size must match base model
            llm_config = (
                model.config.llm_config if hasattr(model.config, "llm_config") else model.config
            )
            config["eagle_architecture_config"].update(
                {
                    "hidden_size": llm_config.hidden_size,
                    "vocab_size": llm_config.vocab_size,
                    # we also overwrite max_pos_embedding for deployment compatibility
                    "max_position_embeddings": llm_config.max_position_embeddings,
                    "draft_vocab_size": custom_config["draft_vocab_size"]
                    if eagle_args.eagle_config and "draft_vocab_size" in custom_config
                    else llm_config.vocab_size,
                }
            )

            mtsp.convert(model, [("eagle", config)])

            # read draft vocab cache
            if model.eagle_config.draft_vocab_size < model.eagle_config.vocab_size:
                try:
                    model_name = os.path.basename(os.path.normpath(model_args.model_name_or_path))
                    vocab_cache_path = os.path.join(
                        data_args.draft_vocab_cache_dir, model_name, "d2t.pt"
                    )
                    vocab_cache = torch.load(vocab_cache_path)
                    model.eagle_module.d2t = vocab_cache
                    print_rank_0(f"Loaded draft vocab cache from {vocab_cache_path}.")
                except Exception as e:
                    raise e
        else:
            raise Exception(f"{training_args.mode} is not supported!")

    print_rank_0("Loading dataset...")
    if training_args.mode == "medusa":
        data_module = make_medusa_supervised_data_module(tokenizer, data_args)
    elif training_args.mode in ["eagle1", "eagle3"]:
        data_module = make_eagle_supervised_data_module(
            tokenizer, data_args, max_length=training_args.training_seq_len
        )

    lr_kwargs = {"min_lr": 1e-4}
    if isinstance(training_args.lr_scheduler_kwargs, dict):
        training_args.lr_scheduler_kwargs.update(lr_kwargs)
    else:
        training_args.lr_scheduler_kwargs = lr_kwargs

    callbacks = []
    tb_writer = None
    if "tensorboard" in training_args.report_to:
        log_dir = training_args.output_dir
        tb_writer = SummaryWriter(log_dir=log_dir)
        if isinstance(training_args.report_to, list):
            training_args.report_to.remove("tensorboard")
        else:
            training_args.report_to = "none"
        callbacks.append(TensorBoardCallback(tb_writer=tb_writer))
    # callbacks.append(
    #     EagleMoEBalancer(update_interval=1, tb_writer=tb_writer)
    # )
    callbacks.append(
        EagleTrainingPlot(training_args.ar_validate_steps, tb_writer=tb_writer, estimate_ar=True)
    )

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

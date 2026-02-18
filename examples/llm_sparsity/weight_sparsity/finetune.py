# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/3783d18/train.py

# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import transformers
import utils
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.utils.distributed as dist
from modelopt.torch.opt.utils import is_dynamic
from modelopt.torch.utils import print_rank_0

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further"
        " context. Write a response that appropriately completes the request.\n\n###"
        " Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default="facebook/opt-125m")
    use_flash_attn: bool | None = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Set trust_remote_code for Huggingface models and tokenizers."},
    )


@dataclass
class DataArguments:
    train_datapath: str = field(default=None, metadata={"help": "Path to the training data."})
    val_datapath: str = field(default=None, metadata={"help": "Path to the eval data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": (
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            )
        },
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)


@dataclass
class ModelOptArguments:
    # modelopt quantization arguments
    quant_cfg: str | None = field(
        default=None,
        metadata={
            "help": (
                "Specify the quantization format for PTQ/QAT. if specified, PTQ/QAT will be enabled"
                " with the specified quantization format. choices=['INT8_DEFAULT_CFG',"
                " 'INT8_SMOOTHQUANT_CFG', 'FP8_DEFAULT_CFG', 'INT4_AWQ_CFG', 'W4A8_AWQ_BETA_CFG']"
            )
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
    # modelopt sparsity arguments
    sparse_fmt: str | None = field(
        default=None,
        metadata={
            "help": (
                "Specify the sparsity format. if specified, sparsity will be enabled with the"
                " specified sparsity format. choices=['dense', 'sparsegpt', 'sparse_magnitude']"
            )
        },
    )
    modelopt_restore_path: str | None = field(
        default=None,
        metadata={"help": "Path to the modelopt state dict to restore from."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in tqdm(strings, desc="Tokenizing")
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    for i in range(len(input_ids_lens)):
        if input_ids_lens[i] > 2048:
            print_rank_0("Input exceeds model length 2048")
            print_rank_0(input_ids_lens[i])
            print_rank_0(strings[i])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "input_ids_lens": input_ids_lens,
        "labels_lens": labels_lens,
    }


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels}


def get_model_state_dict(trainer: transformers.Trainer):
    """Collects the state dict."""
    state_dict = trainer.model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    del state_dict
    return cpu_state_dict


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        training_args: TrainingArguments,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        split: str,
    ):
        super().__init__()

        with training_args.main_process_first():
            print_rank_0("Loading data...")
            list_data_dict = utils.jload(data_path)

            print_rank_0("Formatting inputs...")
            prompt_input = PROMPT_DICT["prompt_input"]
            sources = [prompt_input.format_map(example) for example in list_data_dict]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

            print_rank_0("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }


def make_supervised_data_module(
    training_args: TrainingArguments,
    train_datapath: str,
    val_datapath: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(training_args, train_datapath, tokenizer, "train")
    val_dataset = SupervisedDataset(training_args, val_datapath, tokenizer, "val")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
    }


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ModelOptArguments)
    )
    model_args, data_args, training_args, modelopt_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(modelopt_args)
    )

    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None and args.resume_from_checkpoint is None:
            print_rank_0(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
    ).to(torch.device("cuda"))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN},
            tokenizer=tokenizer,
            model=model,
        )

    data_module = make_supervised_data_module(
        training_args, args.train_datapath, args.val_datapath, tokenizer=tokenizer
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            print_rank_0(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        for name, mod in model.named_modules():
            # freeze the embedding layer
            if isinstance(mod, torch.nn.Embedding):
                mod.weight.requires_grad = False

        trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )

        # modelopt sparsity
        if args.modelopt_restore_path:
            if not os.path.isfile(args.modelopt_restore_path):
                raise FileNotFoundError(
                    f"Sparsity state file {args.modelopt_restore_path} not found."
                )
            if is_dynamic(model):
                raise ValueError("Cannot restore modelopt state dict for a dynamic model.")

            print_rank_0(f"Loading sparsity state from {args.modelopt_restore_path}")
            mto.restore(trainer.model, args.modelopt_restore_path)
            # retrieve the modelopt state after restoring the modelopt state dict
            # this is necessary to ensure that `_fsdp_wrapped_module` is not exposed to the modelopt state dict
            modelopt_state = mto.modelopt_state(model)

        trainer.train(resume_from_checkpoint=checkpoint)
        modelopt_state_path = os.path.join(args.output_dir, "finetuned_modelopt_state.pth")
        model_state_dict = get_model_state_dict(trainer)
        print_rank_0(f"Saving modelopt state to {modelopt_state_path}")
        if dist.is_master():
            torch.save(
                {"modelopt_state": modelopt_state, "model_state_dict": model_state_dict},
                modelopt_state_path,
            )


if __name__ == "__main__":
    train()

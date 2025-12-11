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
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import evaluate
import nltk
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from packaging.version import Version
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.opt as mto

DEFAULT_PAD_TOKEN = "[PAD]"

# Prompt for GPTJ model input
G_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further"
    " context. Write a response that appropriately completes the request.\n\n###"
    " Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
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


def prepare_tokenizer(accelerator, checkpoint_path, model_max_length, padding_side="left"):
    """
    Prepare the tokenizer for the cnn dailymail
    """
    accelerator.print(f"Initializing tokenizer from {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=False,
    )
    return tokenizer


def preprocess_cnndailymail(accelerator, data_path, calib=False):
    # Load from CNN dailymail
    with open(data_path) as fh:
        list_data_dict = json.load(fh)

    sources = [G_PROMPT_INPUT.format_map(example) for example in list_data_dict]
    targets = [f"{example['output']}" for example in list_data_dict]

    accelerator.print(f"Loaded {len(sources)} samples from {data_path}")

    return sources, targets


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


class CNNDailymailDataset(Dataset):
    def __init__(self, accelerator, data_path: str, tokenizer):
        super().__init__()

        self.sources, self.labels = preprocess_cnndailymail(accelerator, data_path, calib=False)
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        return {"src_idx": self.sources[index], "label_idx": self.labels[index]}

    def __len__(self):
        return len(self.sources)


@dataclass
class DataCollator:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        sources, labels = tuple(
            [instance[key] for instance in instances] for key in ("src_idx", "label_idx")
        )

        batch_encoded = self.tokenizer.batch_encode_plus(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_max_length,
        )

        return dict(labels=labels, **batch_encoded)


def get_dataset(accelerator, data_path, tokenizer):
    dataset = CNNDailymailDataset(accelerator, data_path, tokenizer)
    return dataset


def get_dataloader(accelerator, dataset, tokenizer, model_max_length, batch_size, shuffle):
    """Make dataset and collator for supervised fine-tuning."""
    with accelerator.main_process_first():
        data_collator = DataCollator(tokenizer=tokenizer, model_max_length=model_max_length)
        dataloader = DataLoader(
            dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle
        )
        return dataloader


def calculate_rouge_score(accelerator, model, dataloader, tokenizer, beam_size):
    """Run inference on the dataset."""
    gen_kwargs = {
        "early_stopping": True,
        "max_new_tokens": 128,
        "min_new_tokens": 30,
        "num_beams": beam_size,
    }

    metric = evaluate.load("rouge")
    model, dataloader = accelerator.prepare(model, dataloader)
    accelerator.wait_for_everyone()

    for batch in tqdm(dataloader):
        with torch.inference_mode():
            unwrapped_model = accelerator.unwrap_model(model)
            input_batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            input_batch = {k: v.to(accelerator.device) for k, v in input_batch.items()}
            input_lens = [x.shape[0] for x in input_batch["input_ids"]]
            labels = batch["labels"]

            tokens_generated = unwrapped_model.generate(
                **input_batch, **gen_kwargs, pad_token_id=tokenizer.eos_token_id
            )
            tokens_generated = accelerator.pad_across_processes(
                tokens_generated, dim=1, pad_index=tokenizer.eos_token_id
            )
            tokens_generated = accelerator.gather_for_metrics(tokens_generated).cpu().tolist()

            # Truncate the input portion of the outputs
            output_batch_response_only = [
                data[source_len:] for data, source_len in zip(tokens_generated, input_lens)
            ]

            preds = tokenizer.batch_decode(output_batch_response_only, skip_special_tokens=True)

            preds, labels = postprocess_text(preds, labels)
            metric.add_batch(predictions=preds, references=labels)

    accelerator.wait_for_everyone()
    result = metric.compute(use_stemmer=True, use_aggregator=False)
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    accelerator.print(f"ROUGE scores: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        help="Specify where the PyTorch checkpoint path is",
        default="build/models/GPTJ-6B/04252023-GPTJ6B-ckpt",
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to the validation dataset."
    )
    parser.add_argument(
        "--batch_size",
        help="batch size. 80GB can run a maximum of BS=8 for FP32 greedy",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--beam_size", help="The beam width of the decoding op.", type=int, default=1
    )
    parser.add_argument(
        "--model_max_length",
        help="Maximum sequence length. Sequences will be right padded (and possibly truncated).",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--modelopt_restore_path",
        help="Path to the pruned modelopt checkpoint",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise RuntimeError(
            f"Cannot access {args.model_dir}. Please download the model or mount the scratch path."
        )

    accelerator = Accelerator()
    with accelerator.main_process_first():
        if Version(nltk.__version__) < Version("3.8.2"):
            nltk.download("punkt")
        else:
            nltk.download("punkt_tab")

    tokenizer = prepare_tokenizer(accelerator, args.model_dir, args.model_max_length)
    dataset = get_dataset(accelerator, args.data_path, tokenizer)
    dataloader = get_dataloader(
        accelerator, dataset, tokenizer, args.model_max_length, args.batch_size, shuffle=False
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float16).to(
        accelerator.device
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN},
            tokenizer=tokenizer,
            model=model,
        )

    if args.modelopt_restore_path:
        print(f"Restoring pruned model from {args.modelopt_restore_path}")
        model = mto.restore(model, args.modelopt_restore_path)
    model.eval()

    calculate_rouge_score(accelerator, model, dataloader, tokenizer, args.beam_size)


if __name__ == "__main__":
    main()

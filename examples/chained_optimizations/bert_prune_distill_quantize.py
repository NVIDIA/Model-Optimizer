# NOTE: This is adapted from run_qa_no_trainer.py and utils_qa.py from
# https://github.com/huggingface/transformers/blob/c52b515e/examples/pytorch/question-answering
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""
Example showcasing how to do end-to-end optimization of a BERT model on SQuAD using Model Optimizer.
This includes GradNAS pruning, INT8 quantization, fine-tuning / QAT with distillation, and ONNX export.
"""

import argparse
import collections
import json
import logging
import math
import os
import random
from typing import Any

import datasets
import evaluate
import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

# Model Optimizer: imports
import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.prune as mtp
import modelopt.torch.quantization as mtq
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata

# Enable automatic save/load of modelopt_state with huggingface checkpointing
mto.enable_huggingface_checkpointing()

logger = get_logger(__name__)

SEED = 123


def parse_args(input_args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Question Answering task"
    )

    # Training arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-large-uncased-whole-word-masking-finetuned-squad",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training / fine-tuning."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=2.0,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="The number of processes to use for preprocessing the dataset.",
    )

    # Logging and checkpointing arguments
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        default=None,
        help="Path to save the finetuned (pruned or quantized) model for restoring later with `.from_pretrained()`.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help=(
            "Whether the various states should be saved at the end of every n steps, or 'epoch' for"
            " each epoch."
        ),
    )
    parser.add_argument(
        "--resume_from_last_ckpt",
        action="store_true",
        help="If the training should continue from the latest checkpoint in model_name_or_path.",
    )
    parser.add_argument(
        "--onnx_export_path", type=str, default=None, help="Path to export the ONNX model to."
    )

    # Misc arguments for Bert (should not be modified in most cases)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this"
            " will be truncated, and shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help=(
            "When splitting up a long document into chunks how much stride to take between chunks."
        ),
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the"
            " start and end predictions are not conditioned on one another."
        ),
    )

    # Debugging arguments
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training.",
    )

    # Model Optimizer: pruning arguments
    parser.add_argument(
        "--do_modelopt_prune",
        action="store_true",
        help="Whether or not to use Model Optimizer pruning.",
    )
    parser.add_argument(
        "--modelopt_prune_flops_percent",
        type=float,
        default=None,
        help="The percentage (between 0 and 100) of FLOPs to retain in the pruned model.",
    )
    parser.add_argument(
        "--pruned_model_path",
        type=str,
        default=None,
        help="Path to save the pruned model for further finetuning.",
    )

    # Model Optimizer: quantization arguments
    parser.add_argument(
        "--modelopt_quantize_cfg",
        help="Model Optimizer quantization config.",
        choices=mtq.config.choices,
    )

    # Model Optimizer: Distillation arguments
    parser.add_argument(
        "--do_modelopt_distill",
        action="store_true",
        help="Whether or not to use distillation. A teacher model must be specified.",
    )
    parser.add_argument(
        "--temperature", type=float, default=2.0, help="The temperature to use when distilling."
    )
    parser.add_argument(
        "--ptq_model_path",
        type=str,
        default=None,
        help="Path to save the PTQ quantized model for further QAT.",
    )

    args = parser.parse_args(input_args)

    # Sanity checks
    if args.do_train and not args.finetuned_model_path:
        raise ValueError("`finetuned_model_path` required when `do_train` is passed.")
    if args.do_modelopt_prune and not (
        args.modelopt_prune_flops_percent and args.pruned_model_path
    ):
        raise ValueError(
            "`modelopt_prune_flops_percent` and `pruned_model_path` required when `do_modelopt_prune` is passed."
        )
    if args.modelopt_quantize_cfg and not args.ptq_model_path:
        raise ValueError("`ptq_model_path` required when `modelopt_quantize_cfg` is passed.")

    return args


def get_datasets_and_dataloaders(args, tokenizer: PreTrainedTokenizer, accelerator: Accelerator):
    """Get the examples, dataset, dataloader, answer_column_name

    You can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    (the dataset will be downloaded automatically from the datasets Hub).

    For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    'text' is found. You can easily tweak this behavior (see below).
    """

    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    examples, dataset, dataloader = {}, {}, {}

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # Downloading and loading a dataset from the hub.
    raw_datasets = datasets.load_dataset("squad")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Preprocessing the datasets.
    # Preprocessing is slightly different for training and evaluation.

    column_names = raw_datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length"
            f" for the model ({tokenizer.model_max_length}). Using"
            f" max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    examples["train"] = raw_datasets["train"]
    if args.max_train_samples is not None:
        # We will select sample from whole data if argument is specified
        examples["train"] = examples["train"].select(range(args.max_train_samples))

    # Create train feature from dataset
    with accelerator.main_process_first():
        dataset["train"] = examples["train"].map(
            prepare_train_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )
        # if args.max_train_samples is not None:
        #     # Number of samples might increase during Feature Creation, We select only specified max samples
        #     dataset["train"] = dataset["train"].select(range(args.max_train_samples))

    examples["eval"] = raw_datasets["validation"]
    if args.max_eval_samples is not None:
        # We will select sample from whole data
        examples["eval"] = examples["eval"].select(range(args.max_eval_samples))
    # Validation Feature Creation
    with accelerator.main_process_first():
        dataset["eval"] = examples["eval"].map(
            prepare_validation_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on validation dataset",
        )
        # if args.max_eval_samples is not None:
        #     # During Feature creation dataset samples might increase, we will select required samples again
        #     dataset["eval"] = dataset["eval"].select(range(args.max_eval_samples))

    # Log a random sample from the training set:
    for index in random.sample(range(len(dataset["train"])), 1):
        logger.info(f"Sample {index} of the training set: {dataset['train'][index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done to max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    dataloader["train"] = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )

    dataloader["eval"] = DataLoader(
        dataset["eval"].remove_columns(["example_id", "offset_mapping"]),
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    return examples, dataset, dataloader, answer_column_name


def evaluate_model(
    args,
    model: nn.Module,
    accelerator: Accelerator,
    eval_examples: Any,
    eval_dataset: Any,
    eval_dataloader: DataLoader,
    answer_column_name: str,
    prefix: str = "Eval",
):
    def create_and_fill_np_array(start_or_end_logits, max_len):
        """Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits: This is the output predictions of the model.
                We can only enter either start or end logits.
            max_len: The maximum length of the output tensor. (See the model.eval() part for more details)
        """
        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(eval_dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array we will populate it with the outputs using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step
            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(eval_dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(eval_dataset) - step]

            step += batch_size

        return logits_concat

    def postprocess_qa_predictions(
        examples,
        features,
        predictions: tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: str | None = None,
        prefix: str | None = None,
    ) -> EvalPrediction:
        """Post-processes the predictions of a question-answering model to convert them to answers
        that are substrings of  the original contexts. This is the base postprocessing functions for
        models that only return start and end logits.

        Args:
            examples: The non-preprocessed dataset.
            features: The processed dataset.
            predictions: The predictions of the model: two arrays containing the start logits and the end logits
                respectively. Its first dimension must match the number of elements of `features`.
            version_2_with_negative: Whether or not the underlying dataset contains examples with no answers.
            n_best_size: The total number of n-best predictions to generate when looking for an answer.
            max_answer_length: The maximum length of an answer that can be generated. This is needed
                because the start and end predictions are not conditioned on one another.
            null_score_diff_threshold: The threshold used to select the null answer: if the best answer
                has a score that is less than the score of the null answer minus this threshold, the
                null answer is selected for this example (note that the score of the null answer for
                an example giving several features is the minimum of the scores for the null answer on
                each feature: all features must be aligned on the fact they `want` to predict a null answer).
                Only useful when `version_2_with_negative` is `True`.
            output_dir: If provided, the dictionaries of predictions, n_best predictions (with their scores and logits)
                and, if `version_2_with_negative=True`, the dictionary of the scores differences between best and null
                answers, are saved in `output_dir`.
            prefix: If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        """
        if len(predictions) != 2:
            raise ValueError(
                "`predictions` should be a tuple with two elements (start_logits, end_logits)."
            )
        all_start_logits, all_end_logits = predictions

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        logger.debug(
            f"Post-processing {len(examples)} example predictions split into"
            f" {len(features)} features."
        )

        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum
                # context available in the current feature.
                token_is_max_context = features[feature_index].get("token_is_max_context", None)

                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0]
                if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or
                        # correspond to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if token_is_max_context is not None and not token_is_max_context.get(
                            str(start_index), False
                        ):
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
            if version_2_with_negative and min_null_prediction is not None:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            n_best_preds = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[
                :n_best_size
            ]

            # Add back the minimum null prediction if it was removed because of its low score.
            if (
                version_2_with_negative
                and min_null_prediction is not None
                and not any(p["offsets"] == (0, 0) for p in n_best_preds)
            ):
                n_best_preds.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in n_best_preds:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(n_best_preds) == 0 or (len(n_best_preds) == 1 and n_best_preds[0]["text"] == ""):
                n_best_preds.insert(
                    0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
                )

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file,
            # using the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in n_best_preds])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our n_best_preds.
            for prob, pred in zip(probs, n_best_preds):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example["id"]] = n_best_preds[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while n_best_preds[i]["text"] == "":
                    i += 1
                best_non_null_pred = n_best_preds[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = (
                    null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                )
                scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            # Make `n_best_preds` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {
                    k: float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v
                    for k, v in pred.items()
                }
                for pred in n_best_preds
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise OSError(f"{output_dir} is not a directory.")

            prediction_file = os.path.join(
                output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
            )
            nbest_file = os.path.join(
                output_dir,
                "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json",
            )
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
                )

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
            if version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w") as writer:
                    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in all_predictions.items()
        ]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    logger.info(f"***** Running {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    model.eval()
    all_start_logits = []
    all_end_logits = []
    for _, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if (
                not args.pad_to_max_length
            ):  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, max_len)

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = postprocess_qa_predictions(
        examples=eval_examples,
        features=eval_dataset,
        predictions=outputs_numpy,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        # output_dir=args.finetuned_model_path,
        prefix=prefix,
    )

    metric = evaluate.load("squad")
    eval_metric = metric.compute(
        predictions=prediction.predictions, references=prediction.label_ids
    )
    logger.info(f"{prefix} metrics: {eval_metric}\n")
    return eval_metric


# Model Optimizer: Define a custom distillation loss function that uses start and end logits
class StartEndLogitsDistillationLoss(mtd.LogitsDistillationLoss):
    def forward(self, outputs_s, outputs_t):
        loss_start = super().forward(outputs_s.start_logits, outputs_t.start_logits)
        loss_end = super().forward(outputs_s.end_logits, outputs_t.end_logits)
        loss = (loss_start + loss_end) / 2.0

        return loss


def train_and_evaluate_model(
    args,
    model: nn.Module,
    accelerator: Accelerator,
    examples: dict,
    dataset: dict,
    dataloader: dict[str, DataLoader],
    answer_column_name,
):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(dataloader["train"]) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, dataloader["train"], dataloader["eval"], lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader["train"], dataloader["eval"], lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(dataloader["train"]) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("tensorboard", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    # Potentially load in the weights and states from a previous save
    if args.resume_from_last_ckpt:
        # Get the most recent checkpoint
        dirs = [
            f.path
            for f in os.scandir(args.finetuned_model_path)
            if f.is_dir() and (f.name.startswith("epoch_") or f.name.startswith("step_"))
        ]
        if len(dirs) == 0:
            logger.warning(
                f"No checkpoint found in {args.finetuned_model_path}. Training from scratch!"
            )
        else:
            latest_dir = max(dirs, key=os.path.getctime)
            accelerator.load_state(latest_dir)

            # Extract `epoch_{i}` or `step_{i}`
            latest_dir = os.path.basename(latest_dir)
            if "epoch" in latest_dir:
                starting_epoch = int(latest_dir.replace("epoch_", "")) + 1
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = (
                    int(latest_dir.replace("step_", "")) * args.gradient_accumulation_steps
                )
                starting_epoch = resume_step // len(dataloader["train"])
                completed_steps = resume_step // args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(dataloader["train"])

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # Evaluate before training (e.g. PTQ accuracy before QAT)
    eval_metric = evaluate_model(
        args,
        model,
        accelerator,
        examples["eval"],
        dataset["eval"],
        dataloader["eval"],
        answer_column_name,
    )
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_last_ckpt and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(dataloader["train"], resume_step)
        else:
            active_dataloader = dataloader["train"]
        for batch in active_dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)

            # Model Optimizer: If using distillation, we unwrap the model and extract the custom loss function
            if args.do_modelopt_distill:
                loss = model.module.compute_kd_loss()
            else:
                loss, _, _ = outputs.to_tuple()

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                accelerator.save_state(
                    os.path.join(args.finetuned_model_path, f"step_{completed_steps}")
                )

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(args.finetuned_model_path, f"epoch_{epoch}"))

        eval_metric = evaluate_model(
            args,
            model,
            accelerator,
            examples["eval"],
            dataset["eval"],
            dataloader["eval"],
            answer_column_name,
        )

        if args.with_tracking:
            log = {
                "squad": eval_metric,
                "train_loss": total_loss.item() / len(dataloader["train"]),
                "epoch": epoch,
                "step": completed_steps,
            }
            accelerator.log(log, step=completed_steps)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and eval_metric:
        logger.info(json.dumps(eval_metric, indent=4))
        # Prefix all keys with prefix + '_'
        for key in list(eval_metric.keys()):
            eval_metric[f"Eval_{key}"] = eval_metric.pop(key)

        with open(os.path.join(args.finetuned_model_path, "results.json"), "w") as f:
            json.dump(eval_metric, f, indent=4)


def main(input_args: list[str] | None = None) -> None:
    args = parse_args(input_args)

    # Initialize the accelerator
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = "tensorboard"
        accelerator_log_kwargs["project_dir"] = args.finetuned_model_path
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the training seed
    set_seed(SEED)

    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)
    dummy_input = model.dummy_inputs["input_ids"]

    # Get datasets
    examples, dataset, dataloader, answer_column_name = get_datasets_and_dataloaders(
        args, tokenizer, accelerator
    )

    def save(model, output_path):
        if accelerator.is_main_process:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            model = accelerator.unwrap_model(model)
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            logger.info(f"Saved model and tokenizer to {output_path}")

    # Model Optimizer: Prune the model to given FLOPS target using GradNAS algorithm
    if args.do_modelopt_prune:
        logger.info(f"Pruning model to {args.modelopt_prune_flops_percent}% FLOPS")

        # NOTE: gradnas does not perform synchronization across data parallel groups
        # Use unwrapped model & non-distributed dataloader for gradnas so that all the processes
        # in the data parallel group have the same gradients for pruning
        model = model.to(accelerator.device)
        dummy_input = dummy_input.to(accelerator.device)

        # Search for the best pruned model
        # To use other NAS algorithms, you can use `mtn.convert` + `mtn.search` here.
        model, _ = mtp.prune(
            model=model,
            mode="gradnas",
            constraints={"flops": f"{args.modelopt_prune_flops_percent}%"},
            dummy_input=dummy_input,
            config={
                "data_loader": dataloader["train"],
                "collect_func": lambda batch: (batch,),
                "loss_func": lambda output, batch: (
                    output["loss"] if isinstance(output, dict) else output[0]
                ),
            },
        )
        save(model, args.pruned_model_path)

    # Model Optimizer: Quantize the model to INT8 precision
    if args.modelopt_quantize_cfg:
        logger.info(f"Quantizing model with {args.modelopt_quantize_cfg} config")

        # NOTE: `mtq.quantize` does not perform synchronization across data parallel groups
        # Use unwrapped model & non-distributed dataloader for PTQ calibration so that all the processes
        # in the data parallel group have same calibration statistics
        model = model.to(accelerator.device)

        def forward_loop(model):
            num_samples = 256  # Use only 256 samples for PTQ calibration
            num_batches = num_samples // args.per_device_train_batch_size
            for idx, batch in tqdm(enumerate(dataloader["train"]), total=num_batches):
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                model(**batch)
                if idx >= num_batches:
                    break

        model = mtq.quantize(model, getattr(mtq, args.modelopt_quantize_cfg), forward_loop)
        torch.cuda.empty_cache()
        save(model, args.ptq_model_path)

    if args.do_train:
        # Handle the finetuned_model_path creation
        if accelerator.is_main_process:
            os.makedirs(args.finetuned_model_path, exist_ok=True)

        # Model Optimizer: Convert to a DistillationModel containing teacher to train with distillation
        if args.do_modelopt_distill:
            logger.info(f"Using distillation with teacher {args.model_name_or_path}")

            kd_config = {
                "teacher_model": AutoModelForQuestionAnswering.from_pretrained(
                    args.model_name_or_path,
                ),
                "criterion": StartEndLogitsDistillationLoss(args.temperature),
            }
            model = mtd.convert(model, mode=[("kd_loss", kd_config)])

        train_and_evaluate_model(
            args, model, accelerator, examples, dataset, dataloader, answer_column_name
        )

        # Model Optimizer: Export the distilled model
        if args.do_modelopt_distill:
            model = mtd.export(model)

        save(model, args.finetuned_model_path)

    if accelerator.is_main_process and args.onnx_export_path is not None:
        logger.info(f"Exporting ONNX model to {args.onnx_export_path}")

        # Move the model and dummy_input to the device
        model = model.to(accelerator.device)
        dummy_input = dummy_input.to(accelerator.device)

        model_name = os.path.basename(args.onnx_export_path).replace(".onnx", "")
        onnx_bytes, _ = get_onnx_bytes_and_metadata(
            model, dummy_input, model_name=model_name, onnx_opset=14
        )
        onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
        onnx_bytes_obj.write_to_disk(os.path.dirname(args.onnx_export_path), clean_dir=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()

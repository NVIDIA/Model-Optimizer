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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset
from PIL import Image
from scripts.ar_validate import validate_ar
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.distributed import is_master

try:
    import wandb
except ImportError:
    wandb = None

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def preprocess(examples, tokenizer, **kwargs):
    tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "loss_mask": [],
        "labels": [],
    }
    for i in range(len(examples)):
        messages = []
        source = examples[i]["conversations"]

        # Detect format: either role/content or from/value
        def get_role_content(item):
            if "role" in item and "content" in item:
                return item["role"], item["content"]
            elif "from" in item and "value" in item:
                return item["from"], item["value"]
            else:
                raise ValueError(f"Unknown conversation format: {item}")

        for sentence in source:
            role, content = get_role_content(sentence)
            messages.append({"role": role.lower(), "content": content})
        conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        output = tokenizer(
            conversation,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        )
        input_ids = output.input_ids[0]
        attention_mask = output.attention_mask[0]
        loss_mask = torch.ones_like(input_ids)
        labels = torch.cat([input_ids[1:], torch.tensor([IGNORE_TOKEN_ID], dtype=input_ids.dtype)])
        new_examples["input_ids"].append(input_ids)
        new_examples["attention_mask"].append(attention_mask)
        new_examples["loss_mask"].append(loss_mask)
        new_examples["labels"].append(labels)

    return new_examples


def preprocess_vlm(examples, tokenizer, processor, img_dir):
    tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "loss_mask": [],
        "labels": [],
        "pixel_values": [],
        "image_flags": [],
    }
    for i in range(len(examples)):
        messages = []
        source = examples[i]["conversations"]

        # Detect format: either role/content or from/value
        def get_role_content(item):
            if "role" in item and "content" in item:
                return item["role"], item["content"]
            elif "from" in item and "value" in item:
                return item["from"], item["value"]
            else:
                raise ValueError(f"Unknown conversation format: {item}")

        # align role to user-assistant format
        def convert_role(role):
            role_map = {
                "human": "user",
                "gpt": "assistant",
            }
            return role_map[role.lower()] if role.lower() in role_map else role.lower()

        for sentence in source:
            role, content = get_role_content(sentence)
            new_role = convert_role(role)
            messages.append({"role": new_role, "content": content})
        conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        img_filename = os.path.join(img_dir, examples[i]["image"])
        img = Image.open(img_filename)
        output = processor(images=img, text=conversation, return_tensors="pt")
        input_ids = output.input_ids[0]
        attention_mask = output.attention_mask[0]
        loss_mask = torch.ones_like(input_ids)
        labels = torch.cat([input_ids[1:], torch.tensor([IGNORE_TOKEN_ID], dtype=input_ids.dtype)])
        # TODO: add labels and answer-only loss masking?

        new_examples["input_ids"].append(input_ids)
        new_examples["attention_mask"].append(attention_mask)
        new_examples["loss_mask"].append(loss_mask)
        new_examples["labels"].append(labels)
        new_examples["pixel_values"].append(output.pixel_values)
        new_examples["image_flags"].append(
            torch.ones((output.pixel_values.shape[0],), dtype=torch.int64)
        )
    return new_examples


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        vlm_processor=None,
        img_dir=None,
    ):
        super().__init__()

        print_rank_0("Formatting inputs...")
        sources = raw_data
        self.preprocess_fn = preprocess_vlm if vlm_processor is not None else preprocess
        self.data_dict = self.preprocess_fn(
            sources, tokenizer, processor=vlm_processor, img_dir=img_dir
        )

    def __len__(self):
        return len(self.data_dict["input_ids"])

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return {k: self.data_dict[k][i] for k in self.data_dict}


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        vlm_processor=None,
        img_dir=None,
    ):
        super().__init__()
        print_rank_0("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.vlm_processor = vlm_processor
        self.img_dir = img_dir
        self.preprocess_fn = preprocess_vlm if vlm_processor is not None else preprocess

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        ret = self.preprocess_fn(
            [self.raw_data[i]], self.tokenizer, processor=self.vlm_processor, img_dir=self.img_dir
        )
        ret = {k: ret[k][0] for k in ret}
        self.cached_data_dict[i] = ret

        return ret


class OfflineSupervisedDataset(Dataset):
    """Lazy offline dataset for supervised fine-tuning.

    This dataset loads data on-the-fly from pre-processed .pt data files as well as
    input conversations in JSON format.

    Args:
        data_entries (list): A list of tuples (raw_data_example, file_path).
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(
        self,
        data_entries,
        tokenizer: transformers.PreTrainedTokenizer,
        vlm_processor=None,
        img_dir=None,
    ):
        super().__init__()
        print_rank_0("Formatting inputs...Skip in offline mode")
        self.tokenizer = tokenizer
        self.data_entries = data_entries
        self.vlm_processor = vlm_processor
        self.img_dir = img_dir
        self.preprocess_fn = preprocess_vlm if vlm_processor is not None else preprocess

        # Does not cache the hidden states, as those have an extremely large memory footprint.
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        # Load the conversational data, using the cache
        raw_data, offline_file_path = self.data_entries[i]
        # Extend the data sample with the hidden states from the .pt file
        max_length = self.tokenizer.model_max_length
        offline_data = torch.load(offline_file_path)
        offline_data["input_ids"] = offline_data["input_ids"][:max_length]
        offline_data["hidden_states"] = offline_data["hidden_states"][:max_length, :]
        offline_data["aux_hidden_states"] = offline_data["aux_hidden_states"][:max_length, :]

        ret = {
            "input_ids": offline_data["input_ids"],
            "attention_mask": torch.ones_like(offline_data["input_ids"]),
            "loss_mask": torch.ones_like(offline_data["input_ids"]),
            "labels": torch.full_like(offline_data["input_ids"], IGNORE_TOKEN_ID),
            "kwargs": {
                "base_model_outputs": {
                    "base_model_hidden_states": offline_data["hidden_states"],
                    "aux_hidden_states": offline_data["aux_hidden_states"],
                }
            },
        }
        return ret


def make_eagle_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_length=None,
) -> dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    if data_args.vlm_processor:
        vlm_processor = AutoProcessor.from_pretrained(
            data_args.vlm_processor, trust_remote_code=True, use_fast=True
        )
        vlm_img_dir = data_args.vlm_img_dir
    else:
        vlm_processor, vlm_img_dir = None, None
    # Load the conversations from the source file
    print_rank_0("Loading input conversations...")
    data_json = []
    data_path_p = Path(data_args.data_path)
    if data_path_p.is_dir():
        # Load all .jsonl files in the directory and combine them
        for jsonl_file in sorted(data_path_p.glob("*.jsonl")):
            with open(jsonl_file) as f:
                data_json.extend(json.loads(line) for line in f)
    else:
        with open(data_args.data_path) as f:
            if data_args.data_path.endswith("jsonl"):
                data_json = [json.loads(line) for line in f]
            else:
                data_json = json.load(f)

    if data_args.offline_data_path is not None:
        print_rank_0("Loading pre-processed data for offline training...")
        dataset_cls = OfflineSupervisedDataset

        # Glob for all .pt files in the data_path directory
        assert data_args.offline_data_path is not None, (
            "offline_data_path must be provided for offline training."
        )
        offline_data_path = Path(data_args.offline_data_path)
        # Collect all pt file paths
        all_files = {str(p) for p in offline_data_path.glob("*.pt")}
        all_files |= {str(p) for p in offline_data_path.glob("**/*.pt")}
        if not all_files:
            raise ValueError(f"No .pt files found in {data_args.offline_data_path}")

        # Build a map from conv_id to file_path for fast lookup
        print("building conv_id_to_file map...")
        conv_id_to_file = {}
        for pt_path in all_files:
            pt_name = Path(pt_path).name
            # Expect conv_id.pt
            if pt_name.endswith(".pt"):
                conv_id = pt_name[:-3]
                conv_id_to_file[conv_id] = pt_path

        valid_entries = []
        print("filtering valid entries...")
        for entry in data_json:
            conv_id = entry.get("conversation_id")
            if conv_id is None:
                conv_id = entry.get("uuid")
            if conv_id is None:
                conv_id = entry.get("id")
            if conv_id is None:
                raise ValueError(f"Conversation ID required but not found for entry {entry}")

            file_path = conv_id_to_file.get(str(conv_id))
            if file_path is None:
                continue
            valid_entries.append((entry, file_path))

        if len(valid_entries) == 0:
            msg = """No valid files found in the offline data path that match the conversation IDs
            in the provided data json. Please ensure that the offline data path is correct and
            contains .pt files named after the conversation IDs, and that the input conversations
            json has the correct format (with 'conversation_id' or 'id' fields)."""
            raise ValueError(msg)
        elif len(valid_entries) < len(data_json):
            print_rank_0(
                f"Warning: Only {len(valid_entries)} out of {len(data_json)} conversations"
                " have corresponding .pt files in the offline data path. Continuing..."
            )

        num_train = int(len(valid_entries) * 0.95)
        train_dataset = dataset_cls(
            valid_entries[:num_train],
            tokenizer=tokenizer,
            vlm_processor=vlm_processor,
            img_dir=vlm_img_dir,
        )
        eval_dataset = dataset_cls(
            valid_entries[num_train:],
            tokenizer=tokenizer,
            vlm_processor=vlm_processor,
            img_dir=vlm_img_dir,
        )

        data_collator = DataCollatorForOffline(max_length=max_length)
    else:
        print_rank_0("Loading input conversations...")
        dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset

        train_dataset = dataset_cls(
            data_json[: int(len(data_json) * 0.95)],
            tokenizer=tokenizer,
            vlm_processor=vlm_processor,
            img_dir=vlm_img_dir,
        )
        eval_dataset = dataset_cls(
            data_json[int(len(data_json) * 0.95) :],
            tokenizer=tokenizer,
            vlm_processor=vlm_processor,
            img_dir=vlm_img_dir,
        )

        data_collator = DataCollatorWithPadding(max_length=max_length)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }


class DataCollatorWithPadding:
    def __init__(self, max_length):
        self.max_length = max_length

    def paddingtensor2d(self, intensors, length):
        n, dim = intensors.shape
        if n > length:
            return intensors[:length, :]
        padding_tensor = torch.zeros(length - n, dim, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor))
        return outtensors

    def paddingtensor(self, intensors, length):
        if intensors.shape[0] > length:
            return intensors[:length]
        padding_tensor = torch.zeros(length - intensors.shape[0], dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor))
        return outtensors

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch_input_ids = torch.stack(
            [self.paddingtensor(item["input_ids"], self.max_length) for item in features]
        )
        batch_attention_mask = torch.stack(
            [self.paddingtensor(item["attention_mask"], self.max_length) for item in features]
        )
        batch_loss_mask = torch.stack(
            [self.paddingtensor(item["loss_mask"], self.max_length) for item in features]
        )

        batch_labels = torch.stack(
            [self.paddingtensor(item["labels"], self.max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "labels": batch_labels,
        }

        # Collate VLM data
        if "pixel_values" in features[0]:
            # pixel values and image flags should be flattened inside a batch
            batch["pixel_values"] = torch.cat([item["pixel_values"] for item in features], dim=0)
            batch["image_flags"] = torch.cat([item["image_flags"] for item in features], dim=0)

        return batch


class DataCollatorForOffline(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        base_batch = super().__call__(features)
        if "kwargs" not in features[0]:
            raise ValueError("No kwargs found in batch features. Offline data required.")

        features = [item["kwargs"]["base_model_outputs"] for item in features]

        batch_hidden_states = torch.stack(
            [
                self.paddingtensor2d(item["base_model_hidden_states"], self.max_length)
                for item in features
            ]
        )
        batch_aux_hidden_states = torch.stack(
            [self.paddingtensor2d(item["aux_hidden_states"], self.max_length) for item in features]
        )

        batch = {
            **base_batch,
            "base_model_outputs": {
                "base_model_hidden_states": batch_hidden_states,
                "aux_hidden_states": batch_aux_hidden_states,
            },
        }

        return batch


class EagleTrainerWithAccLog(Trainer):
    """Wrapper around Trainer that logs training accuracy."""

    def compute_loss(self, *args, **kwargs):
        """Override compute_loss to save train accs in trainer state."""
        if not hasattr(self.state, "training_accs"):
            self.state.training_accs = []
        kwargs.pop("num_items_in_batch", None)
        loss, outputs = super().compute_loss(return_outputs=True, *args, **kwargs)
        if hasattr(outputs, "train_acc"):
            self.state.training_accs.append(outputs.train_acc)
        return loss


class EagleTrainingPlot(TrainerCallback):
    """Callback that plot training acc and AR during training."""

    def __init__(self, ar_validate_steps: int = 1000, estimate_ar: bool = False):
        self.ar_validate_steps = ar_validate_steps
        if wandb and is_master():
            wandb.init()
        self.estimate_ar = estimate_ar

    def on_log(self, args, state, control, **kwargs):
        """Log training acc and estimate AR during log step."""
        if not hasattr(state, "training_accs") or len(state.training_accs) == 0:
            return control
        average_acc = np.mean(state.training_accs, axis=0)
        if self.estimate_ar:
            # Calculate mean training AR since last log
            # NOTE: This is only an estimate of the real AR.
            est_ar = 1
            acc_cumprod = 1
            for step_acc in average_acc[0]:
                acc_cumprod *= step_acc
                est_ar += acc_cumprod
            # Parallel draft tokens only used after all eagle tokens
            for draft_acc in average_acc[1:]:
                acc_cumprod *= draft_acc[-1]
                est_ar += acc_cumprod
            print_rank_0(f"Step {state.global_step} Estimated Training AR: {est_ar:.4f}")

        # log to wandb
        if wandb and is_master():
            for i, draft_acc in enumerate(average_acc):
                for j, step_acc in enumerate(draft_acc):
                    wandb.log(
                        {f"parallel_{i}_step_{j}_train_acc": step_acc}, step=state.global_step
                    )
            if self.estimate_ar:
                wandb.log({"estimated_training_ar": est_ar}, step=state.global_step)

        # reset training_accs
        state.training_accs = []
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Run AR validation periodically, if available."""
        if self.ar_validate_steps <= 0:
            return control
        if state.global_step % self.ar_validate_steps == 0 and state.global_step > 0:
            print_rank_0("Running AR validation...")
            try:
                ars = validate_ar(
                    model=kwargs["model"],
                    tokenizer=kwargs["processing_class"],
                    ds=load_dataset("HuggingFaceH4/mt_bench_prompts")["train"],
                    device=kwargs["model"].device,
                )
                print_rank_0(f"Step {state.global_step} AR: {sum(ars) / len(ars):.4f}")
                if wandb and is_master():
                    wandb.log({"validate_ar": sum(ars) / len(ars)}, step=state.global_step)
            except Exception:
                print_rank_0("AR validation not available.")
        return control

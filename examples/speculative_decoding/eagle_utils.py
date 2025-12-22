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
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import io
from PIL import Image
from torchvision.transforms import ToTensor

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
            max_length=4096
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
        if i in self.cached_data_dict:
            preprocessed_base = self.cached_data_dict[i]
        else:
            ret = self.preprocess_fn(
                [raw_data], self.tokenizer, processor=self.vlm_processor, img_dir=self.img_dir
            )
            preprocessed_base = {k: ret[k][0] for k in ret}
            self.cached_data_dict[i] = preprocessed_base

        # Extend the data sample with the hidden states from the .pt file
        max_length = self.tokenizer.model_max_length
        offline_data = torch.load(offline_file_path)
        offline_data["input_ids"] = offline_data["input_ids"][:max_length]
        if "aux_hidden_states" in offline_data:
            offline_data["aux_hidden_states"] = offline_data["aux_hidden_states"][:max_length, :].to(torch.bfloat16)
            hidden_dim = offline_data["aux_hidden_states"].shape[-1]
            assert hidden_dim % 3 == 0, \
                f"Aux hidden states dimension {hidden_dim} is not divisible by 3."
            hidden_dim = hidden_dim // 3
            # take last third of aux hidden states as the main hidden states
            offline_data["hidden_states"] = offline_data["aux_hidden_states"][:, -hidden_dim:].clone()
        else:
            offline_data["hidden_states"] = offline_data["hidden_states"][:max_length, :].to(torch.bfloat16)

        # Make sure the input_ids have the same shape
        if preprocessed_base["input_ids"].shape != offline_data["input_ids"].shape:
            min_len = min(
                preprocessed_base["input_ids"].shape[0],
                offline_data["input_ids"].shape[0],
            )
            offline_data["input_ids"] = offline_data["input_ids"][:min_len]
            offline_data["hidden_states"] = offline_data["hidden_states"][:min_len]
            if "aux_hidden_states" in offline_data:
                offline_data["aux_hidden_states"] = offline_data["aux_hidden_states"][:min_len]
            # Use input ids from offline data for consistency
            preprocessed_base["input_ids"] = offline_data["input_ids"][:min_len]
            preprocessed_base["attention_mask"] = preprocessed_base["attention_mask"][:min_len]
            preprocessed_base["loss_mask"] = preprocessed_base["loss_mask"][:min_len]
            # Use labels from offline data for consistency
            preprocessed_base["labels"] = torch.cat([
                preprocessed_base["input_ids"][1:],
                torch.tensor([IGNORE_TOKEN_ID], dtype=preprocessed_base["input_ids"].dtype),
            ])

            # Check for exact off-by-one where the preprocessed data is 1 longer
            # if preprocessed_base["input_ids"].shape[0] == offline_data["input_ids"].shape[0] + 1:
            #     for k in preprocessed_base:
            #         if isinstance(preprocessed_base[k], torch.Tensor) and \
            #             preprocessed_base[k].shape[0] == offline_data["input_ids"].shape[0] + 1:
            #             preprocessed_base[k] = preprocessed_base[k][:-1]
            # elif preprocessed_base["input_ids"].shape[0] + 1 == offline_data["input_ids"].shape[0]:
            #     # Check for exact off-by-one where the offline data is 1 longer
            #     for k in offline_data:
            #         if isinstance(offline_data[k], torch.Tensor) and \
            #             offline_data[k].shape[0] == preprocessed_base["input_ids"].shape[0] + 1:
            #             offline_data[k] = offline_data[k][:-1]
            # else:
            #     msg = f"""Input IDs from offline data do not match the preprocessed input IDs
            #                         for offline data sample at {offline_file_path}."""
            #     raise ValueError(msg)

        ret = {**preprocessed_base}  # Shallow copy so we don't accidentally modify the cache
        ret["input_ids"] = offline_data["input_ids"]
        ret["kwargs"] = {
            "base_model_outputs": {
                "base_model_hidden_states": offline_data["hidden_states"],
            }
        }
        if "aux_hidden_states" in offline_data:
            ret["kwargs"]["base_model_outputs"]["aux_hidden_states"] = \
                offline_data["aux_hidden_states"]
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
        all_files = {str(p) for p in offline_data_path.glob("*.pt")}
        if not all_files:
            raise ValueError(f"No .pt files found in {data_args.offline_data_path}")

        # Filter to conversations that exist in the offline data and in the provided json
        valid_entries = []
        for entry in data_json:
            conv_id = entry.get("conversation_id")
            if conv_id is None:
                conv_id = entry.get("uuid")
            if conv_id is None:
                conv_id = entry.get("id")
            if conv_id is None:
                raise ValueError(f"Conversation ID required but not found for entry {entry}")
            file_path = str(offline_data_path / f"{conv_id}.pt")
            if file_path in all_files:
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

        batch = {
            **base_batch,
            "base_model_outputs": {
                "base_model_hidden_states": batch_hidden_states,
            },
        }
        if features and "aux_hidden_states" in features[0]:
            batch_aux_hidden_states = torch.stack(
                [self.paddingtensor2d(item["aux_hidden_states"], self.max_length) for item in features]
            )
            batch["base_model_outputs"]["aux_hidden_states"] = batch_aux_hidden_states

        return batch


class EagleTrainerWithAccLog(Trainer):
    """Wrapper around Trainer that logs training accuracy."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, *args, **kwargs):
        """Override compute_loss to save train accs in trainer state."""
        if not hasattr(self.state, "training_accs"):
            self.state.training_accs = []
        kwargs.pop("num_items_in_batch", None)
        return_outputs = kwargs.pop("return_outputs", False)
        loss, outputs = super().compute_loss(*args, return_outputs=True, **kwargs)
        if hasattr(outputs, "train_acc"):
            self.state.training_accs.append(outputs.train_acc)
        return (loss, outputs) if return_outputs else loss


class EagleTrainingPlot(TrainerCallback):
    """Callback that plot training acc and AR during training."""

    def __init__(self, ar_validate_steps: int = 1000, estimate_ar: bool = False, tb_writer: SummaryWriter | None = None):
        self.ar_validate_steps = ar_validate_steps
        if wandb and is_master():
            wandb.init()
        self.estimate_ar = estimate_ar
        self.tb_writer = tb_writer
        self.last_seen_step = -1

    def _report_stats(self, state, eval_mode: bool):
        if not hasattr(state, "training_accs") or len(state.training_accs) == 0:
            return
        average_acc = np.mean(state.training_accs, axis=0)
        if self.estimate_ar:
            # Calculate mean training AR since last log
            # NOTE: This is only a estimate of the real AR.
            est_ar = 1
            acc_cumprod = 1
            for step_acc in average_acc:
                est_ar += acc_cumprod * step_acc
                acc_cumprod *= step_acc
            print_rank_0(f"Step {state.global_step} Estimated {"Eval" if eval_mode else "Training"} AR: {est_ar:.4f}")

        # log to wandb
        if wandb and is_master():
            for i, step_acc in enumerate(average_acc):
                wandb.log({f"step_{i}_{"eval" if eval_mode else "train"}_acc": step_acc}, step=state.global_step)
            if self.estimate_ar:
                wandb.log({f"estimated_{"eval" if eval_mode else "train"}_ar": est_ar}, step=state.global_step)
        
        if self.tb_writer:
            for i, step_acc in enumerate(average_acc):
                self.tb_writer.add_scalar(f"custom/step_{i}_{'eval' if eval_mode else 'train'}_acc", step_acc, state.global_step)
            if self.estimate_ar:
                self.tb_writer.add_scalar(f"custom/estimated_{'eval' if eval_mode else 'train'}_ar", est_ar, state.global_step)

    def on_log(self, args, state, control, **kwargs):
        """Log training acc and estimate AR during log step."""
        if not hasattr(state, "training_accs") or len(state.training_accs) == 0:
            self.last_seen_step = state.global_step
            return control
        
        if state.global_step != self.last_seen_step:
            # Eval mode doesn't increment the global step, so we can use that to detect eval vs training
            self._report_stats(state, eval_mode=False)
            # reset training_accs
            state.training_accs = []
        
        self.last_seen_step = state.global_step
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        """Log eval acc and estimate AR during eval step."""
        if not hasattr(state, "training_accs") or len(state.training_accs) == 0:
            return control
        
        self._report_stats(state, eval_mode=True)
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
                if self.tb_writer:
                    self.tb_writer.add_scalar("custom/validate_ar", sum(ars) / len(ars), state.global_step)
            except Exception:
                print_rank_0("AR validation not available.")
        return control

def _render_bar_chart_to_tensor(data_tensor, title, ylabel, ylim, color_fn=None):
    """
    Internal helper: Converts a 1D tensor into a matplotlib bar chart image tensor.
    """
    # 1. Pre-process data to CPU Numpy
    data_np = data_tensor.detach().float().cpu().numpy()
    experts = range(len(data_np))

    # 2. Determine colors based on optional function
    if color_fn is None:
        colors = 'tab:blue' # Default
    else:
        colors = [color_fn(x) for x in data_np]

    # 3. Create a distinct figure (smaller height since it's single)
    fig, ax = plt.subplots(figsize=(10, 4))

    # 4. Plotting details
    ax.bar(experts, data_np, color=colors, alpha=0.7, width=0.8)
    # Important: Set fixed Y-limits to stop axes jumping around between steps
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Expert Index")
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    # Add a strong baseline at 0
    ax.axhline(0, color='black', linewidth=0.8)

    # 5. Render to memory buffer
    plt.tight_layout()
    buf = io.BytesIO()
    # dpi=100 keeps image size reasonable
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)

    # 6. Convert buffer to Torch Tensor
    image = Image.open(buf)
    image_tensor = ToTensor()(image)

    # 7. Cleanup memory
    plt.close(fig)
    buf.close()

    return image_tensor

def log_expert_snapshots(writer, step, violations, biases, batch_expert_usage):
    # --- Image 1: Bias Scores ---
    # Simple bars
    bias_image_tensor = _render_bar_chart_to_tensor(
        biases,
        title=f"Expert Bias Scores (Step {step})",
        ylabel="Bias",
        ylim=(-1.1, 1.1),
        color_fn=None # Use default blue
    )
    writer.add_image("moe/snapshot_bias", bias_image_tensor, global_step=step)

    # --- Image 2: Violation Scores ---
    violation_color_logic = lambda x: 'tab:red' if x > 10.0 else (
        'tab:green' if (-0.5 < x < 1.5) else 'tab:orange'
    )

    violation_image_tensor = _render_bar_chart_to_tensor(
        violations,
        title=f"Batch Expert Violation Scores (Step {step})",
        ylabel="Violation Score",
        ylim=(-2, 25),
        color_fn=violation_color_logic
    )
    writer.add_image("moe/snapshot_violations", violation_image_tensor, global_step=step)

    # --- Image 3: Batch Expert Usage ---
    total_usage = batch_expert_usage.sum().item()
    batch_usage_image_tensor = _render_bar_chart_to_tensor(
        batch_expert_usage / total_usage,
        title=f"Batch Expert Usage (Step {step})",
        ylabel="Fraction of Load",
        ylim=(0, 1.1),
        color_fn=None # Use default blue
    )
    writer.add_image("moe/snapshot_batch_usage", batch_usage_image_tensor, global_step=step)

class EagleMoEBalancer(TrainerCallback):
    """Callback that balances MoE expert usage during training."""

    def __init__(self, update_interval: int, tb_writer: SummaryWriter | None = None):
        self.tb_writer = tb_writer
        self.seen_last_step = -1
        self.last_was_train = False
        self.update_interval = update_interval

    def on_step_end(self, args, state, control, **kwargs):
        """Balance MoE expert usage at the end of each training step."""
        # Exit if not training
        model = kwargs["model"]
        gate = model.eagle_module.layers[0].mlp.gate
        is_train = self.seen_last_step != state.global_step
        if not is_train:
            gate.clear_temp_correction_accumulator()
            self.seen_last_step = state.global_step
            self.last_was_train = False
            return control
        elif not self.last_was_train:
            # First training step after eval, don't use the expert usage stats from eval
            gate.clear_temp_correction_accumulator()
            self.seen_last_step = state.global_step
            self.last_was_train = True
            return control
        
        # In training with usable data
        self.last_was_train = True
        self.seen_last_step = state.global_step

        if state.global_step % self.update_interval != 0:
            return control
        
        violations = gate.apply_bias_update()

        # Logging
        if self.tb_writer:
            if state.global_step % 10 == 0:
                log_expert_snapshots(
                    self.tb_writer, 
                    state.global_step, 
                    violations, 
                    gate.get_e_score_correction_bias(),
                    gate.e_score_correction_bias_temp_acc.data
                )

            # 3. SCALARS: Track the worst case
            self.tb_writer.add_scalar(
                "moe/max_violation",
                violations.max().item(),
                state.global_step,
            )

        gate.clear_temp_correction_accumulator()
        return control
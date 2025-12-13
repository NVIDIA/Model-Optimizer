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

"""Utility functions for getting samples and forward loop function for different datasets."""

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from warnings import warn

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Use dict to store the config for each dataset.
# If we want to export more options to user like target languages, we need more standardized approach like dataclass.
SUPPORTED_DATASET_CONFIG: dict[str, Any] = {
    "open_code_reasoning": {
        "config": {"path": "nvidia/OpenCodeReasoning", "name": "split_0", "split": ["split_0"]},
        "preprocess": lambda sample: "\n".join([sample["input"], sample["output"]]),
    },
    "open_math_reasoning": {
        "config": {
            "path": "nvidia/OpenMathReasoning",
            "split": ["cot", "tir", "genselect"],
        },
        "preprocess": lambda sample: "\n".join([sample["problem"], sample["generated_solution"]]),
    },
    "llama-nemotron-post-training-dataset": {
        "config": {
            "path": "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "name": "SFT",
            "split": ["code", "math", "science", "chat", "safety"],
        },
        "preprocess": lambda sample: "\n".join(turn["content"] for turn in sample["input"])
        + "\n"
        + sample["output"],
    },
    "nemotron-post-training-dataset-v2": {
        "config": {
            "path": "nvidia/Nemotron-Post-Training-Dataset-v2",
            "split": ["stem", "chat", "math", "code"],
        },
        "preprocess": lambda sample: "\n".join(turn["content"] for turn in sample["messages"]),
    },
    "nemotron-post-training-dataset-v1": {
        "config": {
            "path": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": ["stem", "chat", "math", "code", "tool_calling"],
        },
        "preprocess": lambda sample: "\n".join(turn["content"] for turn in sample["messages"]),
    },
    "magpie": {
        "config": {
            "path": "Magpie-Align/Magpie-Pro-MT-300K-v0.1",
            "split": ["train"],
        },
        "preprocess": lambda sample: "\n".join(turn["value"] for turn in sample["conversations"]),
    },
    "cnn_dailymail": {
        "config": {"path": "cnn_dailymail", "name": "3.0.0", "split": ["train"]},
        "preprocess": lambda sample: sample["article"],
    },
    "pile": {
        "config": {"path": "monology/pile-uncopyrighted", "name": "v1.0", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "pg19": {
        "config": {"path": "pg19", "name": "v1.0", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "wikipedia": {
        "config": {"path": "wikipedia", "name": "20220301.en", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "c4": {
        "config": {"path": "c4", "name": "en", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
}

__all__ = [
    "create_forward_loop",
    "get_dataset_dataloader",
    "get_max_batch_size",
    "get_supported_datasets",
]


def _get_dataset_samples(dataset_name: str, num_samples: int) -> list[str]:
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.

    Returns:
        Samples: The list of samples.
    """
    # Load the dataset
    if dataset_name not in SUPPORTED_DATASET_CONFIG:
        raise NotImplementedError(
            f"dataset {dataset_name} is not supported. Please use one of the following:"
            f" {get_supported_datasets()}."
        )

    from datasets import load_dataset

    dataset_config = SUPPORTED_DATASET_CONFIG[dataset_name]
    # It's unfortunate that the load_dataset function does not support split a list while streaming.
    # So we need to load the dataset for each split.
    config = dataset_config["config"].copy()
    splits = config.pop("split", [None])
    dataset_splits = [
        load_dataset(
            streaming=True,
            **config,
            split=split,
        )
        for split in splits
    ]

    # Split the samples evenly across the splits
    # For streaming datasets, there is no reliable way to get the number of samples in each split
    # other than loading the entire dataset. So, we just use the same number of samples for each split.
    num_samples_splits = [num_samples // len(dataset_splits) for _ in dataset_splits]
    num_samples_splits[-1] += num_samples - sum(num_samples_splits)
    samples = []
    for dataset, num_samples_split in zip(dataset_splits, num_samples_splits):
        for i, sample in enumerate(dataset):
            if i >= num_samples_split:
                break

            # Apply preprocess function to the sample
            samples.append(dataset_config["preprocess"](sample))

    return samples


class _CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {
            key: val[idx] if torch.is_tensor(val[idx]) else torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))


def get_dataset_dataloader(
    dataset_name: str | list[str] = "cnn_dailymail",
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    batch_size: int = 1,
    num_samples: int | list[int] = 512,
    max_sample_length: int = 512,
    device: torch.device | None = None,
    include_labels: bool = False,
) -> DataLoader:
    """Get a dataloader with the dataset name and toknizer of the target model.

    Args:
        dataset_name: Name of the dataset to load.
        tokenizer: Instancne of Hugginface tokenizer.
        batch_size: Batch size of the returned dataloader.
        num_samples: Number of samples from the dataset.
        max_sample_length: Maximum length of a sample.
        device: Target device for the returned dataloader.
        include_labels: Whether to include labels in the dataloader.

    Returns:
        A instance of dataloader.
    """
    assert tokenizer is not None, "Please provide a tokenizer."
    # batch_encode_plus will modify the tokenizer in place, so we need to clone it.
    tokenizer = copy.deepcopy(tokenizer)

    if tokenizer.padding_side != "left":
        warn(
            "Tokenizer with the right padding_side may impact calibration accuracy. Recommend set to left"
        )

    if isinstance(num_samples, int):
        num_samples = [num_samples]

    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    assert len(dataset_name) == len(num_samples), (
        "dataset_name and num_samples must be the same length"
    )

    all_samples = []
    for ds_name, num_sample in zip(dataset_name, num_samples):
        samples = _get_dataset_samples(ds_name, num_sample)
        all_samples.extend(samples)

    batch_encoded = tokenizer.batch_encode_plus(
        all_samples,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_sample_length,
    )
    if device:
        batch_encoded = batch_encoded.to(device)

    if include_labels:
        # Labels are needed when backward is called in the model.
        # The labels should be a shifted version of the input_ids.
        # However, we should not shift the input_ids here since the labels are shifted by
        # Huggingface models during loss calculation as shown here -
        # https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/llama/modeling_llama.py#L1093-L1095
        batch_encoded["labels"] = torch.where(
            batch_encoded["attention_mask"] > 0.5, batch_encoded["input_ids"], -100
        )
        tokenized_dataset = _CustomDataset(batch_encoded)
    else:
        # For backward compatibility, if labels are not needed, we only return the input_ids.
        tokenized_dataset = _CustomDataset({"input_ids": batch_encoded["input_ids"]})

    calib_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    return calib_dataloader


def get_supported_datasets() -> list[str]:
    """Retrieves a list of datasets supported.

    Returns:
        A list of strings, where each string is the name of a supported dataset.

    Example usage:

        .. code-block:: python

            from modelopt.torch.utils import get_supported_datasets

            print("Supported datasets:", get_supported_datasets())
    """
    return list(SUPPORTED_DATASET_CONFIG.keys())


def get_max_batch_size(
    model: torch.nn.Module,
    max_sample_length: int = 512,
    sample_memory_usage_ratio: float = 1.0,
    sample_input_single_batch: torch.Tensor | None = None,
    enable_grad: bool = False,
):
    """Get the maximum batch size that can be used for the model."""

    def _get_free_gpu_mem():
        min_gpu_free_mem = torch.cuda.get_device_properties(0).total_memory
        max_allocated_mem = 0
        for device in range(torch.cuda.device_count()):
            free_mem = torch.cuda.mem_get_info(device)[0]
            if free_mem < min_gpu_free_mem:
                min_gpu_free_mem = free_mem
                max_allocated_mem = torch.cuda.max_memory_allocated(device)
        return min_gpu_free_mem, max_allocated_mem

    torch.cuda.empty_cache()

    free_mem_before, max_allocated_before = _get_free_gpu_mem()
    is_enc_dec = model_type_is_enc_dec(model)
    infer_method = model.generate if is_enc_dec else model.forward

    if sample_input_single_batch is None:
        sample_input_single_batch = (
            torch.ones([1, max_sample_length], dtype=torch.int32, device=model.device) * 100
        )

    # Calculate single batch inference with dummy input.
    with torch.set_grad_enabled(enable_grad):
        infer_method(sample_input_single_batch)
    free_mem_after, max_allocated_after = _get_free_gpu_mem()

    mem_diff_per_data_batch = (
        max(
            (free_mem_before - free_mem_after),
            (max_allocated_after - max_allocated_before),
        )
        * sample_memory_usage_ratio
    )
    if mem_diff_per_data_batch <= 0:
        print(
            "Warning: No measurable memory usage found for a single batch. "
            "Falling back to batch_size=1."
        )
        target_data_batch = 1
    else:
        target_data_batch = max(int(free_mem_before / mem_diff_per_data_batch), 1)
    target_input = sample_input_single_batch.expand(
        [
            target_data_batch if index == 0 else dim
            for index, dim in enumerate(sample_input_single_batch.shape)
        ]
    )

    # For some models on multi GPU, we observe the memory per batch is not a constant.
    # So we just test the target batch size and make sure we do not go OOM.
    while target_data_batch > 1:
        with torch.set_grad_enabled(enable_grad):
            try:
                infer_method(target_input)
                break
            except torch.cuda.OutOfMemoryError:
                target_data_batch = target_data_batch // 2

    # Regulate the data batch target to be 1, 2, 4, 8, 12, ..., capped at 64
    if target_data_batch < 2:
        return 1
    elif target_data_batch < 4:
        return 2
    elif target_data_batch < 512:
        return target_data_batch // 4 * 4
    else:
        return 512


def _process_batch(batch_data, infer_method, max_working_batch_size=None):
    """Process a batch of data through the model's inference method.

    Args:
        batch_data: Dictionary containing the batch data
        infer_method: Model's inference method (either forward or generate)
        max_working_batch_size: Maximum batch size known to work without OOM

    Returns:
        The maximum batch size that worked successfully
    """
    assert all(torch.is_tensor(data) or data is None for data in batch_data.values()), (
        "batch_data values must be tensors"
    )
    # Get the batch size of current data
    batch_size = batch_data[next(iter(batch_data.keys()))].shape[0]

    # If we know a smaller batch size works, preemptively split
    if max_working_batch_size is not None and batch_size > max_working_batch_size:
        # Split the batch to avoid OOM
        for i in range(0, batch_size, max_working_batch_size):
            end_idx = min(i + max_working_batch_size, batch_size)
            split_data = {}
            for key in batch_data:
                if batch_data[key] is None:
                    split_data[key] = None
                else:
                    split_data[key] = batch_data[key][i:end_idx, ...]

            max_working_batch_size = _process_batch(
                split_data, infer_method, max_working_batch_size
            )

        return max_working_batch_size

    # Try processing with current batch size
    try:
        infer_method(**batch_data)
        return (
            batch_size
            if max_working_batch_size is None
            else max(batch_size, max_working_batch_size)
        )  # This batch size worked successfully
    except torch.cuda.OutOfMemoryError:
        assert batch_size > 1, (
            "CUDA out of memory error occurred while processing a single sample. "
            "This indicates the model is too large for the available GPU memory. "
            "Consider reducing the model size, using a smaller max_sample_length, "
            "or using a GPU with more memory."
        )

    # Split the batch in half
    mid = (batch_size + 1) // 2
    warn(f"CUDA out of memory with batch size {batch_size}, trying with batch size {mid}")
    split_data_1 = {key: batch_data[key][:mid, ...] for key in batch_data}
    split_data_2 = {key: batch_data[key][mid:, ...] for key in batch_data}

    # Recursively process each half and track max working batch size
    max_working_batch_size = _process_batch(split_data_1, infer_method)
    max_working_batch_size = _process_batch(split_data_2, infer_method, max_working_batch_size)

    # Return the minimum of the two (to be conservative)
    return max_working_batch_size


def _forward_loop(model: torch.nn.Module, dataloader: DataLoader) -> None:
    """Runs forward passes through the model using data from the dataloader.

    Args:
        model: The PyTorch model to run inference on
        dataloader: DataLoader containing the batched input data
    """
    with torch.no_grad():
        is_enc_dec = model_type_is_enc_dec(model)
        infer_method = model.generate if is_enc_dec else model.forward
        max_working_batch_size = None  # Initialize max working batch size as None

        for _, data in enumerate(tqdm(dataloader)):
            # Process batch and update max working batch size
            max_working_batch_size = _process_batch(data, infer_method, max_working_batch_size)


def create_forward_loop(
    model: torch.nn.Module | None = None,
    dataset_name: str = "cnn_dailymail",
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    batch_size: int = 1,
    num_samples: int = 512,
    max_sample_length: int = 512,
    device: str | None = None,
    include_labels: bool = False,
    dataloader: DataLoader | None = None,
) -> Callable:
    """Creates and returns a forward loop function configured for a specific model, dataset, and tokenizer.

    This function initializes a forward loop function tailored to process batches of data from the specified dataset
    using the given model and tokenizer. The forward loop function, when called, iterates over the dataset, applies the
    tokenizer to prepare the input data, feeds it into the model, and returns the model's predictions.

    Args:
        model: The PyTorch model for inference.
        dataset_name: The name of the dataset to be used. Must be one of the datasets in get_supported_datasets().
        tokenizer: The tokenizer used to preprocess text data into a format suitable
            for the model.
        batch_size: Batch size of the returned dataloader. If 0 is provided, we auto determine the batch_size.
        num_samples: Number of samples from the dataset.
        max_sample_length: Maximum length of a sample.
        device: Target device for the returned dataloader.
        include_labels: Whether to include labels in the dataloader.
        dataloader: If provided, use the provided dataloader instead.

    Example usage for quantization:

    .. code-block:: python

        import modelopt.torch.quantization as mtq
        from modelopt.torch.utils import create_forward_loop

        # Initialize model and tokenizer
        # ...

        # Create forward loop for calibration
        forward_loop = create_forward_loop(
            model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer
        )

        # Quantize the model with the calibration dataset
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    Returns:
        A forward loop function that can be called with no arguments. When called, this function iterates over
            the dataset specified by `dataset_name`.
    """
    if dataloader is None:
        if batch_size == 0:
            # We let the system to determine the max data batch for each forward.
            batch_size = get_max_batch_size(model, max_sample_length)
            print(f"Update calib batch {batch_size}")

        dataloader = get_dataset_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_samples=num_samples,
            max_sample_length=max_sample_length,
            device=device,
            include_labels=include_labels,
        )

    return lambda model: _forward_loop(model, dataloader)


def model_type_is_enc_dec(model):
    enc_dec_model_list = ["t5", "bart", "whisper"]
    return any(model_name in model.__class__.__name__.lower() for model_name in enc_dec_model_list)

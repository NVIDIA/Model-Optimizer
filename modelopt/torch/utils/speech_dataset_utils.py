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

"""Utility functions for getting samples and forward loop function for different speech datasets."""

from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

# Use dict to store the config for each dataset.
# If we want to export more options to user like target languages, we need more standardized approach like dataclass.
SUPPORTED_SPEECH_DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "peoples_speech": {
        "config": {"path": "MLCommons/peoples_speech", "name": "clean", "split": "train"},
    },
}

__all__ = ["get_speech_dataset_dataloader", "get_supported_speech_datasets"]


def _get_speech_dataset(dataset_name: str, num_samples: int):
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.

    Returns:
        A hugging face Dataset.
    """
    # Load the dataset
    if dataset_name in SUPPORTED_SPEECH_DATASET_CONFIG:
        from datasets import load_dataset

        # Use streaming can reduce the downloading time for large datasets
        dataset = load_dataset(
            **SUPPORTED_SPEECH_DATASET_CONFIG[dataset_name]["config"],
            trust_remote_code=True,
            streaming=True,
        )
    else:
        raise NotImplementedError(
            f"dataset {dataset_name} is not supported. Please use one of the following:"
            f" {get_supported_speech_datasets()}."
        )
    return dataset.take(num_samples)


def get_supported_speech_datasets() -> list[str]:
    """Retrieves a list of speech datasets supported.

    Returns:
        A list of strings, where each string is the name of a supported dataset.

    Example usage:

    .. code-block:: python

        from modelopt.torch.utils import get_supported_speech_datasets

        print("Supported datasets:", get_supported_speech_datasets())
    """
    return list(SUPPORTED_SPEECH_DATASET_CONFIG.keys())


def get_speech_dataset_dataloader(
    dataset_name: str = "peoples_speech",
    processor: WhisperProcessor | None = None,
    batch_size: int = 1,
    num_samples: int = 512,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[DataLoader, str]:
    """Get a dataloader with the dataset name and processor of the target model.

    Args:
        dataset_name: Name of the dataset to load.
        processor: Processor used for encoding images and text data.
        batch_size: Batch size of the returned dataloader.
        num_samples: Number of samples from the dataset.
        device: Target device for the returned dataloader.
        dtype: dtype of the returned dataset.

    Returns:
        An instance of dataloader.
    """
    assert processor is not None, "Please provide a valid processor."

    dataset = _get_speech_dataset(dataset_name, num_samples=num_samples)
    first_sample = next(iter(dataset))
    first_text = first_sample["text"]

    def preprocess_and_move_to_cuda(example):
        # Process the audio example using the WhisperProcessor
        inputs = processor(
            example["audio"]["array"],  # The raw audio data
            sampling_rate=example["audio"]["sampling_rate"],  # Sampling rate of the audio
            return_tensors="pt",  # Return as PyTorch tensors
        )

        # Move input_features to the GPU (cuda)
        input_features = inputs.input_features[0].to(device)
        if dtype:
            input_features = input_features.to(dtype)

        return {"input_features": input_features}

    dataset = dataset.map(preprocess_and_move_to_cuda)

    def collate_fn(batch):
        # Stack all tensors (all should be of shape (80, 3000) already)
        input_features = torch.stack([item["input_features"] for item in batch], dim=0)

        return {"input_features": input_features}

    # Define the DataLoader (batches will be created automatically by DataLoader)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn), first_text

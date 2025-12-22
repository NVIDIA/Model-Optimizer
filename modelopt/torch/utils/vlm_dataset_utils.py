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

"""Utility functions for getting samples and forward loop function for different vlm datasets."""

from typing import Any

from torch.utils.data import DataLoader

from .image_processor import MllamaImageProcessor

# Use dict to store the config for each dataset.
# If we want to export more options to user like target languages, we need more standardized approach like dataclass.
SUPPORTED_VLM_DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "scienceqa": {"config": {"path": "derek-thomas/ScienceQA", "split": "train"}},
}

__all__ = ["get_supported_vlm_datasets", "get_vlm_dataset_dataloader"]


def _get_vlm_dataset(dataset_name: str, num_samples: int):
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.

    Returns:
        A hugging face Dataset.
    """
    # Load the dataset
    if dataset_name in SUPPORTED_VLM_DATASET_CONFIG:
        from datasets import load_dataset

        # Use streaming can reduce the downloading time for large datasets
        dataset = load_dataset(
            **SUPPORTED_VLM_DATASET_CONFIG[dataset_name]["config"],
        )
    else:
        raise NotImplementedError(
            f"dataset {dataset_name} is not supported. Please use one of the following:"
            f" {get_supported_vlm_datasets()}."
        )
    return dataset.select(range(num_samples))


def get_supported_vlm_datasets() -> list[str]:
    """Retrieves a list of vlm datasets supported.

    Returns:
        A list of strings, where each string is the name of a supported dataset.

    Example usage:

    .. code-block:: python

        from modelopt.torch.utils import get_supported_vlm_datasets

        print("Supported datasets:", get_supported_vlm_datasets())
    """
    return list(SUPPORTED_VLM_DATASET_CONFIG.keys())


def get_vlm_dataset_dataloader(
    dataset_name: str = "scienceqa",
    processor: MllamaImageProcessor = None,
    batch_size: int = 1,
    num_samples: int = 512,
) -> DataLoader:
    """Get a dataloader with the dataset name and processor of the target model.

    Args:
        dataset_name: Name of the dataset to load.
        processor: Processor used for encoding images and text data.
        batch_size: Batch size of the returned dataloader.
        num_samples: Number of samples from the dataset.

    Returns:
        An instance of dataloader.
    """
    assert processor is not None, "Please provide a valid processor."

    dataset = _get_vlm_dataset(dataset_name, num_samples=num_samples)
    # Apply the preprocessing function to the dataset
    processed_dataset = dataset.map(
        processor.preprocess_function, batched=False, remove_columns=dataset.column_names
    )

    # Create DataLoader with the custom collate function
    return DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=processor.collate_function,
    )

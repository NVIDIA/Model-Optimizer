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

"""Utility functions for getting samples and dataloader for different VLM calibration datasets."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from .image_processor import MllamaImageProcessor

# Use dict to store the config for each dataset.
# If we want to export more options to user like target languages, we need more standardized approach like dataclass.
SUPPORTED_VLM_DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "scienceqa": {"config": {"path": "derek-thomas/ScienceQA", "split": "train"}},
}

__all__ = ["get_supported_vlm_datasets", "get_vlm_dataset_dataloader"]


def _get_vlm_dataset(dataset_name: str, num_samples: int, require_image: bool = True):
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.
        require_image: If True, keep only samples that have an image field.

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

    # `load_dataset` returns a DatasetDict. Use the configured split.
    split = SUPPORTED_VLM_DATASET_CONFIG[dataset_name]["config"].get("split", "train")
    ds = dataset[split] if hasattr(dataset, "__getitem__") and split in dataset else dataset

    if require_image:
        # Keep only samples with a non-null image field (ScienceQA has both).
        try:
            ds = ds.filter(lambda ex: ex.get("image", None) is not None)
        except Exception:
            # Some dataset backends may not support filter; fall back to best-effort selection below.
            pass

    # Select the first `num_samples` entries (or fewer if dataset is smaller).
    try:
        return ds.select(range(min(num_samples, len(ds))))
    except Exception:
        # For iterable datasets without __len__/select, take first N items.
        collected = []
        for i, ex in enumerate(ds):
            if i >= num_samples:
                break
            if not require_image or ex.get("image", None) is not None:
                collected.append(ex)
        return collected


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
    processor: Any = None,
    batch_size: int = 1,
    num_samples: int = 512,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    require_image: bool = True,
) -> DataLoader:
    """Get a dataloader with the dataset name and processor of the target model.

    Args:
        dataset_name: Name of the dataset to load.
        processor: Processor used for encoding images and text data.
        batch_size: Batch size of the returned dataloader.
        num_samples: Number of samples from the dataset.
        device: Device to move returned tensors to. If None, keep on CPU.
        max_length: Optional max length for text tokenization (if supported by the processor).
        require_image: If True, keep only samples that have an image field.

    Returns:
        An instance of dataloader.
    """
    assert processor is not None, "Please provide a valid processor."

    if device is not None:
        device = torch.device(device)

    dataset = _get_vlm_dataset(dataset_name, num_samples=num_samples, require_image=require_image)

    # Legacy path: our internal image processor wrapper (e.g., Mllama).
    if isinstance(processor, MllamaImageProcessor):
        processed_dataset = dataset.map(
            processor.preprocess_function, batched=False, remove_columns=dataset.column_names
        )
        return DataLoader(
            processed_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=processor.collate_function,
        )

    # Generic HF ProcessorMixin / AutoProcessor path: tokenize & process images at collate-time.
    # This works well for models that need extra multimodal kwargs (e.g., image_flags) in addition to pixel_values.
    def _build_prompt(proc: Any, question: str) -> str:
        tok = getattr(proc, "tokenizer", None)
        # Prefer a chat template if present; it typically inserts the correct image placeholder tokens.
        if tok is not None and getattr(tok, "chat_template", None) is not None:
            try:
                return tok.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": question}],
                        }
                    ],
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        # Fallback: plain question. Many processors still correctly handle `images=...`.
        return question

    def _collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor] | dict[str, Any]:
        questions = [ex.get("question", "Describe this image.") for ex in examples]
        images = [ex.get("image", None) for ex in examples]
        prompts = [_build_prompt(processor, q) for q in questions]

        kwargs: dict[str, Any] = {"text": prompts, "images": images, "return_tensors": "pt", "padding": True}
        if max_length is not None:
            kwargs.update({"truncation": True, "max_length": max_length})

        enc = processor(**kwargs)

        # Some processors return BatchEncoding; normalize to plain dict of tensors.
        if hasattr(enc, "data"):
            enc = enc.data
        out: dict[str, Any] = dict(enc)

        # Move tensors to device if requested.
        if device is not None:
            for k, v in list(out.items()):
                if torch.is_tensor(v):
                    out[k] = v.to(device)
        return out

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

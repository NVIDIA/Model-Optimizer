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

"""Utility functions for getting samples and dataloader for different VLM calibration datasets.

This module supports both:
- Small non-streaming VLM datasets (e.g., ScienceQA)
- Large streaming VLM datasets (e.g., Nemotron-VLM-Dataset-v2) where we want to avoid downloading everything.
"""

import contextlib
import itertools
from typing import Any

import torch
from torch.utils.data import DataLoader

from .image_processor import MllamaImageProcessor

# Use dict to store the config for each dataset.
# If we want to export more options to user like target languages, we need more standardized approach like dataclass.
SUPPORTED_VLM_DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "scienceqa": {"config": {"path": "derek-thomas/ScienceQA", "split": "train"}},
    # Large multi-subset dataset (use streaming to avoid downloading the entire dataset)
    "nemotron_vlm_dataset_v2": {
        "config": {"path": "nvidia/Nemotron-VLM-Dataset-v2", "split": "train", "streaming": True},
        # Provide a sane default that is easy to extend from the CLI.
        "default_subsets": ["docvqa_cot", "chartqa_cot"],
    },
}

__all__ = ["get_supported_vlm_datasets", "get_vlm_dataset_dataloader"]


class _HFDatasetsIterableWrapper(torch.utils.data.IterableDataset):
    """Wrap a HF streaming IterableDataset to be compatible with torch DataLoader."""

    def __init__(self, hf_iterable, num_samples: int):
        super().__init__()
        self._hf_iterable = hf_iterable
        self._num_samples = num_samples

    def __iter__(self):
        return itertools.islice(iter(self._hf_iterable), self._num_samples)

    def __len__(self):
        return self._num_samples


def _extract_text_from_messages(messages: Any) -> str | None:
    """Best-effort extraction of a user text prompt from a chat-style `messages` field."""
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Common multimodal format: [{"type":"image"}, {"type":"text","text":"..."}]
            texts = [
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ]
            if texts:
                return "\n".join(texts)
    return None


def _get_vlm_dataset(
    dataset_name: str,
    num_samples: int,
    require_image: bool = True,
    subsets: list[str] | None = None,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
):
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.
        require_image: If True, keep only samples that have an image field.
        subsets: Optional subset/config names for multi-subset datasets (e.g., Nemotron-VLM-Dataset-v2).
        shuffle_buffer_size: Shuffle buffer size for streaming datasets (higher is "more random").
        seed: RNG seed for streaming dataset shuffle.

    Returns:
        A hugging face Dataset.
    """
    # Load the dataset
    if dataset_name in SUPPORTED_VLM_DATASET_CONFIG:
        from datasets import load_dataset

        cfg = SUPPORTED_VLM_DATASET_CONFIG[dataset_name]["config"].copy()
        streaming = bool(cfg.pop("streaming", False))

        if dataset_name == "nemotron_vlm_dataset_v2":
            # This dataset contains many subsets; load only the requested ones via `name=...`.
            if not subsets:
                subsets = SUPPORTED_VLM_DATASET_CONFIG[dataset_name].get("default_subsets", [])
            if not subsets:
                raise ValueError("No VLM subsets provided for nemotron_vlm_dataset_v2.")

            # Load each subset as a separate (streaming) dataset, then interleave.
            streams = [
                load_dataset(
                    cfg["path"],
                    name=subset,
                    split=cfg.get("split", "train"),
                    streaming=streaming,
                )
                for subset in subsets
            ]
            try:
                from datasets import interleave_datasets

                ds = interleave_datasets(streams)
            except Exception:
                # Fallback: round-robin by chaining (less balanced than interleave).
                ds = itertools.chain.from_iterable(streams)
        else:
            dataset = load_dataset(**cfg, streaming=streaming)
            split = cfg.get("split", "train")
            ds = dataset[split] if hasattr(dataset, "__getitem__") and split in dataset else dataset
    else:
        raise NotImplementedError(
            f"dataset {dataset_name} is not supported. Please use one of the following:"
            f" {get_supported_vlm_datasets()}."
        )

    # Streaming datasets: shuffle with bounded buffer and wrap into a torch IterableDataset.
    if dataset_name == "nemotron_vlm_dataset_v2":
        with contextlib.suppress(Exception):
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    if require_image:
        # Keep only samples with a non-null image field (ScienceQA has both).
        with contextlib.suppress(Exception):
            ds = ds.filter(
                lambda ex: ex.get("image", None) is not None or ex.get("images", None) is not None
            )

    # Select the first `num_samples` entries (or fewer if dataset is smaller).
    try:
        return ds.select(range(min(num_samples, len(ds))))
    except Exception:
        # For streaming/iterable datasets without __len__/select, wrap for DataLoader iteration.
        return _HFDatasetsIterableWrapper(ds, num_samples=num_samples)


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
    subsets: list[str] | None = None,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
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

    dataset = _get_vlm_dataset(
        dataset_name,
        num_samples=num_samples,
        require_image=require_image,
        subsets=subsets,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )

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
        questions = []
        images = []
        for ex in examples:
            q = ex.get("question")
            if q is None and "messages" in ex:
                q = _extract_text_from_messages(ex.get("messages"))
            if q is None:
                q = "Describe this image."
            questions.append(q)

            img = ex.get("image", None)
            if img is None:
                img = ex.get("images", None)
                if isinstance(img, list) and img:
                    img = img[0]
            images.append(img)
        prompts = [_build_prompt(processor, q) for q in questions]

        kwargs: dict[str, Any] = {
            "text": prompts,
            "images": images,
            "return_tensors": "pt",
            "padding": True,
        }
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

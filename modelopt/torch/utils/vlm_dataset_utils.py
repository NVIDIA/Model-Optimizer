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
import copy
import itertools
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .image_processor import BaseImageProcessor, MllamaImageProcessor
from .nemotron_vlm_dataset_utils import NemotronTarPlusJsonlIterable, list_repo_files_cached

# Use dict to store the config for each dataset.
# If we want to export more options to user like target languages, we need more standardized approach like dataclass.
SUPPORTED_VLM_DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "scienceqa": {"config": {"path": "derek-thomas/ScienceQA", "split": "train"}},
    # Large multi-subset dataset (use streaming to avoid downloading the entire dataset)
    "nemotron_vlm_dataset_v2": {
        "config": {"path": "nvidia/Nemotron-VLM-Dataset-v2", "split": "train", "streaming": True},
        # Provide a sane default that (a) includes in-repo media shards and (b) is document-centric.
        # Subsets like docvqa_cot/chartqa_cot are JSONL-only in the dataset repo and require --vlm_image_root.
        "default_subsets": ["sparsetables", "plotqa_cot", "wiki_en"],
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


def _messages_up_to_last_user(messages: Any) -> list[dict[str, Any]] | None:
    """Return messages truncated to the last user turn (inclusive)."""
    if not isinstance(messages, list):
        return None
    last_user_idx = None
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_idx = i
    if last_user_idx is None:
        return None
    trimmed = messages[: last_user_idx + 1]
    return [m for m in trimmed if isinstance(m, dict)]


def _extract_first_image_from_messages(messages: Any) -> Any:
    """Best-effort extraction of an image object from a chat-style `messages` field."""
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "image"):
                continue
            # Common keys used by HF datasets / chat templates
            for key in ("image", "images", "value", "data", "path", "image_url", "url"):
                if key in part:
                    val = part[key]
                    if isinstance(val, list) and val:
                        return val[0]
                    return val
            # Fallback: return the dict itself (some processors may accept it)
            return part
    return None


def _extract_image_ref_from_example(example: dict[str, Any]) -> Any:
    """Best-effort extraction of an image reference from a dataset example."""
    img = example.get("image")
    if img is None:
        img = example.get("images")
    if img is None:
        img = _extract_first_image_from_messages(example.get("messages"))
    return img


def _maybe_load_image(image_obj: Any, repo_id: str | None, image_root: str | Path | None) -> Any:
    """Convert common image references (path/bytes) into a PIL image if possible.

    For some streaming datasets, images are stored as file paths inside the dataset repo.
    In that case, we lazily download just the referenced files via `hf_hub_download`.
    """
    if image_obj is None:
        return None

    # If it's a list, take the first (some formats store a list for multi-image samples).
    if isinstance(image_obj, list) and image_obj:
        image_obj = image_obj[0]

    # Path-like reference
    if isinstance(image_obj, str):
        # First, try resolving against a local image root (best option for datasets that only ship JSONL refs).
        if image_root is not None:
            try:
                from PIL import Image

                local_path = Path(image_root) / image_obj
                if local_path.exists():
                    return Image.open(local_path).convert("RGB")
            except Exception:
                pass

        if repo_id is None:
            return image_obj
        try:
            from huggingface_hub import hf_hub_download
            from PIL import Image

            local_path = hf_hub_download(repo_id=repo_id, filename=image_obj, repo_type="dataset")
            return Image.open(local_path).convert("RGB")
        except Exception:
            return None

    # Dict-like reference (common in chat content items)
    if isinstance(image_obj, dict):
        # bytes payload
        if "bytes" in image_obj and isinstance(image_obj["bytes"], (bytes, bytearray)):
            try:
                from PIL import Image

                return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
            except Exception:
                return None

        # path/url-ish payloads
        for key in ("path", "image", "image_path", "file", "url", "image_url"):
            if key in image_obj and isinstance(image_obj[key], str):
                return _maybe_load_image(image_obj[key], repo_id=repo_id, image_root=image_root)

    # If it's already a PIL/numpy/torch image-like object, just return it and let the processor validate.
    return image_obj


def _get_vlm_dataset(
    dataset_name: str,
    num_samples: int,
    require_image: bool = True,
    subsets: list[str] | None = None,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    use_media_shards: bool = True,
    max_shards: int | None = None,
):
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.
        require_image: If True, keep only samples that have an image field.
        subsets: Optional subset/config names for multi-subset datasets (e.g., Nemotron-VLM-Dataset-v2).
        shuffle_buffer_size: Shuffle buffer size for streaming datasets (higher is "more random").
        seed: RNG seed for streaming dataset shuffle.
        use_media_shards: If True, prefer reading in-repo `media/shard_*.tar` files when available.
        max_shards: Optional cap on the number of tar shards to download/use.

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

            repo_id = cfg["path"]

            # Prefer in-repo media tar shards when present. HF `datasets` streaming alone does not join media.
            if use_media_shards:
                all_files = list_repo_files_cached(repo_id, repo_type="dataset")
                shard_paths: list[str] = []
                for subset in subsets:
                    prefix = f"{subset}/media/"
                    shard_paths.extend(
                        [
                            p
                            for p in all_files
                            if p.startswith(prefix) and p.lower().endswith(".tar")
                        ]
                    )

                shard_paths = sorted(set(shard_paths))
                if shard_paths:
                    return NemotronTarPlusJsonlIterable(
                        repo_id=repo_id,
                        subsets=subsets,
                        shard_paths=shard_paths,
                        num_samples=num_samples,
                        seed=seed,
                        shuffle_buffer_size=shuffle_buffer_size,
                        max_shards=max_shards,
                    )

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
                lambda ex: ex.get("image", None) is not None
                or ex.get("images", None) is not None
                or _extract_image_ref_from_example(ex) is not None
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
    processor: BaseImageProcessor | Any = None,
    batch_size: int = 1,
    num_samples: int = 512,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    require_image: bool = True,
    subsets: list[str] | None = None,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    image_root: str | Path | None = None,
    use_media_shards: bool = True,
    max_shards: int | None = None,
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

    # Optional: allow callers to set a local image root for datasets that only ship JSON references.
    # We store it on the processor instance to avoid threading it through a bunch of nested closures.
    if image_root is not None:
        setattr(processor, "_modelopt_vlm_image_root", image_root)

    if device is not None:
        device = torch.device(device)

    dataset = _get_vlm_dataset(
        dataset_name,
        num_samples=num_samples,
        require_image=require_image,
        subsets=subsets,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        use_media_shards=use_media_shards,
        max_shards=max_shards,
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
    # For Nemotron VLM datasets, we prefer to follow the model-card flow:
    #   prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #   inputs = processor(text=[prompt], images=[pil_image], ...)

    def _collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor] | dict[str, Any]:
        repo_id = None
        if dataset_name == "nemotron_vlm_dataset_v2":
            repo_id = SUPPORTED_VLM_DATASET_CONFIG[dataset_name]["config"]["path"]
        image_root = getattr(processor, "_modelopt_vlm_image_root", None)

        pairs: list[tuple[str, Any]] = []
        for ex in examples:
            messages = ex.get("messages")

            # Image extraction
            img_ref = _extract_image_ref_from_example(ex)
            img = _maybe_load_image(img_ref, repo_id=repo_id, image_root=image_root)
            if require_image and img is None:
                continue

            # Prompt extraction
            prompt = None
            tok = getattr(processor, "tokenizer", None)
            if tok is not None and messages is not None:
                trimmed = _messages_up_to_last_user(messages) or []
                # For some Nemotron-style templates, the image content expects an empty string.
                # Keep the actual image path separate for loading; blank it in the prompt message.
                prompt_msgs = copy.deepcopy(trimmed)
                for msg in prompt_msgs:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image":
                                part["image"] = ""
                with contextlib.suppress(Exception):
                    prompt = tok.apply_chat_template(
                        prompt_msgs, tokenize=False, add_generation_prompt=True
                    )

            if prompt is None:
                # Fallback: best-effort question-only prompt.
                q = ex.get("question")
                if q is None and messages is not None:
                    q = _extract_text_from_messages(messages)
                prompt = q or "Describe the image."

            pairs.append((prompt, img))

        if not pairs:
            raise ValueError(
                "No usable images found in the current batch. "
                "If you're using JSONL-only subsets (e.g., docvqa_cot/chartqa_cot), provide "
                "`--vlm_image_root <dir>` so referenced paths can be resolved. "
                "If you're using asset-included subsets, keep media shard loading enabled "
                "(default) and consider increasing `--vlm_max_shards`."
            )

        prompts, images = zip(*pairs)

        kwargs: dict[str, Any] = {
            "text": list(prompts),
            "images": list(images),
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

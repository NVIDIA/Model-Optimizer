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

"""Utility functions for getting samples and forward loop function for video datasets."""

import os
import tempfile
from typing import Any

import torch
from torch.utils.data import DataLoader

from .image_processor import BaseImageProcessor

# Use dict to store the config for each dataset.
SUPPORTED_VIDEO_DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "finevideo": {
        "config": {"path": "HuggingFaceFV/finevideo", "split": "train", "streaming": True}
    },
}

__all__ = [
    "Qwen3OmniVideoProcessor",
    "get_supported_video_datasets",
    "get_video_dataset_dataloader",
]


def _get_video_dataset(dataset_name: str, num_samples: int):
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.

    Returns:
        A hugging face Dataset.
    """
    if dataset_name in SUPPORTED_VIDEO_DATASET_CONFIG:
        from datasets import Dataset, load_dataset

        config = SUPPORTED_VIDEO_DATASET_CONFIG[dataset_name]["config"]
        is_streaming = config.get("streaming", False)

        dataset = load_dataset(**config)

        if is_streaming:
            # For streaming datasets, use take() and convert to list then Dataset
            samples = list(dataset.take(num_samples))
            return Dataset.from_list(samples)
        else:
            return dataset.select(range(num_samples))
    else:
        raise NotImplementedError(
            f"dataset {dataset_name} is not supported. Please use one of the following:"
            f" {get_supported_video_datasets()}."
        )


def get_supported_video_datasets() -> list[str]:
    """Retrieves a list of video datasets supported.

    Returns:
        A list of strings, where each string is the name of a supported dataset.

    Example usage:

    .. code-block:: python

        from modelopt.torch.utils import get_supported_video_datasets

        print("Supported video datasets:", get_supported_video_datasets())
    """
    return list(SUPPORTED_VIDEO_DATASET_CONFIG.keys())


def get_video_dataset_dataloader(
    dataset_name: str = "finevideo",
    processor: "Qwen3OmniVideoProcessor" = None,
    batch_size: int = 1,
    num_samples: int = 512,
    cache_dir: str | None = None,
) -> DataLoader:
    """Get a dataloader with the dataset name and processor of the target model.

    Args:
        dataset_name: Name of the dataset to load.
        processor: Processor used for encoding video and text data.
        batch_size: Batch size of the returned dataloader.
        num_samples: Number of samples from the dataset.
        cache_dir: Directory to cache the processed dataset. Defaults to a temp directory.
            If the cache exists, it will be loaded instead of reprocessing.

    Returns:
        An instance of dataloader.
    """
    assert processor is not None, "Please provide a valid processor."

    # Default cache_dir to temp directory
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "modelopt_video_dataset_cache")

    processed_dataset = None

    # Try to load from cache
    if cache_dir is not None:
        from datasets import load_from_disk

        cache_path = os.path.join(cache_dir, f"{dataset_name}_n{num_samples}_processed")
        if os.path.exists(cache_path):
            try:
                processed_dataset = load_from_disk(cache_path)
                print(f"Loaded processed dataset from cache: {cache_path}")
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}. Reprocessing...")
                processed_dataset = None

    # Process dataset if not loaded from cache
    if processed_dataset is None:
        dataset = _get_video_dataset(dataset_name, num_samples=num_samples)
        # Apply the preprocessing function to the dataset
        processed_dataset = dataset.map(
            processor.preprocess_function, batched=False, remove_columns=dataset.column_names
        )

        # Save to cache if cache_dir is provided
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            # Use num_shards=1 to avoid off-by-one sharding bug with complex nested structures
            processed_dataset.save_to_disk(cache_path, num_shards=1)
            print(f"Saved processed dataset to cache: {cache_path}")

    # Create DataLoader with the custom collate function
    return DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=processor.collate_function,
    )


class Qwen3OmniVideoProcessor(BaseImageProcessor):
    """Video processor for Qwen3-Omni multimodal model with finevideo dataset support."""

    def __init__(self, tokenizer, device="cuda", dtype=None, use_audio_in_video=True):
        """Constructor.

        Args:
            tokenizer: The Qwen3OmniMoeProcessor for tokenizing and processing inputs.
            device: Device to move tensors to.
            dtype: dtype for float tensors (e.g., torch.bfloat16). If None, uses default.
            use_audio_in_video: Whether to extract and use audio from video files.
        """
        super().__init__(tokenizer, device)
        self.dtype = dtype
        self.use_audio_in_video = use_audio_in_video
        self._temp_dir = tempfile.mkdtemp(prefix="qwen3omni_video_")
        self._video_counter = 0
        # Try to import qwen_omni_utils for multimodal processing
        try:
            from qwen_omni_utils import process_mm_info

            self.process_mm_info = process_mm_info
        except ImportError:
            raise ImportError(
                "qwen_omni_utils is required for Qwen3OmniVideoProcessor. "
                "Please install it from https://github.com/QwenLM/Qwen3-Omni"
            )

    def _save_video_bytes_to_file(self, video_bytes: bytes) -> str:
        """Save video bytes to a temporary file and return the path.

        Args:
            video_bytes: Raw video bytes (e.g., from finevideo's 'mp4' field).

        Returns:
            Path to the temporary video file.
        """
        video_path = os.path.join(self._temp_dir, f"video_{self._video_counter}.mp4")
        self._video_counter += 1
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        return video_path

    def preprocess_function(self, examples):
        """Preprocess function for Qwen3-Omni with video support.

        Handles both standard video paths and raw video bytes (finevideo format).
        """
        # Get question/prompt - finevideo has metadata in 'json' field
        if "json" in examples and examples["json"] is not None:
            metadata = examples["json"]
            # Try to get a meaningful question from metadata
            category = metadata.get("content_fine_category", "")
            question = f"/no_think Describe what is happening in this video in detail. Category hint: {category}"
        else:
            question = examples.get("question", "/no_think Describe this video in detail.")

        # Build conversation in Qwen format
        content = []

        # Handle video - check for raw bytes (finevideo format) or path
        video_path = None
        if examples.get("mp4") is not None:
            # finevideo format: raw video bytes in 'mp4' field
            video_path = self._save_video_bytes_to_file(examples["mp4"])
        elif examples.get("video") is not None:
            # Standard format: video path or URL
            video_path = examples["video"]

        if video_path is not None:
            content.append({"type": "video", "video": video_path})

        content.append({"type": "text", "text": question})

        conversation = [{"role": "user", "content": content}]

        # Apply chat template (tokenize=False to get string)
        text = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Extract multimodal info using qwen_omni_utils
        audios, images, videos = self.process_mm_info(
            conversation, use_audio_in_video=self.use_audio_in_video
        )

        # Process inputs with the processor
        values = self.tokenizer(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        # Define all possible keys to ensure consistent schema for Arrow serialization
        all_keys = [
            "input_ids",
            "attention_mask",
            "pixel_values_videos",
            "video_grid_thw",
            "video_second_per_grid",
            "feature_attention_mask",
            "input_features",
        ]

        # Convert tensors to lists for Arrow serialization compatibility
        # Tensor conversion back happens in collate_function
        result = dict.fromkeys(all_keys)  # Initialize all keys to None
        for key, val in values.items():
            if val is not None and hasattr(val, "tolist"):
                result[key] = val.tolist()
            elif val is not None:
                result[key] = val

        return result

    def collate_function(self, batch):
        """Collate function to process inputs during data loading."""
        result = {}

        # Take first item from batch (batch_size handling)
        first = batch[0]

        # Convert lists to tensors and move to device
        if first.get("input_ids") is not None:
            result["input_ids"] = torch.LongTensor(first["input_ids"]).to(self.device)
        if first.get("attention_mask") is not None:
            result["attention_mask"] = torch.LongTensor(first["attention_mask"]).to(self.device)

        # Handle pixel values for video frames
        if first.get("pixel_values_videos") is not None:
            pv = torch.tensor(first["pixel_values_videos"])
            if self.dtype is not None:
                pv = pv.to(self.dtype)
            result["pixel_values_videos"] = pv.to(self.device)

        # Handle video grid thw (tile height width info)
        if first.get("video_grid_thw") is not None:
            result["video_grid_thw"] = torch.LongTensor(first["video_grid_thw"]).to(self.device)

        # Handle video second per grid (temporal info for rope)
        if first.get("video_second_per_grid") is not None:
            result["video_second_per_grid"] = torch.tensor(first["video_second_per_grid"]).to(
                self.device
            )

        # Handle audio features if present
        if first.get("feature_attention_mask") is not None:
            result["feature_attention_mask"] = torch.LongTensor(first["feature_attention_mask"]).to(
                self.device
            )
        if first.get("input_features") is not None:
            inp_feat = torch.tensor(first["input_features"])
            if self.dtype is not None:
                inp_feat = inp_feat.to(self.dtype)
            result["input_features"] = inp_feat.to(self.device)

        # Pass use_audio_in_video flag to model.generate() for Qwen3Omni
        result["use_audio_in_video"] = self.use_audio_in_video

        return result

    def cleanup(self):
        """Clean up temporary video files."""
        import shutil

        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)

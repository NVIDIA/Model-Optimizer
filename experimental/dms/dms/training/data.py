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

"""Data pipeline for DMS training: dataset loading, tokenization, blending, and concatenation.

To add a new dataset:
1. Define a filter_fn and extract_fn for your dataset
2. Create a DatasetInfo instance at the bottom of this file
3. Reference it by name in your YAML config's data.blend field
   (e.g. "MyNewDataset:0.5,OpenR1Math220k:0.5")
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from dms.logging import get_logger

logger = get_logger("Data")

# Cache directory structure constants
_CACHE_DIR_NAME = "datasets_cache"
_CACHE_PLAIN = "plain"
_CACHE_TOKENIZED = "tokenized"
_CACHE_SHUFFLED = "shuffled"
_CACHE_CONCATENATED = "concatenated"

# Default parallelism for dataset operations
_DEFAULT_FILTER_NUM_PROC = 8
_DEFAULT_TOKENIZE_NUM_PROC = 32


# =============================================================================
# Dataset pipeline utilities
# =============================================================================


def _get_module_dir() -> Path:
    """Return the directory containing this module."""
    return Path(__file__).parent


def _load_or_create_cached_dataset(
    cache_path: Path,
    create_fn: Callable[[], Dataset],
    description: str,
    caching_enabled: bool,
) -> Dataset:
    """Load dataset from cache if available, otherwise create and optionally cache it.

    Args:
        cache_path: Path to the cached dataset.
        create_fn: Function that creates the dataset if not cached.
        description: Human-readable description for logging.
        caching_enabled: Whether to use/save cache.

    Returns:
        The loaded or newly created dataset.
    """
    if caching_enabled and cache_path.exists():
        logger.info(f"Loading {description} from cache: {cache_path}")
        return load_from_disk(str(cache_path))

    logger.info(f"Processing {description}")
    dataset = create_fn()

    if caching_enabled:
        logger.info(f"Saving {description} to cache: {cache_path}")
        dataset.save_to_disk(str(cache_path))

    return dataset


@dataclass
class ConfiguredTokenizer:
    """A tokenizer with pre-configured kwargs for chat template and encoding.

    Attributes:
        tokenizer: The underlying HuggingFace tokenizer.
        apply_chat_template_kwargs: Additional kwargs for apply_chat_template.
        encode_kwargs: Additional kwargs for encode.
    """

    tokenizer: PreTrainedTokenizerBase
    apply_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    encode_kwargs: dict[str, Any] = field(default_factory=dict)

    def apply_chat_template(self, conversation: list[dict[str, str]]) -> str:
        """Apply chat template to a conversation without tokenizing."""
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, **self.apply_chat_template_kwargs
        )

    def encode(self, prompt: str) -> list[int]:
        """Encode a prompt string to token IDs."""
        return self.tokenizer.encode(prompt, **self.encode_kwargs)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to a string."""
        return self.tokenizer.decode(token_ids)

    def get_hash(self) -> str:
        """Generate a unique hash based on tokenizer configuration."""
        config = {
            "tokenizer": self.tokenizer.name_or_path,
            "apply_chat_template_kwargs": self.apply_chat_template_kwargs,
            "encode_kwargs": self.encode_kwargs,
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


@dataclass
class DatasetInfo:
    """Configuration for loading and processing a dataset.

    To add a new dataset, create a DatasetInfo instance with:
    - args/kwargs for HuggingFace load_dataset
    - filter_fn to select relevant samples
    - extract_fn to transform samples into chat format

    See the dataset definitions at the bottom of this file for examples.

    Attributes:
        args: Positional arguments for load_dataset.
        kwargs: Keyword arguments for load_dataset.
        filter_fn: Function to filter dataset samples.
        extract_fn: Function to extract/transform samples into chat format.
        caching_enabled: Whether to cache intermediate results to disk.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    filter_fn: Callable[[Any], bool]
    extract_fn: Callable[[Any], Any]
    caching_enabled: bool = True

    def get_str_identifier(self) -> str:
        """Return a JSON string uniquely identifying this dataset configuration."""
        identifier_parts = {
            "args": self.args,
            "kwargs": self.kwargs,
            "filter_fn": self.filter_fn.__name__,
            "extract_fn": self.extract_fn.__name__,
        }
        return json.dumps(identifier_parts, sort_keys=True)

    def get_hash(self) -> str:
        """Generate a unique hash based on dataset configuration."""
        return hashlib.sha256(self.get_str_identifier().encode()).hexdigest()

    def _get_cache_base_path(self) -> Path:
        """Return the base cache directory path for this dataset."""
        return _get_module_dir() / _CACHE_DIR_NAME / self.get_hash()

    def _load_plain_dataset(self) -> Dataset:
        """Load and process the raw dataset (filter + extract)."""
        cache_path = self._get_cache_base_path() / _CACHE_PLAIN

        def create_dataset() -> Dataset:
            dataset = load_dataset(*self.args, **self.kwargs)
            dataset = dataset.filter(self.filter_fn, num_proc=_DEFAULT_FILTER_NUM_PROC)
            dataset = dataset.map(self.extract_fn, num_proc=_DEFAULT_FILTER_NUM_PROC)
            return dataset

        return _load_or_create_cached_dataset(
            cache_path=cache_path,
            create_fn=create_dataset,
            description=f"plain dataset {self.get_str_identifier()}",
            caching_enabled=self.caching_enabled,
        )

    def _tokenize_dataset(
        self,
        dataset: Dataset,
        configured_tokenizer: ConfiguredTokenizer,
    ) -> Dataset:
        """Apply tokenization to a dataset."""
        cache_path = (
            self._get_cache_base_path() / _CACHE_TOKENIZED / configured_tokenizer.get_hash()
        )

        def create_tokenized() -> Dataset:
            def apply_tokenizer(sample: dict[str, Any]) -> dict[str, Any]:
                conversation = sample["conversation"]
                prompt = configured_tokenizer.apply_chat_template(conversation)
                encoded_prompt = configured_tokenizer.encode(prompt)
                return {
                    "conversation": conversation,
                    "prompt": prompt,
                    "encoded_prompt": encoded_prompt,
                }

            return dataset.map(apply_tokenizer, num_proc=_DEFAULT_TOKENIZE_NUM_PROC)

        return _load_or_create_cached_dataset(
            cache_path=cache_path,
            create_fn=create_tokenized,
            description=f"tokenized dataset {self.get_str_identifier()}",
            caching_enabled=self.caching_enabled,
        )

    def _shuffle_dataset(
        self,
        dataset: Dataset,
        configured_tokenizer: ConfiguredTokenizer,
        shuffle_seed: int,
    ) -> Dataset:
        """Shuffle dataset with a given seed."""
        cache_path = (
            self._get_cache_base_path()
            / _CACHE_SHUFFLED
            / configured_tokenizer.get_hash()
            / f"shuffle_seed_{shuffle_seed}"
        )

        return _load_or_create_cached_dataset(
            cache_path=cache_path,
            create_fn=lambda: dataset.shuffle(seed=shuffle_seed),
            description=f"shuffled dataset {self.get_str_identifier()}",
            caching_enabled=self.caching_enabled,
        )

    def _concatenate_dataset(
        self,
        dataset: Dataset,
        configured_tokenizer: ConfiguredTokenizer,
        shuffle_seed: int | None,
        concat_up_to: int,
        concat_always_start_new: bool,
    ) -> Dataset:
        """Concatenate samples to create fixed-length contexts.

        Args:
            dataset: The tokenized dataset.
            configured_tokenizer: Tokenizer used (for cache path).
            shuffle_seed: Shuffle seed used (for cache path).
            concat_up_to: Target context length in tokens.
            concat_always_start_new: If True, discard tokens that overflow;
                otherwise, carry them to the next context.

        Returns:
            Dataset with concatenated samples.
        """
        cache_path = (
            self._get_cache_base_path()
            / _CACHE_CONCATENATED
            / configured_tokenizer.get_hash()
            / f"shuffle_seed_{shuffle_seed}"
            / f"concat_up_to_{concat_up_to}"
            / f"always_start_new_{concat_always_start_new}"
        )

        def create_concatenated() -> Dataset:
            concatenated_samples: list[dict[str, Any]] = []
            current_context: dict[str, Any] = {
                "prompt": "",
                "encoded_prompt": [],
                "num_samples": 0,
            }

            for sample in tqdm(dataset, desc="Concatenating dataset"):
                current_context["prompt"] += sample["prompt"]
                current_context["encoded_prompt"] += sample["encoded_prompt"]
                current_context["num_samples"] += 1

                while len(current_context["encoded_prompt"]) >= concat_up_to:
                    # Store the full context before trimming
                    full_encoded = current_context["encoded_prompt"]
                    current_context["encoded_prompt_untrimmed"] = full_encoded
                    current_context["encoded_prompt"] = full_encoded[:concat_up_to]
                    concatenated_samples.append(current_context)

                    # Handle overflow tokens
                    remaining_tokens = full_encoded[concat_up_to:]
                    current_context = {
                        "prompt": "",
                        "encoded_prompt": [],
                        "num_samples": 0,
                    }

                    if not concat_always_start_new and remaining_tokens:
                        current_context["encoded_prompt"] = remaining_tokens
                        current_context["num_samples"] = 1

            result = Dataset.from_list(concatenated_samples)
            logger.info(f"Created concatenated dataset with {len(result)} samples")
            return result

        return _load_or_create_cached_dataset(
            cache_path=cache_path,
            create_fn=create_concatenated,
            description=f"concatenated dataset {self.get_str_identifier()}",
            caching_enabled=self.caching_enabled,
        )

    def get_dataset(
        self,
        configured_tokenizer: ConfiguredTokenizer,
        concat_up_to: int | None,
        concat_always_start_new: bool = True,
        shuffle_seed: int | None = None,
    ) -> Dataset:
        """Load and process dataset through the full pipeline.

        The pipeline stages are:
        1. Load raw dataset and apply filter/extract functions
        2. Tokenize using the configured tokenizer
        3. Optionally shuffle with given seed
        4. Optionally concatenate samples to fixed-length contexts

        Each stage is cached independently for efficient reprocessing.

        Args:
            configured_tokenizer: Tokenizer configuration for encoding.
            concat_up_to: Target context length in tokens. If None, no concatenation.
            concat_always_start_new: If True, discard overflow tokens when concatenating;
                otherwise, carry them to the next context.
            shuffle_seed: Random seed for shuffling. If None, no shuffling.

        Returns:
            Processed HuggingFace Dataset.
        """
        # Stage 1: Load and filter/extract
        dataset = self._load_plain_dataset()

        # Stage 2: Tokenize
        dataset = self._tokenize_dataset(dataset, configured_tokenizer)

        # Stage 3: Shuffle (optional)
        if shuffle_seed is not None:
            dataset = self._shuffle_dataset(dataset, configured_tokenizer, shuffle_seed)

        # Stage 4: Concatenate (optional)
        if concat_up_to is not None:
            dataset = self._concatenate_dataset(
                dataset,
                configured_tokenizer,
                shuffle_seed,
                concat_up_to,
                concat_always_start_new,
            )

        return dataset


# =============================================================================
# Data blending
# =============================================================================


@dataclass
class DataBlendElement:
    """A single dataset element with its blend weight."""

    dataset: DatasetInfo
    weight: float  # weight in the datablend (should be > 0.0)

    def __post_init__(self):
        assert self.weight > 0.0, f"weight: {self.weight} is not greater than 0.0"


class DataBlend:
    """Blends multiple datasets with configurable weights.

    Args:
        data_blend_elements: list of datasets along with their weights in the datablend
        configured_tokenizer: the tokenizer to use for the dataset
        train_samples: the number of samples to provide
        seed: used for datasets and datablend shuffling
        concat_up_to: each sample is concatenated to match this length
        concat_always_start_new: if true then suffixes of documents
            that do not fit in concat_up_to context will be discarded,
            otherwise they will be put at the beginning of the next context.
    """

    def __init__(
        self,
        data_blend_elements: list[DataBlendElement],
        configured_tokenizer: ConfiguredTokenizer,
        train_samples: int,
        seed: int = 42,
        concat_up_to: int | None = None,
        concat_always_start_new: bool = True,
    ):
        """Initialize the data blend with weighted datasets."""
        self.configured_tokenizer = configured_tokenizer
        logger.info(f"Configured tokenizer: {self.configured_tokenizer}")

        logger.info(f"Initializing DataBlend with {len(data_blend_elements)} data blend elements")

        self.dataset_weights = []
        self.datasets = []
        self.dataset_iterators = []

        for dbe in tqdm(data_blend_elements, desc="Processing data blend elements"):
            logger.info(f"Data blend element: {dbe.dataset.get_str_identifier()}")
            logger.info(f"Data blend element weight: {dbe.weight}")
            self.dataset_weights.append(dbe.weight)
            self.datasets.append(
                dbe.dataset.get_dataset(
                    configured_tokenizer=self.configured_tokenizer,
                    concat_up_to=concat_up_to,
                    shuffle_seed=seed,
                    concat_always_start_new=concat_always_start_new,
                )
            )
            self.dataset_iterators.append(0)

        self.normalized_weights = np.array(self.dataset_weights, dtype=np.float64)
        self.normalized_weights /= self.normalized_weights.sum()

        # self[id] ->  dataset id
        self.sample_mapping = []
        for i, nw in enumerate(self.normalized_weights):
            nw = nw.item()
            self.sample_mapping.append(np.full(int(nw * train_samples), i))

        self.sample_mapping = np.concatenate(self.sample_mapping, axis=0)
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(self.sample_mapping)

    def __len__(self):
        return len(self.sample_mapping)

    def __getitem__(self, index: int):
        ds_idx = self.sample_mapping[index]
        ds_sample_idx = self.dataset_iterators[ds_idx]
        ds_sample_idx %= len(self.datasets[ds_idx])

        ds_sample = self.datasets[ds_idx][ds_sample_idx]
        self.dataset_iterators[ds_idx] += 1

        input_ids = np.array(ds_sample["encoded_prompt"], dtype=np.int64)

        ds_sample_augmented = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids, dtype=bool),
        }
        return ds_sample_augmented


# =============================================================================
# Dataset definitions
#
# To add a new dataset:
# 1. Define filter_fn and extract_fn functions
# 2. Create a DatasetInfo instance
# 3. Reference it by name in YAML config data.blend field
# =============================================================================


## OpenR1-Math-220k
def openr1_math_220k_filter_fn(ds_elem: Any) -> bool:
    """Filter function to keep only samples with verified correct solutions."""
    return any(ds_elem["correctness_math_verify"])


def openr1_math_220k_extract_fn(ds_elem: Any) -> dict[str, Any]:
    """Extract problem-solution chat format from a dataset element."""
    problem = ds_elem["problem"]
    solution = None
    for gen, correctness in zip(ds_elem["generations"], ds_elem["correctness_math_verify"]):
        if correctness:
            solution = gen

    assert solution is not None, (
        "solution is None, filtering should remove problems without correct solutions"
    )

    chat = {
        "conversation": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
    }

    return chat


OpenR1Math220k = DatasetInfo(
    args=("open-r1/OpenR1-Math-220k",),
    kwargs={"split": "train"},
    filter_fn=openr1_math_220k_filter_fn,
    extract_fn=openr1_math_220k_extract_fn,
)

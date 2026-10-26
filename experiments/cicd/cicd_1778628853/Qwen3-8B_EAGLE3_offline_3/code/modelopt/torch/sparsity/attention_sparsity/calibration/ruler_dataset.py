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

# Copied and Adapted from https://github.com/NVIDIA/RULER
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""RULER dataset for sparse attention calibration.

This module contains the RULER dataset builder and core generation logic adapted
from the RULER benchmark (https://github.com/NVIDIA/RULER) for calibration purposes.
The generation logic closely follows the official RULER implementation to ensure
dataset consistency.

Key adaptations from official RULER:
- Converted from CLI scripts to library functions
- Works with HuggingFace tokenizers directly
- Removed file I/O, returns data structures
- Simplified for calibration use case (primarily NIAH tasks)
"""

import hashlib
import json
import logging
import random
import re
import string
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


# Needle/Haystack template from official RULER
NEEDLE_TEMPLATE = "One of the special magic {type_needle_v} for {key} is: {value}."

# Depth positions for needle insertion (from official RULER)
DEPTHS = [
    0,
    2,
    5,
    7,
    10,
    12,
    15,
    18,
    20,
    23,
    25,
    28,
    30,
    33,
    35,
    38,
    40,
    43,
    45,
    48,
    50,
    53,
    55,
    58,
    60,
    62,
    65,
    67,
    70,
    72,
    75,
    77,
    80,
    82,
    85,
    87,
    90,
    92,
    95,
    97,
    100,
]


def _load_paul_graham_essays_from_files(data_dir: Path) -> str:
    """Load Paul Graham essays from local files.

    Reads essay .txt files from data_dir/essays.
    Files must be downloaded first using download_ruler_data.sh.

    Args:
        data_dir: Base directory for RULER data (contains an 'essays' subdir with .txt files).

    Returns:
        Combined essay text

    Raises:
        RuntimeError: If essays directory doesn't exist or is empty
    """
    essays_dir = data_dir / "essays"
    if not essays_dir.exists():
        raise RuntimeError(
            f"Essays directory not found at {essays_dir}.\n"
            "Please run the download script first:\n"
            "  bash examples/llm_sparsity/attention_sparsity/download_ruler_data.sh"
        )

    essay_files = list(essays_dir.glob("*.txt"))
    if not essay_files:
        raise RuntimeError(
            f"No essay files found in {essays_dir}.\n"
            "Please run the download script first:\n"
            "  bash examples/llm_sparsity/attention_sparsity/download_ruler_data.sh"
        )

    logger.info(f"Loading {len(essay_files)} Paul Graham essays from {essays_dir}...")

    all_essays = []
    for filepath in essay_files:
        text = filepath.read_text()
        all_essays.append(text)

    combined_text = " ".join(all_essays)
    logger.info(f"Loaded {len(all_essays)} essays successfully")

    return combined_text


def _load_paul_graham_essays(data_dir: Path) -> str:
    """Load Paul Graham essays from local files.

    Essay files must be downloaded first using download_ruler_data.sh.

    Args:
        data_dir: Base directory for RULER data (contains an 'essays' subdir).

    Returns:
        Essay text as string
    """
    essay_text = _load_paul_graham_essays_from_files(data_dir)
    return re.sub(r"\s+", " ", essay_text)


def _load_word_lists():
    """Load word lists for random word generation.

    Returns:
        List of words (adj-noun combinations)
    """
    import wonderwords

    # Load wonderwords lists (same as official RULER)
    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
    words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
    words = sorted(set(words))
    return words


# Global word list (loaded once)
_WORD_LIST = None


def generate_random_number(num_digits=7) -> str:
    """Generate random number (from official RULER)."""
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))


def generate_random_word() -> str:
    """Generate random word (from official RULER)."""
    global _WORD_LIST
    if _WORD_LIST is None:
        _WORD_LIST = _load_word_lists()
    return random.choice(_WORD_LIST)


def generate_random_uuid() -> str:
    """Generate random UUID (from official RULER)."""
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def generate_random(type_needle: str) -> str:
    """Generate random needle value based on type (from official RULER).

    Args:
        type_needle: Type of needle ('numbers', 'words', 'uuids')

    Returns:
        Random value as string
    """
    if type_needle == "numbers":
        return generate_random_number()
    elif type_needle == "words":
        return generate_random_word()
    elif type_needle == "uuids":
        return generate_random_uuid()
    else:
        raise ValueError(f"Unknown needle type: {type_needle}")


def generate_niah_sample(
    num_haystack: int,
    tokenizer,
    template: str,
    answer_prefix: str,
    tokens_to_generate: int = 128,
    type_haystack: str = "essay",
    type_needle_k: str = "words",
    type_needle_v: str = "numbers",
    num_needle_k: int = 1,
    num_needle_v: int = 1,
    num_needle_q: int = 1,
    random_seed: int = 42,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a single NIAH (Needle in a Haystack) sample.

    This function implements the core generation logic from official RULER's niah.py,
    adapted to work as a library function.

    Args:
        num_haystack: Number of haystack items/words
        tokenizer: HuggingFace tokenizer (AutoTokenizer instance)
        template: NIAH question template
        answer_prefix: Answer prefix template
        tokens_to_generate: Expected number of generation tokens
        type_haystack: Type of haystack ('essay', 'noise', 'needle')
        type_needle_k: Type of needle keys ('numbers', 'words', 'uuids')
        type_needle_v: Type of needle values ('numbers', 'words', 'uuids')
        num_needle_k: Number of needle keys
        num_needle_v: Number of needle values per key
        num_needle_q: Number of needles to query
        random_seed: Random seed for this sample
        data_dir: Base directory for RULER data (required when type_haystack='essay').
            Must contain an 'essays' subdir with Paul Graham .txt files.

    Returns:
        Dictionary with 'input', 'outputs', 'length' keys
    """
    import nltk
    from nltk.tokenize import sent_tokenize

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    if random_seed is not None:
        random.seed(random_seed)

    # Ensure num_needle_k >= num_needle_q
    num_needle_k = max(num_needle_k, num_needle_q)

    # Generate needles (keys and values)
    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        keys.append(generate_random(type_needle_k))
        value = []
        for _ in range(num_needle_v):
            value.append(generate_random(type_needle_v))
            needles.append(
                NEEDLE_TEMPLATE.format(
                    type_needle_v=type_needle_v,
                    key=keys[-1],
                    value=value[-1],
                )
            )
        values.append(value)

    random.shuffle(needles)

    # Generate context based on haystack type
    if type_haystack == "essay":
        if data_dir is None:
            raise ValueError(
                "data_dir is required when type_haystack='essay'. "
                "Pass the path to the RULER data directory (containing an 'essays' subdir)."
            )
        # Load essay corpus
        essay_text = _load_paul_graham_essays(Path(data_dir))
        haystack = essay_text.split(" ")

        # Create text from haystack
        if num_haystack <= len(haystack):
            text = " ".join(haystack[:num_haystack])
        else:
            # Repeat haystack as needed
            repeats = (num_haystack + len(haystack) - 1) // len(haystack)
            text = " ".join((haystack * repeats)[:num_haystack])

        # Insert needles at various depths
        document_sents = sent_tokenize(text.strip())
        insertion_positions = [
            0,
            *sorted(
                int(len(document_sents) * (depth / 100))
                for depth in random.sample(DEPTHS, len(needles))
            ),
            len(document_sents),
        ]

        document_sents_list = []
        for i in range(1, len(insertion_positions)):
            last_pos = insertion_positions[i - 1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i - 1 < len(needles):
                document_sents_list.append(needles[i - 1])

        context = " ".join(document_sents_list)

    elif type_haystack == "noise":
        haystack_sent = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        sentences = [haystack_sent] * num_haystack
        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)

    elif type_haystack == "needle":
        sentences = [
            NEEDLE_TEMPLATE.format(
                type_needle_v=type_needle_v,
                key=generate_random(type_needle_k),
                value=generate_random(type_needle_v),
            )
            for _ in range(num_haystack)
        ]

        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)

    else:
        raise ValueError(f"Unknown haystack type: {type_haystack}")

    # Generate query and answer
    indices = random.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]

    # Format template (adjust for singular vs plural)
    type_needle_v_display = type_needle_v
    formatted_template = template
    if num_needle_q * num_needle_v == 1:
        formatted_template = formatted_template.replace("Some", "A")
        formatted_template = formatted_template.replace("are all", "is")
        formatted_template = formatted_template.replace("are", "is")
        formatted_template = formatted_template.replace("answers", "answer")
        type_needle_v_display = type_needle_v[:-1]  # remove "s"

    input_text = formatted_template.format(
        type_needle_v=type_needle_v_display,
        context=context,
        query=query,
    )

    # Add answer prefix
    formatted_answer_prefix = answer_prefix.format(
        type_needle_v=type_needle_v_display,
        query=query,
    )
    input_text = input_text + formatted_answer_prefix

    # Calculate actual length
    if hasattr(tokenizer, "encode"):
        # HuggingFace tokenizer
        tokens = tokenizer.encode(input_text, add_special_tokens=False)
        length = len(tokens) + tokens_to_generate
    else:
        # Fallback
        length = len(input_text.split()) + tokens_to_generate

    return {
        "input": input_text,
        "outputs": answers,
        "length": length,
    }


def find_optimal_haystack_size(
    tokenizer,
    max_seq_length: int,
    template: str,
    answer_prefix: str,
    tokens_to_generate: int = 128,
    type_haystack: str = "essay",
    data_dir: Path | None = None,
    **kwargs,
) -> int:
    """Find optimal haystack size using binary search (from official RULER).

    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        tokens_to_generate: Expected generation tokens
        type_haystack: Type of haystack
        template: NIAH question template
        answer_prefix: Answer prefix template
        data_dir: Base directory for RULER data (required when type_haystack='essay').
        **kwargs: Additional arguments for generate_niah_sample

    Returns:
        Optimal number of haystack items
    """
    # Determine incremental step based on haystack type
    if type_haystack == "essay":
        incremental = 500
    elif type_haystack in ["noise", "needle"]:
        incremental = 25
    else:
        incremental = 100

    if max_seq_length < 4096 and type_haystack != "essay":
        incremental = 5

    # Estimate tokens per haystack item
    sample = generate_niah_sample(
        incremental,
        tokenizer,
        template,
        answer_prefix,
        tokens_to_generate,
        type_haystack=type_haystack,
        data_dir=data_dir,
        **kwargs,
    )

    if hasattr(tokenizer, "encode"):
        sample_tokens = len(tokenizer.encode(sample["input"], add_special_tokens=False))
    else:
        sample_tokens = len(sample["input"].split())

    tokens_per_haystack = sample_tokens / incremental
    estimated_max = int((max_seq_length / tokens_per_haystack) * 3)

    # Binary search for optimal size
    lower_bound = incremental
    upper_bound = max(estimated_max, incremental * 2)
    optimal_num_haystack = None

    logger.debug(f"Estimated {tokens_per_haystack:.1f} tokens per haystack")
    logger.debug(f"Binary search bounds: {lower_bound} to {upper_bound}")

    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        sample = generate_niah_sample(
            mid,
            tokenizer,
            template,
            answer_prefix,
            tokens_to_generate,
            type_haystack=type_haystack,
            data_dir=data_dir,
            **kwargs,
        )
        total_tokens = sample["length"]

        logger.debug(f"Testing haystack size: {mid}, tokens: {total_tokens}/{max_seq_length}")

        if total_tokens <= max_seq_length:
            optimal_num_haystack = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1

    final_size = optimal_num_haystack if optimal_num_haystack is not None else incremental
    logger.debug(f"Optimal haystack size: {final_size}")

    return final_size


def _generate_target_lengths(
    max_seqlen: int, num_length_bins: int = 4, min_seqlen: int = 1024
) -> list[int]:
    """Generate target lengths as descending powers of 2.

    Args:
        max_seqlen: Maximum sequence length
        num_length_bins: Maximum number of length bins to generate
        min_seqlen: Minimum sequence length threshold

    Returns:
        List of target lengths in descending order

    Examples:
        >>> _generate_target_lengths(32768, 4)
        [32768, 16384, 8192, 4096]
        >>> _generate_target_lengths(2048, 4)
        [2048, 1024]
    """
    target_lengths = []
    current = max_seqlen

    for _ in range(num_length_bins):
        if current < min_seqlen:
            break
        target_lengths.append(current)
        current = current // 2

    return target_lengths


@dataclass
class RulerTask:
    """Configuration for a RULER task."""

    name: str
    task_type: str  # niah, variable_tracking, freq_words_extraction, qa
    tokens_to_generate: int
    template: str
    answer_prefix: str
    args: dict[str, Any]


# Task configurations based on RULER benchmark
RULER_TASKS = {
    "niah_multikey_2": RulerTask(
        name="niah_multikey_2",
        task_type="niah",
        tokens_to_generate=128,
        template=(
            "Some special magic {type_needle_v} are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
            "{context}\n"
            "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
        ),
        answer_prefix=(
            " The special magic {type_needle_v} for {query} mentioned in the provided text are"
        ),
        args={
            "type_haystack": "needle",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
    ),
    "niah_multikey_3": RulerTask(
        name="niah_multikey_3",
        task_type="niah",
        tokens_to_generate=128,
        template=(
            "Some special magic {type_needle_v} are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
            "{context}\n"
            "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
        ),
        answer_prefix=(
            " The special magic {type_needle_v} for {query} mentioned in the provided text are"
        ),
        args={
            "type_haystack": "needle",
            "type_needle_k": "uuids",
            "type_needle_v": "uuids",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
    ),
    "vt": RulerTask(
        name="vt",
        task_type="variable_tracking",
        tokens_to_generate=30,
        template=(
            "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
            "{context}\n"
            "Question: Find all variables that are assigned the value {query} in the text above."
        ),
        answer_prefix=(
            " Answer: According to the chain(s) of variable assignment in the text above, "
            "{num_v} variables are assigned the value {query}, they are: "
        ),
        args={"num_chains": 1, "num_hops": 4},
    ),
    "fwe": RulerTask(
        name="fwe",
        task_type="freq_words_extraction",
        tokens_to_generate=50,
        template=(
            "Read the following coded text and track the frequency of each coded word. "
            "Find the three most frequently appeared coded words. {context}\n"
            "Question: Do not provide any explanation. Please ignore the dots '....'. "
            "What are the three most frequently appeared words in the above coded text?"
        ),
        answer_prefix=(
            " Answer: According to the coded text above, "
            "the three most frequently appeared words are:"
        ),
        args={"alpha": 2.0},
    ),
    "qa_1": RulerTask(
        name="qa_1",
        task_type="qa",
        tokens_to_generate=32,
        template=(
            "Answer the question based on the given documents. "
            "Only give me the answer and do not output any other words.\n\n"
            "The following are given documents.\n\n{context}\n\n"
            "Answer the question based on the given documents. "
            "Only give me the answer and do not output any other words.\n\n"
            "Question: {query}"
        ),
        answer_prefix=" Answer:",
        args={"dataset": "squad"},
    ),
    "qa_2": RulerTask(
        name="qa_2",
        task_type="qa",
        tokens_to_generate=32,
        template=(
            "Answer the question based on the given documents. "
            "Only give me the answer and do not output any other words.\n\n"
            "The following are given documents.\n\n{context}\n\n"
            "Answer the question based on the given documents. "
            "Only give me the answer and do not output any other words.\n\n"
            "Question: {query}"
        ),
        answer_prefix=" Answer:",
        args={"dataset": "hotpotqa"},
    ),
}


class RulerDatasetBuilder:
    """Builder for RULER calibration datasets."""

    def __init__(
        self,
        samples: int,
        max_seqlen: int,
        tokenizer_name_or_path: str | object,
        num_length_bins: int = 4,
        max_length_filter: int = 65536,
        seed: int = 42,
        cache_dir: str | None = None,
        data_dir: str | Path | None = None,
    ):
        """Initialize RULER dataset builder.

        Args:
            samples: Total number of samples to generate (distributed evenly across length bins)
            max_seqlen: Maximum sequence length (length bins auto-generated as powers of 2)
            tokenizer_name_or_path: HuggingFace tokenizer path or tokenizer object
            seed: Random seed for reproducibility
            num_length_bins: Number of length bins to generate (default: 4)
            max_length_filter: Maximum sequence length to keep (default: 65536)
            cache_dir: Optional cache directory. If None, uses ~/.cache/modelopt/data/
            data_dir: Optional path to RULER data directory (contains 'essays' subdir).
                Required for NIAH tasks with essay haystack when not using pip default layout.

        Note:
            Length bins are auto-generated as descending powers of 2:
            [max_seqlen, max_seqlen/2, max_seqlen/4, ...]
            Generation stops when num_length_bins is reached or length < 1024.
            Subtasks are set to all the difficult tasks defined in RULER_TASKS.
        """
        # Validate inputs
        if samples <= 0:
            raise ValueError(f"samples must be positive, got {samples}")
        if max_seqlen < 1024:
            raise ValueError(f"max_seqlen must be >= 1024, got {max_seqlen}")

        # Store parameters
        self.total_samples = samples
        self.max_seqlen = max_seqlen
        self.num_length_bins = num_length_bins
        self.subtasks = list(RULER_TASKS.keys())
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.seed = seed
        self.max_length_filter = max_length_filter
        self.cache_dir = cache_dir
        self.data_dir = Path(data_dir) if data_dir is not None else None

        # Generate target lengths and validate
        self.target_lengths = _generate_target_lengths(max_seqlen, num_length_bins, min_seqlen=1024)
        if not self.target_lengths:
            raise ValueError(f"No valid target lengths generated from max_seqlen={max_seqlen}")

        # Distribute samples evenly across lengths
        self.samples_per_length = [samples // len(self.target_lengths)] * len(self.target_lengths)

        # Initialize tokenizer
        if isinstance(tokenizer_name_or_path, str):
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        else:
            self.tokenizer = tokenizer_name_or_path
        random.seed(seed)

    def _get_cache_path(self) -> Path:
        """Generate cache file path based on calibration parameters."""
        tokenizer_path = (
            self.tokenizer_name_or_path
            if isinstance(self.tokenizer_name_or_path, str)
            else str(self.tokenizer_name_or_path)
        )
        key = f"{tokenizer_path}_{self.total_samples}_{self.max_seqlen}"
        hash_str = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]
        filename = f"ruler_cache_{self.total_samples}s_{self.max_seqlen}l_{hash_str}.json"
        if self.cache_dir:
            base_dir = Path(self.cache_dir)
        else:
            base_dir = Path.home() / ".cache" / "modelopt" / "data"
        return base_dir / filename

    def _load_cached_data(self, cache_path: Path) -> list[dict[str, Any]] | None:
        """Load calibration data from cache if it exists."""
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                print(f"Loaded {len(data)} cached calibration samples from {cache_path}")
                return data
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
        return None

    def _save_cached_data(self, cache_path: Path, data: list[dict[str, Any]]) -> None:
        """Save calibration data to cache."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
            print(f"Saved calibration samples to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def build_calibration_dataset(self) -> list[dict[str, Any]]:
        """Build the complete calibration dataset.

        If cache_dir was set, checks cache first and returns cached data if present.
        Otherwise generates the dataset, saves to cache (if cache_dir set), and returns.

        Returns:
            List of calibration samples with 'input' and 'length' fields
        """
        cache_path = self._get_cache_path()
        cached = self._load_cached_data(cache_path)
        if cached is not None:
            return cached

        all_samples = []

        print(
            f"Generating {self.total_samples} calibration samples "
            f"across {len(self.target_lengths)} length bins: {self.target_lengths}"
        )

        # Generate calibration samples with sample-level progress
        with tqdm(total=self.total_samples, desc="Generating RULER samples") as pbar:
            for num_samples, target_length in zip(self.samples_per_length, self.target_lengths):
                samples_per_task = max(num_samples // len(self.subtasks), 1)

                for task_name in self.subtasks:
                    for sample_idx in range(samples_per_task):
                        sample = self._generate_sample(task_name, target_length, sample_idx)
                        if sample and sample["length"] <= self.max_length_filter:
                            all_samples.append(sample)
                        pbar.update(1)

        random.shuffle(all_samples)
        print(f"Generated {len(all_samples)} valid samples")

        self._save_cached_data(cache_path, all_samples)
        return all_samples

    def _generate_sample(
        self, task_name: str, target_length: int, sample_idx: int
    ) -> dict[str, Any]:
        """Generate a single RULER sample.

        Args:
            task_name: Name of the RULER task
            target_length: Target sequence length in tokens
            sample_idx: Index of the sample (for uniqueness)

        Returns:
            Dict with 'input', 'length', and metadata fields
        """
        task = RULER_TASKS[task_name]

        if task.task_type == "niah":
            return self._generate_niah_sample(task, target_length, sample_idx)
        elif task.task_type == "variable_tracking":
            return self._generate_vt_sample(task, target_length, sample_idx)
        elif task.task_type == "freq_words_extraction":
            return self._generate_fwe_sample(task, target_length, sample_idx)
        elif task.task_type == "qa":
            return self._generate_qa_sample(task, target_length, sample_idx)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    def _generate_niah_sample(
        self, task: RulerTask, target_length: int, sample_idx: int
    ) -> dict[str, Any]:
        """Generate a needle-in-haystack sample."""
        args = task.args

        # Find optimal haystack size for target length
        optimal_haystack = find_optimal_haystack_size(
            tokenizer=self.tokenizer,
            max_seq_length=target_length,
            template=task.template,
            answer_prefix=task.answer_prefix,
            tokens_to_generate=task.tokens_to_generate,
            type_haystack=args.get("type_haystack", "essay"),
            type_needle_k=args.get("type_needle_k", "words"),
            type_needle_v=args.get("type_needle_v", "numbers"),
            num_needle_k=args.get("num_needle_k", 1),
            num_needle_v=args.get("num_needle_v", 1),
            num_needle_q=args.get("num_needle_q", 1),
            data_dir=self.data_dir,
        )

        # Generate sample using official RULER implementation
        sample = generate_niah_sample(
            num_haystack=optimal_haystack,
            tokenizer=self.tokenizer,
            template=task.template,
            answer_prefix=task.answer_prefix,
            tokens_to_generate=task.tokens_to_generate,
            type_haystack=args.get("type_haystack", "essay"),
            type_needle_k=args.get("type_needle_k", "words"),
            type_needle_v=args.get("type_needle_v", "numbers"),
            num_needle_k=args.get("num_needle_k", 1),
            num_needle_v=args.get("num_needle_v", 1),
            num_needle_q=args.get("num_needle_q", 1),
            random_seed=self.seed + sample_idx,
            data_dir=self.data_dir,
        )

        # Add task metadata
        sample["task"] = task.name
        sample["target_length"] = target_length
        sample["sample_idx"] = sample_idx

        return sample

    def _generate_vt_sample(
        self, task: RulerTask, target_length: int, sample_idx: int
    ) -> dict[str, Any]:
        """Generate a variable tracking sample."""
        args = task.args
        num_chains = args["num_chains"]
        num_hops = args["num_hops"]

        # Generate variable chains
        variables = []
        chains = []
        for _ in range(num_chains):
            chain = [self._generate_random_variable() for _ in range(num_hops + 1)]
            variables.extend(chain)
            chains.append(chain)

        # Generate assignments
        assignments = [
            f"VAR {chain[i]} = {chain[i + 1]}" for chain in chains for i in range(len(chain) - 1)
        ]

        # Create context with padding
        context = self._pad_context_with_text(
            "\n".join(assignments), target_length, "variable tracking context"
        )

        # Select a query value
        query_value = random.choice([chain[-1] for chain in chains])

        # Format template
        template = task.template.format(context=context, query=query_value)

        # Count variables with the query value
        num_v = sum(1 for chain in chains if chain[-1] == query_value)

        # Add answer prefix
        full_input = template + task.answer_prefix.format(num_v=num_v, query=query_value)

        # Tokenize to get actual length
        tokens = self.tokenizer.encode(full_input, add_special_tokens=False)

        return {
            "input": full_input,
            "length": len(tokens),
            "task": task.name,
            "target_length": target_length,
            "sample_idx": sample_idx,
        }

    def _generate_fwe_sample(
        self, task: RulerTask, target_length: int, sample_idx: int
    ) -> dict[str, Any]:
        """Generate a frequency word extraction sample."""
        # Generate coded words with frequencies
        num_unique_words = 50
        coded_words = [self._generate_coded_word() for _ in range(num_unique_words)]

        # Assign frequencies (make top 3 clearly more frequent)
        frequencies = {}
        for i, word in enumerate(coded_words):
            if i < 3:
                frequencies[word] = random.randint(20, 30)  # High frequency
            else:
                frequencies[word] = random.randint(1, 10)  # Low frequency

        # Generate the coded text
        word_list = []
        for word, freq in frequencies.items():
            word_list.extend([word] * freq)
        random.shuffle(word_list)

        # Add dots for separation
        coded_text = " .... ".join(word_list)

        # Pad to target length
        context = self._pad_context_with_text(coded_text, target_length, "coded text padding")

        # Format template
        template = task.template.format(context=context)
        full_input = template + task.answer_prefix

        # Tokenize to get actual length
        tokens = self.tokenizer.encode(full_input, add_special_tokens=False)

        return {
            "input": full_input,
            "length": len(tokens),
            "task": task.name,
            "target_length": target_length,
            "sample_idx": sample_idx,
        }

    def _generate_qa_sample(
        self, task: RulerTask, target_length: int, sample_idx: int
    ) -> dict[str, Any]:
        """Generate a QA sample."""
        # Generate synthetic documents
        num_docs = 5
        documents = []

        # Create a simple QA pair
        answer = self._generate_random_phrase()
        answer_doc_idx = random.randint(0, num_docs - 1)
        question = f"What is the special code mentioned in document {answer_doc_idx + 1}?"

        for i in range(num_docs):
            doc_text = self._generate_document_text(200)  # Base document
            if i == answer_doc_idx:  # Insert answer in the correct document
                doc_text += f" The special code is {answer}. "
            documents.append(f"Document {i + 1}:\n{doc_text}\n")

        # Combine documents
        context_base = "\n".join(documents)

        # Pad to target length
        context = self._pad_context_with_text(
            context_base, target_length, "additional document text"
        )

        # Format template
        template = task.template.format(context=context, query=question)
        full_input = template + task.answer_prefix

        # Tokenize to get actual length
        tokens = self.tokenizer.encode(full_input, add_special_tokens=False)

        return {
            "input": full_input,
            "length": len(tokens),
            "task": task.name,
            "target_length": target_length,
            "sample_idx": sample_idx,
        }

    def _pad_context_with_text(
        self, base_context: str, target_length: int, padding_type: str
    ) -> str:
        """Pad context to approach target length."""
        tokens = self.tokenizer.encode(base_context, add_special_tokens=False)

        while len(tokens) < target_length * 0.7:  # Leave room for template
            if padding_type == "variable tracking context":
                padding = (
                    f" VAR {self._generate_random_variable()} = {self._generate_random_variable()}."
                )
            elif padding_type == "coded text padding":
                padding = f" .... {self._generate_coded_word()} .... "
            else:
                padding = " " + self._generate_essay_text(50)

            base_context += padding
            tokens = self.tokenizer.encode(base_context, add_special_tokens=False)

        if len(tokens) > target_length * 0.9:
            # Truncate if too long
            base_context = self.tokenizer.decode(tokens[: int(target_length * 0.8)])

        return base_context

    def _generate_random_word(self) -> str:
        """Generate a random word."""
        return "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))

    def _generate_random_variable(self) -> str:
        """Generate a random variable name."""
        return "".join(random.choices(string.ascii_uppercase, k=1)) + "".join(
            random.choices(string.digits, k=3)
        )

    def _generate_coded_word(self) -> str:
        """Generate a coded word."""
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    def _generate_random_phrase(self) -> str:
        """Generate a random phrase."""
        words = [self._generate_random_word() for _ in range(random.randint(2, 4))]
        return " ".join(words)

    def _generate_essay_text(self, num_words: int) -> str:
        """Generate essay-like text."""
        topics = [
            "technology",
            "science",
            "nature",
            "history",
            "culture",
            "education",
            "health",
            "economics",
            "politics",
            "philosophy",
            "art",
            "literature",
        ]

        sentences = []
        words_generated = 0

        while words_generated < num_words:
            topic = random.choice(topics)
            word1 = self._generate_random_word()
            word2 = self._generate_random_word()
            word3 = self._generate_random_word()
            sentence = f"The {topic} of {word1} is {word2} and {word3}. "
            sentences.append(sentence)
            words_generated += len(sentence.split())

        return " ".join(sentences)

    def _generate_document_text(self, num_words: int) -> str:
        """Generate document-like text."""
        return self._generate_essay_text(num_words)

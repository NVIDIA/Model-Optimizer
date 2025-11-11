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

"""RULER dataset builder for sparse attention calibration."""

import random
import string
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

from . import ruler_utils


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
            "{num_v} variables are assgined the value {query}, they are: "
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
    ):
        """Initialize RULER dataset builder.

        Args:
            samples: Total number of samples to generate (distributed evenly across length bins)
            max_seqlen: Maximum sequence length (length bins auto-generated as powers of 2)
            tokenizer_name_or_path: HuggingFace tokenizer path or tokenizer object
            seed: Random seed for reproducibility
            num_length_bins: Number of length bins to generate (default: 4)
            max_length_filter: Maximum sequence length to keep (default: 65536)

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

        # Generate target lengths and validate
        self.target_lengths = _generate_target_lengths(max_seqlen, num_length_bins, min_seqlen=1024)
        if not self.target_lengths:
            raise ValueError(f"No valid target lengths generated from max_seqlen={max_seqlen}")

        # Distribute samples evenly across lengths
        self.samples_per_length = [samples // len(self.target_lengths)] * len(self.target_lengths)

        # Initialize tokenizer
        if isinstance(tokenizer_name_or_path, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        else:
            self.tokenizer = tokenizer_name_or_path
        random.seed(seed)

    def build_calibration_dataset(self) -> list[dict[str, Any]]:
        """Build the complete calibration dataset.

        Returns:
            List of calibration samples with 'input' and 'length' fields
        """
        all_samples = []

        # Generate calibration samples
        for num_samples, target_length in tqdm(
            zip(self.samples_per_length, self.target_lengths),
            desc="Generating RULER calibration samples",
            total=len(self.target_lengths),
        ):
            samples_per_task = max(num_samples // len(self.subtasks), 1)

            # Generate equal samples for each task
            for task_name in self.subtasks:
                for sample_idx in range(samples_per_task):
                    sample = self._generate_sample(task_name, target_length, sample_idx)
                    if sample and sample["length"] <= self.max_length_filter:
                        all_samples.append(sample)

        random.shuffle(all_samples)
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
        optimal_haystack = ruler_utils.find_optimal_haystack_size(
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
        )

        # Generate sample using official RULER implementation
        sample = ruler_utils.generate_niah_sample(
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
        question = f"What is the special code mentioned in document {random.randint(1, num_docs)}?"

        for i in range(num_docs):
            doc_text = self._generate_document_text(200)  # Base document
            if i == 2:  # Insert answer in one document
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

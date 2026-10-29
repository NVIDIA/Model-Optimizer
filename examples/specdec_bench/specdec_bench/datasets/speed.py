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

# mypy: disable-error-code="index"
# Paper-derived prompt strings carry intentional long lines; keep them verbatim.
# ruff: noqa: E501
import ast
import json
import re
from enum import StrEnum
from itertools import chain
from pathlib import Path
from typing import Any, Literal, get_args

from .base import Dataset, Request

try:
    import numpy as np
    import pandas as pd
    import tiktoken
    from datasets import concatenate_datasets, load_dataset

    not_installed = False
except ImportError:
    not_installed = True


config_type = Literal[
    "qualitative",
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
    "throughput_100k",
]
TURNS_PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"


class BenchmarkDataset(StrEnum):
    """Enum for benchmark datasets used in SPEED-Bench.

    Each enum value represents a HuggingFace dataset identifier used for
    loading external benchmark datasets.
    """

    BAMBOO = "RUCAIBox/BAMBOO"
    CNN_DAILYMAIL = "abisee/cnn_dailymail"
    HLE = "cais/hle"
    LIVECODEBENCH = "livecodebench/code_generation_lite"
    CODE_CONTESTS = "deepmind/code_contests"
    MTBENCH_101 = "mtbench101/mt-bench-101"
    OPUS100 = "Helsinki-NLP/opus-100"
    CHATRAG_BENCH = "nvidia/ChatRAG-Bench"
    MMLU_PRO = "TIGER-Lab/MMLU-Pro"
    ADALEVAL_STACKSELECT = "AdaLEval/stackselect"
    ADALEVAL_TEXTSORT = "AdaLEval/textsort"
    ROLEBENCH = "ZenMoore/RoleBench"
    ROLEBENCH_ROLES = "ZenMoore/RoleBench/roles"
    COSER = "Neph0s/CoSER"


DATASETS_AND_LOADERS_FUNCTIONS = {
    BenchmarkDataset.BAMBOO.value: lambda dataset_name, config_name: load_dataset(
        "json", data_files={"test": config_name}, split="test"
    ),
    BenchmarkDataset.CNN_DAILYMAIL.value: lambda dataset_name, config_name: load_dataset(
        dataset_name, config_name, split="test"
    ),
    BenchmarkDataset.HLE.value: lambda dataset_name, config_name: load_dataset(
        dataset_name, split="test", revision="021a3d71f516a7ac28ceb8d284969902edf1edeb"
    ),
    BenchmarkDataset.LIVECODEBENCH.value: lambda dataset_name, config_name: load_dataset(
        "json",
        data_files={
            "test": [
                f"https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/{file_name}.jsonl"
                for file_name in ["test", "test2", "test3", "test4", "test5", "test6"]
            ]
        },
        split="test",
    ),
    BenchmarkDataset.CODE_CONTESTS.value: lambda dataset_name, config_name: load_dataset(
        dataset_name, split="test", revision="802411c3010cb00d1b05bad57ca77365a3c699d6"
    ),
    BenchmarkDataset.MTBENCH_101.value: lambda dataset_name, config_name: load_dataset(
        "json", data_files={"test": config_name}, split="test"
    ),
    BenchmarkDataset.OPUS100.value: lambda dataset_name, config_name: load_dataset(
        dataset_name,
        config_name,
        split="test",
        revision="805090dc28bf78897da9641cdf08b61287580df9",
    ),
    BenchmarkDataset.CHATRAG_BENCH.value: lambda dataset_name, config_names: concatenate_datasets(
        [
            load_dataset(
                dataset_name,
                config_name,
                split="test",
                revision="af6c7d420ddddf21f54f8ab3394bbf462aad2577",
            )
            for config_name in config_names
        ]
    ),
    BenchmarkDataset.MMLU_PRO.value: lambda dataset_name, config_name: load_dataset(
        dataset_name, split="test", revision="30527804ea8854662078e457808040d872ecdf29"
    ),
    BenchmarkDataset.ADALEVAL_STACKSELECT.value: lambda dataset_name, config_name: load_dataset(
        "json", data_files={"test": config_name}, split="test"
    ),
    BenchmarkDataset.ADALEVAL_TEXTSORT.value: lambda dataset_name, config_name: load_dataset(
        "json", data_files={"test": config_name}, split="test"
    ),
    BenchmarkDataset.ROLEBENCH.value: lambda dataset_name, config_name: pd.read_json(
        config_name, lines=True
    ),
    BenchmarkDataset.ROLEBENCH_ROLES.value: lambda dataset_name, config_name: load_dataset(
        "json", data_files={"test": config_name}, split="test"
    ),
    BenchmarkDataset.COSER.value: lambda dataset_name, config_name: load_dataset(
        "json",
        data_files={"test": config_name.replace("tree", "raw") + "/test/test_set.json"},
        split="test",
    ),
}


class SPEEDBench(Dataset):
    def __init__(
        self,
        config_name: config_type = "qualitative",
        num_samples: int | None = None,
        _prepare_mode: bool = False,
        **kwargs,
    ):
        if not_installed:
            raise ImportError(
                "Additional packages are required to use SPEED-Bench. Please run `pip install -r requirements.txt`"
            )
        self.data: list[Request] = []
        self.num_samples = num_samples
        self.external_datasets: dict[str, Any] = {}
        self._config_name = config_name
        self._resolved_dataset = None
        self._preprocess(config_name, _prepare_mode=_prepare_mode)

    def _get_external_dataset(self, dataset_name: str, config_name: str | list[str] = "default"):
        full_name = f"{dataset_name}_{config_name}"
        if full_name not in self.external_datasets:
            self.external_datasets[full_name] = DATASETS_AND_LOADERS_FUNCTIONS[dataset_name](
                dataset_name, config_name
            )
        return self.external_datasets[full_name]

    @staticmethod
    def _generate_stackselect_prompt(question: str, answers: list[str]) -> str:
        # Original prompt as given in Ada-LEval paper: https://arxiv.org/pdf/2404.06480
        prompt = """
You are an AI assistant. Your job is to find out the most helpful answer to a given question.
Each time, you will be provided with a question and n answers to this question.
Each answer begins with an 'A' and a number(e.g. A4), which represents its designation.
You need to determine which answer is the most helpful one to the question.
The case sample is shown below and you should give me the answer in the format exactly the same as the sample.

However, you should NOT focus on the content of sample answer.

Sample Input (format only):

The question is given below.
XXX(The content of question)
Possible answers are given below.
A1:
XXX(The content of answer 1)
A2:
XXX(The content of answer 2)
.
.
.
An:
XXX(The content of answer n)
Now the answers are over, please decide which answer is the most helpful one to the question.
You must give me the designation of the MOST helpful answer and the reason why you choose this answer.
For every other answer, you must give me the reason why you do not choose this answer.

Sample Output (format only):

Answer: The designation of the most helpful answer.(e.g. A4 means answer 4 is the most helpful answer)
Explanation:
A4: The reason why you choose this answer.
A1: The reason why you do not choose this answer.
A2: The reason why you do not choose this answer.
.
.
.
An: The reason why you do not choose this answer.
"""
        prompt += "The question is given below.\n"
        prompt += question + "\n\n"
        prompt += "Possible answers are given below.\n"
        end_prompt = "Now the answers are over, please decide which answer is the most helpful one to the question. \n"
        end_prompt += "You must give me the designation of the MOST helpful answer and the reason why you choose this answer.\n"
        end_prompt += "For every other answer, you must give me the reason why you do not choose this answer.\n"
        for i, answer in enumerate(answers):
            prompt += f"A{i + 1}:\n\n{answer}\n\n"
        prompt += end_prompt
        return prompt

    @staticmethod
    def _generate_textsort_prompt(prompt: str) -> str:
        # Original prompt as given in Ada-LEval paper: https://arxiv.org/pdf/2404.06480
        original_instruction = "\n    You are an AI assistant. Your job is to sort multiple book sections into the correct order.\n    Each time, you will be provided with 4 pieces of text.\n    These texts form a continuous part of a book, but are provided in random order.\n    You need to find the correct order and return the answer in a string.\n    For example, if you output [4, 1, 3, 2], that means the correct order is: Part 4 -> Part 1 -> Part 3 -> Part 2.\n    You will also be provided with the neighboring paragraphs before and after the 4 pieces of texts. \n\n    The case sample is shown below and you should give me the answer in the format exactly the same as the sample. \n\n    However, you should NOT focus on the content of sample answer. \n\n    Please do NOT output any extra content. \n    Sample Input (format only): \n\n    Before: XXX (Text before the continuous book part)\n\n\n    Part 1: XXX\n\n\n    Part 2: XXX\n\n\n    Part 3: XXX\n\n\n    Part 4: XXX\n\n\n    After: XXX (Text after the continuous book part)\n\n\n    Sample Output (format only): \n\n    Answer: [4, 1, 3, 2] \n\n\n\n"

        new_instruction = """
You are an AI assistant. Your job is to sort multiple book sections into the correct order.
    Each time, you will be provided with 4 pieces of text.
    These texts form a continuous part of a book, but are provided in random order.
    You need to find the correct order and write the all the parts in the correct order.
    For example, if the correct order is: Part 4 -> Part 1 -> Part 3 -> Part 2, you need to answer with a continuous text of all the parts in the correct order.
    You should NOT change the text, just write it in the order it should appear.
    You will also be provided with the neighboring paragraphs before and after the 4 pieces of texts.
    You should NOT output the before and after paragraphs, just the text in the correct order.

    The case sample is shown below and you should give me the answer in the format exactly the same as the sample.

    However, you should NOT focus on the content of sample answer.

    Please do NOT output any extra content.

    Sample Input (format only):

    Before: BBB (Text before the continuous book part)


    Part 1: XXX


    Part 2: YYY


    Part 3: ZZZ


    Part 4: WWW


    After: AAA (Text after the continuous book part)

    Sample Output (format only):

   Answer:


   WWW

   XXX

   ZZZ

   YYY


"""
        return prompt.replace(original_instruction, new_instruction, 1)

    @staticmethod
    def _generate_writing_prompt(contents: list[str]) -> str:
        content = "\n\n".join(
            [
                f"START CONTENT {i + 1}\n\n{content}\n\nEND CONTENT"
                for i, content in enumerate(contents)
            ]
        )
        # Inspired by the prompt used in BAMBOO paper: https://arxiv.org/pdf/2309.13345
        prompt = f"""
I want you to act as a long dialogue completer.
Given a long dialogue(s), your objectives are:
1. Add one speaker mentioned in the past dialogue(s) at the end of the last sentence of each dialogue (between START CONTENT and END CONTENT) to complete the sentence and ensure its semantic integrity.  At here, the added word must be a person's name which appears in the dialogue.
2. Continue the dialogue(s) with one or more speakers who appeared in the dialogue(s) before. Be coherent with the previous dialogue(s) and be creative in your response.
The content of the dialogue(s) is given below.


{content}
"""
        return prompt

    @staticmethod
    def _pad_or_truncate_prompt(
        prompt: str, target_num_tokens: int, padding: str = "Answer now please.\n"
    ) -> str:
        encoder = tiktoken.get_encoding("o200k_base")

        tokens = encoder.encode(prompt, disallowed_special=())
        current_num_tokens = len(tokens)

        if current_num_tokens > target_num_tokens:
            # Truncate if too long
            tokens = encoder.encode(prompt, disallowed_special=())
            return encoder.decode(tokens[:target_num_tokens])
        elif current_num_tokens < target_num_tokens:
            # Add padding if too short
            padding_tokens = encoder.encode(padding, disallowed_special=())
            tokens_needed = target_num_tokens - current_num_tokens
            # Calculate how many full padding sequences we need
            num_padding_repeats = (tokens_needed + len(padding_tokens) - 1) // len(padding_tokens)
            padded_prompt = prompt + (padding * num_padding_repeats)
            # Truncate to exact target length
            padded_tokens = encoder.encode(padded_prompt, disallowed_special=())
            return encoder.decode(padded_tokens[:target_num_tokens])
        else:
            return prompt

    @staticmethod
    def _generate_bamboo_prompt(external_dataset: "Dataset", num_tokens: int) -> str:
        prompt = SPEEDBench._generate_writing_prompt(external_dataset["content"])
        return SPEEDBench._pad_or_truncate_prompt(prompt, num_tokens)

    @staticmethod
    def _generate_chatrag_bench_prompt(external_dataset: "Dataset") -> list[Any]:
        prompt = "Please give a full and complete answer for the questions. \n\nContext:\n{context}\n\nQuestion:\n{question}"
        context = "\n\n".join([ctx["text"] for ctx in external_dataset["ctxs"][0]])
        questions = [
            message["content"]
            for message in external_dataset["messages"][0]
            if message["role"] == "user"
        ]

        return [prompt.format(context=context, question=questions[0]), *questions[1:]]

    @staticmethod
    def _generate_coser_prompt(external_dataset: "Dataset") -> str:
        rng = np.random.default_rng(seed=12347)
        # Original prompt as given in CoSER paper: https://arxiv.org/pdf/2404.06480
        prompt = """You are {character} from {book_name}.
==={character}'s Profile===
{character_profile}
===Current Scenario===
{scenario}
===Information about the other Characters===
{other_character_profiles_str}
===Your Inner Thoughts===
{motivation}

===Requirements===
Your output should include **thought**, **speech**, and **action**. Use [your thought]
for thoughts, which others can't see, e.g. [I'm terrified, but I must appear strong.]. Use
(your action) for actions, which others can see, such as (watches silently, trying to control
her fear and anger)."""
        character = rng.choice(external_dataset["major_characters"][0])
        character_profile = external_dataset["character_profiles"][0][character]
        scenario = external_dataset["scenario"][0]
        book_name = external_dataset["book"][0]
        motivation = next(
            (
                key_character["motivation"]
                for key_character in external_dataset["key_characters"][0]
                if key_character["name"] == character
            ),
            "No motivation provided",
        )
        if motivation == "No motivation provided":
            print("warning: no motivation provided for character", character)
        other_character_profiles_str = "\n\n".join(
            [
                f"{character_name}: {character_profile}"
                for character_name, character_profile in external_dataset["character_profiles"][
                    0
                ].items()
                if character_name != character and character_profile is not None
            ]
        )
        return prompt.format(
            character=character,
            character_profile=character_profile,
            book_name=book_name,
            scenario=scenario,
            other_character_profiles_str=other_character_profiles_str,
            motivation=motivation,
        )

    @staticmethod
    def _generate_mmlu_pro_prompt(external_dataset: "Dataset", subject: str) -> list[Any]:
        def get_question_and_options(question, options):
            options = [(chr(ord("A") + i), a) for i, a in enumerate(options)]
            options_str = "\n".join([f"({letter}) {option}" for letter, option in options])
            return f"Question: {question}\n\nOptions: {options_str}\n\n"

        # Original prompt as given in MMLU-Pro paper: https://arxiv.org/pdf/2406.01574
        prompt = 'The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n'
        first_question = prompt.format(subject=subject) + get_question_and_options(
            external_dataset["question"][0], external_dataset["options"][0]
        )
        return [first_question] + [
            get_question_and_options(question, options)
            for question, options in zip(
                external_dataset["question"][1:], external_dataset["options"][1:]
            )
        ]

    @staticmethod
    def _generate_hle_prompt(
        example: dict[str, Any], demonstrations: list[dict[str, Any]], context_tokens: int
    ) -> str:
        encoder = tiktoken.get_encoding("o200k_base")
        prefix = encoder.encode(
            "Please answer the question below.\n\nHere are some examples of question and answer pairs in the category of "
            + example["category"]
            + ":\n\n",
            disallowed_special=(),
        )
        suffix = encoder.encode(
            "\n\nQuestion: " + example["question"] + "\n\nAnswer: ", disallowed_special=()
        )
        context = list(
            chain.from_iterable(
                encoder.encode(
                    "Question: " + demo["question"] + "\n\nAnswer: " + demo["rationale"] + "\n\n",
                    disallowed_special=(),
                )
                for demo in demonstrations
            )
        )
        if not 0 < context_tokens <= len(context):
            raise ValueError("Invalid HLE demonstration-token cutoff")
        return encoder.decode(prefix + context[:context_tokens] + suffix)

    @staticmethod
    def _trim_textsort_prompt(prompt: str, part_four: str, num_tokens: int) -> str:
        encoder = tiktoken.get_encoding("o200k_base")

        def encode(text: str) -> list[int]:
            return encoder.encode(text, disallowed_special=())

        length = len(encode(prompt))
        if num_tokens - 1 <= length <= num_tokens:
            return prompt
        tokens = encode(part_four)
        start = prompt.index(part_four)
        marker = "\n\n[... middle of Part 4 omitted to fit the 100K prompt ...]\n\n"
        maximum = len(tokens) - 200
        if length < num_tokens or maximum <= 0:
            raise ValueError("TextSort Part 4 cannot be trimmed to the target length")

        def candidate(remove_count):
            remaining = len(tokens) - remove_count
            left, right = remaining // 2, remaining - remaining // 2
            result = (
                prompt[:start]
                + encoder.decode(tokens[:left])
                + marker
                + encoder.decode(tokens[-right:])
                + prompt[start + len(part_four) :]
            )
            return result, len(encode(result))

        remove_count = min(maximum, length - num_tokens + len(encode(marker)))
        tried, best = set(), None
        for _ in range(12):
            remove_count = max(1, min(maximum, remove_count))
            if remove_count in tried:
                break
            tried.add(remove_count)
            result, length = candidate(remove_count)
            if best is None or abs(length - num_tokens) < abs(best[1] - num_tokens):
                best = (result, length)
            if length == num_tokens:
                return result
            remove_count += length - num_tokens
        center = max(1, min(maximum, remove_count))
        for radius in (8, 32, 128):
            for remove_count in range(max(1, center - radius), min(maximum, center + radius) + 1):
                if remove_count in tried:
                    continue
                tried.add(remove_count)
                result, length = candidate(remove_count)
                if best is None or abs(length - num_tokens) < abs(best[1] - num_tokens):
                    best = (result, length)
                if length == num_tokens:
                    return result
        if best is not None and best[1] == num_tokens - 1:
            return best[0]
        raise ValueError("Could not fit TextSort Part 4 to the target length")

    @staticmethod
    def _get_num_tokens_from_config(speed_config: config_type | str) -> int:
        match = re.search(r"throughput_(\d+)k", speed_config)
        if match:
            return int(match.group(1)) * 1000
        else:
            raise ValueError(f"Could not determine num_tokens from speed_config: {speed_config}")

    def _fetch_all_turns_data(
        self, example: dict[str, Any], speed_config: config_type | str
    ) -> dict[str, Any]:
        turns = example["turns"]
        if not turns[0].startswith(TURNS_PLACEHOLDER):
            return example

        if BenchmarkDataset.BAMBOO.value in example["source"]:
            num_tokens = self._get_num_tokens_from_config(speed_config)
            src_ids = [int(match) for match in re.findall(r"_(\d+)", example["src_id"])]
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.BAMBOO.value, config_name=example["source"]
            )
            external_dataset = external_dataset.select(src_ids)
            example["turns"] = [self._generate_bamboo_prompt(external_dataset, num_tokens)]

        elif BenchmarkDataset.CNN_DAILYMAIL.value in example["source"]:
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.CNN_DAILYMAIL.value, config_name="3.0.0"
            ).to_pandas()
            src_id = example["src_id"]
            article = external_dataset[external_dataset["id"] == src_id]["article"].iloc[0]
            example["turns"] = [
                example["turns"][0].removeprefix(f"{TURNS_PLACEHOLDER}\n\n").format(article=article)
            ]

        elif BenchmarkDataset.HLE.value in example["source"]:
            if "qualitative" in speed_config:
                external_dataset = self._get_external_dataset(
                    BenchmarkDataset.HLE.value, config_name="test"
                ).to_pandas()
                src_id = example["src_id"]
                example["turns"] = [
                    external_dataset[external_dataset["id"] == src_id]["question"].iloc[0]
                ]
            elif "throughput" in speed_config:
                fields = json.loads(example["src_id"])
                if "hle_by_id" not in self.external_datasets:
                    dataset = self._get_external_dataset(
                        BenchmarkDataset.HLE.value, config_name="test"
                    )
                    self.external_datasets["hle_by_id"] = (
                        dataset.to_pandas().set_index("id").to_dict("index")
                    )
                rows = self.external_datasets["hle_by_id"]
                prompt = self._generate_hle_prompt(
                    rows[fields["target"]],
                    [rows[src_id] for src_id in fields["demos"]],
                    fields["tokens"],
                )
                num_tokens = self._get_num_tokens_from_config(speed_config)
                actual_tokens = len(
                    tiktoken.get_encoding("o200k_base").encode(prompt, disallowed_special=())
                )
                if not num_tokens - 1 <= actual_tokens <= num_tokens:
                    raise ValueError(
                        f"HLE prompt for {example['src_id']} has {actual_tokens} tokens; expected {num_tokens - 1} to {num_tokens}"
                    )
                example["turns"] = [prompt]
            else:
                raise ValueError(f"Invalid speed_config: {speed_config}")

        elif BenchmarkDataset.LIVECODEBENCH.value in example["source"]:
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.LIVECODEBENCH.value, config_name="test"
            ).to_pandas()
            src_id = example["src_id"]
            external_dataset_example = external_dataset[
                external_dataset["question_id"] == src_id
            ].iloc[0]
            example["turns"] = [
                example["turns"][0]
                .removeprefix(f"{TURNS_PLACEHOLDER}\n\n")
                .format(
                    question=external_dataset_example["question_content"],
                    starter_code=external_dataset_example["starter_code"],
                )
            ]

        elif BenchmarkDataset.CODE_CONTESTS.value in example["source"]:
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.CODE_CONTESTS.value, config_name="test"
            ).to_pandas()
            src_id = example["src_id"]
            external_dataset_example = external_dataset[external_dataset["name"] == src_id].iloc[0]
            example["turns"] = [
                example["turns"][0]
                .removeprefix(f"{TURNS_PLACEHOLDER}\n\n")
                .format(question=external_dataset_example["description"])
            ]

        elif BenchmarkDataset.MTBENCH_101.value in example["source"]:
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.MTBENCH_101.value, config_name=example["source"]
            )
            src_id = example["src_id"].rsplit("_", 1)[1]
            external_dataset_example = external_dataset.select([int(src_id)])
            example["turns"] = [entry["user"] for entry in external_dataset_example["history"][0]]

        elif BenchmarkDataset.OPUS100.value in example["source"]:
            _, config_name, src_id = example["src_id"].split("_")
            src_lang = config_name.split("-")[0]
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.OPUS100.value, config_name=config_name
            )
            external_dataset_example = external_dataset.select([int(src_id)])
            example["turns"] = [
                example["turns"][0]
                .removeprefix(f"{TURNS_PLACEHOLDER}\n\n")
                .format(question=external_dataset_example["translation"][0][src_lang])
            ]

        elif BenchmarkDataset.CHATRAG_BENCH.value in example["source"]:
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.CHATRAG_BENCH.value, config_name=["hybridial", "sqa"]
            )
            src_id = example["src_id"].rsplit("_", 1)[1]
            external_dataset_example = external_dataset.select([int(src_id)])
            example["turns"] = self._generate_chatrag_bench_prompt(external_dataset_example)

        elif BenchmarkDataset.MMLU_PRO.value in example["source"]:
            if "mmlu_pro_grouped" not in self.external_datasets:
                external_dataset = self._get_external_dataset(
                    BenchmarkDataset.MMLU_PRO.value, config_name="test"
                ).to_pandas()
                external_dataset["group"] = external_dataset.index // 2
                self.external_datasets["mmlu_pro_grouped"] = external_dataset.groupby(
                    ["group", "category"]
                ).agg({"question": list, "options": list})
            external_dataset = self.external_datasets["mmlu_pro_grouped"]
            src_id = ast.literal_eval(example["src_id"].removeprefix("mmlu_pro_"))
            external_dataset_example = external_dataset.loc[src_id]
            example["turns"] = self._generate_mmlu_pro_prompt(
                external_dataset_example, example["sub_category"]
            )

        elif BenchmarkDataset.ADALEVAL_STACKSELECT.value in example["source"]:
            num_tokens = self._get_num_tokens_from_config(speed_config)
            fields = json.loads(example["src_id"])
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.ADALEVAL_STACKSELECT.value, config_name=example["source"]
            )
            row = external_dataset[fields["row"]]
            answers = [row["all_answers"][index] for index in fields["answers"]]
            prompt = self._generate_stackselect_prompt(row["question"], answers)
            # Match v1.1 padding/truncation before validating the reconstructed length.
            prompt = self._pad_or_truncate_prompt(prompt, num_tokens)
            actual_tokens = len(
                tiktoken.get_encoding("o200k_base").encode(prompt, disallowed_special=())
            )
            if actual_tokens != num_tokens:
                raise ValueError(
                    f"StackSelect prompt for {example['src_id']} has {actual_tokens} tokens; expected {num_tokens}"
                )
            example["turns"] = [prompt]

        elif BenchmarkDataset.ADALEVAL_TEXTSORT.value in example["source"]:
            num_tokens = self._get_num_tokens_from_config(speed_config)
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.ADALEVAL_TEXTSORT.value, config_name=example["source"]
            )
            src_id = example["src_id"].split("_")[1]
            external_dataset_example = external_dataset.select([int(src_id)])
            prompt = self._generate_textsort_prompt(external_dataset_example["prompt"][0])
            # Preserve exact v1.1 reconstruction: only 100K trims Part 4, and short
            # prompts must fail rather than introduce padding absent from the benchmark.
            example["turns"] = [
                self._trim_textsort_prompt(
                    prompt, external_dataset_example["chapters"][0][3], num_tokens
                )
                if num_tokens == 100_000
                else self._pad_or_truncate_prompt(prompt, num_tokens)
            ]

        elif BenchmarkDataset.ROLEBENCH.value in example["source"]:
            config_name = example["src_id"].split("_")[1]
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.ROLEBENCH.value,
                config_name=example["source"].replace("tree", "raw")
                + f"/{config_name}/role_specific/test.jsonl",
            )
            roles_dataset = self._get_external_dataset(
                BenchmarkDataset.ROLEBENCH_ROLES.value,
                config_name="https://huggingface.co/datasets/ZenMoore/RoleBench/raw/a57ed54f9613921e4a5f1b63601a558cd5acf971/profiles-eng/desc.json",
            )
            src_ids = [int(match) for match in re.findall(r"_(\d+)", example["src_id"])][
                : len(example["turns"])
            ]
            external_dataset_example = external_dataset.iloc[src_ids]
            role_name = external_dataset_example["role"].iloc[0]
            role_description_and_catchphrases = roles_dataset[role_name][0]
            example["turns"] = [
                example["turns"][0]
                .removeprefix(f"{TURNS_PLACEHOLDER}\n\n")
                .format(
                    role_name=role_name,
                    role_description_and_catchphrases=role_description_and_catchphrases,
                )
                + "\n"
                + external_dataset_example["question"].iloc[0]
            ] + [
                question.removeprefix(f"{role_name}, ").removeprefix(f" {role_name},")
                for question in external_dataset_example["question"].iloc[1:]
            ]

        elif BenchmarkDataset.COSER.value in example["source"]:
            external_dataset = self._get_external_dataset(
                BenchmarkDataset.COSER.value, config_name=example["source"]
            )
            src_id = example["src_id"].split("_")[1]
            external_dataset_example = external_dataset.select([int(src_id)])
            example["turns"] = [self._generate_coser_prompt(external_dataset_example)]

        return example

    def _load_dataset(self, config_name_or_dataset_path: config_type | str) -> "Dataset":
        """Load the raw HuggingFace dataset from a config name or local path.

        Args:
            config_name_or_dataset_path: Either a SPEED-Bench config name
                (e.g. ``"qualitative"``) or a path to a local parquet file /
                directory.
            category: If provided, filter the dataset to this category only.

        Returns:
            The loaded (and optionally filtered / truncated) HuggingFace dataset.
        """
        if config_name_or_dataset_path in get_args(config_type):
            dataset = load_dataset(
                "nvidia/SPEED-Bench", config_name_or_dataset_path, split="test", revision="v1.1"
            )
        else:
            config_name_or_dataset_path_path = Path(config_name_or_dataset_path)
            if not config_name_or_dataset_path_path.exists():
                msg = ", ".join(get_args(config_type))
                raise ValueError(
                    f"Dataset path {config_name_or_dataset_path_path} does not exist or not one of the supported configs {msg}"
                )
            if config_name_or_dataset_path_path.is_dir():
                data_files = {
                    "test": [
                        str(path) for path in config_name_or_dataset_path_path.rglob("*.parquet")
                    ]
                }
            else:
                data_files = {"test": [str(config_name_or_dataset_path_path)]}
            try:
                dataset = load_dataset("parquet", data_files=data_files, split="test")
            except (TypeError, ValueError):
                # Fallback: parquet metadata may be incompatible with the installed
                # ``datasets`` version.  Read via PyArrow and convert directly.
                import pyarrow
                import pyarrow.parquet as pq
                from datasets import Dataset as HFDataset

                tables = [pq.read_table(f) for f in data_files["test"]]
                table = pyarrow.concat_tables(tables) if len(tables) > 1 else tables[0]
                # Strip HF metadata from the schema to avoid Feature parsing errors
                schema = table.schema
                if schema.metadata and b"huggingface" in schema.metadata:
                    new_meta = {k: v for k, v in schema.metadata.items() if k != b"huggingface"}
                    table = table.replace_schema_metadata(new_meta or None)
                dataset = HFDataset(table)
        if self.num_samples is not None and self.num_samples < len(dataset):
            dataset = self._stratified_select(dataset, self.num_samples)
        return dataset

    @staticmethod
    def _stratified_select(dataset: "Dataset", n: int) -> "Dataset":
        """Select ``n`` samples uniformly across the ``category`` column.

        Round-robin across categories until ``n`` rows are collected. The
        resulting prefix is balanced; once a smaller category is exhausted
        the remaining categories continue contributing, so exactly ``n``
        rows are returned whenever ``n`` does not exceed the dataset size.
        Falls back to ``range(n)`` when ``category`` is absent or there is
        only one category. Indices come from ``range(category_size)`` (not
        random) so behavior is deterministic.
        """
        if "category" not in dataset.column_names:
            return dataset.select(range(n))
        cat_to_rows: dict[str, list[int]] = {}
        for i, c in enumerate(dataset["category"]):
            cat_to_rows.setdefault(c, []).append(i)
        if len(cat_to_rows) <= 1:
            return dataset.select(range(n))
        cat_lists = list(cat_to_rows.values())
        interleaved: list[int] = []
        max_len = max(len(c) for c in cat_lists)
        for i in range(max_len):
            for c in cat_lists:
                if i < len(c):
                    interleaved.append(c[i])
                    if len(interleaved) == n:
                        return dataset.select(interleaved)
        return dataset.select(interleaved)

    def _resolve_external_data(
        self, dataset: "Dataset", speed_config: config_type | str
    ) -> "Dataset":
        """Resolve all external data references in the dataset.

        Applies ``_fetch_all_turns_data`` to every example so that turn
        placeholders are replaced with fully-resolved prompt text.

        Args:
            dataset: The HuggingFace dataset with potentially unresolved turns.
            speed_config: The SPEED-Bench config name used to determine
                token-length parameters for throughput configs.

        Returns:
            The dataset with all turns fully resolved.
        """
        return dataset.map(self._fetch_all_turns_data, fn_kwargs={"speed_config": speed_config})

    def _preprocess(
        self,
        config_name_or_dataset_path: config_type | str,
        *,
        _prepare_mode: bool = False,
    ):
        dataset = self._load_dataset(config_name_or_dataset_path)

        if _prepare_mode:
            # Resolve all external data references (only allowed during prepare)
            dataset = self._resolve_external_data(dataset, config_name_or_dataset_path)
        else:
            # Validate that all turns are fully resolved (no placeholders remaining)
            for example in dataset:
                for turn in example["turns"]:
                    if turn.startswith(TURNS_PLACEHOLDER):
                        raise ValueError(
                            f"Unresolved data placeholder found in question_id={example['question_id']} "
                            f"(category={example['category']}). Please run "
                            f"`python prepare_data.py --config <config>` first to download "
                            f"and resolve all external data references."
                        )

        self._resolved_dataset = dataset
        self.data = [
            Request(
                system_prompt=None,
                turns=example["turns"],
                category=example["category"],
                question_id=example["question_id"],
            )
            for example in dataset
        ]
        assert len(self.data) == len(dataset), (  # type: ignore[arg-type]
            f"Number of requests {len(self.data)} does not match number of requests in the dataset {len(dataset)}"  # type: ignore[arg-type]
        )

    @classmethod
    def prepare_data(
        cls,
        output_dir: str | Path,
        config_name: config_type = "qualitative",
    ) -> Path:
        """Download, resolve, and save the SPEED-Bench dataset as parquet.

        This is the **only** entry-point that fetches external data and
        resolves turn placeholders.  The resulting parquet file can then be
        loaded directly by the normal ``SPEEDBench(config_name=<path>)``
        constructor without any further network access.

        Args:
            output_dir: Directory where the parquet file will be written.
            config_name: SPEED-Bench configuration to prepare.

        Returns:
            Path to the saved parquet file.
        """
        instance = cls(config_name=config_name, _prepare_mode=True)

        # Persist to parquet
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "test.parquet"
        instance._resolved_dataset.to_parquet(output_path)
        return output_path

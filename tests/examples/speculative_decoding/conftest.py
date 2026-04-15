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

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_conversations_path(tmp_path_factory):
    """Tiny JSONL with short synthetic conversations for compute_hidden_states_hf tests.

    Uses minimal single-turn conversations so that tokenized lengths stay well
    within the tiny test model's max_position_embeddings (32) even after chat
    template formatting.
    """
    tmp_dir = tmp_path_factory.mktemp("tiny_convs")
    output_file = tmp_dir / "train.jsonl"
    conversations = [
        {
            "conversation_id": f"test-{i}",
            "conversations": [
                {"role": "user", "content": "What is 2 plus 2?"},
                {"role": "assistant", "content": "4"},
            ],
        }
        for i in range(5)
    ]
    with open(output_file, "w") as f:
        f.writelines(json.dumps(conv) + "\n" for conv in conversations)
    return output_file


@pytest.fixture(scope="session", autouse=True)
def tiny_daring_anteater_path():
    """Return path to synthetic test data in OpenAI messages format.

    Uses examples/dataset/synthetic_conversations_1k.jsonl (1000 samples,
    900 single-turn + 100 two-turn). Synthesized by Claude (Anthropic),
    Apache-2.0 licensed.
    """
    return Path(__file__).parents[3] / "examples" / "dataset" / "synthetic_conversations_1k.jsonl"

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dataset tests that reach the HuggingFace Hub.

These live under ``tests/gpu`` (networked infra) rather than ``tests/unit``,
which is kept hermetic with toy local fixtures.
"""

import pytest
from huggingface_hub import get_token

from modelopt.torch.utils.dataset_utils import SUPPORTED_DATASET_CONFIG, get_dataset_samples

_NEW_NEMOTRON_KEYS = [
    "nemotron-sft-instruction-following-chat-v2",
    "nemotron-science-v1",
    "nemotron-competitive-programming-v1",
    "nemotron-sft-agentic-v2",
    "nemotron-math-v2",
    "nemotron-sft-swe-v2",
    "nemotron-sft-multilingual-v1",
]


@pytest.mark.parametrize("dataset_key", _NEW_NEMOTRON_KEYS)
def test_new_nemotron_registry_shape(dataset_key):
    """Shape check on the 7 newly registered nvidia/Nemotron-* entries.

    Complements the gated smoke test below — catches typos in dataset paths or
    split names even when the runner has no HF credentials.
    """
    assert dataset_key in SUPPORTED_DATASET_CONFIG
    entry = SUPPORTED_DATASET_CONFIG[dataset_key]
    config = entry["config"]
    assert config["path"].startswith("nvidia/Nemotron-")
    splits = config["split"]
    assert isinstance(splits, list) and splits
    assert all(isinstance(s, str) and s for s in splits)
    assert len(set(splits)) == len(splits)
    assert callable(entry["preprocess"])
    assert entry["chat_key"] == "messages"


@pytest.mark.parametrize("dataset_key", _NEW_NEMOTRON_KEYS)
def test_get_dataset_samples_new_nemotron(dataset_key):
    """Smoke-test the 7 newly registered nvidia/Nemotron-* calibration datasets.

    Skipped when no HF token is available because these datasets live behind the HF Hub.
    ``huggingface_hub.get_token()`` covers both the ``HF_TOKEN`` env var and tokens
    cached by ``hf auth login``.
    """
    if not get_token():
        pytest.skip(
            "No HF token (env HF_TOKEN or `hf auth login`); skipping gated Nemotron smoke test"
        )

    samples = get_dataset_samples(dataset_key, num_samples=2)
    assert isinstance(samples, list)
    assert len(samples) == 2
    assert all(isinstance(s, str) and len(s) > 0 for s in samples)

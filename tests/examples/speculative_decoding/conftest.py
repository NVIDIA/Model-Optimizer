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

import os

import pytest
from _test_utils.examples.run_command import MODELOPT_ROOT, run_example_command


@pytest.fixture(scope="session", autouse=True)
def tiny_daring_anteater_path(tmp_path_factory):
    dataset_path = MODELOPT_ROOT / "examples/speculative_decoding/Daring-Anteater"
    if not os.path.exists(dataset_path):
        try:
            run_example_command(
                ["git", "clone", "https://huggingface.co/datasets/nvidia/Daring-Anteater"],
                "speculative_decoding",
            )
        except Exception as e:
            # Ignore rate-limiting errors
            pytest.skip(f"Failed to clone Daring-Anteater dataset: {e}")
    output_path = tmp_path_factory.mktemp("daring_anteater") / "train.jsonl"
    with open(dataset_path / "train.jsonl") as src, open(output_path, "w") as dst:
        for i, line in enumerate(src):
            if i >= 128:
                break
            dst.write(line)
    return output_path

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

import pytest
import yaml
from _test_utils.examples.run_command import run_example_command


@pytest.fixture(scope="session", autouse=True)
def tiny_daring_anteater_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("daring_anteater")
    output_file = tmp_dir / "train.jsonl"

    config = {
        "outputs": [
            {
                "filename": str(output_file),
                "global_limit": 100,
                "sources": [{"name": "daring-anteater", "splits": {"all": 100}}],
            }
        ]
    }
    config_path = tmp_dir / "data_config.yaml"
    config_path.write_text(yaml.dump(config))

    run_example_command(
        ["python", "../dataset/make_dataset.py", "-f", str(config_path), "--full-conversations"],
        "speculative_decoding",
    )

    return output_file

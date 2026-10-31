# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Check README command extraction and explicit end-to-end input overrides."""

import pytest
from _test_utils.doc_tests.parser import DocTestError
from _test_utils.examples.readme import readme_command


def test_readme_command_preserves_flags_and_substitutes_inputs(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("""<!-- modelopt-snippet: quickstart -->
```bash
torchrun --nproc_per_node 8 train.py --model_path pretrained \
    --max_steps 100 --seq_length "${SEQ_LENGTH:-512}" --new-flag
python export.py --checkpoint trained
```
""")
    command = readme_command(
        path,
        "quickstart",
        "train.py",
        overrides={
            "nproc_per_node": 1,
            "model_path": "local model",
            "max_steps": 2,
            "save_steps": 1,
        },
    )
    assert command == [
        "torchrun",
        "--nproc_per_node",
        "1",
        "train.py",
        "--model_path",
        "local model",
        "--max_steps",
        "2",
        "--seq_length",
        "512",
        "--new-flag",
        "--save_steps",
        "1",
    ]
    assert readme_command(path, "quickstart", "export.py") == [
        "python",
        "export.py",
        "--checkpoint",
        "trained",
    ]
    with pytest.raises(DocTestError):
        readme_command(path, "renamed", "train.py")
    path.write_text(path.read_text().replace("${SEQ_LENGTH:-512}", "$UNSET_LENGTH"))
    with pytest.raises(DocTestError):
        readme_command(path, "quickstart", "train.py")

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

"""Check real script help without executing README commands or launchers."""

import pytest
from _test_utils.doc_tests.cli import check_cli_options, python_commands
from _test_utils.doc_tests.parser import DocTestError


@pytest.mark.parametrize(
    "launcher",
    [
        "python",
        "python3",
        "torchrun --nproc_per_node 8",
        "accelerate launch --config_file unused.yaml",
    ],
)
def test_readme_flags_against_cached_script_help(tmp_path, launcher):
    nested = tmp_path / "nested"
    nested.mkdir()
    script = nested / "cli.py"
    script.write_text("""import argparse
import os
from pathlib import Path
p = argparse.ArgumentParser(description="An obsolete --removed flag is mentioned here.")
p.add_argument('--alpha', '-a', required=True)
p.add_argument('--beta')
with Path('help-calls').open('a') as calls:
    calls.write('help\\n')
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
p.parse_args()
raise AssertionError('Only --help should execute')
""")
    readme = tmp_path / "README.md"
    readme.write_text(f"""<!-- modelopt-snippet: cli -->
```bash
cd nested
{launcher} cli.py --alpha "$(touch forbidden)" \\
    --beta=value
python3 cli.py -a two
```
""")
    commands = python_commands(readme, cwd=tmp_path)
    assert len(commands) == 2
    assert commands[0].line == 4
    for command in commands:
        check_cli_options(readme, command)
    assert (nested / "help-calls").read_text() == "help\n"
    assert not (nested / "forbidden").exists()
    readme.write_text(readme.read_text().replace("--alpha", "--removed"))
    with pytest.raises(
        DocTestError, match=r"README.md:4: cli: cli.py: unknown options:.*--removed"
    ):
        check_cli_options(readme, python_commands(readme, cwd=tmp_path)[0])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (None, "does not exist"),
        ("raise SystemExit('help unavailable')", "help unavailable"),
        ("import time; time.sleep(10)", "timed out"),
        ("print('No option definitions')", "did not list any options"),
    ],
)
def test_unusable_script_help_reports_readme_location(tmp_path, body, message):
    if body is not None:
        (tmp_path / "cli.py").write_text(body)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- modelopt-snippet: cli -->\n```sh\npython cli.py --alpha=one\n```\n")
    command = python_commands(readme, cwd=tmp_path)[0]
    with pytest.raises(DocTestError, match=rf"(?s)README.md:3: cli:.*{message}"):
        check_cli_options(readme, command, timeout=0.2)

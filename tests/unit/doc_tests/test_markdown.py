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

"""Focused checks for README parsing, failure locations, and subprocess cleanup."""

import os
import time
from pathlib import Path

import pytest
from _test_utils.doc_tests.parser import DocTestError, Scenario, parse_markdown, resolve_scenarios
from _test_utils.doc_tests.runner import run_scenario


def _scenario(tmp_path, code, *, language="sh", timeout_seconds=60):
    path = tmp_path / "README.md"
    path.write_text(f"<!-- modelopt-snippet: command -->\n```{language}\n{code}\n```\n")
    return resolve_scenarios(
        path, [Scenario("smoke", ["command"], timeout_seconds=timeout_seconds)]
    )[0]


def test_named_fence_parsing_and_selection(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("""<!-- modelopt-snippet: shell -->
~~~bash
printf 'hello'
~~~
> <!-- modelopt-snippet: python -->
> ```python
> answer = 42
> ```
""")
    snippets = parse_markdown(path)
    assert [(s.id, s.kind, s.line, s.code) for s in snippets.values()] == [
        ("shell", "run", 3, "printf 'hello'\n"),
        ("python", "python", 7, "answer = 42\n"),
    ]
    (scenario,) = resolve_scenarios(
        path, [Scenario("smoke", ["python"])], excluded={"shell": "Illustrative command"}
    )
    assert scenario.path == path
    assert scenario.steps == [snippets["python"]]


@pytest.mark.skipif(os.name != "posix", reason="Requires bash/POSIX")
@pytest.mark.parametrize(
    ("language", "code"),
    [("sh", "false | cat\ntouch forbidden"), ("python", "raise RuntimeError('snippet failed')")],
)
def test_failure_reports_snippet_and_markdown_line(tmp_path, language, code):
    scenario = _scenario(tmp_path, code, language=language)
    with pytest.raises(DocTestError) as error:
        run_scenario(scenario, tmp_path, tmp_path)
    assert f"{scenario.path}:3:" in str(error.value)
    assert "command" in str(error.value)
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.skipif(os.name != "posix", reason="Requires bash/POSIX")
def test_timeout_cleans_up_descendants(tmp_path):
    if not Path("/proc/self/stat").exists():
        pytest.skip("Requires /proc to verify descendant exit")
    scenario = _scenario(tmp_path, "sleep 30 &\necho $! > child.pid\nsleep 30", timeout_seconds=1)
    with pytest.raises(DocTestError, match="timed out"):
        run_scenario(scenario, tmp_path, tmp_path)
    assert not list(tmp_path.glob("doc-test-*"))
    pid = int((tmp_path / "child.pid").read_text())
    for _ in range(100):
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            return
        time.sleep(0.01)
    pytest.fail(f"Child {pid} survived scenario cleanup")

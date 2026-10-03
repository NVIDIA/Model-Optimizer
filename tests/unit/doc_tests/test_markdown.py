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

"""Offline acceptance tests for the Markdown parser and real subprocess runner."""

import os
import time
from pathlib import Path

import pytest
from _test_utils.doc_tests.parser import DocTestError, parse_markdown
from _test_utils.doc_tests.runner import run_scenario

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Documentation commands require bash/POSIX"
)


def _document(tmp_path, body, timeout=10):
    path = tmp_path / "README.md"
    path.write_text(
        '<!-- modelopt-doc-test:begin\nid = "smoke"\nprofile = "cpu"\n'
        f"timeout_seconds = {timeout}\n-->\n{body}\n<!-- modelopt-doc-test:end -->\n"
    )
    return path


def test_explicit_execution_and_persistence(tmp_path):
    tmp_path = tmp_path / "docs with 'quotes'"
    tmp_path.mkdir()
    path = _document(
        tmp_path,
        """
<!-- modelopt-doc-test:setup python
ctx.cwd = ctx.tmp
ctx.env["FROM_SETUP"] = "present"
(ctx.tmp / "nested").mkdir()
(ctx.repo / "doc_test_module.py").write_text("VALUE = 17")
-->
```sh
exit 91
```
<!-- modelopt-doc-test:run -->
```bash
cd nested
export VALUE="two words"
local_value=retained
printf '%s' "$FROM_SETUP" > setup.txt
```
An explanation between fences is fine.
<!-- modelopt-doc-test:run -->
~~~sh
printf '%s' "$VALUE/$local_value" > result.txt
python -c 'import doc_test_module; assert doc_test_module.VALUE == 17'
printf 'captured output\\n'
~~~
<!-- modelopt-doc-test:verify python
assert ctx.cwd == ctx.tmp / "nested"
assert ctx.env["VALUE"] == "two words"
assert (ctx.cwd / "setup.txt").read_text() == "present"
assert (ctx.cwd / "result.txt").read_text() == "two words/retained"
-->
""",
    )
    output = run_scenario(parse_markdown(path)[0], tmp_path, tmp_path)
    assert "captured output" in output


@pytest.mark.parametrize("command", ["false", "false | cat", "printf 'before failure\\n'; false"])
def test_shell_failure_reports_markdown_line(tmp_path, command):
    path = _document(
        tmp_path, f"<!-- modelopt-doc-test:run -->\n```sh\n{command}\ntouch forbidden\n```"
    )
    with pytest.raises(DocTestError, match=rf"{path}:8: shell command failed"):
        run_scenario(parse_markdown(path)[0], tmp_path, tmp_path)
    assert not (tmp_path / "forbidden").exists()


def test_python_failure_reports_markdown_line(tmp_path):
    path = _document(
        tmp_path,
        """<!-- modelopt-doc-test:run -->
```sh
true
```
<!-- modelopt-doc-test:verify python
assert False, "verification failed"
-->""",
    )
    with pytest.raises(DocTestError, match="verification failed") as error:
        run_scenario(parse_markdown(path)[0], tmp_path, tmp_path)
    assert f'File "{path}", line 11' in str(error.value)


@pytest.mark.parametrize(
    "body",
    [
        "<!-- modelopt-doc-test:run -->\nprose\n```sh\ntrue\n```",
        "<!-- modelopt-doc-test:run -->\n```python\npass\n```",
        "<!-- modelopt-doc-test:run -->\n```sh\ntrue",
        "<!-- modelopt-doc-test:unknown -->",
        "<!-- modelopt-doc-test:setup python\nthis is invalid python!\n-->",
        "<!-- modelopt-doc-test:run -->\n```sh\ntrue\n```\n<!-- modelopt-doc-test:setup python\npass\n-->",
        "<!-- modelopt-doc-test:verify python\npass\n-->",
        '<!-- modelopt-doc-test:begin\nid="nested"\n-->',
        "",
    ],
)
def test_malformed_directives(tmp_path, body):
    path = _document(tmp_path, body)
    with pytest.raises(DocTestError, match=str(path)):
        parse_markdown(path)


@pytest.mark.parametrize(
    "replacement", ['profile = "tpu"', "timeout_seconds = 0", "timeout_seconds = true", "extra = 1"]
)
def test_invalid_metadata(tmp_path, replacement):
    path = _document(tmp_path, "<!-- modelopt-doc-test:run -->\n```sh\ntrue\n```")
    field = replacement.split(" = ")[0]
    text = path.read_text()
    if field == "profile":
        text = text.replace('profile = "cpu"', replacement)
    elif field == "timeout_seconds":
        text = text.replace("timeout_seconds = 10", replacement)
    else:
        text = text.replace('profile = "cpu"', 'profile = "cpu"\n' + replacement)
    path.write_text(text)
    with pytest.raises(DocTestError):
        parse_markdown(path)


def test_unmarked_and_documented_directives_are_ignored(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("```sh\nexit 99\n```\n````markdown\n<!-- modelopt-doc-test:run -->\n````\n")
    assert parse_markdown(path) == []


@pytest.mark.parametrize("mutation", ["duplicate", "missing-end", "outside"])
def test_scenario_boundaries(tmp_path, mutation):
    path = _document(tmp_path, "<!-- modelopt-doc-test:run -->\n```sh\ntrue\n```")
    text = path.read_text()
    path.write_text(
        {
            "duplicate": text + text,
            "missing-end": text.replace("<!-- modelopt-doc-test:end -->", ""),
            "outside": "<!-- modelopt-doc-test:run -->\n" + text,
        }[mutation]
    )
    with pytest.raises(DocTestError):
        parse_markdown(path)


@pytest.mark.parametrize("ending", ["sleep 30", "false", "true"])
def test_process_group_cleanup(tmp_path, ending):
    path = _document(
        tmp_path,
        f"""<!-- modelopt-doc-test:run -->
```sh
sleep 30 &
echo $! > child.pid
{ending}
```""",
        timeout=2,
    )
    scenario = parse_markdown(path)[0]
    if ending == "true":
        run_scenario(scenario, tmp_path, tmp_path)
    else:
        with pytest.raises(DocTestError, match="timed out" if ending == "sleep 30" else "exit"):
            run_scenario(scenario, tmp_path, tmp_path)
    assert not list(tmp_path.glob("doc-test-*"))
    pid = int((tmp_path / "child.pid").read_text())
    for _ in range(100):
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"Child {pid} survived scenario cleanup")


def test_timeout_includes_hidden_python(tmp_path):
    path = _document(
        tmp_path,
        """<!-- modelopt-doc-test:setup python
import time
time.sleep(30)
-->
<!-- modelopt-doc-test:run -->
```sh
true
```""",
        timeout=1,
    )
    with pytest.raises(DocTestError, match="timed out"):
        run_scenario(parse_markdown(path)[0], tmp_path, tmp_path)


def test_pilot_collection_and_recipe_listing(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    scenarios = parse_markdown(repo / "examples/llm_qat/README.md")
    assert [scenario.id for scenario in scenarios] == ["llm-qat-recipes", "llm-qat-quickstart"]
    assert [scenario.profile for scenario in scenarios] == ["cpu", "gpu"]
    assert "nvfp4" in run_scenario(scenarios[0], repo, tmp_path)

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

"""Offline acceptance tests for named snippets and the real subprocess runner."""

import importlib.util
import os
import time
from pathlib import Path

import pytest
from _test_utils.doc_tests.parser import DocTestError, Scenario, parse_markdown, resolve_scenarios
from _test_utils.doc_tests.runner import run_scenario

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Requires bash/POSIX")
REPO = Path(__file__).resolve().parents[3]


def _scenario(tmp_path, body, *, handler=None, snippets=None, **metadata):
    path = tmp_path / "README.md"
    path.write_text(body.lstrip("\n"))
    if handler is not None:
        (tmp_path / "doc_test_hooks.py").write_text(handler)
        metadata["handler"] = "doc_test_hooks:run"
    return resolve_scenarios(path, [Scenario("smoke", snippets or ["command"], **metadata)])[0]


def _example(example):
    path = REPO / f"tests/examples/{example}/test_{example}_readme.py"
    spec = importlib.util.spec_from_file_location("readme_" + example, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return resolve_scenarios(module.README, module.SCENARIOS, excluded=module.EXCLUDED)


def test_explicit_selection_and_shell_persistence(tmp_path):
    tmp_path = tmp_path / "docs with 'quotes'"
    tmp_path.mkdir()
    path = tmp_path / "README.md"
    path.write_text("""<!-- modelopt-snippet: second -->
~~~bash
printf '%s' "$VALUE/$local_value" > result.txt
python -c 'import doc_test_module; assert doc_test_module.VALUE == 17'
printf 'captured output\\n'
~~~
<!-- modelopt-snippet: first -->
```sh
cd nested
export VALUE="two words"
local_value=retained
```
<!-- modelopt-snippet: excluded -->
```sh
exit 91
```
""")
    (tmp_path / "doc_test_hooks.py").write_text("""def run(ctx):
    ctx.cwd = ctx.tmp
    (ctx.tmp / 'nested').mkdir()
    (ctx.repo / 'doc_test_module.py').write_text('VALUE = 17')
    ctx.run()
    assert ctx.cwd == ctx.tmp / 'nested'
    assert ctx.env['VALUE'] == 'two words'
    assert (ctx.cwd / 'result.txt').read_text() == 'two words/retained'
""")
    definitions = [Scenario("smoke", ["first", "second"], handler="doc_test_hooks:run")]
    scenario = resolve_scenarios(path, definitions, excluded={"excluded": "Never executed"})[0]
    assert "captured output" in run_scenario(scenario, tmp_path, tmp_path)
    # A second test configuration can reuse the same IDs.
    assert (
        len(
            resolve_scenarios(
                path,
                [*definitions, Scenario("other", ["first", "second"])],
                excluded={"excluded": "Never executed"},
            )
        )
        == 2
    )


def test_worker_uses_resolved_plan_without_reparsing_readme(tmp_path):
    scenario = _scenario(
        tmp_path,
        "<!-- modelopt-snippet: command -->\n```sh\nprintf original\n```\n",
    )
    scenario.path.write_text("<!-- modelopt-snippet: renamed -->\n```sh\nexit 99\n```\n")
    assert "original" in run_scenario(scenario, tmp_path, tmp_path)


@pytest.mark.parametrize("command", ["false", "false | cat"])
def test_shell_failure_reports_snippet_and_markdown_line(tmp_path, command):
    scenario = _scenario(
        tmp_path, f"<!-- modelopt-snippet: command -->\n```sh\n{command}\ntouch forbidden\n```\n"
    )
    with pytest.raises(DocTestError, match=rf"{scenario.path}:3: command: shell command failed"):
        run_scenario(scenario, tmp_path, tmp_path)
    assert not (tmp_path / "forbidden").exists()


def test_verification_failure_reports_python_source(tmp_path):
    scenario = _scenario(
        tmp_path,
        "<!-- modelopt-snippet: command -->\n```sh\ntrue\n```\n",
        handler="def run(ctx):\n    ctx.run()\n    assert False, 'verification failed'\n",
    )
    with pytest.raises(DocTestError, match="verification failed") as error:
        run_scenario(scenario, tmp_path, tmp_path)
    assert f'File "{tmp_path / "doc_test_hooks.py"}", line 3' in str(error.value)


@pytest.mark.parametrize(
    "body",
    [
        "<!-- modelopt-snippet: command -->\nprose\n```sh\ntrue\n```",
        "<!-- modelopt-snippet: command -->\n```sh\ntrue",
        "<!-- modelopt-snippet: command -->",
        "<!-- modelopt-snippet: command -->\n<!-- modelopt-snippet: second -->",
        "<!-- modelopt-snippet: Uppercase -->\n```sh\ntrue\n```",
        "<!-- modelopt-snippet: command -->\n```sh\ntrue\n```\n<!-- modelopt-snippet: command -->\n```sh\nfalse\n```",
        "<!-- modelopt-doc-test:run -->\n```sh\ntrue\n```",
    ],
)
def test_malformed_markers(tmp_path, body):
    path = tmp_path / "README.md"
    path.write_text(body)
    with pytest.raises(DocTestError, match=str(path)):
        parse_markdown(path)


@pytest.mark.parametrize(
    "metadata",
    [
        {"profile": "tpu"},
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
        {"manual": 1},
        {"background": 1},
        {"min_gpus": -1},
        {"requires": ["torch..cuda"]},
        {"requires": "torch"},
    ],
)
def test_invalid_metadata(tmp_path, metadata):
    with pytest.raises(DocTestError):
        _scenario(tmp_path, "<!-- modelopt-snippet: command -->\n```sh\ntrue\n```", **metadata)


def test_invalid_handler_reference(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("<!-- modelopt-snippet: command -->\n```sh\ntrue\n```\n")
    with pytest.raises(DocTestError, match="module:function"):
        resolve_scenarios(path, [Scenario("smoke", ["command"], handler="lambda:bad.name")])


def test_unmarked_and_documented_markers_never_execute(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(
        "```sh\nexit 99\n```\n````markdown\n<!-- modelopt-snippet: command -->\n```sh\ntrue\n```\n````\n"
    )
    assert parse_markdown(path) == {}
    with pytest.raises(DocTestError, match="unaccounted fence"):
        resolve_scenarios(path, [])


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "unknown snippets"),
        ("unaccounted", "unaccounted snippets"),
        ("duplicate", "duplicate scenario ID"),
        ("overlap", "both tested and excluded"),
        ("reason", "nonempty reason"),
        ("unknown-exclusion", "unknown excluded snippets"),
    ],
)
def test_reference_and_coverage_validation(tmp_path, case, message):
    path = tmp_path / "README.md"
    path.write_text("<!-- modelopt-snippet: command -->\n```sh\ntrue\n```\n")
    scenarios = [Scenario("smoke", ["missing" if case == "missing" else "command"])]
    excluded = {}
    if case == "unaccounted":
        scenarios = []
    if case == "duplicate":
        scenarios *= 2
    if case == "overlap":
        excluded = {"command": "illustration"}
    if case == "reason":
        scenarios, excluded = [], {"command": " "}
    if case == "unknown-exclusion":
        excluded = {"missing": "illustration"}
    with pytest.raises(DocTestError, match=message):
        resolve_scenarios(path, scenarios, excluded=excluded)


@pytest.mark.parametrize("ending", ["sleep 30", "false", "true"])
def test_process_group_cleanup(tmp_path, ending):
    scenario = _scenario(
        tmp_path,
        f"<!-- modelopt-snippet: command -->\n```sh\nsleep 30 &\necho $! > child.pid\n{ending}\n```",
        timeout_seconds=2,
    )
    if ending == "true":
        run_scenario(scenario, tmp_path, tmp_path)
    else:
        with pytest.raises(DocTestError, match="timed out" if ending == "sleep 30" else "exit"):
            run_scenario(scenario, tmp_path, tmp_path)
    assert not list(tmp_path.glob("doc-test-*"))
    _assert_dead(int((tmp_path / "child.pid").read_text()))


def _assert_dead(pid):
    for _ in range(100):
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            return
        time.sleep(0.01)
    pytest.fail(f"Child {pid} survived scenario cleanup")


@pytest.mark.parametrize("phase", ["setup", "verify"])
def test_deadline_includes_handler(tmp_path, phase):
    call = "    ctx.run()\n"
    sleep = "    time.sleep(30)\n"
    handler = "import time\ndef run(ctx):\n" + (sleep + call if phase == "setup" else call + sleep)
    scenario = _scenario(
        tmp_path,
        "<!-- modelopt-snippet: command -->\n```sh\ntrue\n```",
        handler=handler,
        timeout_seconds=1,
    )
    with pytest.raises(DocTestError, match="timed out"):
        run_scenario(scenario, tmp_path, tmp_path)


def test_quoted_python_snippets_share_namespace_and_context(tmp_path):
    body = """<!-- modelopt-snippet: first -->
```python
if __name__ == "__main__":
    answer += 1
```
<!-- modelopt-snippet: second -->
~~~python
import os
answer += 1
os.environ['ANSWER'] = str(answer)
assert __file__ == str(ctx.repo / 'README.md')
~~~
"""
    scenario = _scenario(
        tmp_path,
        "".join("> " + line for line in body.splitlines(keepends=True)),
        snippets=["first", "second"],
        handler="""def run(ctx):
    ctx.cwd = ctx.tmp
    namespace = {'answer': 40}
    assert ctx.run(namespace) is namespace
    assert namespace['answer'] == 42
    assert ctx.cwd == ctx.tmp
    assert ctx.env['ANSWER'] == '42'
""",
    )
    run_scenario(scenario, tmp_path, tmp_path)


def test_indented_fence_preserves_python_error_location(tmp_path):
    scenario = _scenario(
        tmp_path,
        "   <!-- modelopt-snippet: command -->\n   ```python\n   assert False, 'snippet failed'\n   ```\n",
    )
    with pytest.raises(DocTestError, match="snippet failed") as error:
        run_scenario(scenario, tmp_path, tmp_path)
    assert f'File "{scenario.path}", line 3' in str(error.value)


def test_background_service_is_verified_then_cleaned_up(tmp_path):
    scenario = _scenario(
        tmp_path,
        "<!-- modelopt-snippet: command -->\n```sh\nsleep 30 &\necho $! > child.pid\nwait\n```",
        background=True,
        handler="""import os
import time

def run(ctx):
    ctx.cwd = ctx.tmp
    ctx.run()
    for _ in range(100):
        if (ctx.cwd / 'child.pid').exists(): break
        time.sleep(0.01)
    os.kill(ctx.background_pid, 0)
    assert (ctx.cwd / 'child.pid').is_file()
""",
    )
    run_scenario(scenario, tmp_path, tmp_path)
    _assert_dead(int((tmp_path / "child.pid").read_text()))


@pytest.mark.parametrize(
    ("body", "metadata"),
    [
        ("<!-- modelopt-snippet: command -->\n```javascript\ntrue\n```", {}),
        ("<!-- modelopt-snippet: command -->\n```run\ntrue\n```", {}),
        ("<!-- modelopt-snippet: command -->\n```sh\nsleep 30\n```", {"background": True}),
        ("<!-- modelopt-snippet: command -->\n```python\nx =\n```", {}),
        (
            "<!-- modelopt-snippet: command -->\n```python\nx = 1\n```\n"
            "<!-- modelopt-snippet: shell -->\n```sh\ntrue\n```",
            {"snippets": ["command", "shell"]},
        ),
    ],
)
def test_unsupported_execution(tmp_path, body, metadata):
    with pytest.raises(DocTestError):
        _scenario(tmp_path, body, **metadata)


@pytest.mark.parametrize(
    ("handler", "code", "message"),
    [
        (None, "import sys; sys.exit(0)", "exited before completing"),
        ("def run(ctx):\n    return\n", "pass", "did not complete ctx.run"),
        ("def run(ctx):\n    ctx.run()\n    ctx.run()\n", "pass", "exactly once"),
        (
            "def run(ctx):\n    ctx.run()\n    raise SystemExit(0)\n",
            "pass",
            "exited before completing",
        ),
    ],
)
def test_handler_cannot_bypass_completion(tmp_path, handler, code, message):
    scenario = _scenario(
        tmp_path, f"<!-- modelopt-snippet: command -->\n```python\n{code}\n```", handler=handler
    )
    with pytest.raises(DocTestError, match=message):
        run_scenario(scenario, tmp_path, tmp_path)


@pytest.mark.parametrize(
    ("example", "count", "manual"), [("llm_qat", 15, 3), ("megatron_bridge", 16, 10)]
)
def test_example_collection_and_fence_coverage(example, count, manual):
    scenarios = _example(example)
    assert len(scenarios) == count
    assert sum(s.manual for s in scenarios) == manual
    assert all(s.path == REPO / f"examples/{example}/README.md" for s in scenarios)


def test_cpu_example_commands_and_verification(tmp_path):
    scenarios = _example("llm_qat")
    for key in ["llm-qat-recipes", "llm-qat-copy-config"]:
        scenario = next(s for s in scenarios if s.id == key)
        output = run_scenario(scenario, REPO, tmp_path / key)
        if key == "llm-qat-recipes":
            assert "nvfp4" in output

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

"""Exercise CI reuse decisions with real commits and fixture GitHub responses."""

import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gate


class ChangelogGateTest(unittest.TestCase):
    """Check reuse policy across real repository and workflow states."""

    def setUp(self):
        """Create a passing source revision followed by a changelog-only commit."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.old_cwd = Path.cwd()
        os.chdir(self.directory.name)
        self.addCleanup(os.chdir, self.old_cwd)
        self._git("init", "-q")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "CI test")
        self._git("remote", "add", "origin", self.directory.name)
        self.base = self._commit("model.py", "original code")
        self.old_head = self._commit("model.py", "PR code")
        self.old_head = self._commit("CHANGELOG.rst", "entry")
        self.old_key = gate.fingerprint(self.old_head, self.base, "main", "42")
        self.head = self._commit("CHANGELOG.rst", "updated entry")
        self.current = {
            "id": 200,
            "run_number": 20,
            "head_branch": "feature",
            "event": "pull_request",
            "workflow_id": 3,
            "head_repository": {"id": 1},
            "pull_requests": [{"number": 42}],
            "head_sha": self.head,
        }
        self.previous = dict(
            self.current,
            id=100,
            run_number=19,
            status="completed",
            conclusion="success",
            head_sha=self.old_head,
            html_url="https://github.com/example/repo/actions/runs/100",
        )
        self.artifacts = [{"name": f"ci-equivalence-{self.old_key}", "expired": False}]
        self.env = {
            "GITHUB_REPOSITORY": "example/repo",
            "GITHUB_RUN_ID": "200",
            "CI_HEAD_SHA": self.head,
            "CI_BASE_SHA": self.base,
            "CI_BASE_REF": "main",
            "CI_PR": "42",
            "CI_ALLOW_REUSE": "true",
            "GITHUB_OUTPUT": "output",
            "GITHUB_STEP_SUMMARY": "summary",
            "CI_MARKER": "marker",
        }

    def _git(self, *args):
        return subprocess.check_output(["git", *args], stderr=subprocess.PIPE).decode().strip()

    def _commit(self, filename, contents):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text(contents)
        self._git("add", filename)
        self._git("commit", "-qm", filename)
        return self._git("rev-parse", "HEAD")

    def _api(self, endpoint):
        if endpoint.endswith("/actions/runs/200"):
            return self.current
        if "/workflows/3/runs?" in endpoint:
            return {"workflow_runs": [self.current, self.previous]}
        if "/actions/runs/100/artifacts?" in endpoint:
            return {"artifacts": self.artifacts}
        self.fail(f"Unexpected endpoint: {endpoint}")

    def _decision(self, expected):
        with patch.dict(os.environ, self.env), patch.object(gate, "_api", side_effect=self._api):
            gate.main()
        assert f"reuse={str(expected).lower()}\n" in Path("output").read_text()
        assert Path("marker").read_text().strip()
        Path("output").unlink()

    def test_changelog_updates_reuse_success_for_both_pr_routes(self):
        """Allow both native PR runs and copied PR branch runs."""
        for event in ["pull_request", "push"]:
            with self.subTest(event=event):
                for run in [self.current, self.previous]:
                    run["event"] = event
                    run["pull_requests"] = [{"number": 42}] if event == "pull_request" else []
                    run["head_branch"] = "feature" if event == "pull_request" else "pull-request/42"
                self.env["CI_PR"] = "42" if event == "pull_request" else "pull-request/42"
                key = gate.fingerprint(self.old_head, self.base, "main", self.env["CI_PR"])
                self.artifacts = [{"name": f"ci-equivalence-{key}", "expired": False}]
                self._decision(True)

    def test_no_passing_equivalent_run_uses_normal_tests(self):
        """Reject unavailable, unsuccessful or unrelated prior evidence."""
        original = copy.deepcopy(self.previous)
        for field, value in [
            ("conclusion", "failure"),
            ("conclusion", "cancelled"),
            ("status", "in_progress"),
            ("head_repository", {"id": 2}),
            ("pull_requests", [{"number": 43}]),
            ("run_number", 21),
        ]:
            with self.subTest(field=field, value=value):
                self.previous = dict(original, **{field: value})
                self._decision(False)
        self.previous = original
        for artifacts in [
            [],
            [{"name": f"ci-equivalence-{self.old_key}", "expired": True}],
            [{"name": "ci-equivalence-wrong", "expired": False}],
        ]:
            with self.subTest(artifacts=artifacts):
                self.artifacts = artifacts
                self._decision(False)

    def test_code_workflow_and_other_docs_changes_cannot_reuse(self):
        """Reject non-changelog source changes even if an artifact claims equivalence."""
        for filename in ["model.py", ".github/workflows/unit_tests.yml", "README.md"]:
            with self.subTest(filename=filename):
                self.env["CI_HEAD_SHA"] = self._commit(filename, "changed")
                # Even a forged matching marker cannot override the independent source diff.
                key = gate.fingerprint(self.env["CI_HEAD_SHA"], self.base, "main", "42")
                self.artifacts = [{"name": f"ci-equivalence-{key}", "expired": False}]
                self._decision(False)
                self._git("reset", "--hard", self.head)

    def test_rename_into_changelog_cannot_hide_deleted_source(self):
        """Do not treat a code file renamed to CHANGELOG.rst as changelog-only."""
        Path("CHANGELOG.rst").unlink()
        self._git("mv", "model.py", "CHANGELOG.rst")
        self._git("commit", "-qam", "rename source into changelog")
        self.env["CI_HEAD_SHA"] = self._git("rev-parse", "HEAD")
        key = gate.fingerprint(self.env["CI_HEAD_SHA"], self.base, "main", "42")
        self.artifacts = [{"name": f"ci-equivalence-{key}", "expired": False}]
        self._decision(False)

    def test_changed_target_or_test_merge_cannot_reuse(self):
        """Reject new target context or code introduced by the test merge."""
        for variable, value in [
            ("CI_BASE_SHA", self.old_head),
            ("CI_BASE_REF", "release/test"),
            ("CI_PR", "43"),
        ]:
            with self.subTest(variable=variable), patch.dict(self.env, {variable: value}):
                self._decision(False)
        # Source head stayed the same, but the checked-out merge contains new base code.
        self._commit("base-change.py", "new base code")
        self._decision(False)

    def test_unchanged_test_merge_reuses_previous_success(self):
        """Match equivalent test merges even when the target is ahead of the source head."""
        self._git("reset", "--hard", self.base)
        target = self._commit("base-change.py", "target code")
        self._git("checkout", "--detach", self.old_head)
        self._git("merge", "--no-ff", "-m", "previous test merge", target)
        old_key = gate.fingerprint(self.old_head, target, "main", "42")
        self.artifacts = [{"name": f"ci-equivalence-{old_key}", "expired": False}]
        self._git("checkout", "--detach", self.head)
        self._git("merge", "--no-ff", "-m", "current test merge", target)
        self.env["CI_BASE_SHA"] = target
        self._decision(True)

    def test_api_errors_and_forced_runs_use_normal_tests(self):
        """Keep API failures and forced runs on the normal test path."""
        with patch.object(self, "_api", side_effect=subprocess.CalledProcessError(1, "gh")):
            self._decision(False)
        self.env["CI_ALLOW_REUSE"] = "false"
        with patch.object(self, "_api", side_effect=AssertionError("must not query")):
            self._decision(False)


if __name__ == "__main__":
    unittest.main()

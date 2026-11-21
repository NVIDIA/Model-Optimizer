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

"""Reuse successful CI only for changelog-only updates to the same tested code."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlencode


def _git(*args):
    # Argument lists avoid shell evaluation of commit IDs and filenames.
    return subprocess.check_output(["git", *args], stderr=subprocess.PIPE)


def _api(endpoint):
    # gh supplies authentication without exposing the token in command arguments.
    return json.loads(subprocess.check_output(["gh", "api", endpoint], stderr=subprocess.PIPE))


def _source_tree(sha):
    entries = _git("ls-tree", "-z", sha).split(b"\0")
    return b"\0".join(entry for entry in entries if not entry.endswith(b"\tCHANGELOG.rst"))


def fingerprint(head_sha, base_sha, base_ref, pr, execution_policy=""):
    """Identify source, tested checkout and PR target while ignoring the root changelog."""
    context = [
        _source_tree(head_sha).hex(),
        _source_tree("HEAD").hex(),
        base_sha,
        base_ref,
        pr,
        execution_policy,
    ]
    return hashlib.sha256(json.dumps(context).encode()).hexdigest()


def reusable_run(repo, run_id, head_sha, key):
    """Return the prior run URL only when its success and equivalent inputs are verified."""
    current = _api(f"repos/{repo}/actions/runs/{run_id}")
    query = urlencode({"branch": current["head_branch"], "event": current["event"], "per_page": 10})
    runs = _api(f"repos/{repo}/actions/workflows/{current['workflow_id']}/runs?{query}")[
        "workflow_runs"
    ]
    previous = next((run for run in runs if run["run_number"] < current["run_number"]), None)
    if not previous or previous["status"] != "completed" or previous["conclusion"] != "success":
        return None
    # Run artifacts survive failed attempts; a later success cannot certify their policy.
    if previous["run_attempt"] != 1:
        return None
    if previous["head_repository"]["id"] != current["head_repository"]["id"]:
        return None
    if {pr["number"] for pr in previous["pull_requests"]} != {
        pr["number"] for pr in current["pull_requests"]
    }:
        return None
    old_sha = previous["head_sha"]
    if old_sha == head_sha:
        return None
    # Verify the old source independently of its artifact; workflows and the gate must match too.
    _git("fetch", "--no-tags", "origin", old_sha)
    if _git("diff", "--no-renames", "--name-only", "-z", old_sha, head_sha) != b"CHANGELOG.rst\0":
        return None
    artifacts = _api(f"repos/{repo}/actions/runs/{previous['id']}/artifacts?per_page=100")[
        "artifacts"
    ]
    if any(item["name"] == f"ci-equivalence-{key}" and not item["expired"] for item in artifacts):
        return previous["html_url"]
    return None


def main():
    """Publish reuse evidence and fall back to normal tests if evidence is unavailable."""
    key = ""
    previous = None
    failure_reason = ""
    try:
        key = fingerprint(
            os.environ["CI_HEAD_SHA"],
            os.environ["CI_BASE_SHA"],
            os.environ["CI_BASE_REF"],
            os.environ["CI_PR"],
            os.environ["CI_EXECUTION_POLICY"],
        )
        if os.environ["CI_ALLOW_REUSE"] == "true":
            previous = reusable_run(
                os.environ["GITHUB_REPOSITORY"],
                os.environ["GITHUB_RUN_ID"],
                os.environ["CI_HEAD_SHA"],
                key,
            )
    except (OSError, subprocess.CalledProcessError, KeyError, TypeError, ValueError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        # Keep subprocess output on one prefixed line so it cannot emit workflow commands.
        failure_reason = " ".join(detail.split()) or type(error).__name__
        print(f"CI reuse unavailable: {failure_reason}; running tests normally.")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"reuse={str(previous is not None).lower()}\nfingerprint={key}\n")
    if key:
        Path(os.environ["CI_MARKER"]).write_text(f"{key}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        if failure_reason:
            summary.write(f"CI reuse unavailable: {failure_reason}. Running tests normally.\n")
        elif previous:
            summary.write(f"Changelog-only update: reusing [successful CI]({previous}).\n")
        else:
            summary.write("No equivalent successful CI run found; using normal test selection.\n")


if __name__ == "__main__":
    main()

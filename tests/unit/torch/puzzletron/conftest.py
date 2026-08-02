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

import json
import platform
from pathlib import Path

import pytest


_QUARANTINE_PATH = Path(__file__).with_name("quarantined_tests.json")


def pytest_addoption(parser):
    group = parser.getgroup("puzzletron")
    group.addoption(
        "--validate-puzzletron-quarantine",
        action="store_true",
        help="Require every registered Puzzletron quarantine node to be collected exactly once.",
    )


def _quarantine_entries():
    payload = json.loads(_QUARANTINE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise pytest.UsageError("unsupported Puzzletron quarantine schema")

    entries = {}
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise pytest.UsageError("Puzzletron quarantine groups must be an object")

    for group_id, metadata in groups.items():
        if not isinstance(metadata, dict):
            raise pytest.UsageError(f"Puzzletron quarantine group {group_id!r} must be an object")
        for field in ("observed", "remove_when", "safety"):
            if not isinstance(metadata.get(field), str) or not metadata[field].strip():
                raise pytest.UsageError(
                    f"Puzzletron quarantine group {group_id!r} requires non-empty {field!r}"
                )
        nodes = metadata.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise pytest.UsageError(
                f"Puzzletron quarantine group {group_id!r} requires a non-empty node list"
            )
        for node_id in nodes:
            if not isinstance(node_id, str) or not node_id.startswith(
                "tests/unit/torch/puzzletron/"
            ):
                raise pytest.UsageError(
                    f"Puzzletron quarantine group {group_id!r} has an invalid node ID"
                )
            if node_id in entries:
                raise pytest.UsageError(f"duplicate Puzzletron quarantine node: {node_id}")
            entries[node_id] = (group_id, metadata)

    return entries


def pytest_collection_modifyitems(config, items):
    entries = _quarantine_entries()
    matched = set()
    for item in items:
        entry = entries.get(item.nodeid)
        if entry is None:
            continue
        group_id, metadata = entry
        item.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{group_id}: {metadata['observed']} "
                    f"Remove when: {metadata['remove_when']}"
                ),
            )
        )
        matched.add(item.nodeid)

    if config.getoption("--validate-puzzletron-quarantine"):
        missing = sorted(set(entries) - matched)
        if missing:
            formatted = "\n".join(f"  - {node_id}" for node_id in missing)
            raise pytest.UsageError(
                "registered Puzzletron quarantine nodes were not collected:\n" + formatted
            )


# `import fcntl` fails on Windows
def pytest_ignore_collect(collection_path, config):
    return platform.system() == "Windows"

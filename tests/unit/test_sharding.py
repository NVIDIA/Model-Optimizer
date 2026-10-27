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

from pathlib import Path
from types import SimpleNamespace

import pytest
from _test_utils.sharding import assign_files, select_shard

DURATIONS = {"a.py": 100.0, "b.py": 60.0, "c.py": 50.0, "d.py": 40.0, "e.py": 10.0}


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_assign_files_partitions_the_files_and_balances_them(count):
    files = sorted(DURATIONS)
    shards = assign_files(DURATIONS, files, count)
    assert sorted(f for shard in shards for f in shard) == files
    loads = [sum(DURATIONS[f] for f in shard) for shard in shards]
    # Longest-first never leaves a shard more than the largest file above the ideal load
    assert max(loads) <= sum(DURATIONS.values()) / count + max(DURATIONS.values())


def test_assign_files_is_deterministic_and_counts_an_unknown_file_as_the_median():
    files = [*sorted(DURATIONS), "new.py"]
    assert assign_files(DURATIONS, files, 2) == assign_files(DURATIONS, list(reversed(files)), 2)
    # median of the known durations is 50 s: "new.py" sits next to "c.py" instead of being ignored
    loads = [
        sum(DURATIONS.get(f, 50.0) for f in shard) for shard in assign_files(DURATIONS, files, 2)
    ]
    assert max(loads) - min(loads) <= 50.0


def _items(root, names):
    return [SimpleNamespace(path=root / name) for name in names]


@pytest.mark.parametrize("index", [1, 2])
def test_select_shard_keeps_whole_files_and_the_shards_partition_the_items(tmp_path, index):
    names = ["a.py", "a.py", "b.py", "c.py", "c.py", "d.py", "e.py"]
    deselected = []
    config = SimpleNamespace(
        rootpath=tmp_path,
        hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)),
    )
    items = _items(tmp_path, names)
    select_shard(config, items, index, 2)
    kept_files = {item.path.name for item in items}
    assert not kept_files & {item.path.name for item in deselected}
    assert len(items) + len(deselected) == len(names)
    assert [item.path.name for item in items] == [n for n in names if n in kept_files]


def test_the_two_shards_run_every_item_exactly_once(tmp_path):
    names = [f"{c}.py" for c in "abcde" for _ in range(3)]
    ran = []
    for index in (1, 2):
        config = SimpleNamespace(
            rootpath=tmp_path, hook=SimpleNamespace(pytest_deselected=lambda items: None)
        )
        items = _items(tmp_path, names)
        select_shard(config, items, index, 2)
        ran += [(item.path.name, i) for i, item in enumerate(items)]
    assert sorted(n for n, _ in ran) == sorted(names)


@pytest.mark.parametrize("index", [0, 3])
def test_select_shard_rejects_an_index_outside_the_count(tmp_path, index):
    config = SimpleNamespace(rootpath=Path(tmp_path), hook=None)
    with pytest.raises(pytest.UsageError, match="--shard-index"):
        select_shard(config, _items(tmp_path, ["a.py"]), index, 2)


def test_shard_durations_parse_takes_the_time_between_result_lines_per_file():
    from _test_utils.shard_durations import parse

    log = "\n".join(
        [
            "2026-10-23T22:00:00.0000000Z tests/gpu/a.py::test_1 PASSED [ 10%]",
            "2026-10-23T22:00:05.5000000Z tests/gpu/a.py::test_2 PASSED [ 20%]",
            "2026-10-23T22:00:07.0000000Z some other line",
            "2026-10-23T22:00:09.0000000Z tests/gpu/b.py::test_1[x] SKIPPED (reason) [ 30%]",
        ]
    )
    assert parse(log) == {"tests/gpu/a.py": 5.5, "tests/gpu/b.py": 3.5}

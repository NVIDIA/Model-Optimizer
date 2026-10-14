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

"""Offline tests for the NemotronH omni VL tutorial's QAD blend builder.

The Hugging Face reader and Megatron-Bridge's chat filter are replaced by fakes, so these run on
CPU with neither ``datasets`` nor Megatron-Bridge installed.
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "megatron_bridge"
    / "tutorials"
    / "NemotronH-Omni-W4A4-QAD"
    / "build_qad_blend.py"
)
_SPEC = importlib.util.spec_from_file_location("build_qad_blend", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
blend = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(blend)


def _chat(i: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    }


@pytest.fixture
def two_sources(monkeypatch):
    """Two ``messages`` sources of equal weight, each with 100 rows; returns the per-source stream counts."""
    monkeypatch.setattr(
        blend,
        "SOURCES",
        [
            (1, "org/a", "default", "train", "messages"),
            (1, "org/b", "default", "train", "messages"),
        ],
    )
    monkeypatch.setattr(blend, "is_usable", lambda chat: True)
    calls = {"org/a": 0, "org/b": 0}

    def stream(repo, config, split):
        calls[repo] += 1
        yield from (_chat(i) for i in range(100))

    monkeypatch.setattr(blend, "stream", stream)
    return calls


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_to_chat_handles_all_three_row_shapes():
    msgs = [{"role": "user", "content": "hi"}]
    assert blend.to_chat({"messages": msgs}, "messages") == {"messages": msgs}
    assert blend.to_chat({"messages": []}, "messages") is None

    text = "<extra_id_0>System\n<extra_id_1>User\nhello\n<extra_id_1>Assistant\nworld\n"
    assert blend.to_chat({"text": text}, "text") == {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }
    assert blend.to_chat({"text": "def f(): pass"}, "text") == {
        "messages": [{"role": "assistant", "content": "def f(): pass"}]
    }
    assert blend.to_chat({"text": ""}, "text") is None
    # Markers that parse to no turns: dropped, never trained as assistant text.
    assert blend.to_chat({"text": "<extra_id_0>\nraw body\n"}, "text") is None


def test_take_skips_rejected_rows_and_gives_up_after_too_many(monkeypatch):
    monkeypatch.setattr(blend, "stream", lambda *a: iter([_chat(i) for i in range(50)]))
    monkeypatch.setattr(blend, "is_usable", lambda chat: chat["messages"][0]["content"] != "q1")
    got, dropped = blend.take("org/a", "default", "train", "messages", want=3)
    assert [c["messages"][0]["content"] for c in got] == ["q0", "q2", "q3"]
    assert dropped == 1

    monkeypatch.setattr(blend, "is_usable", lambda chat: False)
    got, dropped = blend.take("org/a", "default", "train", "messages", want=2)
    assert got == [] and dropped == 21  # stops after 10 * want rejects


def test_build_splits_weighted_rows(tmp_path, two_sources):
    sizes = blend.build(str(tmp_path), total=100, seed=0)
    assert sizes == {"validation": 5, "train": 95}
    rows = _read(tmp_path / "train.jsonl") + _read(tmp_path / "validation.jsonl")
    assert len(rows) == 100 and all(set(r) == {"messages"} for r in rows)
    # Equal weights: each source contributes its first 50 rows, q0..q49.
    counts = Counter(r["messages"][0]["content"] for r in rows)
    assert counts == {f"q{i}": 2 for i in range(50)}


def test_build_reuses_cached_parts(tmp_path, two_sources):
    blend.build(str(tmp_path), total=100, seed=0)
    first = (tmp_path / "train.jsonl").read_text()
    blend.build(str(tmp_path), total=100, seed=0)
    assert two_sources == {"org/a": 1, "org/b": 1}  # second run streams nothing
    assert (tmp_path / "train.jsonl").read_text() == first


def test_build_rebuilds_a_truncated_part(tmp_path, two_sources):
    blend.build(str(tmp_path), total=100, seed=0)
    first = (tmp_path / "train.jsonl").read_text()
    part = tmp_path / "parts" / "a__default__train.jsonl"
    part.write_text(part.read_text()[:-20])  # an interrupted write, cut mid-record
    blend.build(str(tmp_path), total=100, seed=0)
    assert two_sources == {"org/a": 2, "org/b": 1}
    assert len(_read(part)) == 50
    assert (tmp_path / "train.jsonl").read_text() == first


def test_build_writes_parts_atomically(tmp_path, two_sources, monkeypatch):
    real_dumps = blend.json.dumps
    calls = {"n": 0}

    def flaky_dumps(obj, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 75:  # partway through the second source's part
            raise KeyboardInterrupt
        return real_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(blend.json, "dumps", flaky_dumps)
    with pytest.raises(KeyboardInterrupt):
        blend.build(str(tmp_path), total=100, seed=0)
    assert not (tmp_path / "parts" / "b__default__train.jsonl").exists()
    assert len(_read(tmp_path / "parts" / "a__default__train.jsonl")) == 50


def test_build_fails_on_a_source_that_cannot_load(tmp_path, two_sources, monkeypatch):
    real_stream = blend.stream

    def stream(repo, config, split):
        if repo == "org/b":
            yield from real_stream(repo, config, split)
            return
        yield from (_chat(i) for i in range(10))  # yields some rows, then the reader fails
        raise TypeError("cannot cast")

    monkeypatch.setattr(blend, "stream", stream)
    with pytest.raises(RuntimeError, match="org/a train: failed to load"):
        blend.build(str(tmp_path), total=100, seed=0)
    assert not (tmp_path / "parts" / "a__default__train.jsonl").exists()
    assert not (tmp_path / "train.jsonl").exists()


def test_build_fails_on_a_short_source(tmp_path, two_sources):
    with pytest.raises(RuntimeError, match="org/a train: only 100/250 usable rows"):
        blend.build(str(tmp_path), total=500, seed=0)
    assert not (tmp_path / "parts" / "a__default__train.jsonl").exists()


def test_build_rejects_a_blend_too_small_to_split(tmp_path, two_sources):
    with pytest.raises(RuntimeError, match="too few rows"):
        blend.build(str(tmp_path), total=10, seed=0)

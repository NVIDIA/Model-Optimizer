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

import gzip
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import huggingface_hub
import pytest
import yaml

from modelopt.torch.utils.plugins.prepare_megatron_data_blend import main

# The plugins package re-exports the function under the module's name, so import the module itself.
blend_module = importlib.import_module("modelopt.torch.utils.plugins.prepare_megatron_data_blend")


def _setup_test(
    tiny_tokenizer_path: str, tmp_path: Path, monkeypatch, target_tokens: int | None
) -> tuple[Path, Path, Mock]:
    output_dir = tmp_path / "tokenized"
    config = {
        "tokenizer": tiny_tokenizer_path,
        "output_dir": str(output_dir),
        "sources": [
            {
                "hf_dataset": "nanotron/minipile_100_samples",
                "split": "train",
                "max_samples": 100,
                "content_field": "text",
                "weight": 60,
            },
            {
                "hf_dataset": "nvidia/Nemotron-SFT-Competitive-Programming-v2",
                "files": ["data/competitive_programming_python_00.jsonl"],
                "content_field": "messages",
                "weight": 40,
            },
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_yaml = yaml.safe_dump(config)
    if target_tokens is not None:
        config_yaml += f"target_tokens: {target_tokens:_}\n"
    config_path.write_text(config_yaml, encoding="utf-8")
    jsonl_path = tmp_path / "competitive_programming_python_00.jsonl"
    conversation = {
        "messages": [
            {"role": "user", "content": "Write a Python function that adds two integers."},
            {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
        ]
    }
    jsonl_path.write_text(
        "".join(json.dumps(conversation) + "\n" for _ in range(20)), encoding="utf-8"
    )
    download = Mock(return_value=str(jsonl_path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(
        sys, "argv", ["prepare_megatron_data_blend.py", "--config", str(config_path)]
    )
    return output_dir, config_path, download


@pytest.mark.parametrize("target_tokens", [1_000, None], ids=["token-budget", "all-data"])
def test_prepare_megatron_data_blend_with_split_and_files_sources(
    tiny_tokenizer_path: str, tmp_path: Path, monkeypatch, target_tokens: int | None
):
    output_dir, config_path, download = _setup_test(
        tiny_tokenizer_path, tmp_path, monkeypatch, target_tokens
    )

    # Run in-process so the CLI entry point uses the mocked NVIDIA download.
    main()

    download.assert_called_once_with(
        repo_id="nvidia/Nemotron-SFT-Competitive-Programming-v2",
        filename="data/competitive_programming_python_00.jsonl",
        repo_type="dataset",
        local_dir=tmp_path / "raw/nvidia--Nemotron-SFT-Competitive-Programming-v2",
    )
    blend = [
        line.split(maxsplit=1)
        for line in (output_dir / "data_blend.txt").read_text(encoding="utf-8").splitlines()
    ]
    assert [weight for weight, _ in blend] == ["60", "40"]
    for _, prefix in blend:
        assert Path(prefix + ".bin").exists()
        assert Path(prefix + ".idx").exists()
    token_suffixes = ["_tokens600", "_tokens400"] if target_tokens is not None else ["", ""]
    # HF split prefixes use {dataset}_{config}_{split}_{field}_max{samples}.
    assert [Path(prefix).name for _, prefix in blend] == [
        f"nanotron--minipile_100_samples_default_train_text_max100{token_suffixes[0]}",
        f"competitive_programming_python_00_messages{token_suffixes[1]}",
    ]
    assert (output_dir / "config.yaml").read_bytes() == config_path.read_bytes()


def _chat(i: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    }


def _image_chat(i: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": "x.png"}]},
            {"role": "assistant", "content": f"a{i}"},
        ]
    }


def _source(name: str, **fields) -> dict:
    source = {
        "hf_dataset": f"org/{name}",
        "split": "train",
        "content_field": "messages",
        "weight": 1,
    }
    source.update(fields)
    return {k: v for k, v in source.items() if v is not None}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def chat_blend(tmp_path: Path, monkeypatch):
    """Sources of 100 ``messages`` rows each; returns (config writer, stream counts per source)."""
    calls = Counter()

    def iter_rows(source, output_dir, shuffle, seed):
        calls[source["hf_dataset"]] += 1
        yield from (_chat(i) for i in range(100))

    monkeypatch.setattr(blend_module, "_iter_rows", iter_rows)

    def write_config(**overrides) -> Path:
        config = {
            "output_dir": str(tmp_path / "blend"),
            "output_format": "chat_jsonl",
            "target_samples": 100,
            "shuffle": False,
            "seed": 0,
            "sources": [_source("a"), _source("b")],
        }
        config.update(overrides)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    return write_config, calls


def test_to_chat_handles_all_row_shapes():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert blend_module._to_chat({"messages": msgs}, "messages") == {"messages": msgs}
    assert blend_module._to_chat({"messages": []}, "messages") is None
    assert blend_module._to_chat({"messages": "hello"}, "messages") is None
    # Nothing would be trained under assistant-only loss; loss_mode full trains the user turn.
    assert blend_module._to_chat({"messages": msgs[:1]}, "messages") is None
    assert blend_module._to_chat({"messages": msgs[:1]}, "messages", loss_mode="full") == {
        "messages": msgs[:1]
    }
    # Tool definitions and a top-level system prompt reach the chat path.
    tools = [{"type": "function", "function": {"name": "add"}}]
    row = {"messages": msgs, "tools": tools, "system": "Be brief.", "id": 7}
    chat = blend_module._to_chat(row, "messages")
    assert chat == {"messages": msgs, "tools": tools, "system": "Be brief."}
    # Unlike the tokenizing mode, reasoning_content stays on the message by default (native).
    reasoning = [{"role": "assistant", "content": "4", "reasoning_content": "2+2"}]
    assert blend_module._to_chat({"messages": reasoning}, "messages")["messages"] == reasoning
    inline = blend_module._to_chat({"messages": reasoning}, "messages", "inline")
    assert inline["messages"] == [{"role": "assistant", "content": "<think>\n2+2\n</think>\n4"}]

    expected = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }
    text = "<extra_id_0>System\n<extra_id_1>User\nhello\n<extra_id_1>Assistant\nworld\n"
    assert blend_module._to_chat({"text": text}, "text") == expected
    # Any marker ends a turn, so a trailing marker is not trained as content.
    assert blend_module._to_chat({"text": text + "<extra_id_1>"}, "text") == expected
    assert blend_module._to_chat({"text": "def f(): pass"}, "text") == {
        "messages": [{"role": "assistant", "content": "def f(): pass"}]
    }
    assert blend_module._to_chat({"text": ""}, "text") is None
    assert blend_module._to_chat({"text": ["x"]}, "text") is None
    # Markers that parse to no turns: dropped, never trained as assistant text.
    assert blend_module._to_chat({"text": "<extra_id_0>\nraw body\n"}, "text") is None
    # A role without a mapping would cut its turn out of the middle: the whole row is dropped.
    text = "<extra_id_1>User\nq\n<extra_id_1>Tool\nresult\n<extra_id_1>Assistant\na\n"
    assert blend_module._to_chat({"text": text}, "text") is None
    for text in (
        # Text before the first marker would be cut from the conversation.
        "preamble\n<extra_id_1>User\nq\n<extra_id_1>Assistant\na\n",
        # A SteerLM label line is no role either.
        "<extra_id_1>User\nq\n<extra_id_1>Assistant\n<extra_id_2>quality:4\na\n",
        # A role must end at a newline: this is no User turn.
        "<extra_id_1>Userguide\nq\n<extra_id_1>Assistant\na\n",
        # An empty known turn would be cut out of the middle of the conversation.
        "<extra_id_1>User\nq1\n<extra_id_1>Assistant\n<extra_id_1>User\nq2\n<extra_id_1>Assistant\na2\n",
        # No assistant turn to train on.
        "<extra_id_1>User\nq\n<extra_id_1>Assistant\n",
    ):
        assert blend_module._to_chat({"text": text}, "text") is None, text


def test_take_chat_rows_skips_non_text_rows_and_fails_without_enough(
    tmp_path: Path, monkeypatch, capsys
):
    source = _source("a")
    # Media is rejected inside message content and as a top-level key the conversion drops.
    rows = [_chat(0), _image_chat(1), {**_chat(9), "images": ["a.png"]}, _chat(2), _chat(3)]
    monkeypatch.setattr(blend_module, "_iter_rows", lambda *a: iter(rows))
    got = blend_module._take_chat_rows(source, tmp_path, 3, False, 0, "native", "assistant")
    assert [c["messages"][0]["content"] for c in got] == ["q0", "q2", "q3"]

    images = [_image_chat(i) for i in range(200)]
    monkeypatch.setattr(blend_module, "_iter_rows", lambda *a: iter(images))
    with pytest.raises(RuntimeError, match=r"only 0/2 usable rows \(gave up after 121 rejected"):
        blend_module._take_chat_rows(source, tmp_path, 2, False, 0, "native", "assistant")

    # Unconvertible rows are skipped, but only the first few are warned about.
    bad = [{"messages": ["hello"]} for _ in range(20)]
    monkeypatch.setattr(blend_module, "_iter_rows", lambda *a: iter([*bad, _chat(0)]))
    capsys.readouterr()
    assert (
        len(blend_module._take_chat_rows(source, tmp_path, 1, False, 0, "native", "assistant")) == 1
    )
    out = capsys.readouterr().out
    assert out.count("cannot be converted") == 5
    assert "1 rows (20 rejected, 20 unconvertible)" in out
    # A config error is not a bad row: it fails before any row is skipped.
    monkeypatch.setattr(blend_module, "_iter_rows", lambda *a: iter(bad))
    no_field = {k: v for k, v in source.items() if k != "content_field"}
    with pytest.raises(KeyError, match="content_field"):
        blend_module._take_chat_rows(no_field, tmp_path, 1, False, 0, "native", "assistant")


def test_chat_blend_splits_weighted_rows(tmp_path: Path, chat_blend, monkeypatch):
    write_config, calls = chat_blend
    sources = [_source("a", weight=3), _source("b"), _source("c", weight=0)]
    config_path = write_config(sources=sources)
    monkeypatch.setattr(
        sys, "argv", ["prepare_megatron_data_blend.py", "--config", str(config_path)]
    )
    main()
    out = tmp_path / "blend"
    assert len(_read_jsonl(out / "validation.jsonl")) == 5
    rows = _read_jsonl(out / "train.jsonl") + _read_jsonl(out / "validation.jsonl")
    # 3:1 weights: the first 75 rows of org/a and 25 of org/b; org/c rounds to zero and is skipped.
    expected = Counter({f"q{i}": 1 for i in range(75)}) + Counter(f"q{i}" for i in range(25))
    assert Counter(r["messages"][0]["content"] for r in rows) == expected
    assert calls["org/c"] == 0
    assert (out / "config.yaml").read_bytes() == config_path.read_bytes()


def test_chat_blend_reuses_and_rebuilds_cached_parts(tmp_path: Path, chat_blend):
    write_config, calls = chat_blend
    config_path = write_config()
    blend_module.prepare_chat_data_blend(config_path)
    first = (tmp_path / "blend" / "train.jsonl").read_text(encoding="utf-8")
    blend_module.prepare_chat_data_blend(config_path)
    assert calls == {"org/a": 1, "org/b": 1}  # the second run streams nothing

    (part,) = (tmp_path / "blend" / "parts").glob("org--a_*.jsonl")
    part.write_text(part.read_text(encoding="utf-8")[:-20], encoding="utf-8")  # cut mid-record
    blend_module.prepare_chat_data_blend(config_path)
    assert calls == {"org/a": 2, "org/b": 1}
    assert (tmp_path / "blend" / "train.jsonl").read_text(encoding="utf-8") == first


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({}, {"sources": [_source("a", split="test")]}),
        (
            {"sources": [_source("a", split=None, files=["a.jsonl"])]},
            {"sources": [_source("a", split=None, files=["b.jsonl"])]},
        ),
        ({}, {"shuffle": True}),
        ({"shuffle": True, "seed": 0}, {"shuffle": True, "seed": 1}),
        ({}, {"reasoning_content": "inline"}),
        ({}, {"loss_mode": "full"}),
    ],
    ids=["split", "files", "shuffle", "seed", "reasoning_content", "loss_mode"],
)
def test_chat_blend_reselects_after_a_selection_change(chat_blend, first, second):
    write_config, calls = chat_blend
    blend_module.prepare_chat_data_blend(write_config(**first))
    blend_module.prepare_chat_data_blend(write_config(**second))
    assert calls["org/a"] == 2


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_samples": 0}, "positive integer"),
        ({"target_samples": 1.5}, "positive integer"),
        # A whole source would be held in memory.
        ({"target_samples": None}, "target_samples is required"),
        ({"output_format": "megatron"}, "writes output_format: chat_jsonl"),
        ({"sources": [_source("a", weight=-1)]}, "nonnegative"),
        ({"sources": [_source("a", weight=0)]}, "not all zero"),
        ({"sources": [_source("a", weight=float("inf"))]}, "finite"),
        ({"tokenizer": "x"}, "both tokenizer and max_length"),
        ({"max_length": 64}, "both tokenizer and max_length"),
        ({"pack_max_tokens": 64}, "pack_max_tokens needs tokenizer and max_length"),
        ({"tokenizer": "x", "max_length": 0}, "max_length must be a positive integer"),
        ({"tokenizer": "x", "max_length": 64, "loss_mode": "last"}, "loss_mode must be"),
        ({"target_tokens": 1000}, "does not use"),
        ({"reasoning_content": "drop"}, "reasoning_content must be one of"),
        # A row with both columns must not silently be read from ``text``.
        ({"sources": [_source("a", content_field="conversations")]}, "messages or text"),
    ],
)
def test_chat_blend_rejects_invalid_configs(chat_blend, overrides, match):
    write_config, calls = chat_blend
    with pytest.raises(ValueError, match=match):
        blend_module.prepare_chat_data_blend(write_config(**overrides))
    assert not calls


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({}, "Unknown output_format 'chat_jsonl'"),
        ({"output_format": "chat-jsonl"}, "Unknown output_format 'chat-jsonl'"),
        ({"output_format": "megatron"}, r"Tokenized blends do not use \['seed', 'shuffle'"),
    ],
)
def test_tokenized_blend_rejects_chat_options(chat_blend, overrides, match):
    write_config, _ = chat_blend
    with pytest.raises(ValueError, match=match):
        blend_module.prepare_megatron_data_blend(write_config(**overrides))


def test_chat_blend_fails_on_a_bad_short_or_tiny_source(tmp_path: Path, chat_blend, monkeypatch):
    write_config, _ = chat_blend
    with pytest.raises(RuntimeError, match="org/a train: only 100/250 usable rows"):
        blend_module.prepare_chat_data_blend(write_config(target_samples=500))
    with pytest.raises(RuntimeError, match="Too few rows"):
        blend_module.prepare_chat_data_blend(write_config(target_samples=10))

    def iter_rows(source, output_dir, shuffle, seed):
        yield from (_chat(i) for i in range(10))  # yields some rows, then the reader fails
        raise TypeError("cannot cast")

    monkeypatch.setattr(blend_module, "_iter_rows", iter_rows)
    with pytest.raises(RuntimeError, match="org/a train: failed to read or convert"):
        blend_module.prepare_chat_data_blend(write_config())
    assert not (tmp_path / "blend" / "train.jsonl").exists()


def test_write_jsonl_is_atomic(tmp_path: Path):
    path = tmp_path / "part.jsonl"
    with pytest.raises(TypeError):
        blend_module._write_jsonl(path, [_chat(0), {"bad": object()}])
    assert not path.exists()


def test_chat_blend_reads_files_sources(tmp_path: Path, monkeypatch):
    plain, packed = tmp_path / "a.jsonl", tmp_path / "b.jsonl.gz"
    lines = [json.dumps(_chat(i)) for i in range(20)]
    # Malformed records, unparseable or unconvertible, are skipped with a warning as in the
    # tokenizing reader.
    bad = ['{"messages": [', '{"messages": ["hello"]}']
    plain.write_text("\n".join([*lines[:10], *bad, *lines[10:]]) + "\n", "utf-8")
    with gzip.open(packed, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(_chat(i)) + "\n" for i in range(20, 40))
    paths = {"data/a.jsonl": str(plain), "data/b.jsonl.gz": str(packed)}
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kwargs: paths[filename]
    )
    config_path = tmp_path / "config.yaml"
    config = {
        "output_dir": str(tmp_path / "blend"),
        "output_format": "chat_jsonl",
        "target_samples": 40,
        "sources": [_source("a", split=None, files=list(paths))],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert blend_module.prepare_chat_data_blend(config_path) == {"validation": 2, "train": 38}
    out = tmp_path / "blend"
    rows = _read_jsonl(out / "train.jsonl") + _read_jsonl(out / "validation.jsonl")
    assert sorted(r["messages"][0]["content"] for r in rows) == sorted(f"q{i}" for i in range(40))
    # The part is named after its first file.
    assert len(list((out / "parts").glob("org--a_a_*.jsonl"))) == 1


def test_chat_blend_drops_untrainable_rows_and_pairs_the_rest(
    tiny_tokenizer_path: str, tmp_path: Path, chat_blend, monkeypatch
):
    def iter_rows(source, output_dir, shuffle, seed):
        # Prompts of 0..99 words; past max_length the truncation leaves no assistant token.
        yield from (
            {
                "messages": [
                    {"role": "user", "content": "w " * i},
                    {"role": "assistant", "content": "a"},
                ]
            }
            for i in range(100)
        )

    monkeypatch.setattr(blend_module, "_iter_rows", iter_rows)
    write_config, _ = chat_blend
    config = {"tokenizer": tiny_tokenizer_path, "max_length": 64, "num_workers": 1}
    sizes = blend_module.prepare_chat_data_blend(write_config(**config))
    out = tmp_path / "blend"
    blend_module._init_sft_tokenizer(tiny_tokenizer_path, 64, "assistant")
    rows = _read_jsonl(out / "train.jsonl") + _read_jsonl(out / "validation.jsonl")
    assert 40 < len(rows) < 100  # the 50 + 50 selected rows, minus the long prompts
    assert all(blend_module._sft_length(row) for row in rows)
    # A setup error would drop every row, so it is raised instead.
    monkeypatch.setitem(blend_module._SFT_TOKENIZER, "loss_mode", "unknown")
    with pytest.raises(ValueError, match="loss_mode"):
        blend_module._sft_length(rows[0])

    # A cap of 2 * max_length fits any two rows, so only an odd row per split is left out.
    paired = blend_module.prepare_chat_data_blend(write_config(**config, pack_max_tokens=128))
    assert paired == {split: n - n % 2 for split, n in sizes.items()}


def test_pair_rows_pairs_long_with_short_under_the_cap():
    lengths = {"a": 10, "b": 60, "c": 30, "d": 50, "e": 95, "f": 20}
    rows = blend_module._pair_rows(list(lengths), list(lengths.values()), cap=80, seed=0)
    # e fits with no row and c is left without a partner; rows (2k, 2k + 1) are a pack.
    assert {frozenset(rows[k : k + 2]) for k in range(0, len(rows), 2)} == {
        frozenset("ab"),
        frozenset("df"),
    }

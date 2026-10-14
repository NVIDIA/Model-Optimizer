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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def chat_blend(tmp_path: Path, monkeypatch):
    """Two equal-weight ``messages`` sources of 100 rows each; returns (config writer, stream counts)."""
    calls = {"org/a": 0, "org/b": 0}

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
            "sources": [
                {
                    "hf_dataset": f"org/{n}",
                    "split": "train",
                    "content_field": "messages",
                    "weight": 1,
                }
                for n in ("a", "b")
            ],
        }
        config.update(overrides)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    return write_config, calls


def test_to_chat_handles_all_row_shapes():
    msgs = [{"role": "user", "content": "hi"}]
    assert blend_module._to_chat({"messages": msgs}, "messages") == {"messages": msgs}
    assert blend_module._to_chat({"messages": []}, "messages") is None

    expected = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }
    text = "<extra_id_0>System\n<extra_id_1>User\nhello\n<extra_id_1>Assistant\nworld\n"
    assert blend_module._to_chat({"text": text}, "text") == expected
    # Any marker ends a turn, so a trailing or label marker is not trained as content.
    text = "<extra_id_1>User\nhello\n<extra_id_1>Assistant\nworld\n<extra_id_1>"
    assert blend_module._to_chat({"text": text}, "text") == expected
    assert blend_module._to_chat({"text": "def f(): pass"}, "text") == {
        "messages": [{"role": "assistant", "content": "def f(): pass"}]
    }
    assert blend_module._to_chat({"text": ""}, "text") is None
    # Markers that parse to no turns: dropped, never trained as assistant text.
    assert blend_module._to_chat({"text": "<extra_id_0>\nraw body\n"}, "text") is None
    # A role without a mapping would cut its turn out of the middle: the whole row is dropped.
    text = "<extra_id_1>User\nq\n<extra_id_1>Tool\nresult\n<extra_id_1>Assistant\na\n"
    assert blend_module._to_chat({"text": text}, "text") is None


@pytest.mark.parametrize(
    ("mode", "assistant"),
    [
        ("native", {"role": "assistant", "content": "4", "reasoning_content": "2+2"}),
        ("inline", {"role": "assistant", "content": "<think>\n2+2\n</think>\n4"}),
        ("strip", {"role": "assistant", "content": "4"}),
    ],
)
def test_to_chat_prepares_messages_like_the_tokenizing_mode(mode, assistant):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "2+2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "add", "arguments": '{"a": 2, "b": 2}'}}],
        },
    ]
    chat = blend_module._to_chat({"messages": messages}, "messages", mode)
    assert chat["messages"][1] == assistant
    # JSON-string tool-call arguments become dicts, as the Nemotron v3 chat templates expect.
    assert chat["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"a": 2, "b": 2}


def test_take_chat_rows_skips_non_text_rows_and_gives_up(tmp_path: Path, monkeypatch):
    source = {"hf_dataset": "org/a", "split": "train", "content_field": "messages"}
    rows = [_chat(0), _image_chat(1), _chat(2), _chat(3)]
    monkeypatch.setattr(blend_module, "_iter_rows", lambda *a: iter(rows))
    got = blend_module._take_chat_rows(source, tmp_path, 3, False, 0, "native")
    assert [c["messages"][0]["content"] for c in got] == ["q0", "q2", "q3"]

    monkeypatch.setattr(
        blend_module, "_iter_rows", lambda *a: iter([_image_chat(i) for i in range(50)])
    )
    with pytest.raises(RuntimeError, match=r"only 0/2 usable rows \(gave up after 21 rejected"):
        blend_module._take_chat_rows(source, tmp_path, 2, False, 0, "native")


def test_chat_blend_splits_weighted_rows(tmp_path: Path, chat_blend, monkeypatch):
    write_config, _ = chat_blend
    config_path = write_config()
    monkeypatch.setattr(
        sys, "argv", ["prepare_megatron_data_blend.py", "--config", str(config_path)]
    )
    main()
    out = tmp_path / "blend"
    assert len(_read_jsonl(out / "validation.jsonl")) == 5
    rows = _read_jsonl(out / "train.jsonl") + _read_jsonl(out / "validation.jsonl")
    # Equal weights: each source contributes its first 50 rows, q0..q49.
    assert Counter(r["messages"][0]["content"] for r in rows) == {f"q{i}": 2 for i in range(50)}
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
    "change",
    [
        {"shuffle": True},
        {
            "sources": [
                {
                    "hf_dataset": f"org/{n}",
                    "split": "test",
                    "content_field": "messages",
                    "weight": 1,
                }
                for n in ("a", "b")
            ]
        },
    ],
    ids=["shuffle", "split"],
)
def test_chat_blend_does_not_reuse_parts_of_another_selection(chat_blend, change):
    write_config, calls = chat_blend
    blend_module.prepare_chat_data_blend(write_config())
    blend_module.prepare_chat_data_blend(write_config(**change))
    assert calls == {"org/a": 2, "org/b": 2}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"shuffle": True, "seed": 0}, {"shuffle": True, "seed": 1}),
        ({}, {"reasoning_content": "inline"}),
        (
            {
                "sources": [
                    {
                        "hf_dataset": "org/a",
                        "files": ["a.jsonl"],
                        "content_field": "messages",
                        "weight": 1,
                    }
                ]
            },
            {
                "sources": [
                    {
                        "hf_dataset": "org/a",
                        "files": ["b.jsonl"],
                        "content_field": "messages",
                        "weight": 1,
                    }
                ]
            },
        ),
    ],
    ids=["seed", "reasoning_content", "files"],
)
def test_chat_blend_reselects_after_a_settings_change(chat_blend, first, second):
    write_config, calls = chat_blend
    blend_module.prepare_chat_data_blend(write_config(**first))
    blend_module.prepare_chat_data_blend(write_config(**first))  # cached
    blend_module.prepare_chat_data_blend(write_config(**second))
    assert calls["org/a"] == 2


def test_chat_blend_fails_on_a_source_without_usable_rows(chat_blend, monkeypatch):
    write_config, _ = chat_blend

    def iter_rows(source, output_dir, shuffle, seed):
        make_row = _chat if source["hf_dataset"] == "org/a" else _image_chat
        yield from (make_row(i) for i in range(100))

    monkeypatch.setattr(blend_module, "_iter_rows", iter_rows)
    # Without target_samples, an all-rejected source must not drop out of the blend silently.
    with pytest.raises(
        RuntimeError, match=r"org/b train: only 0/any usable rows \(source exhausted"
    ):
        blend_module.prepare_chat_data_blend(write_config(target_samples=None))


def test_chat_blend_does_not_reuse_a_capped_part_without_a_budget(tmp_path: Path, chat_blend):
    write_config, calls = chat_blend
    assert blend_module.prepare_chat_data_blend(write_config()) == {"validation": 5, "train": 95}
    # Every usable row of both sources, not the 50 per source cached by the capped run.
    sizes = blend_module.prepare_chat_data_blend(write_config(target_samples=None))
    assert sizes == {"validation": 10, "train": 190}
    assert calls == {"org/a": 2, "org/b": 2}


def test_chat_blend_skips_sources_rounded_to_zero_rows(tmp_path: Path, chat_blend):
    write_config, calls = chat_blend
    config = write_config(
        sources=[
            {"hf_dataset": "org/a", "split": "train", "content_field": "messages", "weight": 999},
            {"hf_dataset": "org/b", "split": "train", "content_field": "messages", "weight": 1},
        ]
    )
    assert blend_module.prepare_chat_data_blend(config) == {"validation": 5, "train": 95}
    assert calls == {"org/a": 1, "org/b": 0}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_samples": 0}, "positive integer"),
        ({"target_samples": 1.5}, "positive integer"),
        (
            {
                "sources": [
                    {
                        "hf_dataset": "org/a",
                        "split": "train",
                        "content_field": "messages",
                        "weight": -1,
                    }
                ]
            },
            "nonnegative",
        ),
        (
            {
                "sources": [
                    {
                        "hf_dataset": "org/a",
                        "split": "train",
                        "content_field": "messages",
                        "weight": 0,
                    }
                ]
            },
            "not all zero",
        ),
        (
            {
                "sources": [
                    {
                        "hf_dataset": "org/a",
                        "split": "train",
                        "content_field": "messages",
                        "weight": float("nan"),
                    }
                ]
            },
            "finite",
        ),
    ],
)
def test_chat_blend_rejects_invalid_budgets(chat_blend, overrides, match):
    write_config, calls = chat_blend
    with pytest.raises(ValueError, match=match):
        blend_module.prepare_chat_data_blend(write_config(**overrides))
    assert calls == {"org/a": 0, "org/b": 0}


def test_chat_blend_writes_parts_atomically(tmp_path: Path, chat_blend, monkeypatch):
    write_config, _ = chat_blend
    real_dumps = blend_module.json.dumps
    calls = {"n": 0}

    def flaky_dumps(obj, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 75:  # partway through the second source's part
            raise KeyboardInterrupt
        return real_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(blend_module.json, "dumps", flaky_dumps)
    with pytest.raises(KeyboardInterrupt):
        blend_module.prepare_chat_data_blend(write_config())
    parts = tmp_path / "blend" / "parts"
    assert not list(parts.glob("org--b_*.jsonl"))  # only the interrupted .tmp file was written
    (part,) = parts.glob("org--a_*.jsonl")
    assert len(_read_jsonl(part)) == 50


def test_chat_blend_fails_on_a_bad_source(tmp_path: Path, chat_blend, monkeypatch):
    write_config, _ = chat_blend
    real_iter_rows = blend_module._iter_rows

    def iter_rows(source, output_dir, shuffle, seed):
        if source["hf_dataset"] == "org/b":
            yield from real_iter_rows(source, output_dir, shuffle, seed)
            return
        yield from (_chat(i) for i in range(10))  # yields some rows, then the reader fails
        raise TypeError("cannot cast")

    monkeypatch.setattr(blend_module, "_iter_rows", iter_rows)
    with pytest.raises(RuntimeError, match="org/a train: failed to load"):
        blend_module.prepare_chat_data_blend(write_config())
    assert not (tmp_path / "blend" / "train.jsonl").exists()


def test_chat_blend_fails_on_a_short_or_tiny_blend(chat_blend):
    write_config, _ = chat_blend
    with pytest.raises(RuntimeError, match="org/a train: only 100/250 usable rows"):
        blend_module.prepare_chat_data_blend(write_config(target_samples=500))
    with pytest.raises(RuntimeError, match="Too few rows"):
        blend_module.prepare_chat_data_blend(write_config(target_samples=10))


def test_chat_blend_reads_files_sources(tmp_path: Path, monkeypatch):
    jsonl_path = tmp_path / "chat.jsonl"
    jsonl_path.write_text("".join(json.dumps(_chat(i)) + "\n" for i in range(40)), encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", Mock(return_value=str(jsonl_path)))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "blend"),
                "output_format": "chat_jsonl",
                "sources": [
                    {
                        "hf_dataset": "org/a",
                        "files": ["data/chat.jsonl"],
                        "content_field": "messages",
                        "weight": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # No target_samples: every usable row, in file order before mixing.
    assert blend_module.prepare_chat_data_blend(config_path) == {"validation": 2, "train": 38}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"tokenizer": "x"}, "does not use"),
        ({"target_tokens": 1000}, "does not use"),
        ({"reasoning_content": "drop"}, "reasoning_content must be one of"),
        # A row with both columns must not silently be read from ``text``.
        (
            {
                "sources": [
                    {
                        "hf_dataset": "org/a",
                        "split": "train",
                        "content_field": "conversations",
                        "weight": 1,
                    }
                ]
            },
            "content_field messages or text",
        ),
    ],
)
def test_chat_blend_rejects_tokenizer_options(chat_blend, overrides, match):
    write_config, _ = chat_blend
    with pytest.raises(ValueError, match=match):
        blend_module.prepare_chat_data_blend(write_config(**overrides))


def test_tokenized_blend_rejects_chat_options(chat_blend):
    write_config, _ = chat_blend
    with pytest.raises(ValueError, match="Tokenized blends do not use"):
        blend_module.prepare_megatron_data_blend(write_config())


def test_chat_blend_reads_gzip_files_and_skips_malformed_records(tmp_path: Path, monkeypatch):
    plain, packed = tmp_path / "a.jsonl", tmp_path / "b.jsonl.gz"
    lines = [json.dumps(_chat(i)) for i in range(20)]
    plain.write_text(
        "\n".join([*lines[:10], '{"messages": [', *lines[10:]]) + "\n", encoding="utf-8"
    )
    with gzip.open(packed, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(_chat(i)) + "\n" for i in range(20, 40))
    paths = {"data/a.jsonl": str(plain), "data/b.jsonl.gz": str(packed)}
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kwargs: paths[filename]
    )
    source = {"hf_dataset": "org/a", "files": list(paths), "content_field": "messages"}
    rows = list(blend_module._iter_rows(source, tmp_path / "blend", False, 0))
    assert [r["messages"][0]["content"] for r in rows] == [f"q{i}" for i in range(40)]

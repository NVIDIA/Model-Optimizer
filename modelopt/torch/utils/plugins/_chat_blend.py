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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic selection and question-isolated splits for chat blends."""

import collections
import hashlib
import importlib.metadata
import inspect
import json
import platform
import random
import re
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset

__all__ = []
_IMAGE = re.compile(
    r"<image\b|<img\b|\[image\]|!\[[^\]]*\]\s*\(|<\|(?:image|vision_start)\|>", re.IGNORECASE
)


def _digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


@lru_cache(maxsize=8)
def _questions(dataset, revision, split, kind):
    base = f"hf://datasets/{dataset}@{revision}"
    if kind == "taco":
        n = 9 if split == "train" else 1
        files = [f"{base}/ALL/{split}-{i:05d}-of-{n:05d}.parquet" for i in range(n)]
        rows = load_dataset(
            "parquet", data_files=files, split="train", streaming=True, columns=["question"]
        )
    elif kind == "apps":
        rows = load_dataset(
            "json", data_files=f"{base}/{split}.jsonl", split="train", streaming=True
        )
    else:
        rows = load_dataset(dataset, revision=revision, split=split, streaming=True)
    result = []
    for row in rows:
        text = row["question"] if kind in ("taco", "apps") else row["description"]
        if kind == "open-r1/codeforces":
            for key in ("input_format", "output_format", "note"):
                if row.get(key):
                    text += f"\n\n{key}\n\n{row[key]}"
            for example in row.get("examples") or []:
                text += f"\n\nInput\n\n{example['input']}\n\nOutput\n\n{example['output']}"
        result.append(text)
    return result


def _restore(row, source, question_sources):
    missing = [
        i
        for i, m in enumerate(row.get("messages") or [])
        if isinstance(m, dict)
        and m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].strip() == "-"
    ]
    if not missing:
        return row, []
    try:
        kind, split, raw_index = row["dataset"], row["split"], row["index"]
        if (
            kind not in ("taco", "apps", "code_contests", "open-r1/codeforces")
            or isinstance(raw_index, bool)
            or not isinstance(raw_index, (int, str))
        ):
            raise ValueError("unsupported problem location")
        index = int(raw_index)
        spec = question_sources[kind]
        if len(missing) != 1 or index < 0 or split not in ("train", "test", "valid", "validation"):
            raise ValueError("ambiguous problem location")
        question = _questions(spec["hf_dataset"], spec["revision"], split, kind)[index]
        if not isinstance(question, str) or not question.strip() or question.strip() == "-":
            raise ValueError("empty original question")
        selection = source.get("split", "") + " ".join(source.get("files", []))
        lang = "python" if "python" in selection else "cpp" if "cpp" in selection else None
        if lang is None:
            raise ValueError("missing programming language")
        messages = [dict(m) for m in row["messages"]]
        comment = "# Your code here" if lang == "python" else "// Your code here"
        messages[missing[0]]["content"] = (
            "You are a helpful and harmless assistant. You should think step-by-step before "
            "responding to the instruction below.\n\n"
            f"Please use {'python' if lang == 'python' else 'c++'} programming language only.\n\n"
            f"You must use ```{lang} for just the final solution code block with the following "
            f"format:\n```{lang}\n{comment}\n```\n\n{question}"
        )
        keys = [
            ("problem", spec["hf_dataset"], spec["revision"], split, index),
            ("problem_text", _digest(question)),
        ]
        return {**row, "messages": messages}, keys
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError(f"Cannot restore {source['hf_dataset']}: uuid={row.get('uuid')}") from e


def _reject(chat, config):
    messages = chat["messages"]
    if not all(
        m.get(k) is None or isinstance(m[k], str)
        for m in messages
        for k in ("content", "reasoning_content")
    ):
        return "non_text_content"
    text = "\n".join(
        str(m.get(k) or "") for m in messages for k in ("content", "reasoning_content")
    )
    if config.get("filter_image_references") and _IMAGE.search(text):
        return "image_reference"
    if config.get("filter_tool_calls") and (
        chat.get("tools") or any(m.get("tool_calls") or m.get("role") == "tool" for m in messages)
    ):
        return "tool_calling"
    if config.get("filter_invalid_reasoning") and any(
        (m.get("content") or "").count("<think>") != (m.get("content") or "").count("</think>")
        for m in messages
        if m.get("role") == "assistant"
    ):
        return "unbalanced_reasoning"
    if not any(
        m.get("role") == "assistant"
        and ((m.get("content") or "").strip() or (m.get("reasoning_content") or "").strip())
        for m in messages
    ):
        return "empty_assistant"
    return None


def _shape(chat):
    messages = chat["messages"]
    users = sum(m.get("role") == "user" for m in messages)
    reasoning = any(
        m.get("reasoning_content") or "<think>" in (m.get("content") or "")
        for m in messages
        if m.get("role") == "assistant"
    )
    return ("document" if not users else "single" if users == 1 else "multi", bool(reasoning))


def _allocate(counts, total):
    n = sum(counts)
    if not n:
        return [0] * len(counts)
    quotas = [total * c // n for c in counts]
    for i in sorted(range(len(counts)), key=lambda i: -(total * counts[i] % n))[
        : total - sum(quotas)
    ]:
        quotas[i] += 1
    return quotas


def prepare_grouped_blend(config_path, config, api):
    """Select unique records, reserve whole validation question groups, and refill source budgets."""
    # Bridge is optional outside the chat SFT path.
    from megatron.bridge.data.conversation_processing import is_text_only_chat_example

    for key in (
        "group_split",
        "shuffle",
        "filter_image_references",
        "filter_tool_calls",
        "filter_invalid_reasoning",
    ):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{key} must be a boolean.")
    for source in [*config["sources"], *config.get("question_sources", {}).values()]:
        if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
            raise ValueError("Grouped chat blends require full dataset revision SHAs in YAML.")
    if config.get("loss_mode", "assistant") != "assistant":
        raise ValueError("Grouped chat blends currently support assistant loss only.")
    if "max_length" in config or "pack_max_tokens" in config:
        raise ValueError("Grouped blends retain master records; omit length filtering/packing.")
    buffer_size = config.get("shuffle_buffer_size", 10000)
    validation_size = config.get("validation_samples", 1000)
    if (
        type(buffer_size) is not int
        or buffer_size < 1
        or type(validation_size) is not int
        or validation_size < 0
    ):
        raise ValueError("Require positive shuffle_buffer_size and nonnegative validation_samples.")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted(config["sources"], key=_digest)
    if len({_digest(s) for s in sources}) != len(sources):
        raise ValueError("Duplicate source configuration.")
    budgets = _allocate([Fraction(str(s["weight"])) for s in sources], config["target_samples"])
    seed = int(config.get("seed", 1234))
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    versions["python"] = platform.python_version()
    recipe = {k: v for k, v in config.items() if k != "output_dir"}
    code = Path(__file__).read_bytes() + Path(api.__file__).read_bytes()
    code += Path(inspect.getfile(api.prepare_messages_for_chat_template)).read_bytes()
    code += Path(inspect.getfile(is_text_only_chat_example)).read_bytes()
    fingerprint = _digest([recipe, versions, hashlib.sha256(code).hexdigest()])
    report_path = output / "report.json"
    if report_path.exists():
        cached = api._read_jsonl(report_path)
        previous = cached[0] if cached and isinstance(cached[0], dict) else {}
        if (
            previous.get("fingerprint") == fingerprint
            and set(previous.get("output_sha256", {}))
            == {"train.jsonl", "validation.jsonl", "manifest.jsonl"}
            and all(
                (output / name).exists()
                and hashlib.file_digest((output / name).open("rb"), "sha256").hexdigest() == sha
                for name, sha in previous["output_sha256"].items()
            )
        ):
            for source in previous["sources"]:
                if source["shortfall"]:
                    print(f"[WARN] Cached source shortfall: {source}", flush=True)
            api._copy_config(config_path, output / "config.yaml")
            return previous["sizes"]
    streams = [
        iter(
            api._iter_rows(
                {**s, "shuffle_buffer_size": buffer_size},
                output,
                bool(config.get("shuffle", True)),
                seed,
            )
        )
        for s in sources
    ]
    rows, provenance, pools = [], [], [[] for _ in sources]
    stats = [collections.Counter() for _ in sources]
    exhausted, seen, identities, parent = set(), {}, {}, []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def fill(source_index, need):
        source, pool, count = sources[source_index], pools[source_index], stats[source_index]
        label = f"{source['hf_dataset']} {source.get('split', source.get('files'))}"
        while len(pool) < need and source_index not in exhausted:
            try:
                raw = next(streams[source_index])
            except StopIteration:
                exhausted.add(source_index)
                break
            count["read"] += 1
            if not isinstance(raw, dict):
                count["invalid_chat"] += 1
                continue
            original = raw
            raw, keys = _restore(raw, source, config.get("question_sources", {}))
            try:
                chat = api._to_chat(
                    raw, source["content_field"], config.get("reasoning_content", "native")
                )
                reason = (
                    "invalid_chat"
                    if chat is None
                    else "media"
                    if not is_text_only_chat_example({**raw, **chat})
                    else _reject(chat, config)
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                reason, chat = "conversion_failed", None
            if reason:
                count[reason] += 1
                continue
            try:
                raw_sha, digest = _digest(original), _digest(chat)
            except TypeError:
                count["non_json_record"] += 1
                continue
            task = {**chat, "messages": chat["messages"][:-1]}
            if (
                not any(m.get("role") == "user" for m in chat["messages"])
                or chat["messages"][-1].get("role") != "assistant"
            ):
                task = chat
            keys.append(("input", _digest(task)))
            i = seen.get(digest, len(rows))
            duplicate = i < len(rows)
            if not duplicate:
                parent.append(i)
                seen[digest] = i
            for key in keys:
                if key in identities:
                    left, right = root(i), root(identities[key])
                    parent[max(left, right)] = min(left, right)
                identities[key] = i
            if duplicate:
                count["exact_duplicate"] += 1
                aliases = provenance[i].setdefault("duplicate_sources", [])
                if source != provenance[i]["source"] and source not in aliases:
                    aliases.append(source)
                continue
            rows.append(chat)
            provenance.append(
                {
                    "source": source,
                    "sample_index": count["read"] - 1,
                    "raw_id": raw.get("uuid", raw.get("id")),
                    "raw_sha256": raw_sha,
                    "record_sha256": digest,
                    "problem_keys": keys,
                }
            )
            pool.append(i)
            count["accepted"] += 1
            if count["accepted"] % 1000 == 0:
                print(f"{label}: {count['accepted']} accepted", flush=True)

    def select():
        sizes = [min(b, len(p)) for b, p in zip(budgets, pools)]
        quotas = _allocate(budgets, min(validation_size, sum(budgets)))
        validation, used, val_counts = [], set(), collections.Counter()
        for pool_index, (p, quota) in enumerate(zip(pools, quotas)):
            strata = collections.defaultdict(list)
            for i in p:
                strata[_shape(rows[i])].append(i)
            categories = sorted(strata)
            targets = _allocate([len(strata[k]) for k in categories], quota)
            for category, want in zip(categories, targets):
                order = sorted(
                    strata[category], key=lambda i: _digest([seed, provenance[i]["record_sha256"]])
                )
                for i in order:
                    if want and root(i) not in used:
                        validation.append(i)
                        val_counts[pool_index] += 1
                        used.add(root(i))
                        want -= 1
            # If a stratum has too few distinct questions, use other strata, never split a group.
            for i in sorted(p, key=lambda i: _digest([seed, provenance[i]["record_sha256"]])):
                if val_counts[pool_index] >= quota:
                    break
                if root(i) not in used:
                    validation.append(i)
                    val_counts[pool_index] += 1
                    used.add(root(i))
        train = []
        for pool_index, (p, size) in enumerate(zip(pools, sizes)):
            n_val = val_counts[pool_index]
            train.extend([i for i in p if root(i) not in used][: size - n_val])
        return train, validation, quotas

    for i, budget in enumerate(budgets):
        fill(i, budget)
    while True:
        train, validation, quotas = select()
        selected_set, val_set = set(train + validation), set(validation)
        deficits = []
        for i, (pool, budget, quota) in enumerate(zip(pools, budgets, quotas)):
            have = sum(j in selected_set for j in pool)
            val_have = sum(j in val_set for j in pool)
            if (have < budget or val_have < quota) and i not in exhausted:
                deficits.append(i)
        if not deficits:
            break
        for i in deficits:
            fill(i, len(pools[i]) + max(32, budgets[i] // 10))
    streams.clear()
    splits = {"train": train, "validation": validation}
    manifest = []
    for name, indices in splits.items():
        random.Random(seed).shuffle(indices)
        api._write_jsonl(output / f"{name}.jsonl", [rows[i] for i in indices])
        manifest.extend(
            {**provenance[i], "split": name, "row": j, "group_id": root(i)}
            for j, i in enumerate(indices)
        )
    api._write_jsonl(output / "manifest.jsonl", manifest)
    source_report = []
    split_sets = {name: set(indices) for name, indices in splits.items()}
    for source, pool, budget, count in zip(sources, pools, budgets, stats):
        actual = {name: sum(i in indices for i in pool) for name, indices in split_sets.items()}
        shortage = budget - sum(actual.values())
        source_report.append(
            {
                "source": source,
                "target": budget,
                **actual,
                "shortfall": shortage,
                "counts": dict(count),
                "formats": {
                    name: dict(
                        collections.Counter(str(_shape(rows[i])) for i in pool if i in indices)
                    )
                    for name, indices in split_sets.items()
                },
            }
        )
        if shortage:
            print(
                f"[WARN] {source}: target={budget}, accepted={sum(actual.values())}, "
                f"shortfall={shortage}; source exhausted",
                flush=True,
            )
    sizes = {name: len(indices) for name, indices in splits.items()}
    if not sizes["train"] or sizes["validation"] < validation_size:
        print(f"[WARN] Split sizes {sizes}; requested validation={validation_size}", flush=True)
    report = {
        "fingerprint": fingerprint,
        "pipeline_sha256": hashlib.sha256(code).hexdigest(),
        "sources": source_report,
        "sizes": sizes,
        "versions": versions,
        "config": recipe,
        "output_sha256": {
            name: hashlib.file_digest((output / name).open("rb"), "sha256").hexdigest()
            for name in ("train.jsonl", "validation.jsonl", "manifest.jsonl")
        },
    }
    api._copy_config(config_path, output / "config.yaml")
    api._write_jsonl(output / "report.tmp.jsonl", [report])
    (output / "report.tmp.jsonl").replace(report_path)
    return sizes

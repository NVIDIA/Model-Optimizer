# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize and filter generation logs for DeepSeek-V4 native-MTP training.

The Nemotron generation logs keep the prompt in ``messages`` and the generated
assistant turn in top-level ``reasoning_content`` / ``generation`` fields.  The
hidden-state collectors expect a complete conversation, so this script appends
that assistant turn, verifies it against ``serialized_output``, tokenizes with
the exact DSV4 template, and rejects incomplete or overlong trajectories.

Selection is deterministic and token-budgeted.  A bounded hash reservoir is
maintained independently for math/chat/STEM/code so the much larger chat source
does not dominate the training mixture.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import load_chat_template, tokenize_with_loss_mask, verify_generation_tags

DEFAULT_CATEGORY_WEIGHTS = {"math": 1.0, "chat": 1.0, "stem": 1.0, "code": 1.0}
DSV4_CONTROL_MARKERS = (
    "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
    "<\uff5cend\u2581of\u2581sentence\uff5c>",
    "<\uff5cUser\uff5c>",
    "<\uff5cAssistant\uff5c>",
    "<think>",
    "</think>",
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UNSAFE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class Candidate:
    """One fully validated conversation eligible for deterministic selection."""

    conversation_id: str
    category: str
    num_tokens: int
    score: int
    serialized: str
    canonical_hash: str


class ShardWriter:
    """Write JSONL records into fixed-size shards without overwriting prior output."""

    def __init__(self, output_dir: Path, split: str, records_per_shard: int) -> None:
        self.output_dir = output_dir
        self.split = split
        self.records_per_shard = records_per_shard
        self.num_records = 0
        self.num_tokens = 0
        self._file = None

    def write(self, candidate: Candidate) -> None:
        shard_index = self.num_records // self.records_per_shard
        if self.num_records % self.records_per_shard == 0:
            if self._file is not None:
                self._file.close()
            path = self.output_dir / f"{self.split}-{shard_index:05d}.jsonl"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing shard: {path}")
            self._file = path.open("w", encoding="utf-8")
        assert self._file is not None
        self._file.write(candidate.serialized)
        self._file.write("\n")
        self.num_records += 1
        self.num_tokens += candidate.num_tokens

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _parse_args() -> argparse.Namespace:
    default_template = Path(__file__).parents[1] / "input_conversations" / "dsv4_thinking.jinja"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        action="append",
        required=True,
        help="Generation-log JSONL. Repeat once per source file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, default=default_template)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=1_000_000,
        help="Maximum selected total tokens; 0 writes every eligible row.",
    )
    parser.add_argument(
        "--category-weight",
        action="append",
        default=[],
        metavar="CATEGORY=WEIGHT",
        help="Override the default equal math/chat/stem/code token mixture.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--records-per-shard", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--debug-max-input-rows",
        type=int,
        default=None,
        help="Stop after this many input rows; intended only for pipeline smoke tests.",
    )
    parser.add_argument(
        "--write-rejects",
        action="store_true",
        help="Write conversation id and rejection reason to rejected.jsonl.",
    )
    return parser.parse_args()


def _contains_control_marker(text: str) -> bool:
    return _UNSAFE_CONTROL.search(text) is not None or any(
        marker in text for marker in DSV4_CONTROL_MARKERS
    )


def _is_degenerate(text: str) -> bool:
    """Detect only high-confidence repetition failures, without judging content quality."""
    if re.search(r"(.)\1{255,}", text, flags=re.DOTALL):
        return True
    tokens = text.split()
    if len(tokens) < 64:
        return False
    return Counter(tokens).most_common(1)[0][1] / len(tokens) > 0.5


def normalize_generation_row(row: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return a canonical first-turn conversation or a stable rejection reason."""
    if not isinstance(row, Mapping):
        return None, "row_not_object"

    conversation_id = row.get("uuid") or row.get("conversation_id") or row.get("id")
    if not isinstance(conversation_id, str) or not _SAFE_ID.fullmatch(conversation_id):
        return None, "invalid_conversation_id"
    category = str(row.get("category", "")).strip().lower()
    if category not in DEFAULT_CATEGORY_WEIGHTS:
        return None, "unsupported_category"
    if row.get("finish_reason") != "stop":
        return None, "incomplete_finish_reason"

    reasoning = row.get("reasoning_content")
    content = row.get("generation")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, "empty_reasoning"
    if not isinstance(content, str) or not content.strip():
        return None, "empty_final_answer"
    if _contains_control_marker(reasoning) or _contains_control_marker(content):
        return None, "embedded_control_marker"
    if _is_degenerate(reasoning) or _is_degenerate(content):
        return None, "degenerate_repetition"

    serialized_output = row.get("serialized_output")
    if (
        not isinstance(serialized_output, list)
        or len(serialized_output) != 1
        or not isinstance(serialized_output[0], Mapping)
    ):
        return None, "invalid_serialized_output"
    assistant = serialized_output[0]
    if assistant.get("role") != "assistant":
        return None, "serialized_output_not_assistant"
    if assistant.get("reasoning_content") != reasoning or assistant.get("content") != content:
        return None, "serialized_output_mismatch"
    if assistant.get("tool_calls") or assistant.get("function_call"):
        return None, "tool_or_function_call"

    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, "messages_not_list"
    roles = [message.get("role") if isinstance(message, Mapping) else None for message in messages]
    if roles not in (["user"], ["system", "user"]):
        return None, "unsupported_prompt_roles"

    canonical_messages: list[dict[str, str]] = []
    for message in messages:
        assert isinstance(message, Mapping)
        prompt_content = message.get("content")
        if not isinstance(prompt_content, str):
            return None, "non_string_prompt_content"
        if message.get("role") == "user" and not prompt_content.strip():
            return None, "empty_user_prompt"
        if _contains_control_marker(prompt_content):
            return None, "embedded_control_marker"
        canonical_messages.append({"role": str(message["role"]), "content": prompt_content})

    canonical_messages.append(
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": content,
        }
    )
    return {
        "conversation_id": conversation_id,
        "category": category,
        "messages": canonical_messages,
    }, None


def _category_weights(overrides: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_CATEGORY_WEIGHTS)
    for value in overrides:
        try:
            category, raw_weight = value.split("=", 1)
            category = category.strip().lower()
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(
                f"Invalid --category-weight {value!r}; expected CATEGORY=WEIGHT"
            ) from error
        if category not in weights:
            raise ValueError(f"Unknown category in --category-weight: {category!r}")
        if weight < 0:
            raise ValueError(f"Category weight must be non-negative, got {value!r}")
        weights[category] = weight
    if sum(weights.values()) <= 0:
        raise ValueError("At least one category weight must be positive")
    return weights


def _budgets(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total == 0:
        return dict.fromkeys(weights, 0)
    total_weight = sum(weights.values())
    budgets = {category: int(total * weight / total_weight) for category, weight in weights.items()}
    remainder = total - sum(budgets.values())
    for category in sorted(weights, key=lambda key: (-weights[key], key)):
        if remainder == 0:
            break
        budgets[category] += 1
        remainder -= 1
    return budgets


def _hash_int(seed: int, namespace: str, value: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{namespace}:{value}".encode(), digest_size=8, person=b"dsv4-mtp"
    ).digest()
    return int.from_bytes(digest, "big")


def _canonical_hash(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        record["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _iter_rows(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if line.strip():
                    yield path, line_number, line


def _length_bucket(num_tokens: int) -> str:
    if num_tokens <= 512:
        return "0001-0512"
    if num_tokens <= 1024:
        return "0513-1024"
    if num_tokens <= 1536:
        return "1025-1536"
    return "1537-2048+"


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_seq_len < 2:
        raise ValueError("--max-seq-len must be at least 2")
    if args.token_budget < 0:
        raise ValueError("--token-budget must be non-negative")
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if args.records_per_shard < 1:
        raise ValueError("--records-per-shard must be positive")
    if args.debug_max_input_rows is not None and args.debug_max_input_rows < 1:
        raise ValueError("--debug-max-input-rows must be positive")
    missing = [str(path) for path in args.input_file if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input JSONL files do not exist: {missing}")


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    weights = _category_weights(args.category_weight)
    category_budgets = _budgets(args.token_budget, weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    occupied = list(args.output_dir.glob("*.jsonl")) + list(args.output_dir.glob("report.json"))
    if occupied:
        raise FileExistsError(
            f"Refusing to mix with existing preparation outputs in {args.output_dir}: {occupied[:3]}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = load_chat_template(args.chat_template)
    verify_generation_tags(tokenizer.chat_template)

    rejections: Counter[str] = Counter()
    input_by_category: Counter[str] = Counter()
    eligible_by_category: Counter[str] = Counter()
    eligible_tokens_by_category: Counter[str] = Counter()
    length_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    tokenization_errors: list[str] = []
    seen_ids: set[str] = set()
    seen_conversations: set[str] = set()
    heaps: dict[str, list[tuple[int, str, Candidate]]] = defaultdict(list)
    heap_tokens: Counter[str] = Counter()
    all_candidates: list[Candidate] = []
    input_rows = 0
    malformed_json = 0

    reject_file = None
    if args.write_rejects:
        reject_file = (args.output_dir / "rejected.jsonl").open("w", encoding="utf-8")

    def reject(reason: str, path: Path, line_number: int, conversation_id: Any = None) -> None:
        rejections[reason] += 1
        if reject_file is not None:
            reject_file.write(
                json.dumps(
                    {
                        "source": str(path),
                        "line": line_number,
                        "conversation_id": conversation_id,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    try:
        for path, line_number, line in _iter_rows(args.input_file):
            if (
                args.debug_max_input_rows is not None
                and input_rows >= args.debug_max_input_rows
            ):
                break
            input_rows += 1
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                malformed_json += 1
                reject("invalid_json", path, line_number)
                continue
            if isinstance(row, Mapping):
                input_by_category[str(row.get("category", "")).strip().lower()] += 1
            record, reason = normalize_generation_row(row)
            if reason is not None:
                reject(
                    reason,
                    path,
                    line_number,
                    row.get("uuid") if isinstance(row, Mapping) else None,
                )
                continue
            assert record is not None
            conversation_id = record["conversation_id"]
            if conversation_id in seen_ids:
                reject("duplicate_conversation_id", path, line_number, conversation_id)
                continue
            seen_ids.add(conversation_id)
            canonical_hash = _canonical_hash(record)
            if canonical_hash in seen_conversations:
                reject("duplicate_conversation", path, line_number, conversation_id)
                continue
            seen_conversations.add(canonical_hash)

            try:
                input_ids, loss_mask = tokenize_with_loss_mask(
                    tokenizer, record["messages"], answer_only_loss=True
                )
            except Exception as error:
                if len(tokenization_errors) < 10:
                    tokenization_errors.append(
                        f"{path}:{line_number}: {type(error).__name__}: {error}"
                    )
                reject("tokenization_error", path, line_number, conversation_id)
                continue
            num_tokens = int(input_ids.shape[1])
            if num_tokens > args.max_seq_len:
                reject("overlong", path, line_number, conversation_id)
                continue
            supervised_tokens = int(loss_mask.sum())
            if supervised_tokens in (0, num_tokens) or int(loss_mask[-1]) != 1:
                reject("invalid_assistant_mask", path, line_number, conversation_id)
                continue

            category = record["category"]
            eligible_by_category[category] += 1
            eligible_tokens_by_category[category] += num_tokens
            length_buckets[category][_length_bucket(num_tokens)] += 1
            record["num_tokens"] = num_tokens
            record["num_supervised_tokens"] = supervised_tokens
            serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            score = _hash_int(args.seed, "selection", canonical_hash)
            candidate = Candidate(
                conversation_id=conversation_id,
                category=category,
                num_tokens=num_tokens,
                score=score,
                serialized=serialized,
                canonical_hash=canonical_hash,
            )

            if args.token_budget == 0:
                all_candidates.append(candidate)
                continue
            category_budget = category_budgets[category]
            if category_budget == 0 or num_tokens > category_budget:
                continue
            heapq.heappush(heaps[category], (-score, conversation_id, candidate))
            heap_tokens[category] += num_tokens
            while heap_tokens[category] > category_budget:
                _, _, removed = heapq.heappop(heaps[category])
                heap_tokens[category] -= removed.num_tokens
    finally:
        if reject_file is not None:
            reject_file.close()

    if args.token_budget > 0:
        all_candidates = [entry[2] for heap in heaps.values() for entry in heap]
    all_candidates.sort(key=lambda candidate: (candidate.score, candidate.conversation_id))

    train_writer = ShardWriter(args.output_dir, "train", args.records_per_shard)
    validation_writer = ShardWriter(args.output_dir, "validation", args.records_per_shard)
    selected_by_category: Counter[str] = Counter()
    selected_tokens_by_category: Counter[str] = Counter()
    try:
        for candidate in all_candidates:
            selected_by_category[candidate.category] += 1
            selected_tokens_by_category[candidate.category] += candidate.num_tokens
            split_score = _hash_int(args.seed, "split", candidate.canonical_hash) / 2**64
            writer = validation_writer if split_score < args.validation_fraction else train_writer
            writer.write(candidate)
    finally:
        train_writer.close()
        validation_writer.close()

    report = {
        "configuration": {
            "input_files": [str(path) for path in args.input_file],
            "model_dir": str(args.model_dir),
            "chat_template": str(args.chat_template),
            "max_seq_len": args.max_seq_len,
            "token_budget": args.token_budget,
            "category_weights": weights,
            "category_budgets": category_budgets,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "debug_max_input_rows": args.debug_max_input_rows,
        },
        "input_rows": input_rows,
        "malformed_json": malformed_json,
        "input_by_category": dict(sorted(input_by_category.items())),
        "rejections": dict(sorted(rejections.items())),
        "eligible_rows": sum(eligible_by_category.values()),
        "eligible_tokens": sum(eligible_tokens_by_category.values()),
        "eligible_by_category": dict(sorted(eligible_by_category.items())),
        "eligible_tokens_by_category": dict(sorted(eligible_tokens_by_category.items())),
        "length_buckets_by_category": {
            category: dict(sorted(buckets.items()))
            for category, buckets in sorted(length_buckets.items())
        },
        "selected_rows": len(all_candidates),
        "selected_tokens": sum(selected_tokens_by_category.values()),
        "selected_by_category": dict(sorted(selected_by_category.items())),
        "selected_tokens_by_category": dict(sorted(selected_tokens_by_category.items())),
        "train_rows": train_writer.num_records,
        "train_tokens": train_writer.num_tokens,
        "validation_rows": validation_writer.num_records,
        "validation_tokens": validation_writer.num_tokens,
        "tokenization_error_samples": tokenization_errors,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(_parse_args())

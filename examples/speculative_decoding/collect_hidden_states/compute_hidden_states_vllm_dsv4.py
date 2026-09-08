# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect native DeepSeek-V4 MTP target features from an instrumented vLLM server.

The server must run vLLM's ``extract_hidden_states`` speculative method with
the DSV4 MTP feature patch documented next to this script.  It returns a
request-scoped safetensors file whose ``hidden_states`` tensor is
``[sequence, hc_mult + 1, hidden]``: the raw HC streams followed by the
normalized LM-head input.  This client sends token IDs, never decoded text, so
server and client tokenization cannot drift.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import torch
from common import (
    add_answer_only_loss_args,
    load_chat_template,
    tokenize_with_loss_mask,
    verify_generation_tags,
)
from safetensors import safe_open
from transformers import AutoTokenizer

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Tokenizer directory.")
    parser.add_argument("--served-model", required=True, help="Model name accepted by vLLM.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shared-storage-root",
        type=Path,
        required=True,
        help="Allowlisted root used by vLLM's ExampleHiddenStatesConnector.",
    )
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--file-wait-timeout", type=float, default=600.0)
    parser.add_argument("--debug-max-num-conversations", type=int, default=None)
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--dp-world-size", type=int, default=1)
    parser.add_argument("--hc-mult", type=int, default=4)
    add_answer_only_loss_args(parser)
    return parser.parse_args()


def _iter_jsonl(input_data: Path):
    if input_data.is_file() and input_data.suffix == ".jsonl":
        paths = [input_data]
    elif input_data.is_dir():
        paths = sorted(input_data.glob("*.jsonl"))
    else:
        raise ValueError(f"--input-data must be a JSONL file or directory, got {input_data}")
    if not paths:
        raise ValueError(f"No JSONL files found under {input_data}")
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except ValueError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
                if not isinstance(entry, dict):
                    raise TypeError(f"Expected an object at {path}:{line_number}")
                yield entry


def _path_under_root(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _wait_for_connector_file(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    lock_path = Path(f"{path}.lock")
    while not path.exists() and not lock_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for vLLM feature file {path}")
        time.sleep(0.05)


def _load_connector_features(
    path: Path,
    *,
    shared_storage_root: Path,
    hc_mult: int,
    wait_timeout: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load, validate, and remove one request-scoped vLLM feature file."""
    if not _path_under_root(path, shared_storage_root):
        raise ValueError(f"vLLM returned a path outside --shared-storage-root: {path}")
    _wait_for_connector_file(path, wait_timeout)
    lock_path = Path(f"{path}.lock")
    lock_file = lock_path.open("r") if lock_path.exists() else None
    try:
        if lock_file is not None:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
        with safe_open(path, framework="pt") as handle:
            token_ids = handle.get_tensor("token_ids")
            hidden_states = handle.get_tensor("hidden_states")
    finally:
        if lock_file is not None:
            lock_file.close()

    with contextlib.suppress(OSError):
        path.unlink()
    with contextlib.suppress(OSError):
        lock_path.unlink()

    if token_ids.ndim != 1:
        raise ValueError(f"token_ids must be [S], got {tuple(token_ids.shape)}")
    expected = (token_ids.shape[0], hc_mult + 1)
    if hidden_states.ndim != 3 or tuple(hidden_states.shape[:2]) != expected:
        raise ValueError(
            "Instrumented vLLM must return hidden_states shaped "
            f"[S, hc_mult + 1, H] with hc_mult={hc_mult}; got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(f"DSV4 MTP target features must remain BF16; got {hidden_states.dtype}")
    return (
        token_ids,
        hidden_states[:, :hc_mult, :].contiguous(),
        hidden_states[:, hc_mult, :].contiguous(),
    )


def _atomic_torch_save(payload: dict[str, Any], output_file: Path) -> None:
    """Write a feature dump atomically so interrupted jobs leave no valid-looking partial file."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_file.parent,
        prefix=f".{output_file.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_file)
    finally:
        with contextlib.suppress(OSError):
            temporary_path.unlink()


async def main(args: argparse.Namespace) -> None:
    if args.max_seq_len < 2:
        raise ValueError("--max-seq-len must be at least 2")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.dp_world_size < 1 or not 0 <= args.dp_rank < args.dp_world_size:
        raise ValueError("--dp-rank must be in [0, --dp-world-size)")
    if args.hc_mult < 1:
        raise ValueError("--hc-mult must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template = load_chat_template(args.chat_template)
    if template is not None:
        tokenizer.chat_template = template
    if args.answer_only_loss:
        verify_generation_tags(tokenizer.chat_template)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_storage_root = args.shared_storage_root.resolve()
    if not shared_storage_root.is_dir():
        raise FileNotFoundError(
            f"--shared-storage-root must be an existing directory: {shared_storage_root}"
        )
    timeout = httpx.Timeout(args.request_timeout, connect=30.0)
    counters: Counter[str] = Counter()
    error_samples: list[str] = []

    async def process(client: httpx.AsyncClient, entry: dict[str, Any]) -> None:
        conversation_id = entry.get("conversation_id") or entry.get("uuid") or entry.get("id")
        conversations = entry.get("messages") or entry.get("conversations")
        if (
            not isinstance(conversation_id, str)
            or not _SAFE_ID.fullmatch(conversation_id)
            or not isinstance(conversations, list)
        ):
            counters["invalid"] += 1
            return
        output_file = args.output_dir / f"{conversation_id}.pt"
        if output_file.exists():
            counters["existing"] += 1
            return
        input_ids, loss_mask = await asyncio.to_thread(
            tokenize_with_loss_mask,
            tokenizer,
            conversations,
            args.answer_only_loss,
        )
        num_tokens = input_ids.shape[1]
        if num_tokens <= 1:
            counters["too_short"] += 1
            return
        if num_tokens > args.max_seq_len:
            counters["overlong"] += 1
            return

        response = await client.post(
            f"{args.base_url.rstrip('/')}/v1/completions",
            json={
                "model": args.served_model,
                "prompt": input_ids.squeeze(0).tolist(),
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        response.raise_for_status()
        body = response.json()
        feature_path = (body.get("kv_transfer_params") or {}).get("hidden_states_path")
        if not isinstance(feature_path, str):
            raise RuntimeError(
                "vLLM response did not contain kv_transfer_params.hidden_states_path"
            )
        server_ids, raw_hiddens, teacher_hiddens = await asyncio.to_thread(
            _load_connector_features,
            Path(feature_path),
            shared_storage_root=shared_storage_root,
            hc_mult=args.hc_mult,
            wait_timeout=args.file_wait_timeout,
        )
        client_ids = input_ids.squeeze(0).to(server_ids.dtype)
        if not torch.equal(server_ids, client_ids):
            raise RuntimeError(
                f"Server token IDs drifted for {conversation_id}: "
                f"client={client_ids.shape[0]}, server={server_ids.shape[0]}"
            )
        await asyncio.to_thread(
            _atomic_torch_save,
            {
                "input_ids": input_ids.squeeze(0).cpu(),
                "target_mtp_hidden_states": raw_hiddens.cpu(),
                "target_lm_head_hidden_states": teacher_hiddens.cpu(),
                "loss_mask": loss_mask.cpu(),
                "conversation_id": conversation_id,
                "feature_backend": "vllm",
            },
            output_file,
        )
        counters["written"] += 1

    async with httpx.AsyncClient(timeout=timeout) as client:
        pending: set[asyncio.Task] = set()
        selected = 0
        for row_index, entry in enumerate(_iter_jsonl(args.input_data)):
            if row_index % args.dp_world_size != args.dp_rank:
                continue
            if (
                args.debug_max_num_conversations is not None
                and selected >= args.debug_max_num_conversations
            ):
                break
            selected += 1
            pending.add(asyncio.create_task(process(client, entry)))
            if len(pending) < args.concurrency:
                continue
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            results = await asyncio.gather(*done, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    counters["errors"] += 1
                    if len(error_samples) < 10:
                        error_samples.append(f"{type(result).__name__}: {result}")
        if pending:
            results = await asyncio.gather(*pending, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    counters["errors"] += 1
                    if len(error_samples) < 10:
                        error_samples.append(f"{type(result).__name__}: {result}")

    summary = {
        "dp_rank": args.dp_rank,
        "dp_world_size": args.dp_world_size,
        "counts": dict(sorted(counters.items())),
        "error_samples": error_samples,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if counters["errors"]:
        raise RuntimeError(f"vLLM feature collection encountered {counters['errors']} errors")


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))

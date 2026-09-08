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

"""Dump DeepSeek-V4 target features for native MTP boost training.

The DeepSeek-V4 release's inference implementation is intentionally used only
under ``torch.inference_mode`` here.  It supplies the target's final
Hyper-Connections state and the normalized input to its language-model head;
the native MTP training implementation consumes these tensors directly or
from offline dumps.

Before running this script, convert the downloaded Hugging Face checkpoint
with the release's ``inference/convert.py`` script. Multi-rank launches consume
the matching model-parallel checkpoint directory and write features on rank 0.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch
from common import (
    add_answer_only_loss_args,
    load_chat_template,
    tokenize_with_loss_mask,
    verify_generation_tags,
)
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from modelopt.torch.speculative.mtp.deepseek_v4 import dsv4_target_features, load_dsv4_target_model

# Some DeepSeek tokenizer templates intentionally discard the reasoning prefix from
# *previous* assistant turns.  Native MTP training needs those tokens, so remove that
# exact opt-out fragment when it is present.  This does not remove thinking content.
STRIP_REASONING_CHAT_TEMPLATE_FRAGMENT = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _atomic_torch_save(payload: dict[str, Any], output_file: Path) -> None:
    """Write a feature dump atomically so interrupted jobs cannot leave a partial sample."""
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect native DeepSeek-V4 MTP target features from conversations."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Downloaded DeepSeek-V4 checkpoint directory containing inference/ and the tokenizer.",
    )
    parser.add_argument(
        "--converted-checkpoint",
        type=Path,
        required=True,
        help=(
            "Vendor-converted checkpoint file, or an MP directory containing "
            "model{rank}-mp{world_size}.safetensors files."
        ),
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="JSONL file or directory of JSONL conversation files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=3072)
    parser.add_argument("--debug-max-num-conversations", type=int, default=None)
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--dp-world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    add_answer_only_loss_args(parser)
    return parser.parse_args()


def _load_conversations(input_data: Path):
    if input_data.is_file() and input_data.suffix == ".jsonl":
        return load_dataset("json", data_files=str(input_data), split="train")
    if input_data.is_dir():
        return load_dataset("json", data_files={"train": f"{input_data}/*.jsonl"}, split="train")
    raise ValueError(f"--input-data must be a .jsonl file or JSONL directory, got {input_data}")


def _prepare_tokenizer(model_dir: Path, args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template = load_chat_template(args.chat_template)
    if template is not None:
        tokenizer.chat_template = template
    if tokenizer.chat_template is not None:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            STRIP_REASONING_CHAT_TEMPLATE_FRAGMENT, ""
        )
    if args.answer_only_loss:
        verify_generation_tags(tokenizer.chat_template)
    return tokenizer


def main(args: argparse.Namespace) -> None:
    if args.dp_world_size < 1 or not 0 <= args.dp_rank < args.dp_world_size:
        raise ValueError("--dp-rank must be in [0, --dp-world-size)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but CUDA is unavailable")

    mp_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    mp_rank = int(os.environ.get("RANK", "0"))
    if mp_world_size > 1:
        if args.dp_rank != 0 or args.dp_world_size != 1:
            raise ValueError("--dp-rank/--dp-world-size cannot be combined with model parallelism")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        args.device = f"cuda:{local_rank}"

    dataset = _load_conversations(args.input_data)
    if args.dp_world_size > 1:
        dataset = dataset.shard(num_shards=args.dp_world_size, index=args.dp_rank)
    if args.debug_max_num_conversations is not None:
        dataset = dataset.select(range(min(args.debug_max_num_conversations, len(dataset))))

    is_writer = mp_rank == 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _prepare_tokenizer(args.model_dir, args)
    device = torch.device(args.device)
    model = load_dsv4_target_model(
        args.model_dir,
        args.converted_checkpoint,
        device,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
    )

    skipped = 0
    written = 0
    skipped_too_long = 0
    for entry in tqdm(dataset, desc=f"DSV4 MTP features DP#{args.dp_rank}", disable=not is_writer):
        conversation_id = entry.get("conversation_id") or entry.get("uuid") or entry.get("id")
        if not isinstance(conversation_id, str) or not _SAFE_ID.fullmatch(conversation_id):
            skipped += 1
            continue
        output_file = args.output_dir / f"{conversation_id}.pt"
        output_exists = torch.tensor(int(output_file.exists()), device=device)
        if mp_world_size > 1:
            torch.distributed.broadcast(output_exists, src=0)
        if output_exists.item():
            continue

        conversations = entry.get("messages") or entry.get("conversations")
        if not isinstance(conversations, list):
            skipped += 1
            continue
        input_ids, loss_mask = tokenize_with_loss_mask(
            tokenizer, conversations, args.answer_only_loss
        )
        if input_ids.shape[1] <= 1:
            skipped += 1
            continue
        if input_ids.shape[1] > args.max_seq_len:
            # Never retain a prefix that ends in the middle of reasoning or before
            # the final answer. Data preparation is responsible for selecting a
            # complete conversation that fits the training sequence length.
            skipped_too_long += 1
            continue

        raw_hiddens, teacher_hiddens = dsv4_target_features(model, input_ids.to(device))
        if is_writer:
            _atomic_torch_save(
                {
                    "input_ids": input_ids.squeeze(0).cpu(),
                    "target_mtp_hidden_states": raw_hiddens.squeeze(0).cpu(),
                    "target_lm_head_hidden_states": teacher_hiddens.squeeze(0).cpu(),
                    "loss_mask": loss_mask.cpu(),
                    "conversation_id": str(conversation_id),
                },
                output_file,
            )
        written += 1

    if mp_world_size > 1:
        torch.distributed.barrier()
    if is_writer:
        print(
            f"Wrote {written} DSV4 MTP feature files; skipped {skipped} invalid rows; "
            f"skipped {skipped_too_long} overlong rows."
        )


if __name__ == "__main__":
    main(_parse_args())

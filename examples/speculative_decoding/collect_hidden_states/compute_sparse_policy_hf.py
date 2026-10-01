# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dump a base model's top-k deployment policy for external draft training.

The counterpart to ``compute_hidden_states_hf.py``: instead of a hidden state per
token it stores the base's truncated next-token distribution, which is what the
TVD objectives actually consume. The dump is far smaller and needs no base
lm_head at training time; feed it to ``main.py`` via ``data.sparse_data_path``.

Output is gzipped JSONL, one conversation per line, matching the schema
``modelopt.torch.speculative.external.sparse_data`` reads.
"""

import argparse
import gzip
import json
import os
from pathlib import Path

import torch
from common import (
    add_answer_only_loss_args,
    load_chat_template,
    tokenize_with_loss_mask,
    verify_generation_tags,
)
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump a base model's top-k deployment policy over conversation responses."
    )
    parser.add_argument("--model", type=str, required=True, help="Base (target) model.")
    parser.add_argument(
        "--input-data", type=str, required=True, help=".jsonl file or directory of .jsonl files."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=3072)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Width of the stored policy. Must match external.external_top_k at training time.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus applied after top-k, matching external.external_top_p.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Serving temperature of the base."
    )
    parser.add_argument(
        "--shard-size", type=int, default=2000, help="Conversations per output shard."
    )
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--dp-world-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--debug-max-num-conversations", type=int, default=None)
    add_answer_only_loss_args(parser)
    return parser.parse_args()


def deployment_policy(logits, top_k: int, top_p: float, temperature: float):
    """Top-k then top-p over the base's next-token distribution, renormalised.

    Returns ``(ids, log_probs)`` of width ``top_k``. Candidates the nucleus excluded keep
    their slot with ``-inf``, the convention ``sparse_data`` decodes to zero probability,
    so every row has the same width and sums to 1 -- it warns when a row does not, because
    a policy that is itself a truncation of a wider nucleus makes the objective charge the
    draft for mass on tokens the teacher actually supported.
    """
    p = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
    top_p_vals, top_ids = p.topk(top_k, dim=-1)
    # Renormalise before the nucleus, exactly as the serving filter does: applying
    # top-p to un-redistributed top-k mass makes it a no-op whenever that mass is
    # already below top_p.
    top_p_vals = top_p_vals / top_p_vals.sum(-1, keepdim=True).clamp_min(1e-9)
    if top_p and top_p < 1.0:
        keep = (top_p_vals.cumsum(-1) - top_p_vals) < top_p
        top_p_vals = top_p_vals * keep
        top_p_vals = top_p_vals / top_p_vals.sum(-1, keepdim=True).clamp_min(1e-9)
    return top_ids, top_p_vals.log()


def scored_spans(loss_mask):
    """Contiguous runs of supervised positions, as half-open ``(start, end)`` pairs.

    A multi-turn conversation masks each assistant turn separately, so the supervised
    positions are disjoint; treating everything after the first as the response would
    score the later user turns too.
    """
    spans, run_start, prev = [], None, None
    for i in loss_mask.squeeze(0).nonzero().flatten().tolist():
        if run_start is None:
            run_start = i
        elif i != prev + 1:
            spans.append((run_start, prev + 1))
            run_start = i
        prev = i
    if run_start is not None:
        spans.append((run_start, prev + 1))
    return spans


def policy_record(
    conversation_id,
    ids,
    first: int,
    end: int,
    logits,
    top_k: int,
    top_p: float,
    temperature: float,
    source=None,
):
    """Build one shard record for a single supervised span.

    Args:
        first: index of the span's first supervised token; everything before it is context.
            Must be >= 1 -- position 0 has no preceding context to condition on.
        end: one past the span's last supervised token.
        logits: the base's logits over the whole sequence.
    """
    if first < 1:
        raise ValueError(
            f"supervised span must start at index >= 1, got {first}: position 0 has no "
            "preceding context, so no policy can be derived for it."
        )
    # Position t of the stored policy is the distribution token ids[t] was drawn from,
    # conditioned on ids[:t] -- that is the model's logits at t - 1. Dropping this shift
    # still yields a falling loss curve, against the wrong target.
    tok, lp = deployment_policy(logits[first - 1 : end - 1], top_k, top_p, temperature)
    record = {
        "id": str(conversation_id),
        "prompt_ids": ids[:first].tolist(),
        "gen_ids": ids[first:end].tolist(),
        "topk_tok": tok.cpu().tolist(),
        "topk_lp": lp.float().cpu().tolist(),
    }
    if source is not None:
        record["source"] = source
    return record


def main(args: argparse.Namespace) -> None:
    if args.input_data.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=args.input_data, split="train")
    elif os.path.isdir(args.input_data):
        dataset = load_dataset(
            "json", data_files={"train": f"{args.input_data}/*.jsonl"}, split="train"
        )
    else:
        raise ValueError(f"input_data must be a .jsonl file or directory, got: {args.input_data}")
    print(f"Loaded {len(dataset)} conversations from {args.input_data}")

    if args.dp_world_size > 1:
        dataset = dataset.shard(num_shards=args.dp_world_size, index=args.dp_rank)
    if args.debug_max_num_conversations is not None:
        dataset = dataset.select(range(args.debug_max_num_conversations))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto", trust_remote_code=args.trust_remote_code
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    override_template = load_chat_template(args.chat_template)
    if override_template is not None:
        tokenizer.chat_template = override_template
    if args.answer_only_loss:
        verify_generation_tags(tokenizer.chat_template)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_skipped = num_invalid = num_success = 0
    shard_idx = 0
    written_in_shard = 0
    shard_file = None

    def open_shard(idx: int):
        # Write-then-rename: a partial shard left by an interrupted run would otherwise
        # be picked up by the trainer's glob and decoded as damaged input.
        tmp = args.output_dir / f"policy-{args.dp_rank:03d}-{idx:05d}.jsonl.gz.tmp"
        return gzip.open(tmp, "wt"), tmp

    try:
        shard_file, shard_tmp = open_shard(shard_idx)
        for entry in tqdm(dataset, desc=f"DP#{args.dp_rank} dumping policy"):
            conversation_id = entry.get("conversation_id", entry.get("uuid"))
            conversations = entry.get("messages") or entry.get("conversations")
            if not conversations or not isinstance(conversations, list):
                num_invalid += 1
                continue

            input_ids, loss_mask = tokenize_with_loss_mask(
                tokenizer, conversations, args.answer_only_loss
            )
            if input_ids.shape[1] <= 10 or input_ids.shape[1] > args.max_seq_len:
                num_skipped += 1
                continue

            # The supervised span is the response; everything before it is context.
            # One record per supervised span: a multi-turn conversation masks each
            # assistant turn separately, and each is scored against the context before it.
            spans = scored_spans(loss_mask)
            # Position 0 has no preceding context, so it can carry no policy. Without
            # --answer-only-loss the mask covers the whole sequence and would start at 0.
            spans = [(max(a, 1), b) for a, b in spans if b > max(a, 1)]
            if not spans:
                num_invalid += 1
                continue
            ids = input_ids.squeeze(0)

            with torch.inference_mode():
                logits = model(input_ids=input_ids.to(model.device)).logits.squeeze(0)
            for turn, (span_start, span_end) in enumerate(spans):
                record = policy_record(
                    f"{conversation_id}-{turn}" if len(spans) > 1 else conversation_id,
                    ids,
                    span_start,
                    span_end,
                    logits,
                    args.top_k,
                    args.top_p,
                    args.temperature,
                    entry.get("source"),
                )
                shard_file.write(json.dumps(record) + "\n")
                written_in_shard += 1
            num_success += 1

            if written_in_shard >= args.shard_size:
                shard_file.close()
                os.replace(shard_tmp, shard_tmp.with_suffix(""))
                shard_idx += 1
                written_in_shard = 0
                shard_file, shard_tmp = open_shard(shard_idx)
    finally:
        if shard_file is not None:
            shard_file.close()
            if written_in_shard:
                os.replace(shard_tmp, shard_tmp.with_suffix(""))
            else:
                Path(shard_tmp).unlink(missing_ok=True)

    if num_skipped:
        print(f"Skipped {num_skipped} conversations due to length constraints.")
    if num_invalid:
        print(f"Skipped {num_invalid} conversations without a usable response span.")
    print(f"Wrote {num_success} conversations to {args.output_dir}")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)

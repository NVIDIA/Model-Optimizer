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
import sys
from pathlib import Path

import torch
from common import (
    add_answer_only_loss_args,
    load_conversations,
    load_tokenizer,
    tokenize_with_loss_mask,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from modelopt.torch.speculative.plugins.hf_external import HFExternalDraftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump a base model's top-k deployment policy over conversation responses."
    )
    parser.add_argument("--model", type=str, required=True, help="Base (target) model.")
    parser.add_argument(
        "--input-data", type=Path, required=True, help=".jsonl file or directory of .jsonl files."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help=(
            "Skip conversations longer than this. Keep it <= training.training_seq_len: the "
            "loader right-truncates to that length, so a record whose context alone exceeds "
            "it carries no scored position."
        ),
    )
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
        "--shard-size",
        type=int,
        default=2000,
        help="Records per output shard. One record per supervised span, so a multi-turn "
        "conversation contributes one per assistant turn.",
    )
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--dp-world-size", type=int, default=1)
    # compute_hidden_states_hf.py spells this --trust_remote_code; accept both so the
    # dumpers can be driven from one script.
    parser.add_argument(
        "--trust-remote-code", "--trust_remote_code", action="store_true", dest="trust_remote_code"
    )
    parser.add_argument("--debug-max-num-conversations", type=int, default=None)
    add_answer_only_loss_args(parser)
    return parser.parse_args()


def deployment_policy(logits, top_k: int, top_p: float):
    """``(ids, log_probs)`` of width ``top_k`` under the filter ``tvd_deploy`` applies to the draft.

    Reuses that filter so the two cannot drift. Slots the nucleus drops hold ``-inf``,
    which ``sparse_data`` decodes to zero probability.
    """
    probs = HFExternalDraftModel._deployment_probs(logits, top_k, top_p)
    ids = logits.float().topk(min(top_k, logits.shape[-1]), dim=-1).indices
    return ids, probs.gather(-1, ids).log()


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
    # Position 0 has no preceding context, so it can carry no policy. Without
    # --answer-only-loss the mask covers the whole sequence and would start at 0.
    return [(max(a, 1), b) for a, b in spans if b > max(a, 1)]


def policy_record(
    conversation_id,
    ids,
    first: int,
    end: int,
    logits,
    top_k: int,
    top_p: float,
    source=None,
):
    """Build one shard record for a single supervised span.

    Args:
        first: index of the span's first supervised token (>= 1); everything before it is
            context.
        end: one past the span's last supervised token.
        logits: the base's logits over the whole sequence.
    """
    # Position t of the stored policy is the distribution token ids[t] was drawn from,
    # conditioned on ids[:t] -- that is the model's logits at t - 1. Dropping this shift
    # still yields a falling loss curve, against the wrong target.
    tok, lp = deployment_policy(logits[first - 1 : end - 1], top_k, top_p)
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
    if args.top_k < 1:
        raise ValueError(
            f"--top-k must be at least 1, got {args.top_k}: the dump stores a "
            "fixed-width policy and a zero-width row carries no teacher."
        )
    dataset = load_conversations(args.input_data, args.dp_rank, args.dp_world_size)
    if args.debug_max_num_conversations is not None:
        # After DP sharding this rank may hold fewer rows than the debug cap, and
        # select() raises on an out-of-range index rather than clamping.
        dataset = dataset.select(range(min(args.debug_max_num_conversations, len(dataset))))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto", trust_remote_code=args.trust_remote_code
    )
    model.eval()
    tokenizer = load_tokenizer(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A second, shorter export overwrites only the shard names it reaches, so the extra
    # shards from the first run stay visible to the trainer's glob and it reads records
    # from both. Refuse rather than silently mix two dumps.
    existing = sorted(args.output_dir.glob(f"policy-{args.dp_rank:03d}-*.jsonl.gz"))
    if existing:
        raise FileExistsError(
            f"{args.output_dir} already holds {len(existing)} shard(s) for DP rank "
            f"{args.dp_rank}, starting with {existing[0].name}. Remove them or pass a "
            "fresh --output-dir; a shorter run here would leave the earlier dump's "
            "extra shards behind and training would read both."
        )
    num_skipped = num_invalid = num_success = 0
    shard_idx = 0
    written_in_shard = 0
    shard_file = None

    def open_shard(idx: int):
        # Write-then-rename: a partial shard left by an interrupted run would otherwise
        # be picked up by the trainer's glob and decoded as damaged input.
        tmp = args.output_dir / f"policy-{args.dp_rank:03d}-{idx:05d}.jsonl.gz.tmp"
        # Exclusive create: the existing-shard check above is a point-in-time test, so
        # two exports for the same rank can both pass it and then write the same file.
        # "xt" makes the second one fail here instead of interleaving records.
        try:
            return gzip.open(tmp, "xt"), tmp
        except FileExistsError:
            raise FileExistsError(
                f"{tmp} already exists: another export is writing DP rank "
                f"{args.dp_rank} into {args.output_dir}, or a previous one was killed. "
                "Remove the .tmp files or use a fresh --output-dir."
            ) from None

    completed = False
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
        completed = True
    finally:
        if shard_file is not None:
            shard_file.close()
            # Publish the last shard only on a clean pass. Renaming it after a failure
            # hands the trainer a truncated export that looks complete.
            if written_in_shard and completed:
                os.replace(shard_tmp, shard_tmp.with_suffix(""))
            else:
                Path(shard_tmp).unlink(missing_ok=True)
        if not completed:
            done = sorted(args.output_dir.glob(f"policy-{args.dp_rank:03d}-*.jsonl.gz"))
            print(
                f"Export failed after {num_success} conversations. {len(done)} completed "
                f"shard(s) remain in {args.output_dir} and are a partial dump; remove "
                "them before training or re-exporting.",
                file=sys.stderr,
            )

    if num_skipped:
        print(f"Skipped {num_skipped} conversations due to length constraints.")
    if num_invalid:
        print(f"Skipped {num_invalid} conversations without a usable response span.")
    print(f"Wrote {num_success} conversations to {args.output_dir}")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)

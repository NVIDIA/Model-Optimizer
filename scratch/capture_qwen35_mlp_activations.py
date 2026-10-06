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

# ruff: noqa: D103
# mypy: ignore-errors

"""Capture MLP-input activations from Qwen3.5-9B on a configurable calibration dataset.

Hooks a forward pre-hook on ``model.layers[0].mlp`` and ``model.layers[31].mlp``
(first and last decoder block). Saves bf16 activation tensors to disk so the
NVFP4 input_scale calibration sweep can be re-run on real data.

Default dataset is the ``cnn_nemotron_v2_mix`` combo used by modelopt's
``hf_ptq.py`` (50/50 CNN/DailyMail articles + Nemotron-Post-Training v2 chat),
matching what real production PTQ calibration would see.

Output: two ``.pt`` files in this directory:
  qwen35_9b_mlp_input_layer0.pt   — list of (seq_len, hidden) bf16 tensors
  qwen35_9b_mlp_input_layer31.pt  — same shape, last layer
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Member datasets of modelopt's DATASET_COMBOS we care about here.
# Each tuple is (path, config_name, splits, key) — matches
# modelopt/torch/utils/dataset_utils.py:DATASET_COMBOS.
COMBO_MEMBERS = {
    "cnn_nemotron_v2_mix": [
        ("abisee/cnn_dailymail", "3.0.0", ["train"], "article"),
        (
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            None,
            ["stem", "chat", "math", "code"],
            "messages",
        ),
    ],
    "nemotron-post-training-v3": [
        (
            "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
            None,
            ["reasoning_off"],
            "messages",
        ),
        ("nvidia/Nemotron-Science-v1", None, ["MCQ", "RQA"], "messages"),
        (
            "nvidia/Nemotron-Competitive-Programming-v1",
            None,
            [
                "competitive_coding_cpp_part00",
                "competitive_coding_cpp_part01",
                "competitive_coding_python_part00",
                "competitive_coding_python_part01",
            ],
            "messages",
        ),
        ("nvidia/Nemotron-SFT-Agentic-v2", None, ["search"], "messages"),
        (
            "nvidia/Nemotron-Math-v2",
            None,
            ["high_part00", "high_part01", "high_part02", "medium", "low"],
            "messages",
        ),
        ("nvidia/Nemotron-SFT-SWE-v2", None, ["agentless"], "messages"),
        (
            "nvidia/Nemotron-SFT-Multilingual-v1",
            None,
            [
                "code_de",
                "code_es",
                "code_fr",
                "code_it",
                "code_ja",
                "code_zh",
                "math_de",
                "math_es",
                "math_fr",
                "math_it",
                "math_ja",
                "math_zh",
                "stem_de",
                "stem_es",
                "stem_fr",
                "stem_it",
                "stem_ja",
                "stem_zh",
            ],
            "messages",
        ),
    ],
}

MODEL_PATH = "/models/Qwen/Qwen3.5-9B"
OUT_DIR = Path(__file__).parent
LAYER_IDS = [0, 31]


def iter_combo_samples(combo_name, tokenizer, n_per_member):
    """Yield (text_str, source_name) interleaving member datasets in round-robin order."""
    members = COMBO_MEMBERS[combo_name]
    # Build one streaming iterator per member, multi-split chained.
    member_iters = []
    for path, name, splits, key in members:

        def _stream(p=path, nm=name, sp=splits, k=key):
            for split in sp:
                ds = load_dataset(p, name=nm, split=split, streaming=True)
                for ex in ds:
                    yield ex, k, p

        member_iters.append(_stream())

    yielded = [0] * len(members)
    targets = [n_per_member] * len(members)
    active = [True] * len(members)
    while any(active):
        for mi, it in enumerate(member_iters):
            if not active[mi] or yielded[mi] >= targets[mi]:
                active[mi] = False
                continue
            try:
                ex, key, src = next(it)
            except StopIteration:
                active[mi] = False
                continue
            if key == "messages":
                msgs = ex.get("messages")
                if not msgs:
                    continue
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    continue
            else:
                text = ex.get(key)
                if not text:
                    continue
            yielded[mi] += 1
            yield text, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seqs", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--dataset", type=str, default="cnn_nemotron_v2_mix")
    args = ap.parse_args()

    print(f"Loading tokenizer from {MODEL_PATH}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading model (bf16, cuda:0)", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    # Set up sample iterator. For combo datasets, sample evenly per member.
    if args.dataset in COMBO_MEMBERS:
        members = COMBO_MEMBERS[args.dataset]
        n_per_member = (args.n_seqs + len(members) - 1) // len(members)
        print(
            f"Combo dataset {args.dataset}: {len(members)} members, "
            f"target {n_per_member} samples each (total ~{n_per_member * len(members)})",
            flush=True,
        )
        sample_iter = iter_combo_samples(args.dataset, tok, n_per_member)
    else:
        # Single Nemotron-style chat dataset, taking a single split for backward compat.
        ds = load_dataset(
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            split=args.dataset,
            streaming=True,
        )

        def _single():
            for ex in ds:
                msgs = ex.get("messages")
                if not msgs:
                    continue
                try:
                    text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    continue
                yield text, args.dataset

        sample_iter = _single()

    # Hooks: capture forward INPUT to mlp.
    captured = {lid: [] for lid in LAYER_IDS}

    def make_hook(lid):
        def hook(_module, inputs):
            # inputs is a tuple; first is the (B, S, H) tensor.
            x = inputs[0].detach()
            # Drop the batch dim (B=1 in this script) and move to CPU as bf16.
            captured[lid].append(x[0].to("cpu", torch.bfloat16))

        return hook

    handles = []
    for lid in LAYER_IDS:
        h = model.model.layers[lid].mlp.register_forward_pre_hook(make_hook(lid))
        handles.append(h)

    # Iterate sample stream, run forward.
    n_done = 0
    source_counts = {}
    for text, source in sample_iter:
        if n_done >= args.n_seqs:
            break
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_tokens,
        ).to("cuda:0")
        input_ids = enc["input_ids"]
        if input_ids.shape[-1] < 32:
            continue
        with torch.no_grad():
            _ = model(**enc, use_cache=False)
        n_done += 1
        source_counts[source] = source_counts.get(source, 0) + 1
        if n_done % 50 == 0:
            print(
                f"  [{n_done}/{args.n_seqs}] seq_len={input_ids.shape[-1]} sources={source_counts}",
                flush=True,
            )

    for h in handles:
        h.remove()

    print(f"Captured {len(captured[LAYER_IDS[0]])} sequences  sources={source_counts}", flush=True)
    for lid in LAYER_IDS:
        out_path = OUT_DIR / f"qwen35_9b_mlp_input_layer{lid}.pt"
        torch.save(captured[lid], out_path)
        n_tokens = sum(t.shape[0] for t in captured[lid])
        amax = max(t.float().abs().max().item() for t in captured[lid])
        print(
            f"  saved {out_path.name}: {len(captured[lid])} seqs, "
            f"{n_tokens} tokens, global amax={amax:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())

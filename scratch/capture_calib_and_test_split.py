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

"""Capture disjoint calibration + held-out test activations for Qwen3.5-9B.

Runs the model once and captures four activation sets:
  - calib from cnn_nemotron_v2_mix (positions 0..N_CALIB-1 of each member)
  - test  from cnn_nemotron_v2_mix (positions N_CALIB..N_CALIB+N_TEST-1)
  - calib from nemotron-post-training-v3 (same skip/take logic per member)
  - test  from nemotron-post-training-v3 (positions strictly after calib)

The two test sets are concatenated downstream to form one shared 512-sample
test tensor evaluated under every calibration option. Because positions are
disjoint per member dataset, no test sample appears in any calibration set.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mirrors modelopt's DATASET_COMBOS for the two combos under study.
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


def render_text(ex, key, tokenizer):
    if key == "messages":
        msgs = ex.get("messages")
        if not msgs:
            return None
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            return None
    val = ex.get(key)
    return val if val else None


def iter_combo_with_skip(combo_name, tokenizer, skip_per_member, take_per_member):
    """Yield (text, source) pairs with per-member skip+take semantics.

    Skips the first ``skip_per_member`` rows that would otherwise be yielded
    (i.e. that successfully render), then yields the next ``take_per_member``
    rows. Round-robin across members.
    """
    members = COMBO_MEMBERS[combo_name]
    member_iters = []
    for path, name, splits, key in members:

        def _stream(p=path, nm=name, sp=splits, k=key):
            for split in sp:
                ds = load_dataset(p, name=nm, split=split, streaming=True)
                for ex in ds:
                    yield ex, k, p

        member_iters.append(_stream())

    skipped = [0] * len(members)
    yielded = [0] * len(members)
    active = [True] * len(members)
    while any(active):
        for mi, it in enumerate(member_iters):
            if not active[mi] or yielded[mi] >= take_per_member:
                active[mi] = False
                continue
            try:
                ex, key, src = next(it)
            except StopIteration:
                active[mi] = False
                continue
            text = render_text(ex, key, tokenizer)
            if text is None:
                continue
            if skipped[mi] < skip_per_member:
                skipped[mi] += 1
                continue
            yielded[mi] += 1
            yield text, src


def run_segment(model, tokenizer, sample_iter, n_target, max_tokens, label):
    """Run forward on samples from iter; return per-layer list of activations."""
    captured = {lid: [] for lid in LAYER_IDS}

    def make_hook(lid):
        def hook(_module, inputs):
            x = inputs[0].detach()
            captured[lid].append(x[0].to("cpu", torch.bfloat16))

        return hook

    handles = []
    for lid in LAYER_IDS:
        h = model.model.layers[lid].mlp.register_forward_pre_hook(make_hook(lid))
        handles.append(h)

    n_done = 0
    source_counts = {}
    for text, source in sample_iter:
        if n_done >= n_target:
            break
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(
            "cuda:0"
        )
        if enc["input_ids"].shape[-1] < 32:
            continue
        with torch.no_grad():
            _ = model(**enc, use_cache=False)
        n_done += 1
        source_counts[source] = source_counts.get(source, 0) + 1
        if n_done % 100 == 0:
            print(f"  [{label}] {n_done}/{n_target}  sources={source_counts}", flush=True)

    for h in handles:
        h.remove()

    return captured, n_done, source_counts


def save_captured(captured, prefix):
    paths = []
    for lid in LAYER_IDS:
        out_path = OUT_DIR / f"{prefix}_layer{lid}.pt"
        torch.save(captured[lid], out_path)
        n_tokens = sum(t.shape[0] for t in captured[lid])
        amax = max(t.float().abs().max().item() for t in captured[lid]) if captured[lid] else 0.0
        print(
            f"  saved {out_path.name}: {len(captured[lid])} seqs, "
            f"{n_tokens} tokens, amax={amax:.3f}",
            flush=True,
        )
        paths.append(out_path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_calib", type=int, default=1024, help="Calibration samples per combo.")
    ap.add_argument("--n_test", type=int, default=256, help="Test samples per combo.")
    ap.add_argument("--max_tokens", type=int, default=512)
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

    for combo in ("cnn_nemotron_v2_mix", "nemotron-post-training-v3"):
        members = COMBO_MEMBERS[combo]
        n_members = len(members)
        calib_per_member = (args.n_calib + n_members - 1) // n_members
        test_per_member = (args.n_test + n_members - 1) // n_members
        print(
            f"\n=== {combo}: {n_members} members, calib_per_member={calib_per_member}, "
            f"test_per_member={test_per_member}",
            flush=True,
        )

        print(f"  --> calib: positions [0, {calib_per_member}) per member")
        calib_iter = iter_combo_with_skip(
            combo, tok, skip_per_member=0, take_per_member=calib_per_member
        )
        captured, n_done, src = run_segment(
            model,
            tok,
            calib_iter,
            n_target=args.n_calib,
            max_tokens=args.max_tokens,
            label=f"{combo}/calib",
        )
        print(f"  captured calib: {n_done} sequences  sources={src}")
        save_captured(captured, f"qwen35_{combo.replace('-', '_')}_calib")

        print(
            f"  --> test: positions "
            f"[{calib_per_member}, {calib_per_member + test_per_member}) per member"
        )
        test_iter = iter_combo_with_skip(
            combo, tok, skip_per_member=calib_per_member, take_per_member=test_per_member
        )
        captured, n_done, src = run_segment(
            model,
            tok,
            test_iter,
            n_target=args.n_test,
            max_tokens=args.max_tokens,
            label=f"{combo}/test",
        )
        print(f"  captured test:  {n_done} sequences  sources={src}")
        save_captured(captured, f"qwen35_{combo.replace('-', '_')}_test")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())

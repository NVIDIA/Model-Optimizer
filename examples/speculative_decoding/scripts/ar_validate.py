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

"""AR validation for speculative decoding models (EAGLE3, DFlash, Medusa).

Supports per-category MT-Bench evaluation and online (context-dependent) validation.
"""

import argparse
from collections import defaultdict

from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import modelopt.torch.opt as mto
from modelopt.torch.speculative.plugins.hf_eagle import HFARValidation
from modelopt.torch.speculative.utils import load_vlm_or_llm

mto.enable_huggingface_checkpointing()


def validate_ar(
    model,
    tokenizer,
    ds,
    steps=3,
    osl=20,
    num_samples=80,
    device=None,
):
    """Validate acceptance rate on MT-Bench prompts using online validation.

    Online validation recomputes ground truth after each accepted draft token
    (context-dependent), matching actual speculative decoding behavior.

    Args:
        model: Speculative decoding model (EAGLE3, DFlash, etc.)
        tokenizer: Tokenizer for the model.
        ds: MT-Bench dataset (HuggingFace dataset with 'prompt' and optional 'category').
        steps: Number of draft tokens per speculative step.
        osl: Output sequence length.
        num_samples: Max number of samples to evaluate.
        device: Device to run on.

    Returns:
        ``(results, length_histogram)`` -- a list of ``(category, ar)`` tuples and the
        pooled per-step acceptance-length histogram across all samples. The histogram
        is what a per-position ``speculation_profile.json`` is built from; the scalar
        ARs alone cannot express where acceptance falls off.
    """
    validator = HFARValidation(model, tokenizer)
    num_samples = min(num_samples, len(ds))
    results = []
    length_histogram: dict[int, int] = {}
    failures = 0
    for i in tqdm(range(num_samples), desc="Validating AR"):
        prompt = ds[i]["prompt"][0]
        category = ds[i].get("category", "unknown")
        if hasattr(tokenizer, "apply_chat_template"):
            chat_messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if device:
            input_ids = input_ids.to(device)

        try:
            _, ar, hist = validator.validate_online(osl, input_ids=input_ids, steps=steps)
            results.append((category, ar))
            for length, count in hist.items():
                length_histogram[length] = length_histogram.get(length, 0) + count
        except Exception as e:
            failures += 1
            print(f"  WARNING: sample {i} ({category}) failed: {e}")
    if failures:
        print(f"WARNING: {failures}/{num_samples} samples failed during AR validation")
    return results, dict(sorted(length_histogram.items()))


def _write_speculation_profile(
    path, length_histogram, num_speculative_tokens, per_request_mean, osl, num_samples
):
    """Emit the same speculation_profile.json schema specdec_bench produces.

    Two producers, one schema: this one runs inside the training loop with no serving
    engine, so acceptance can be tracked as a checkpoint trains; specdec_bench measures
    the deployed engine. Consumers should not care which produced a profile.

    The conversion is reimplemented here rather than imported. specdec_bench's copy is
    deliberately importable without modelopt (it runs in engine containers where
    modelopt is absent -- the MiniMax-M2.7 DFlash run recorded modelopt_version: null),
    and importing *into* modelopt from examples/ is not possible either. The shared
    piece is small and pinned by tests on both sides; if a third producer appears, that
    is the point to extract it properly.
    """
    import json

    total = sum(length_histogram.values())
    if total == 0:
        # Every sample failed, or osl was too small to produce a single decode step.
        # Writing measured=true here would advertise a draft that accepts nothing,
        # which is indistinguishable from a genuinely terrible draft.
        print("  WARNING: no decode steps observed; skipping speculation profile")
        return
    # Marginal[i] = P(at least i+1 drafts accepted). Acceptance length counts the
    # target's bonus token, so draft position i corresponds to length i+2.
    marginal, conditional = [], []
    for i in range(num_speculative_tokens):
        at_least = sum(c for length, c in length_histogram.items() if length >= i + 2)
        prev = sum(c for length, c in length_histogram.items() if length >= i + 1)
        marginal.append(at_least / total if total else 0.0)
        conditional.append(at_least / prev if prev else 0.0)
    # Per-step mean, matching what the vectors describe -- not the per-request mean,
    # which weights a short request the same as a long one.
    per_step_mean = (
        sum(length * c for length, c in length_histogram.items()) / total if total else 0.0
    )

    profile = {
        "schema_version": "1.0",
        "measured": True,
        "producer": "ar_validate",
        "num_speculative_tokens": num_speculative_tokens,
        "conditional_accept_rates": [round(x, 6) for x in conditional],
        "marginal_accept_rates": [round(x, 6) for x in marginal],
        "mean_accept_length": round(per_step_mean, 6),
        "mean_accept_length_per_request": round(per_request_mean, 6),
        "acceptance_length_histogram": length_histogram,
        "measurement_conditions": {
            "dataset": "mt_bench",
            "osl": osl,
            "num_samples": num_samples,
            "validation": "online (ground truth recomputed after each accepted token)",
        },
    }
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="AR validation for speculative decoding models.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--steps", type=int, default=3, help="Draft tokens per step")
    parser.add_argument("--osl", type=int, default=32, help="Output sequence length")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples")
    parser.add_argument("--per_category", action="store_true", help="Report per-category AR")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Write a speculation_profile.json here. Without it this script is "
            "print-only, so nothing downstream -- CI regression gating, the export "
            "step, a deployment -- can consume what it measured."
        ),
    )
    parser.add_argument(
        "--ar_lower_bound",
        type=float,
        default=None,
        help="Error if AR is below this threshold.",
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    model = load_vlm_or_llm(
        args.model_path, device_map="auto", trust_remote_code=args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model.eval()
    model = accelerator.prepare(model)

    ds = load_dataset("HuggingFaceH4/mt_bench_prompts")["train"]
    results, length_histogram = validate_ar(
        model,
        tokenizer,
        ds,
        args.steps,
        args.osl,
        args.num_samples,
        accelerator.device,
    )

    if results and accelerator.is_main_process:
        all_ars = [ar for _, ar in results]
        avg_ar = sum(all_ars) / len(all_ars)
        print(f"\n==== AR Validation Results (osl={args.osl}, steps={args.steps}) ====")

        if args.per_category:
            cat_ars = defaultdict(list)
            for cat, ar in results:
                cat_ars[cat].append(ar)
            for cat in sorted(cat_ars):
                cat_avg = sum(cat_ars[cat]) / len(cat_ars[cat])
                print(f"  {cat:>12}: {cat_avg:.4f}")

        print(f"  {'ALL':>12}: {avg_ar:.4f}")
        print(f"  Samples: {len(results)}")

        if args.output_json:
            _write_speculation_profile(
                args.output_json,
                length_histogram,
                num_speculative_tokens=args.steps,
                per_request_mean=avg_ar,
                osl=args.osl,
                num_samples=len(results),
            )
            print(f"  Wrote speculation profile to {args.output_json}")

        # Bound check last: an out-of-bounds AR is still worth having on disk, and
        # raising first would discard the measurement that explains the failure.
        if args.ar_lower_bound and avg_ar < args.ar_lower_bound:
            raise ValueError(f"AR {avg_ar:.4f} is below lower bound {args.ar_lower_bound}.")


if __name__ == "__main__":
    main()

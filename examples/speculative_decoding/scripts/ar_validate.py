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
from modelopt.torch.speculative.plugins.transformers import HFARValidation
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
        List of (category, ar) tuples.
    """
    validator = HFARValidation(model, tokenizer)
    num_samples = min(num_samples, len(ds))
    results = []
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
            _, ar = validator.validate_online(osl, input_ids=input_ids, steps=steps)
            results.append((category, ar))
        except Exception as e:
            failures += 1
            print(f"  WARNING: sample {i} ({category}) failed: {e}")
    if failures:
        print(f"WARNING: {failures}/{num_samples} samples failed during AR validation")
    return results


def main():
    parser = argparse.ArgumentParser(description="AR validation for speculative decoding models.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--steps", type=int, default=3, help="Draft tokens per step")
    parser.add_argument("--osl", type=int, default=32, help="Output sequence length")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples")
    parser.add_argument("--per_category", action="store_true", help="Report per-category AR")
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
    results = validate_ar(
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

        if args.ar_lower_bound and avg_ar < args.ar_lower_bound:
            raise ValueError(f"AR {avg_ar:.4f} is below lower bound {args.ar_lower_bound}.")


if __name__ == "__main__":
    main()

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

import argparse

from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.opt as mto
from modelopt.torch.speculative.plugins.transformers import HFARValidation

mto.enable_huggingface_checkpointing()


def validate_ar(model, tokenizer, ds, steps=3, osl=20, num_samples=80, device=None):
    validator = HFARValidation(model, tokenizer)
    num_samples = min(num_samples, len(ds))
    ars = []
    for i in tqdm(range(num_samples), desc="Validating AR"):
        prompt = ds[i]["prompt"][0]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        # Apply chat template to the prompt, continuing with assistant response
        if hasattr(tokenizer, "apply_chat_template"):
            chat_messages = [
                {"role": "user", "content": prompt},
            ]
            prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if device:
            input_ids = input_ids.to(device)

        # validate AR
        _, ar = validator.validate(osl, input_ids=input_ids, steps=steps)
        ars.append(ar)
    return ars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--steps", type=int, default=3, help="Steps for AR validation")
    parser.add_argument(
        "--osl", type=int, default=32, help="Output sequence length for AR validation"
    )
    parser.add_argument(
        "--num_samples", type=int, default=80, help="Number of MT-Bench samples to use"
    )
    parser.add_argument(
        "--ar_lower_bound",
        type=float,
        default=None,
        help="AR lower bound for validation. If provided, will throw error if AR is below threshold.",
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    model = accelerator.prepare(model)

    # Load MT-Bench prompts from HuggingFace
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts")["train"]
    ars = validate_ar(
        model, tokenizer, ds, args.steps, args.osl, args.num_samples, accelerator.device
    )
    # Optionally, throw error if AR is below lower bound
    if args.ar_lower_bound:
        mean_ar = sum(ars) / len(ars)
        if mean_ar < args.ar_lower_bound:
            raise ValueError(
                f"AR is below lower bound {args.ar_lower_bound}. Mean AR: {mean_ar:.4f}"
            )
    # Print results
    if ars and accelerator.is_main_process:
        avg_ar = sum(ars) / len(ars)
        print("\n==== AR Validation Results on MT-Bench ====")
        print(f"Number of samples: {len(ars)}")
        print(f"Output Sequence Length: {args.osl}")
        print(f"Steps: {args.steps}")
        print(f"Average AR: {avg_ar:.4f}")


if __name__ == "__main__":
    main()

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

"""Compare a fake-quant model with BF16 on shared WikiText-prompted continuations."""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GenerationConfig, set_seed

__all__ = ["evaluate", "mean_kl", "score_continuation"]


@torch.inference_mode()
def mean_kl(reference_logits, quantized_logits, top_k=128, chunk_size=32):
    """Return mean forward KL in nats over [generated positions, vocabulary] logits."""
    if reference_logits.ndim != 2 or reference_logits.shape != quantized_logits.shape:
        raise ValueError("Expected matching [generated positions, vocabulary] logits.")
    num_tokens, vocab_size = reference_logits.shape
    if num_tokens == 0 or chunk_size < 1 or not 1 <= top_k <= vocab_size:
        raise ValueError("Require generated positions, a positive chunk size, and 1 <= top_k <= V.")
    totals = torch.zeros(2, dtype=torch.float32, device=reference_logits.device)
    for start in range(0, num_tokens, chunk_size):
        reference = reference_logits[start : start + chunk_size].float()
        quantized = quantized_logits[start : start + chunk_size].to(
            device=reference.device, dtype=torch.float32
        )
        log_p = F.log_softmax(reference, dim=-1)
        log_q = F.log_softmax(quantized, dim=-1)
        full_kl = F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)
        # Both conditional distributions must use the reference's token IDs, not rank slots.
        ids = reference.topk(top_k, dim=-1).indices
        log_p_head = F.log_softmax(reference.gather(-1, ids), dim=-1)
        log_q_head = F.log_softmax(quantized.gather(-1, ids), dim=-1)
        head_kl = F.kl_div(log_q_head, log_p_head, log_target=True, reduction="none").sum(-1)
        totals += torch.stack((full_kl, head_kl), dim=-1).sum(0)
    means = totals / num_tokens
    if not torch.isfinite(means).all():
        raise ValueError("Non-finite KL: inspect the model logits instead of dropping positions.")
    full_kl, head_kl = means.cpu().tolist()
    return {"full_vocab_kl": full_kl, "conditional_topk_kl": head_kl}


@torch.inference_mode()
def score_continuation(reference, quantized, token_ids, prompt_tokens, top_k=128):
    """Score only continuation predictions, including the first generated token and EOS."""
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("Expected one prompt and its continuation.")
    if not 0 < prompt_tokens < token_ids.shape[1]:
        raise ValueError("Require a nonempty prompt and continuation.")
    logits = []
    for model in (reference, quantized):
        inputs = token_ids[:, :-1].to(model.get_input_embeddings().weight.device)
        outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
        # Position prompt_tokens - 1 predicts the first continuation token.
        logits.append(outputs.logits[0, prompt_tokens - 1 :])
    return mean_kl(logits[0], logits[1], top_k)


@torch.inference_mode()
def evaluate(
    reference, quantized, tokenizer, prompts, max_new_tokens=512, top_k=128, detailed_results=False
):
    """Return overall KL means, optionally with per-example details, on shared BF16 continuations."""
    if not prompts:
        raise ValueError("No evaluation prompts.")
    reference.eval()
    quantized.eval()
    eos = reference.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    generation = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    totals = {"full_vocab_kl": 0.0, "conditional_topk_kl": 0.0}
    results = []
    for index, prompt in enumerate(prompts):
        inputs = torch.tensor(
            [prompt["input_ids"]], device=reference.get_input_embeddings().weight.device
        )
        # Transformers 5 removed use_model_defaults=False. Isolate checkpoint decoding settings.
        model_generation = reference.generation_config
        try:
            reference.generation_config = generation
            sequence = reference.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=generation,
            )
        finally:
            reference.generation_config = model_generation
        generated_tokens = sequence.shape[1] - inputs.shape[1]
        metrics = score_continuation(reference, quantized, sequence, inputs.shape[1], top_k)
        for name, value in metrics.items():
            totals[name] += value
        if detailed_results:
            results.append(
                {
                    "example": index,
                    "block_index": prompt["block_index"],
                    "prompt_ids": prompt["input_ids"],
                    "generated_ids": sequence[0, inputs.shape[1] :].cpu().tolist(),
                    "generated_tokens": generated_tokens,
                    **metrics,
                }
            )
        print(f"{index + 1}/{len(prompts)}: {generated_tokens} tokens; {metrics}", flush=True)
    summary = {name: total / len(prompts) for name, total in totals.items()}
    return {"summary": summary, "examples": results} if detailed_results else summary


def _wikitext_prompts(tokenizer, num_examples, prompt_tokens, seed):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row for row in dataset["text"] if row.strip())
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    num_blocks = len(token_ids) // prompt_tokens
    if num_blocks < num_examples:
        raise ValueError(f"WikiText has only {num_blocks} complete prompt windows.")
    selected = random.Random(seed).sample(range(num_blocks), num_examples)
    prompts = [
        {
            "block_index": block,
            "input_ids": token_ids[block * prompt_tokens : (block + 1) * prompt_tokens],
        }
        for block in selected
    ]
    return prompts, dataset._fingerprint


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--recipe", required=True, help="Built-in PTQ recipe name or YAML path.")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--prompt_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("kl_results.json"))
    parser.add_argument(
        "--detailed_results",
        action="store_true",
        help="Include per-example scores, token counts, token IDs, and run settings in JSON.",
    )
    # None preserves hf_ptq's defaults instead of duplicating them here.
    parser.add_argument("--dataset", help="Calibration dataset; defaults to hf_ptq's mixture.")
    parser.add_argument("--calib_size", help="hf_ptq calibration sample counts.")
    parser.add_argument("--calib_seq", type=int)
    parser.add_argument(
        "--batch_size", type=int, help="Calibration batch size; 0 selects automatically."
    )
    parser.add_argument("--gpu_max_mem_percentage", type=float)
    parser.add_argument("--attn_implementation")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    for name in ("num_examples", "prompt_tokens", "max_new_tokens", "top_k"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive.")
    if args.calib_seq is not None and args.calib_seq < 1:
        parser.error("--calib_seq must be positive.")
    if args.batch_size is not None and args.batch_size < 0:
        parser.error("--batch_size must be nonnegative.")
    return args


def main():
    """Calibrate through hf_ptq and write an evaluation report without exporting weights."""
    args = _parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run this prototype in one process; multiple visible GPUs are supported.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example's calibration workflow.")

    # Defer hf_ptq's heavy export/calibration imports so metric helpers remain usable on CPU.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hf_ptq"))
    import hf_ptq

    from modelopt.torch.quantization.nn import TensorQuantizer

    ptq_argv = ["--model", args.model, "--recipe", args.recipe, "--skip_generate"]
    for name in (
        "dataset",
        "calib_size",
        "calib_seq",
        "batch_size",
        "gpu_max_mem_percentage",
        "attn_implementation",
    ):
        value = getattr(args, name)
        if value is not None:
            ptq_argv.extend([f"--{name}", str(value)])
    if args.trust_remote_code:
        ptq_argv.append("--trust_remote_code")
    ptq_args = hf_ptq.parse_args(ptq_argv)
    ptq_args.dataset = ptq_args.dataset.split(",") if ptq_args.dataset else None
    ptq_args.calib_size = [int(value) for value in ptq_args.calib_size.split(",")]
    if any(size < 1 for size in ptq_args.calib_size):
        raise ValueError("Calibration sample counts must be positive.")
    recipe = hf_ptq.load_recipe(args.recipe)
    if not isinstance(recipe, hf_ptq.ModelOptPTQRecipe):
        raise ValueError("This prototype accepts fixed PTQ recipes, not AutoQuantize recipes.")
    if any(cfg.get("export_dir") is not None for cfg in hf_ptq.recipe_layerwise_blocks(recipe)):
        raise ValueError("Layerwise export recipes cannot retain a live fake-quant model.")

    set_seed(hf_ptq.RAND_SEED)
    torch.compiler.set_stance("force_eager")
    hf_ptq.setup_distributed_args(ptq_args)
    reference = hf_ptq.load_model(copy.deepcopy(ptq_args))[0]
    loaded = hf_ptq.load_model(ptq_args)
    quantized, _, _, _, _, tokenizer, *_ = loaded
    for model in (reference, quantized):
        if model.dtype != torch.bfloat16 or getattr(model.config, "quantization_config", None):
            raise ValueError("Supply an unquantized BF16 base checkpoint.")
        if model.config.is_encoder_decoder or hf_ptq.is_quantized(model):
            raise ValueError("Only unquantized causal language models are supported.")
        model.eval()
    if tokenizer is None or ptq_args.calib_with_images:
        raise ValueError("This prototype requires text-only calibration and generation.")
    config = getattr(reference.config, "text_config", reference.config)
    context_length = getattr(config, "max_position_embeddings", None)
    if context_length and args.prompt_tokens + args.max_new_tokens > context_length:
        raise ValueError("Prompt plus continuation exceeds the model's context length.")
    if args.top_k > config.vocab_size:
        raise ValueError("--top_k exceeds the model vocabulary.")

    prompts, fingerprint = _wikitext_prompts(
        tokenizer, args.num_examples, args.prompt_tokens, args.seed
    )
    quantized = hf_ptq.quantize_main(ptq_args, *loaded, export=False)
    if any(
        isinstance(module, TensorQuantizer) and module.is_enabled and not module.fake_quant
        for module in quantized.modules()
    ):
        raise ValueError("The recipe must use fake quantization, not packed quantized weights.")
    set_seed(args.seed)
    report = evaluate(
        reference,
        quantized,
        tokenizer,
        prompts,
        args.max_new_tokens,
        args.top_k,
        detailed_results=args.detailed_results,
    )
    if args.detailed_results:
        report["settings"] = {
            "model": args.model,
            "model_revision": getattr(reference.config, "_commit_hash", None),
            "recipe": args.recipe,
            "resolved_recipe": recipe.model_dump(mode="json"),
            "backend": "pytorch",
            "torch_version": torch.__version__,
            "model_dtype": "bfloat16",
            "metric_dtype": "float32",
            "kl_direction": "reference_to_quantized",
            "units": "nats",
            "aggregation": "mean_tokens_per_example_then_mean_examples",
            "evaluation_dataset": "Salesforce/wikitext/wikitext-2-raw-v1/test",
            "dataset_fingerprint": fingerprint,
            "num_examples": args.num_examples,
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
            "seed": args.seed,
            "generation": "greedy_until_eos_or_token_cap",
            "calibration_dataset": ptq_args.dataset,
            "calibration_samples": ptq_args.calib_size,
            "calibration_seq_length": ptq_args.calib_seq,
            "calibration_batch_size": ptq_args.batch_size,
            "calibration_seed": hf_ptq.RAND_SEED,
            "attention_implementation": args.attn_implementation,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"] if args.detailed_results else report, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

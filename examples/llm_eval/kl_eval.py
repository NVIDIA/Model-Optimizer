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

"""Compare a fake-quant model with BF16 on shared blended-benchmark continuations."""

import argparse
import copy
import inspect
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GenerationConfig, set_seed

__all__ = ["evaluate", "mean_kl", "score_continuation"]

EVALUATION_DATASET = "nvidia/balanced-benchmark-prompts-300"
EVALUATION_REVISION = "c211674f78e43ecd5aa487b3d3539499efafce78"


@torch.inference_mode()
def mean_kl(reference_logits, quantized_logits, top_k=1024, chunk_size=32):
    """Return full and conditional top-k forward KL in nats; top_k=None skips the head metric."""
    if reference_logits.ndim != 2 or reference_logits.shape != quantized_logits.shape:
        raise ValueError("Expected matching [generated positions, vocabulary] logits.")
    num_tokens, vocab_size = reference_logits.shape
    if num_tokens == 0 or chunk_size < 1:
        raise ValueError("Require generated positions and a positive chunk size.")
    if top_k is not None and (not isinstance(top_k, int) or not 1 <= top_k <= vocab_size):
        raise ValueError("Require integer 1 <= top_k <= V when conditional KL is requested.")
    names = ["full_vocab_kl"] + (["conditional_topk_kl"] if top_k is not None else [])
    totals = torch.zeros(len(names), dtype=torch.float32, device=reference_logits.device)
    for start in range(0, num_tokens, chunk_size):
        reference = reference_logits[start : start + chunk_size].float()
        quantized = quantized_logits[start : start + chunk_size].to(
            device=reference.device, dtype=torch.float32
        )
        log_p = F.log_softmax(reference, dim=-1)
        log_q = F.log_softmax(quantized, dim=-1)
        values = [F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)]
        if top_k is not None:
            # Both conditional distributions use the reference's token IDs, not rank slots.
            ids = reference.topk(top_k, dim=-1).indices
            log_p_head = F.log_softmax(reference.gather(-1, ids), dim=-1)
            log_q_head = F.log_softmax(quantized.gather(-1, ids), dim=-1)
            values.append(
                F.kl_div(log_q_head, log_p_head, log_target=True, reduction="none").sum(-1)
            )
        totals += torch.stack(values, dim=-1).sum(0)
    means = totals / num_tokens
    if not torch.isfinite(means).all():
        raise ValueError("Non-finite KL: inspect the model logits instead of dropping positions.")
    return dict(zip(names, means.cpu().tolist()))


@torch.inference_mode()
def _score_batch(reference, quantized, sequences, prompt_lengths, pad_token_id, top_k):
    generated_lengths = [
        len(sequence) - length for sequence, length in zip(sequences, prompt_lengths)
    ]
    if any(length < 1 for length in generated_lengths):
        raise ValueError("Require a nonempty continuation for each prompt.")
    width = max(len(sequence) - 1 for sequence in sequences)
    scores = []
    for model in (reference, quantized):
        device = model.get_input_embeddings().weight.device
        inputs = torch.full((len(sequences), width), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(inputs)
        for index, sequence in enumerate(sequences):
            length = len(sequence) - 1
            inputs[index, -length:] = torch.tensor(sequence[:-1], device=device)
            attention_mask[index, -length:] = 1
        position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        # Project continuation rows only on models supporting the Transformers optimization.
        kwargs = {}
        if "logits_to_keep" in inspect.signature(model.forward).parameters:
            kwargs["logits_to_keep"] = max(generated_lengths)
        outputs = model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **kwargs,
        )
        # Left padding aligns the final prediction in every row. Clone to release prompt logits.
        scores.append(
            [outputs.logits[i, -length:].clone() for i, length in enumerate(generated_lengths)]
        )
        del outputs
    return [mean_kl(p, q, top_k) for p, q in zip(*scores)]


@torch.inference_mode()
def score_continuation(reference, quantized, token_ids, prompt_tokens, top_k=1024):
    """Score only continuation predictions, including the first generated token and EOS."""
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("Expected one prompt and its continuation.")
    if not 0 < prompt_tokens < token_ids.shape[1]:
        raise ValueError("Require a nonempty prompt and continuation.")
    return _score_batch(reference, quantized, token_ids.cpu().tolist(), [prompt_tokens], 0, top_k)[
        0
    ]


@torch.inference_mode()
def evaluate(
    reference,
    quantized,
    tokenizer,
    prompts,
    max_new_tokens=2048,
    top_k=1024,
    detailed_results=False,
    eval_batch_size=8,
):
    """Generate and score shared continuations in batches, averaging prompt scores equally."""
    if not prompts or eval_batch_size < 1:
        raise ValueError("Require evaluation prompts and a positive evaluation batch size.")
    reference.eval()
    quantized.eval()
    eos = reference.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = next(iter(eos_ids), None)
    if pad_token_id is None:
        raise ValueError("Batched evaluation requires a padding or EOS token ID.")
    generation = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
        pad_token_id=pad_token_id,
        use_cache=True,
    )
    totals = {"full_vocab_kl": 0.0}
    if top_k is not None:
        totals["conditional_topk_kl"] = 0.0
    results = []
    for start in range(0, len(prompts), eval_batch_size):
        batch = prompts[start : start + eval_batch_size]
        lengths = [len(prompt["input_ids"]) for prompt in batch]
        width = max(lengths)
        inputs = torch.full(
            (len(batch), width),
            pad_token_id,
            dtype=torch.long,
            device=reference.get_input_embeddings().weight.device,
        )
        attention_mask = torch.zeros_like(inputs)
        for index, prompt in enumerate(batch):
            inputs[index, -lengths[index] :] = torch.tensor(
                prompt["input_ids"], device=inputs.device
            )
            attention_mask[index, -lengths[index] :] = 1
        # Isolate checkpoint decoding settings from this evaluation's greedy policy.
        model_generation = reference.generation_config
        try:
            reference.generation_config = generation
            sequences = reference.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                generation_config=generation,
            )
        finally:
            reference.generation_config = model_generation
        continuations = []
        for tokens in sequences[:, width:].cpu().tolist():
            end = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
            continuations.append(tokens[:end])
        del sequences
        histories = [prompt["input_ids"] + tokens for prompt, tokens in zip(batch, continuations)]
        metrics = _score_batch(reference, quantized, histories, lengths, pad_token_id, top_k)
        for index, (prompt, tokens, values) in enumerate(zip(batch, continuations, metrics), start):
            for name, value in values.items():
                totals[name] += value
            if detailed_results:
                results.append(
                    {
                        "example": index,
                        "id": prompt["id"],
                        "source": prompt["source"],
                        "cohort_rank": prompt["cohort_rank"],
                        "prompt_ids": prompt["input_ids"],
                        "generated_ids": tokens,
                        "generated_tokens": len(tokens),
                        **values,
                    }
                )
            print(f"{index + 1}/{len(prompts)}: {len(tokens)} tokens; {values}", flush=True)
    summary = {name: total / len(prompts) for name, total in totals.items()}
    return {"summary": summary, "examples": results} if detailed_results else summary


def _blended_prompts(tokenizer, num_examples, revision, context_limit, max_new_tokens):
    dataset = load_dataset(EVALUATION_DATASET, split="test", revision=revision, token=True)
    if not 1 <= num_examples <= len(dataset):
        raise ValueError(f"Require 1 <= num_examples <= {len(dataset)} dataset rows.")
    prompts = []
    for row in dataset.select(range(num_examples)):
        # Only messages enter the model; difficulty labels and other metadata stay outside it.
        input_ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_dict=False,
        )
        if not input_ids:
            raise ValueError(f"Empty prompt for dataset row {row['cohort_rank']}.")
        if context_limit and len(input_ids) + max_new_tokens > context_limit:
            raise ValueError(
                f"Dataset row {row['cohort_rank']} requires {len(input_ids)} prompt tokens "
                f"plus {max_new_tokens} continuation tokens, exceeding context {context_limit}. "
                "Prompts are not truncated; use a model with a sufficient context window."
            )
        prompts.append(
            {
                "id": row["id"],
                "source": row["source"],
                "cohort_rank": row["cohort_rank"],
                "input_ids": input_ids,
            }
        )
    return prompts, dataset._fingerprint


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--recipe", required=True, help="Built-in PTQ recipe name or YAML path.")
    parser.add_argument(
        "--num_examples", type=int, default=300, help="Evaluate this many rows in dataset order."
    )
    parser.add_argument("--eval_revision", default=EVALUATION_REVISION)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Prompts per generation and scoring batch."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1024,
        help="Reference top tokens for conditional KL (default: 1024); full KL is always computed.",
    )
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
    for name in ("num_examples", "max_new_tokens", "eval_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive.")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top_k must be positive when supplied.")
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
    ptq_args.hub_model_id, ptq_args.local_checkpoint_path = hf_ptq.ensure_local_checkpoint(
        ptq_args.pyt_ckpt_path
    )
    with hf_ptq.distributed_run(device=ptq_args.device) as dist_state:
        ptq_args.dist_state = dist_state
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
        if args.top_k is not None and args.top_k > config.vocab_size:
            raise ValueError("--top_k exceeds the model vocabulary.")

        prompts, fingerprint = _blended_prompts(
            tokenizer, args.num_examples, args.eval_revision, context_length, args.max_new_tokens
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
            eval_batch_size=args.eval_batch_size,
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
                "evaluation_dataset": EVALUATION_DATASET,
                "evaluation_revision": args.eval_revision,
                "evaluation_split": "test",
                "prompt_format": "tokenizer_chat_template_with_generation_prompt",
                "dataset_fingerprint": fingerprint,
                "num_examples": args.num_examples,
                "prompt_tokens": [len(prompt["input_ids"]) for prompt in prompts],
                "max_new_tokens": args.max_new_tokens,
                "evaluation_batch_size": args.eval_batch_size,
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

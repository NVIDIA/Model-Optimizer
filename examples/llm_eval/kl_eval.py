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

"""Compare an exported ModelOpt checkpoint with BF16 using offline vLLM and shared continuations."""

import argparse
import json
import multiprocessing
import os
import random
import signal
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

__all__ = ["mean_kl"]


@torch.inference_mode()
def mean_kl(reference_logprobs, quantized_logprobs, top_k=128, chunk_size=32):
    """Return forward KL means in nats from normalized [positions, vocabulary] log-probabilities."""
    if reference_logprobs.ndim != 2 or reference_logprobs.shape != quantized_logprobs.shape:
        raise ValueError("Expected matching [generated positions, vocabulary] log-probabilities.")
    num_tokens, vocab_size = reference_logprobs.shape
    if num_tokens == 0 or chunk_size < 1 or not 1 <= top_k <= vocab_size:
        raise ValueError("Require generated positions, a positive chunk size, and 1 <= top_k <= V.")
    totals = torch.zeros(2, dtype=torch.float32, device=reference_logprobs.device)
    for start in range(0, num_tokens, chunk_size):
        log_p = reference_logprobs[start : start + chunk_size].float()
        log_q = quantized_logprobs[start : start + chunk_size].to(log_p)
        if not torch.isfinite(log_p).all() or not torch.isfinite(log_q).all():
            raise ValueError("Non-finite KL: inspect the scores instead of dropping positions.")
        full_kl = F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)
        # Conditional KL uses the reference token IDs and independently normalizes each head.
        ids = log_p.topk(top_k, dim=-1).indices
        log_p_head = F.log_softmax(log_p.gather(-1, ids), dim=-1)
        log_q_head = F.log_softmax(log_q.gather(-1, ids), dim=-1)
        head_kl = F.kl_div(log_q_head, log_p_head, log_target=True, reduction="none").sum(-1)
        totals += torch.stack((full_kl, head_kl), dim=-1).sum(0)
    means = totals / num_tokens
    if not torch.isfinite(means).all():
        raise ValueError("Non-finite KL: inspect the scores instead of dropping positions.")
    full_kl, head_kl = means.cpu().tolist()
    return {"full_vocab_kl": full_kl, "conditional_topk_kl": head_kl}


def _continuation_logprobs(output, prompt_tokens, vocab_size):
    rows = output.prompt_logprobs
    num_positions = len(output.prompt_token_ids)
    flat = hasattr(rows, "start_indices")
    if rows is None or not 0 < prompt_tokens < num_positions:
        raise ValueError("Require a prompt, continuation, and full prompt log-probabilities.")
    if (len(rows.start_indices) if flat else len(rows)) != num_positions:
        raise ValueError("Missing prompt log-probability positions.")
    scores = np.empty((num_positions - prompt_tokens, vocab_size), dtype=np.float32)
    vocabulary = np.arange(vocab_size)
    # vLLM position i predicts input token i, unlike Transformers' shifted logits.
    for target, position in enumerate(range(prompt_tokens, num_positions)):
        if flat:
            start, end = rows.start_indices[position], rows.end_indices[position]
            ids, values = rows.token_ids[start:end], rows.logprobs[start:end]
            # FlatLogprobs can prepend the observed token to the complete vocabulary.
            if len(ids) == vocab_size + 1:
                if ids[0] not in ids[1:] or values[0] != values[1:][ids[1:].index(ids[0])]:
                    raise ValueError("Inconsistent duplicate observed-token log-probability.")
                ids, values = ids[1:], values[1:]
        else:
            row = rows[position]
            if row is None:
                raise ValueError("Missing continuation log-probabilities.")
            ids, values = list(row), [value.logprob for value in row.values()]
        ids = np.asarray(ids)
        if ids.dtype.kind not in "iu":
            raise ValueError("Vocabulary token IDs must be integers.")
        if len(ids) != vocab_size or not (
            np.array_equal(ids, vocabulary) or np.array_equal(np.sort(ids), vocabulary)
        ):
            raise ValueError("Full-vocabulary KL requires every vocabulary token ID exactly once.")
        scores[target, ids] = values
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite continuation log-probabilities.")
    logprobs = torch.from_numpy(scores)
    if not torch.allclose(logprobs.logsumexp(-1), torch.zeros(len(scores)), atol=1e-4, rtol=0):
        raise ValueError("Expected normalized, unprocessed full-vocabulary log-probabilities.")
    return logprobs


def _build_report(examples, detailed_results=False):
    if not examples:
        raise ValueError("No evaluation examples.")
    summary = {
        name: sum(example[name] for example in examples) / len(examples)
        for name in ("full_vocab_kl", "conditional_topk_kl")
    }
    return {"summary": summary, "examples": examples} if detailed_results else summary


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


def _prepare(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=args.trust_remote_code
    )
    candidate_tokenizer = AutoTokenizer.from_pretrained(
        args.quantized_model,
        revision=args.quantized_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.get_vocab() != candidate_tokenizer.get_vocab():
        raise ValueError(
            "Reference and quantized checkpoints must have identical token-ID mappings."
        )
    try:
        generation = GenerationConfig.from_pretrained(args.model, revision=args.revision)
    except OSError:
        config = AutoConfig.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=args.trust_remote_code
        )
        generation = GenerationConfig.from_model_config(config)
    eos = generation.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    prompts, fingerprint = _wikitext_prompts(
        tokenizer, args.num_examples, args.prompt_tokens, args.seed
    )
    return {
        "prompts": prompts,
        "eos_token_ids": eos if isinstance(eos, list) else ([] if eos is None else [eos]),
        "dataset_fingerprint": fingerprint,
    }


def _score(llm, sampling, example, vocab_size):
    sequence = example["prompt_ids"] + example["generated_ids"]
    output = llm.generate([{"prompt_token_ids": sequence}], sampling, use_tqdm=False)[0]
    if output.prompt_token_ids != sequence:
        raise ValueError("vLLM changed the supplied teacher-forcing token IDs.")
    return _continuation_logprobs(output, len(example["prompt_ids"]), vocab_size)


def _stop_phase(signum, _frame):
    raise SystemExit(128 + signum)


def _run_phase(args, directory, role):
    signal.signal(signal.SIGTERM, _stop_phase)
    # vLLM is an optional CUDA dependency; CPU metric tests do not initialize its runtime.
    import vllm
    from vllm import LLM, SamplingParams

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    is_reference = role == "reference"
    model = args.model if is_reference else args.quantized_model
    revision = args.revision if is_reference else args.quantized_revision
    llm = LLM(
        model=model,
        revision=revision,
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.prompt_tokens + args.max_new_tokens + 1,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        generation_config="vllm",
        seed=args.seed,
        max_logprobs=-1,
        logprobs_mode="raw_logprobs",
        kv_cache_dtype="bfloat16" if is_reference else args.kv_cache_dtype,
    )
    try:
        config = llm.llm_engine.vllm_config
        quantization = config.model_config.quantization
        identical_control = (args.model, args.revision) == (
            args.quantized_model,
            args.quantized_revision,
        )
        if is_reference and quantization is not None:
            raise ValueError("The reference checkpoint must be unquantized BF16.")
        if (
            not is_reference
            and not identical_control
            and not str(quantization).startswith("modelopt")
        ):
            raise ValueError("Supply an exported ModelOpt quantized checkpoint.")
        vocab_size = config.model_config.get_vocab_size()
        if args.top_k > vocab_size:
            raise ValueError("--top_k exceeds the model vocabulary.")
        if not is_reference:
            reference_settings = json.loads((directory / "reference.json").read_text())
            if vocab_size != reference_settings["vocab_size"]:
                raise ValueError("Reference and quantized output vocabularies must match.")
        settings = {
            "model": model,
            "revision": revision or getattr(config.model_config.hf_config, "_commit_hash", None),
            "vocab_size": vocab_size,
            "model_dtype": str(config.model_config.dtype),
            "quantization": quantization,
            "kv_cache_dtype": (
                str(config.model_config.dtype).removeprefix("torch.")
                if config.cache_config.cache_dtype == "auto"
                else config.cache_config.cache_dtype
            ),
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": config.model_config.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "enforce_eager": True,
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
        }
        (directory / f"{role}.json").write_text(json.dumps(settings, indent=2))
        print(f"{role}: {settings}", flush=True)
        scoring = SamplingParams(
            temperature=0, max_tokens=1, prompt_logprobs=-1, detokenize=False, flat_logprobs=True
        )
        generation = SamplingParams(
            temperature=0,
            max_tokens=args.max_new_tokens,
            ignore_eos=True,
            stop_token_ids=manifest["eos_token_ids"],
            detokenize=False,
        )
        examples = [] if is_reference else json.loads((directory / "examples.json").read_text())
        for index, prompt in enumerate(manifest["prompts"]):
            if is_reference:
                output = llm.generate(
                    [{"prompt_token_ids": prompt["input_ids"]}], generation, use_tqdm=False
                )[0]
                generated_ids = list(output.outputs[0].token_ids)
                if not generated_ids:
                    raise ValueError("The reference generated an empty continuation.")
                example = {
                    "example": index,
                    "block_index": prompt["block_index"],
                    "prompt_ids": prompt["input_ids"],
                    "generated_ids": generated_ids,
                    "generated_tokens": len(generated_ids),
                }
                examples.append(example)
            else:
                example = examples[index]
            logprobs = _score(llm, scoring, example, vocab_size)
            cache = directory / f"{index}.npy"
            if is_reference:
                np.save(cache, logprobs.numpy(), allow_pickle=False)
            else:
                reference = torch.from_numpy(np.load(cache, allow_pickle=False))
                example.update(mean_kl(reference, logprobs, args.top_k))
                cache.unlink()
            print(
                f"{role} {index + 1}/{len(manifest['prompts'])}: {example['generated_tokens']} tokens",
                flush=True,
            )
        (directory / "examples.json").write_text(json.dumps(examples))
    finally:
        # Close engine workers before multiprocessing joins this phase.
        llm.llm_engine.engine_core.shutdown()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Unquantized BF16 reference path or Hub ID.")
    parser.add_argument(
        "--quantized_model", required=True, help="Exported ModelOpt path or Hub ID."
    )
    parser.add_argument("--revision")
    parser.add_argument("--quantized_revision")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--prompt_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument(
        "--kv_cache_dtype",
        default="auto",
        help="Candidate KV dtype; auto follows checkpoint configuration.",
    )
    parser.add_argument(
        "--temp_dir", type=Path, help="Scratch directory for temporary FP32 reference scores."
    )
    parser.add_argument("--output", type=Path, default=Path("kl_results.json"))
    parser.add_argument("--detailed_results", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    for name in (
        "num_examples",
        "prompt_tokens",
        "max_new_tokens",
        "top_k",
        "tensor_parallel_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu_memory_utilization must be in (0, 1].")
    return args


def main():
    """Run two isolated model processes and write KL results, removing temporary scores on exit."""
    args = _parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "Launch one evaluator process; use --tensor_parallel_size for multiple GPUs."
        )
    manifest = _prepare(args)
    with tempfile.TemporaryDirectory(prefix="modelopt-kl-", dir=args.temp_dir) as directory:
        work = Path(directory)
        (work / "manifest.json").write_text(json.dumps(manifest))
        for role in ("reference", "quantized"):
            process = multiprocessing.get_context("spawn").Process(
                target=_run_phase, args=(args, directory, role)
            )
            process.start()
            try:
                process.join()
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"{role} vLLM process failed with exit code {process.exitcode}.")
        examples = json.loads((work / "examples.json").read_text())
        report = _build_report(examples, args.detailed_results)
        if args.detailed_results:
            report["settings"] = {
                role: json.loads((work / f"{role}.json").read_text())
                for role in ("reference", "quantized")
            }
            report["settings"].update(
                {
                    "backend": "vllm",
                    "metric_dtype": "float32",
                    "kl_direction": "reference_to_quantized",
                    "units": "nats",
                    "aggregation": "mean_tokens_per_example_then_mean_examples",
                    "evaluation_dataset": "Salesforce/wikitext/wikitext-2-raw-v1/test",
                    "dataset_fingerprint": manifest["dataset_fingerprint"],
                    "num_examples": args.num_examples,
                    "prompt_tokens": args.prompt_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "eos_token_ids": manifest["eos_token_ids"],
                    "generation": "greedy_until_eos_or_token_cap",
                    "scoring": "teacher_forced_prefill_raw_logprobs",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"] if args.detailed_results else report, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

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

"""Teacher-forced fidelity health check between a base LLM and its quantized variant.

Computes per-position KL divergence, EAR (Expected Acceptance Rate), top-1 mismatch
rate, and ΔNLL by forcing both models to score the same token sequence — produced
once by the base model — via the vLLM OpenAI-compatible ``/v1/completions`` endpoint
with ``echo=True`` and ``logprobs=k``. Results are split by prefill vs generation
phase, since serving stacks frequently use different kernels for the two phases.

Why these metrics: aggregate accuracy and perplexity hide significant quantization
drift (a ~24% answer-flip rate has been observed at <1% accuracy delta). Forward
KL on top-k correlates ~0.98 with downstream flip rate while costing only one
forward pass per model on a shared prompt set.

Usage (two vLLM servers already running):
    python examples/llm_eval/fidelity_check.py \\
        --base-url http://localhost:8000/v1 --base-model meta-llama/Llama-3-8B \\
        --quant-url http://localhost:8001/v1 --quant-model llama-3-8b-nvfp4 \\
        --dataset cnn_dailymail --num-prompts 128 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from openai import OpenAI

from modelopt.torch.utils.dataset_utils import get_dataset_samples

# Floor used when a base top-k token is absent from quant's top-k.  log(1e-9) ≈ -20.7.
# Conservative enough that one missing token contributes a bounded but non-tiny KL term.
MISSING_LOGPROB_FLOOR = math.log(1e-9)


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def _position_metrics(
    base_topk: dict[str, float],
    quant_topk: dict[str, float],
    ref_token: str,
) -> dict[str, float]:
    """Compute KL, EAR, top-1 mismatch, ΔNLL at a single token position.

    KL and EAR are evaluated on the union of base's and quant's reported top-k
    supports, with missing entries floored to ``MISSING_LOGPROB_FLOOR``.  Both
    distributions are renormalized over the union so KL is well-defined.
    """
    support = set(base_topk) | set(quant_topk)
    keys = list(support)
    base_lp = np.array([base_topk.get(t, MISSING_LOGPROB_FLOOR) for t in keys], dtype=np.float64)
    quant_lp = np.array([quant_topk.get(t, MISSING_LOGPROB_FLOOR) for t in keys], dtype=np.float64)
    base_lp -= _logsumexp(base_lp)
    quant_lp -= _logsumexp(quant_lp)
    base_p = np.exp(base_lp)
    quant_p = np.exp(quant_lp)

    kl = float(np.sum(base_p * (base_lp - quant_lp)))
    ear = float(np.sum(np.minimum(base_p, quant_p)))
    base_top1 = keys[int(np.argmax(base_lp))]
    quant_top1 = keys[int(np.argmax(quant_lp))]
    mismatch = float(base_top1 != quant_top1)

    # ΔNLL on the reference token (base's chosen token, or input token in prefill).
    base_ref_lp = base_topk.get(ref_token, MISSING_LOGPROB_FLOOR)
    quant_ref_lp = quant_topk.get(ref_token, MISSING_LOGPROB_FLOOR)
    delta_nll = float(-quant_ref_lp - (-base_ref_lp))

    return {"kl": kl, "ear": ear, "mismatch": mismatch, "delta_nll": delta_nll}


def _coerce_topk(entry: Any) -> dict[str, float] | None:
    """Normalize the OpenAI SDK's per-position ``top_logprobs`` entry to a dict."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        return {str(k): float(v) for k, v in entry.items()}
    # Some SDKs expose it as an object with ``.items()`` or as a Pydantic model.
    if hasattr(entry, "items"):
        return {str(k): float(v) for k, v in entry.items()}
    if hasattr(entry, "model_dump"):
        return {str(k): float(v) for k, v in entry.model_dump().items()}
    raise TypeError(f"Unrecognized top_logprobs entry type: {type(entry).__name__}")


def evaluate_prompt(
    prompt: str,
    base_client: OpenAI,
    quant_client: OpenAI,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Score one prompt under teacher forcing and return per-position metrics.

    Uses a 3-call protocol so base and quant logprobs come from the same vLLM
    code path (prefill+echo).  An earlier 2-call protocol mixed base's
    decode-kernel logprobs against quant's prefill-kernel logprobs and gave a
    non-zero KL noise floor (~0.03) even when comparing a model to itself.
    """
    # 1. Generate from base to obtain the teacher-forced token sequence.  We
    #    discard the logprobs from this call to avoid mixing decode-kernel and
    #    prefill-kernel outputs in the comparison.
    base_gen_resp = base_client.completions.create(
        model=args.base_model,
        prompt=prompt,
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        logprobs=1,
        echo=False,
    )
    full_text = prompt + base_gen_resp.choices[0].text
    prompt_token_count = int(base_gen_resp.usage.prompt_tokens)

    # 2. Re-evaluate base on the FULL sequence via echo+max_tokens=0 — same
    #    code path that quant will use, so self-comparison is exactly zero.
    base_resp = base_client.completions.create(
        model=args.base_model,
        prompt=full_text,
        max_tokens=0,
        temperature=0.0,
        logprobs=args.top_k,
        echo=True,
    )
    base_choice = base_resp.choices[0]
    base_lp_obj = base_choice.logprobs
    if base_lp_obj is None:
        return {"skipped": True, "reason": "base server did not return logprobs"}

    base_tokens: list[str] = list(base_lp_obj.tokens)
    base_top: list[Any] = list(base_lp_obj.top_logprobs)

    # 3. Evaluate quant on the same sequence with the same call shape.
    quant_resp = quant_client.completions.create(
        model=args.quant_model,
        prompt=full_text,
        max_tokens=0,
        temperature=0.0,
        logprobs=args.top_k,
        echo=True,
    )
    quant_choice = quant_resp.choices[0]
    quant_lp_obj = quant_choice.logprobs
    if quant_lp_obj is None:
        return {"skipped": True, "reason": "quant server did not return logprobs"}

    quant_tokens: list[str] = list(quant_lp_obj.tokens)
    quant_top: list[Any] = list(quant_lp_obj.top_logprobs)

    if len(base_tokens) != len(quant_tokens):
        return {
            "skipped": True,
            "reason": f"token length mismatch (base={len(base_tokens)}, quant={len(quant_tokens)}): "
            "base and quant must share a tokenizer",
        }

    per_position: list[dict[str, Any]] = []
    for i in range(1, len(base_tokens)):  # position 0 has no preceding context
        base_lp = _coerce_topk(base_top[i])
        quant_lp = _coerce_topk(quant_top[i])
        if base_lp is None or quant_lp is None:
            continue
        m = _position_metrics(base_lp, quant_lp, base_tokens[i])
        m["phase"] = "prefill" if i < prompt_token_count else "generation"
        per_position.append(m)

    return {
        "per_position": per_position,
        "prompt_tokens": prompt_token_count,
        "total_tokens": len(base_tokens),
    }


def _bootstrap_ci(values: list[float], n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _aggregate(per_position_pool: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    values = [p[metric] for p in per_position_pool]
    if not values:
        return None
    arr = np.asarray(values)
    lo, hi = _bootstrap_ci(values)
    return {"mean": float(arr.mean()), "ci95": [lo, hi], "n": len(values)}


def _print_phase(phase: str, block: dict[str, Any]) -> None:
    print(f"\n[{phase}]")
    for k in ("kl", "ear", "mismatch", "delta_nll"):
        v = block.get(k)
        if v is None:
            print(f"  {k:10s}: (no data)")
            continue
        print(
            f"  {k:10s}: {v['mean']:.5f}  "
            f"(95% CI [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}], n={v['n']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher-forced KL / EAR / mismatch / ΔNLL between base and quantized LLMs."
    )
    parser.add_argument(
        "--base-url", required=True, help="vLLM OpenAI-compatible base URL for the base model"
    )
    parser.add_argument(
        "--base-model", required=True, help="Model name as registered on the base server"
    )
    parser.add_argument(
        "--quant-url", required=True, help="vLLM OpenAI-compatible base URL for the quantized model"
    )
    parser.add_argument(
        "--quant-model", required=True, help="Model name as registered on the quant server"
    )
    parser.add_argument(
        "--dataset",
        default="cnn_dailymail",
        help="Dataset name from modelopt.torch.utils.dataset_utils.SUPPORTED_DATASET_CONFIG, "
        "an HF dataset path, or a local .jsonl file",
    )
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="top_logprobs to request from each server. vLLM caps this at --max-logprobs (default 20).",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=2000,
        help="Truncate each prompt to this many characters so prefill stays affordable.",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", default="fidelity_report.json")
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key (vLLM ignores this; placeholder for OpenAI client).",
    )
    args = parser.parse_args()

    if args.top_k > 20:
        print(
            f"[warn] top-k={args.top_k} exceeds vLLM's default --max-logprobs=20. "
            "Start each server with --max-logprobs >= top-k or the request will fail.",
            file=sys.stderr,
        )

    base_client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    quant_client = OpenAI(base_url=args.quant_url, api_key=args.api_key)

    prompts = get_dataset_samples(args.dataset, args.num_prompts)
    prompts = [p[: args.max_prompt_chars] for p in prompts if p]

    print(
        f"[config] base={args.base_model} @ {args.base_url}\n"
        f"         quant={args.quant_model} @ {args.quant_url}\n"
        f"         dataset={args.dataset} n_prompts={len(prompts)} "
        f"max_new_tokens={args.max_new_tokens} top_k={args.top_k}"
    )

    results: list[dict[str, Any]] = []
    skipped: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(evaluate_prompt, p, base_client, quant_client, args): i
            for i, p in enumerate(prompts)
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                r = fut.result()
            except Exception as e:
                skipped.append(f"prompt {futures[fut]}: {type(e).__name__}: {e}")
                continue
            if r.get("skipped"):
                skipped.append(f"prompt {futures[fut]}: {r['reason']}")
                continue
            results.append(r)
            if done % 8 == 0 or done == len(prompts):
                print(f"[{done}/{len(prompts)}] prompts scored")

    if not results:
        raise RuntimeError(
            "All prompts failed. First few skip reasons:\n  " + "\n  ".join(skipped[:5])
        )

    prefill_pool = [p for r in results for p in r["per_position"] if p["phase"] == "prefill"]
    gen_pool = [p for r in results for p in r["per_position"] if p["phase"] == "generation"]
    overall_pool = [p for r in results for p in r["per_position"]]

    metrics = ("kl", "ear", "mismatch", "delta_nll")
    phase_blocks: dict[str, dict[str, Any]] = {
        "prefill": {m: _aggregate(prefill_pool, m) for m in metrics},
        "generation": {m: _aggregate(gen_pool, m) for m in metrics},
        "overall": {m: _aggregate(overall_pool, m) for m in metrics},
    }
    report: dict[str, Any] = {
        "config": {
            "base_model": args.base_model,
            "quant_model": args.quant_model,
            "dataset": args.dataset,
            "num_prompts": args.num_prompts,
            "n_prompts_used": len(results),
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
            "max_prompt_chars": args.max_prompt_chars,
        },
        **phase_blocks,
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:10],
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Fidelity report ===")
    print(f"prompts used: {len(results)}  skipped: {len(skipped)}")
    for phase, block in phase_blocks.items():
        _print_phase(phase, block)
    print(f"\nFull report written to: {args.output}")


if __name__ == "__main__":
    main()

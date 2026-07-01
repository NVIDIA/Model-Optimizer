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

Modes
-----
``full`` (default): both base and quant servers up concurrently, end-to-end.

The remaining modes split the same 3-call protocol into persistable phases so the
two models do not need to be alive at the same time (useful when a single model
fills the node):

* ``collect-base``: launch base, generate teacher sequence (call 1) and re-score
  via ``echo+max_tokens=0`` (call 2). Dump per-position top-k logprobs to JSONL.
* ``collect-quant``: launch quant, replay each saved ``prompt + gen_text`` via
  ``echo+max_tokens=0`` (call 3). Dump per-position top-k logprobs to JSONL.
* ``compare``: pure-CPU. Load two JSONL files, align position-by-position, compute
  the same KL/EAR/mismatch/ΔNLL numbers as ``full`` mode.

Both ``collect-quant`` and ``collect-base``'s call-2 use the identical prefill+echo
code path on the server, so the methodology invariant holds across the split.
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


def _generate_teacher(
    client: OpenAI, model: str, prompt: str, max_new_tokens: int
) -> tuple[str, int]:
    """Call 1: greedy generate the teacher continuation. Returns (gen_text, prompt_token_count)."""
    resp = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        logprobs=1,
        echo=False,
    )
    return resp.choices[0].text, int(resp.usage.prompt_tokens)


def _echo_logprobs(
    client: OpenAI, model: str, full_text: str, top_k: int
) -> tuple[list[str] | None, list[dict[str, float] | None] | None]:
    """Calls 2 and 3: re-score the full sequence via echo and return per-position top-k.

    Both base and quant go through this identical entry point, so their logprobs
    come from the same vLLM code path (prefill kernel).

    SGLang rejects ``max_tokens=0`` (vLLM accepts it). We send ``max_tokens=1`` and
    drop the trailing generated position — the prefix positions we care about all
    come from the prefill kernel either way.
    """
    resp = client.completions.create(
        model=model,
        prompt=full_text,
        max_tokens=1,
        temperature=0.0,
        logprobs=top_k,
        echo=True,
    )
    lp_obj = resp.choices[0].logprobs
    if lp_obj is None:
        return None, None
    tokens = list(lp_obj.tokens)
    topk = [_coerce_topk(t) for t in lp_obj.top_logprobs]
    # Drop the extra generated position so per-position alignment matches the prefix length.
    full_text_len_tokens = int(resp.usage.prompt_tokens)
    if len(tokens) > full_text_len_tokens:
        tokens = tokens[:full_text_len_tokens]
        topk = topk[:full_text_len_tokens]
    return tokens, topk


def _score(
    base_tokens: list[str],
    base_topk: list[dict[str, float] | None],
    quant_tokens: list[str],
    quant_topk: list[dict[str, float] | None],
    prompt_token_count: int,
) -> dict[str, Any]:
    """Pairwise per-position scoring. Returns either {per_position, ...} or {skipped, reason}."""
    if len(base_tokens) != len(quant_tokens):
        return {
            "skipped": True,
            "reason": f"token length mismatch (base={len(base_tokens)}, quant={len(quant_tokens)}): "
            "base and quant must share a tokenizer",
        }
    per_position: list[dict[str, Any]] = []
    for i in range(1, len(base_tokens)):  # position 0 has no preceding context
        base_lp = base_topk[i]
        quant_lp = quant_topk[i]
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
    gen_text, prompt_token_count = _generate_teacher(
        base_client, args.base_model, prompt, args.max_new_tokens
    )
    full_text = prompt + gen_text
    base_tokens, base_topk = _echo_logprobs(base_client, args.base_model, full_text, args.top_k)
    if base_tokens is None or base_topk is None:
        return {"skipped": True, "reason": "base server did not return logprobs"}
    quant_tokens, quant_topk = _echo_logprobs(quant_client, args.quant_model, full_text, args.top_k)
    if quant_tokens is None or quant_topk is None:
        return {"skipped": True, "reason": "quant server did not return logprobs"}
    return _score(base_tokens, base_topk, quant_tokens, quant_topk, prompt_token_count)


def _collect_base_one(
    idx: int, prompt: str, client: OpenAI, model: str, max_new_tokens: int, top_k: int
) -> dict[str, Any]:
    """For ``collect-base``: generate teacher + re-score, return JSONL-ready record."""
    gen_text, prompt_token_count = _generate_teacher(client, model, prompt, max_new_tokens)
    full_text = prompt + gen_text
    tokens, topk = _echo_logprobs(client, model, full_text, top_k)
    if tokens is None:
        return {"prompt_idx": idx, "skipped": True, "reason": "no logprobs"}
    return {
        "prompt_idx": idx,
        "prompt": prompt,
        "gen_text": gen_text,
        "prompt_token_count": prompt_token_count,
        "tokens": tokens,
        "top_logprobs": topk,
    }


def _collect_quant_one(
    teacher_rec: dict[str, Any], client: OpenAI, model: str, top_k: int
) -> dict[str, Any]:
    """For ``collect-quant``: replay teacher's ``prompt + gen_text`` through echo, return record."""
    if teacher_rec.get("skipped"):
        return {
            "prompt_idx": teacher_rec.get("prompt_idx"),
            "skipped": True,
            "reason": "teacher skipped",
        }
    full_text = teacher_rec["prompt"] + teacher_rec["gen_text"]
    tokens, topk = _echo_logprobs(client, model, full_text, top_k)
    if tokens is None:
        return {"prompt_idx": teacher_rec["prompt_idx"], "skipped": True, "reason": "no logprobs"}
    return {
        "prompt_idx": teacher_rec["prompt_idx"],
        "prompt_token_count": teacher_rec["prompt_token_count"],
        "tokens": tokens,
        "top_logprobs": topk,
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


def _build_report(
    results: list[dict[str, Any]], config: dict[str, Any], skipped: list[str]
) -> dict[str, Any]:
    prefill_pool = [p for r in results for p in r["per_position"] if p["phase"] == "prefill"]
    gen_pool = [p for r in results for p in r["per_position"] if p["phase"] == "generation"]
    overall_pool = [p for r in results for p in r["per_position"]]
    metrics = ("kl", "ear", "mismatch", "delta_nll")
    phase_blocks: dict[str, dict[str, Any]] = {
        "prefill": {m: _aggregate(prefill_pool, m) for m in metrics},
        "generation": {m: _aggregate(gen_pool, m) for m in metrics},
        "overall": {m: _aggregate(overall_pool, m) for m in metrics},
    }
    return {
        "config": config,
        **phase_blocks,
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:10],
    }


def _print_report(report: dict[str, Any], n_used: int, n_skipped: int) -> None:
    print("\n=== Fidelity report ===")
    print(f"prompts used: {n_used}  skipped: {n_skipped}")
    for phase in ("prefill", "generation", "overall"):
        if phase in report:
            _print_phase(phase, report[phase])


def _run_full(args: argparse.Namespace) -> None:
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
    config = {
        "mode": "full",
        "base_model": args.base_model,
        "quant_model": args.quant_model,
        "dataset": args.dataset,
        "num_prompts": args.num_prompts,
        "n_prompts_used": len(results),
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "max_prompt_chars": args.max_prompt_chars,
    }
    report = _build_report(results, config, skipped)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    _print_report(report, len(results), len(skipped))
    print(f"\nFull report written to: {args.output}")


def _run_collect_base(args: argparse.Namespace) -> None:
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    prompts = get_dataset_samples(args.dataset, args.num_prompts)
    prompts = [p[: args.max_prompt_chars] for p in prompts if p]
    print(
        f"[collect-base] model={args.base_model} @ {args.base_url}\n"
        f"               dataset={args.dataset} n_prompts={len(prompts)} "
        f"max_new_tokens={args.max_new_tokens} top_k={args.top_k}\n"
        f"               concurrency={args.concurrency} output={args.output_collect}"
    )
    records: dict[int, dict[str, Any]] = {}
    skipped: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                _collect_base_one,
                i,
                p,
                client,
                args.base_model,
                args.max_new_tokens,
                args.top_k,
            ): i
            for i, p in enumerate(prompts)
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                r = fut.result()
            except Exception as e:
                skipped.append(f"prompt {futures[fut]}: {type(e).__name__}: {e}")
                continue
            records[r["prompt_idx"]] = r
            if done % 8 == 0 or done == len(prompts):
                print(f"[{done}/{len(prompts)}] prompts collected")
    # Write in deterministic prompt_idx order — makes downstream alignment robust.
    with open(args.output_collect, "w") as f:
        for i in range(len(prompts)):
            if i in records:
                f.write(json.dumps(records[i]) + "\n")
    print(
        f"\n[collect-base] wrote {len(records)} records to {args.output_collect}; "
        f"skipped: {len(skipped)}"
    )
    if skipped:
        print("  first few skips:")
        for s in skipped[:5]:
            print(f"    {s}")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _run_collect_quant(args: argparse.Namespace) -> None:
    client = OpenAI(base_url=args.quant_url, api_key=args.api_key)
    teacher = _read_jsonl(args.teacher_jsonl)
    print(
        f"[collect-quant] model={args.quant_model} @ {args.quant_url}\n"
        f"                teacher={args.teacher_jsonl} n={len(teacher)} top_k={args.top_k}\n"
        f"                concurrency={args.concurrency} output={args.output_collect}"
    )
    records: dict[int, dict[str, Any]] = {}
    skipped: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_collect_quant_one, rec, client, args.quant_model, args.top_k): i
            for i, rec in enumerate(teacher)
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                r = fut.result()
            except Exception as e:
                skipped.append(f"prompt {futures[fut]}: {type(e).__name__}: {e}")
                continue
            records[r["prompt_idx"]] = r
            if done % 8 == 0 or done == len(teacher):
                print(f"[{done}/{len(teacher)}] prompts scored")
    with open(args.output_collect, "w") as f:
        for rec in teacher:
            i = rec["prompt_idx"]
            if i in records:
                f.write(json.dumps(records[i]) + "\n")
    print(
        f"\n[collect-quant] wrote {len(records)} records to {args.output_collect}; "
        f"skipped: {len(skipped)}"
    )
    if skipped:
        print("  first few skips:")
        for s in skipped[:5]:
            print(f"    {s}")


def _run_compare(args: argparse.Namespace) -> None:
    base = {r["prompt_idx"]: r for r in _read_jsonl(args.base_jsonl) if not r.get("skipped")}
    quant = {r["prompt_idx"]: r for r in _read_jsonl(args.quant_jsonl) if not r.get("skipped")}
    shared = sorted(set(base) & set(quant))
    print(
        f"[compare] base={args.base_jsonl} ({len(base)} records)\n"
        f"          quant={args.quant_jsonl} ({len(quant)} records)\n"
        f"          shared={len(shared)}"
    )
    results: list[dict[str, Any]] = []
    skipped: list[str] = []
    for idx in shared:
        b = base[idx]
        q = quant[idx]
        # Restore from JSON: top_logprobs is list[dict|None] already (None survives round-trip).
        r = _score(
            b["tokens"], b["top_logprobs"], q["tokens"], q["top_logprobs"], b["prompt_token_count"]
        )
        if r.get("skipped"):
            skipped.append(f"prompt {idx}: {r['reason']}")
            continue
        results.append(r)
    if not results:
        raise RuntimeError(
            "All prompts failed compare. First few skip reasons:\n  " + "\n  ".join(skipped[:5])
        )
    config = {
        "mode": "compare",
        "base_jsonl": args.base_jsonl,
        "quant_jsonl": args.quant_jsonl,
        "n_shared": len(shared),
        "n_prompts_used": len(results),
    }
    report = _build_report(results, config, skipped)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    _print_report(report, len(results), len(skipped))
    print(f"\nFull report written to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher-forced KL / EAR / mismatch / ΔNLL between base and quantized LLMs."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "collect-base", "collect-quant", "compare"),
        default="full",
        help="Run mode. 'full' needs both servers up; the others split the protocol across runs.",
    )
    # Server / model identity (used by full + collect-*; arguments are optional per mode).
    parser.add_argument("--base-url", help="vLLM URL for the base model")
    parser.add_argument("--base-model", help="Served model name on the base server")
    parser.add_argument("--quant-url", help="vLLM URL for the quantized model")
    parser.add_argument("--quant-model", help="Served model name on the quant server")
    # Dataset / sampling (used by full + collect-base).
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
    # I/O.
    parser.add_argument(
        "--output",
        default="fidelity_report.json",
        help="Output JSON report path (used by 'full' and 'compare').",
    )
    parser.add_argument(
        "--output-collect",
        help="Output JSONL path (used by 'collect-base' and 'collect-quant').",
    )
    parser.add_argument(
        "--teacher-jsonl",
        help="Input JSONL produced by 'collect-base' (used by 'collect-quant').",
    )
    parser.add_argument(
        "--base-jsonl",
        help="Base JSONL for 'compare' (typically a 'collect-base' output).",
    )
    parser.add_argument(
        "--quant-jsonl",
        help="Quant JSONL for 'compare' (typically a 'collect-quant' output).",
    )
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

    if args.mode == "full":
        missing = [
            n
            for n in ("base_url", "base_model", "quant_url", "quant_model")
            if not getattr(args, n)
        ]
        if missing:
            parser.error(
                f"--mode full requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        _run_full(args)
    elif args.mode == "collect-base":
        missing = [n for n in ("base_url", "base_model", "output_collect") if not getattr(args, n)]
        if missing:
            parser.error(
                f"--mode collect-base requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        _run_collect_base(args)
    elif args.mode == "collect-quant":
        missing = [
            n
            for n in ("quant_url", "quant_model", "teacher_jsonl", "output_collect")
            if not getattr(args, n)
        ]
        if missing:
            parser.error(
                f"--mode collect-quant requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        _run_collect_quant(args)
    elif args.mode == "compare":
        missing = [n for n in ("base_jsonl", "quant_jsonl") if not getattr(args, n)]
        if missing:
            parser.error(
                f"--mode compare requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        _run_compare(args)


if __name__ == "__main__":
    main()

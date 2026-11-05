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

"""Validate, freeze and score JSONL prompt banks with shared greedy continuations."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path

__all__ = []


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checkpoint_hash(directory):
    directory = Path(directory)
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Study checkpoints must be nonempty local directories.")
    return _digest({str(path.relative_to(directory)): _file_hash(path) for path in files})


def _study_manifest(args, bank):
    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("dry_run", "output", "work_dir")
    }
    settings.update(
        {
            "input_sha256": bank["input_sha256"],
            "model_sha256": _checkpoint_hash(args.model),
            "quantized_model_sha256": _checkpoint_hash(args.quantized_model),
            "evaluator_sha256": {
                p.name: _file_hash(p) for p in Path(__file__).parent.glob("kl*.py")
            },
            "greedy_trajectory_repeats": 1,
            "metric_dtype": "float32",
            "units": "nats",
            "aggregation": "mean_tokens_per_example_then_mean_examples",
        }
    )
    if any(
        row.get("reference_sha256", settings["model_sha256"]) != settings["model_sha256"]
        for row in bank["examples"]
    ):
        raise ValueError("Frozen trajectories declare a different reference checkpoint.")
    manifest = {"identity": _digest(settings), "settings": settings}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    path = args.work_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(
            "Incompatible resume: inputs, settings, evaluator or checkpoint bytes changed."
        )
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _run(args, bank, manifest):
    # vLLM and Transformers are optional for validation and artifact inspection.
    from kl_engine import Engine
    from transformers import AutoConfig, AutoTokenizer

    reference = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    candidate = AutoTokenizer.from_pretrained(
        args.quantized_model, trust_remote_code=args.trust_remote_code
    )
    if reference.get_vocab() != candidate.get_vocab():
        raise ValueError("Reference and candidate token-ID mappings differ.")
    tokenizer_sha = _digest(reference.get_vocab())
    for row in bank["examples"]:
        if row.get("tokenizer_sha256", tokenizer_sha) != tokenizer_sha:
            raise ValueError("Frozen tokenizer identity differs from the scoring tokenizer.")
    for model in (args.model, args.quantized_model):
        config = AutoConfig.from_pretrained(model, trust_remote_code=args.trust_remote_code)
        text = getattr(config, "text_config", config)
        if args.context_limit > text.max_position_embeddings:
            raise ValueError("Explicit context limit exceeds the checkpoint's trained context.")
    outputs = args.work_dir / "examples"
    outputs.mkdir(exist_ok=True)
    pending, results = [], []
    for row in bank["examples"]:
        path = outputs / f"{_digest(row['id'])}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["study_identity"] != manifest["identity"] or old["input_identity"] != _digest(
                row
            ):
                raise ValueError("Incompatible resumed result provenance.")
            results.append(old)
        else:
            pending.append((row, path))
    ref = other = None
    try:
        if pending:
            ref = Engine(args, args.model, "reference", args.reference_devices)
            if args.phase == "score":
                other = Engine(args, args.quantized_model, "candidate", args.candidate_devices)
                if ref.runtime["vocab_size"] != other.runtime["vocab_size"]:
                    raise ValueError("Reference and candidate output vocabularies differ.")
            runtime = {
                "reference": ref.runtime,
                "candidate": other.runtime if other else None,
                "tokenizer_sha256": tokenizer_sha,
            }
            runtime_path = args.work_dir / "runtime.json"
            if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
                raise ValueError("Incompatible resumed runtime.")
            _write(runtime_path, runtime)
        for row, path in pending:
            assert ref is not None
            if args.phase == "freeze":
                value = ref.call(op="generate", row=row)
                value["tokenizer_sha256"] = tokenizer_sha
                value["reference_sha256"] = manifest["settings"]["model_sha256"]
            else:
                assert other is not None
                cache = args.work_dir / "reference.f32"
                try:
                    captured = ref.call(op="score", row=row, output=str(cache))
                    value = other.call(op="score", row=row, output=str(cache), reference=str(cache))
                    value["reference_audit"] = captured
                finally:
                    cache.unlink(missing_ok=True)
            result = {
                **row,
                **value,
                "study_identity": manifest["identity"],
                "input_identity": _digest(row),
            }
            _write(path, result)
            results.append(result)
            print(f"Completed {row['id']}", flush=True)
    finally:
        if other is not None:
            other.close()
        if ref is not None:
            ref.close()
    by_id = {r["id"]: r for r in results}
    results = [by_id[r["id"]] for r in bank["examples"]]
    if args.phase == "freeze":
        # JSONL is directly reusable across candidates; no candidate generation is performed.
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in results))
        temporary.replace(args.output)
        return None
    return {
        "manifest": manifest,
        "examples": results,
        "summary": {
            "full_vocab_kl": math.fsum(
                math.fsum(r["metrics"]["full_vocab_kl"]) / len(r["generated_ids"]) for r in results
            )
            / len(results),
            "conditional_topk_kl": {
                str(k): math.fsum(
                    math.fsum(r["metrics"]["conditional_topk_kl"][str(k)]) / len(r["generated_ids"])
                    for r in results
                )
                / len(results)
                for k in args.top_ks
            },
        },
    }


def _read_bank(args):
    with args.input.open() as stream:
        examples = [json.loads(line) for line in stream if line.strip()]
    seen = set()
    for example in examples:
        if not isinstance(example.get("id"), str) or example["id"] in seen:
            raise ValueError("Each prompt needs a unique string id.")
        seen.add(example["id"])
        for field in ("prompt_ids", "generated_ids"):
            ids = example.get(field, [])
            if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("Token IDs must be nonnegative integer lists.")
        prompt, generated = example["prompt_ids"], example.get("generated_ids", [])
        if not prompt or (args.phase != "freeze" and not generated):
            raise ValueError("Require complete prompts and frozen continuations.")
        if len(generated) > args.max_new_tokens:
            raise ValueError("Frozen continuation exceeds the generation cap.")
        if any(token in args.eos_token_ids for token in generated[:-1]):
            raise ValueError("Frozen continuation contains tokens after EOS.")
        available = args.context_limit - len(prompt)
        if available < len(generated) or available < 1:
            raise ValueError("Frozen sequence exceeds the explicit context limit.")
        example["context_limited"] = available < args.max_new_tokens
        if generated:
            example["stop_reason"] = (
                "eos"
                if generated[-1] in args.eos_token_ids
                else "context_limit"
                if len(generated) == available
                else "token_cap"
                if len(generated) == args.max_new_tokens
                else "frozen_prefix"
            )
    if not examples:
        raise ValueError("Empty prompt bank.")
    return {"input_sha256": _digest(examples), "examples": examples}


def main():
    """Execute one reproducible study phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["validate", "freeze", "score"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context_limit", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--eos_token_ids", type=int, nargs="+", default=[])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--quantized_model", type=Path)
    parser.add_argument("--work_dir", type=Path)
    parser.add_argument("--runner_sha256")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--top_ks", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024, 2048]
    )
    parser.add_argument("--reference_devices", default="0,1")
    parser.add_argument("--candidate_devices", default="2,3")
    parser.add_argument("--prefill_chunk_size", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--kv_cache_dtype", default="bfloat16")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Compare bounded capture with dense scores on short controls.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    if args.context_limit < 2 or args.max_new_tokens < 1:
        parser.error("Require a positive generation cap and context limit >= 2.")
    if (
        min(args.top_ks) < 1
        or args.prefill_chunk_size < 1
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        parser.error(
            "Require positive top-k values, prefill chunk size and GPU utilization in (0, 1]."
        )
    args.top_ks = sorted(set(args.top_ks))
    reference_devices = args.reference_devices.split(",")
    candidate_devices = args.candidate_devices.split(",")
    if (
        any(not device.strip() for device in reference_devices + candidate_devices)
        or len(set(reference_devices)) != len(reference_devices)
        or len(set(candidate_devices)) != len(candidate_devices)
        or (args.phase == "score" and set(reference_devices) & set(candidate_devices))
    ):
        parser.error("Resident engines require unique, disjoint GPU device groups.")
    report = _read_bank(args)
    if args.phase in ("freeze", "score"):
        if args.phase == "freeze":
            args.quantized_model = args.model
        if not all((args.model, args.quantized_model, args.work_dir, args.runner_sha256)):
            parser.error("Scoring requires local checkpoints, --work_dir and --runner_sha256.")
        if args.audit and any(
            len(r["prompt_ids"]) + len(r["generated_ids"]) > min(64, args.context_limit)
            for r in report["examples"]
        ):
            parser.error("Dense audits require short sequences of at most 64 tokens.")
        args.work_dir.mkdir(parents=True, exist_ok=True)
        with (args.work_dir / "lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = _study_manifest(args, report)
            if args.dry_run:
                report["manifest"] = manifest
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                report = _run(args, report, manifest)
                if report is None:
                    return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

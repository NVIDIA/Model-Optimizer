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

"""Collect mask-reuse calibration observations through the vLLM V1 path.

ModelOpt derives each trial threshold from an existing vanilla skip-softmax
``(a, b)`` fit, arms the custom backend with one exact prompt/target
invocation, and merges only raw sufficient statistics returned by every TP
rank.  No promoted reuse policy is loaded during collection.

Prompt JSONL schema (one object per line)::

    {"split":"calibration", "partition":"development", "inner_fold":0,
     "prompt_id":"p0", "source":"ruler/niah", "source_group_sha256":"...",
     "prompt":"...", "min_kv_tokens":8192, "max_kv_tokens":65536}

Usage::

    python examples/vllm_serve/collect_mask_reuse.py /path/to/checkpoint \
        --model-id Nemotron-3-Ultra \
        --checkpoint-manifest-sha256 012345... \
        --plan nemotron3_ultra_stride2 \
        --fa4-source /path/to/extracted-fa4-runtime-source \
        --fa4-source-manifest /path/to/fa4-source-manifest.json \
        --fa4-source-manifest-sha256 abcdef... \
        --fa4-commit 4c40766b... \
        --prompts-jsonl prompts.jsonl \
        --vanilla-config /path/to/config.json \
        --target-sparsities 0.5 0.6 0.7 \
        --output compact-captures.jsonl
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import stat
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import cast

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    read_stable_file_snapshot,
    verify_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.source_manifest import (
    SourceManifestError,
    VerifiedSourceManifest,
    verify_source_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_capture import (
    MAX_QUERY_CHUNK_TOKENS,
    CaptureContractError,
    build_capture_invocation,
    canonical_json_sha256,
    merge_rank_captures,
    merge_rank_topology_discovery_captures,
    parse_prompt_specs_jsonl,
    parse_vanilla_prefill_fit,
    validate_begin_acks,
    validate_capture_statuses,
)

CAPTURE_ENV = "MASK_REUSE_FA4_CALIBRATION_CAPTURE"
PLAN_ENV = "MASK_REUSE_FA4_PLAN"
CHECKPOINT_ENV = "MASK_REUSE_FA4_CHECKPOINT_MANIFEST_SHA256"
DENSE_SHADOW_ENV = "MASK_REUSE_FA4_CAPTURE_DENSE_SHADOW"
TOPOLOGY_MAX_REUSE_SPAN_ENV = "MASK_REUSE_FA4_TOPOLOGY_MAX_REUSE_SPAN"
_POLICY_ENVS = (
    "MASK_REUSE_FA4_POLICY",
    "MASK_REUSE_FA4_POLICY_SHA256",
)
_FORBIDDEN_ENGINE_KWARGS = frozenset(
    {
        "additional_config",
        "attention_backend",
        "decode_context_parallel_size",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "enforce_eager",
        "kv_cache_dtype",
        "kv_transfer_config",
        "max_num_batched_tokens",
        "max_num_seqs",
        "pipeline_parallel_size",
        "quantization",
        "speculative_config",
        "worker_cls",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect ModelOpt mask-reuse sufficient statistics through vLLM V1"
    )
    parser.add_argument("model", help="HF checkpoint path loaded by vLLM")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Stable model name stored in observations (default: the model argument)",
    )
    parser.add_argument(
        "--checkpoint-manifest-sha256",
        required=True,
        help="Separately pinned SHA256 of the checkpoint manifest",
    )
    parser.add_argument("--plan", required=True, help="Explicit mask-reuse layer-plan preset")
    parser.add_argument(
        "--max-reuse-span",
        type=int,
        default=None,
        help=(
            "Maximum number of intervening attention-layer positions considered by a "
            "*_topology_discovery plan"
        ),
    )
    parser.add_argument(
        "--fa4-source",
        required=True,
        help="FlashAttention source tree extracted from the exact witnessed git archive",
    )
    parser.add_argument(
        "--fa4-source-manifest",
        required=True,
        help="Canonical full-tree witness generated with create_fa4_source_witness.py",
    )
    parser.add_argument(
        "--fa4-source-manifest-sha256",
        required=True,
        help="Separately pinned SHA256 of --fa4-source-manifest",
    )
    parser.add_argument(
        "--fa4-commit",
        required=True,
        help="Separately pinned 40-hex FlashAttention Git commit",
    )
    parser.add_argument("--prompts-jsonl", required=True, help="Strict prompt-plan JSONL")
    parser.add_argument(
        "--vanilla-config",
        required=True,
        help="ModelOpt config containing the calibrated prefill skip-softmax (a, b) fit",
    )
    parser.add_argument(
        "--target-sparsities",
        type=float,
        nargs="+",
        required=True,
        help="Preregistered target-sparsity menu evaluated for every prompt",
    )
    parser.add_argument("--output", required=True, help="Compact normalized capture JSONL")
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Capture provenance JSON (default: <output>.manifest.json)",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--validate-dense-output",
        action="store_true",
        help="Bitwise-compare every armed capture layer with a second pinned dense FA4 call",
    )
    parser.add_argument(
        "--engine-kwargs",
        default=None,
        help="JSON object of non-contract vLLM kwargs (for example hybrid-model options)",
    )
    return parser


def _target_menu(values: list[float]) -> tuple[float, ...]:
    menu = tuple(sorted(set(values)))
    if not menu or any(not 0.0 < value < 1.0 for value in menu):
        raise CaptureContractError("--target-sparsities values must be finite and in (0, 1)")
    return menu


def _engine_kwargs(args: argparse.Namespace) -> dict[str, object]:
    extra: dict[str, object] = {}
    if args.engine_kwargs is not None:
        raw = json.loads(args.engine_kwargs)
        if not isinstance(raw, dict):
            raise CaptureContractError("--engine-kwargs must be a JSON object")
        conflicts = _FORBIDDEN_ENGINE_KWARGS & raw.keys()
        if conflicts:
            raise CaptureContractError(
                f"--engine-kwargs cannot override capture/precision controls: {sorted(conflicts)}"
            )
        extra.update(raw)
    extra.update(
        {
            "model": args.model,
            "worker_cls": (
                "modelopt.torch.sparsity.attention_sparsity.plugins."
                "vllm_mask_reuse_capture.MaskReuseCaptureWorker"
            ),
            "attention_backend": "CUSTOM",
            "dtype": "bfloat16",
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": MAX_QUERY_CHUNK_TOKENS,
            "max_num_seqs": 1,
            "disable_cascade_attn": True,
            "pipeline_parallel_size": 1,
            "decode_context_parallel_size": 1,
        }
    )
    if args.max_model_len is not None:
        if args.max_model_len <= 0:
            raise CaptureContractError("--max-model-len must be positive")
        extra["max_model_len"] = args.max_model_len
    if args.tensor_parallel_size <= 0:
        raise CaptureContractError("--tensor-parallel-size must be positive")
    extra["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        if not 0.0 < args.gpu_memory_utilization <= 1.0:
            raise CaptureContractError("--gpu-memory-utilization must be in (0, 1]")
        extra["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.trust_remote_code:
        extra["trust_remote_code"] = True
    return extra


def _canonical_capture_line(capture: dict[str, object]) -> bytes:
    return (
        json.dumps(capture, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _verify_fa4_source(
    fa4_source: str,
    fa4_source_manifest: str,
    fa4_source_manifest_sha256: str,
    fa4_commit: str,
) -> VerifiedSourceManifest:
    try:
        verified = verify_source_manifest(
            fa4_source,
            fa4_source_manifest,
            expected_manifest_sha256=fa4_source_manifest_sha256,
            expected_commit=fa4_commit,
            expected_source_kind="flash-attention-4",
        )
    except SourceManifestError as error:
        raise CaptureContractError(f"--fa4-source verification failed: {error}") from error
    required = (
        verified.source_root / "flash_attn/cute/interface.py",
        verified.source_root / "flash_attn/cute/block_sparsity.py",
    )
    try:
        required_are_regular = all(
            stat.S_ISREG(path.stat(follow_symlinks=False).st_mode) for path in required
        )
    except OSError:
        required_are_regular = False
    if not required_are_regular:
        raise CaptureContractError(
            "--fa4-source must contain flash_attn/cute/interface.py and block_sparsity.py"
        )
    return verified


def _configure_capture_environment(
    plan: str,
    fa4_source: str,
    fa4_source_manifest: str,
    fa4_source_manifest_sha256: str,
    fa4_commit: str,
    checkpoint_manifest_sha256: str,
    *,
    validate_dense_output: bool,
    max_reuse_span: int | None = None,
) -> VerifiedSourceManifest:
    verified = _verify_fa4_source(
        fa4_source,
        fa4_source_manifest,
        fa4_source_manifest_sha256,
        fa4_commit,
    )
    plugins = [
        entry
        for entry in importlib.metadata.entry_points(group="vllm.general_plugins")
        if entry.name == "mask_reuse_fa4"
    ]
    if len(plugins) != 1 or plugins[0].value != "mask_reuse_vllm.plugin:register":
        raise CaptureContractError(
            "the mask_reuse_fa4 vLLM plugin entry point is missing or ambiguous"
        )
    os.environ[CAPTURE_ENV] = "1"
    os.environ[PLAN_ENV] = plan
    os.environ[CHECKPOINT_ENV] = checkpoint_manifest_sha256
    os.environ[DENSE_SHADOW_ENV] = "1" if validate_dense_output else "0"
    topology_discovery = plan.endswith("_topology_discovery")
    if topology_discovery:
        if max_reuse_span is None or max_reuse_span <= 0:
            raise CaptureContractError("a *_topology_discovery plan requires --max-reuse-span > 0")
        os.environ[TOPOLOGY_MAX_REUSE_SPAN_ENV] = str(max_reuse_span)
    elif max_reuse_span is not None:
        raise CaptureContractError(
            "--max-reuse-span is valid only with a *_topology_discovery plan"
        )
    else:
        os.environ.pop(TOPOLOGY_MAX_REUSE_SPAN_ENV, None)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.environ["MASK_REUSE_FA4_SOURCE"] = str(verified.source_root)
    os.environ["MASK_REUSE_FA4_FORCE_DENSE"] = "0"
    os.environ["VLLM_PLUGINS"] = "mask_reuse_fa4"
    # Collection is deliberately policy-free.  Remove inherited serving
    # settings so a stale deployment policy cannot become threshold authority.
    for name in _POLICY_ENVS:
        os.environ.pop(name, None)
    return verified


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        # Windows lacks portable directory fsync; no-clobber linking remains
        # atomic, while crash durability of the directory entry is best effort.
        return
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_clobber(temporary: Path, destination: Path) -> tuple[int, int]:
    """Atomically link a complete temp file only when destination is absent."""

    observed = temporary.stat(follow_symlinks=False)
    if observed.st_ino == 0:
        raise RuntimeError("capture temporary file has no stable identity")
    identity = observed.st_dev, observed.st_ino
    os.link(temporary, destination, follow_symlinks=False)
    try:
        published = destination.stat(follow_symlinks=False)
        if published.st_ino == 0 or (published.st_dev, published.st_ino) != identity:
            raise RuntimeError("capture destination changed during publication")
        temporary.unlink()
        _fsync_directory(destination.parent)
    except BaseException:
        _unlink_if_identity(destination, identity)
        raise
    return identity


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> None:
    """Rollback only a file that is still the inode published by this process."""

    if identity[1] == 0:
        return
    try:
        stat = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.st_ino != 0 and (stat.st_dev, stat.st_ino) == identity:
        path.unlink()
        _fsync_directory(path.parent)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    prompt_snapshot = read_stable_file_snapshot(args.prompts_jsonl, label="prompt plan")
    prompt_plan_sha256 = prompt_snapshot.sha256
    prompts = parse_prompt_specs_jsonl(prompt_snapshot.payload)
    vanilla_snapshot = read_stable_file_snapshot(args.vanilla_config, label="vanilla config")
    vanilla_config_sha256 = vanilla_snapshot.sha256
    threshold_scale_factor = parse_vanilla_prefill_fit(vanilla_snapshot.payload)
    targets = _target_menu(args.target_sparsities)
    checkpoint = verify_checkpoint_manifest(args.model, expected_model=args.model_id)
    if checkpoint.sha256 != args.checkpoint_manifest_sha256:
        raise CaptureContractError(
            "checkpoint manifest does not match --checkpoint-manifest-sha256"
        )
    model_id = checkpoint.model
    output_path = Path(args.output)
    manifest_path = (
        Path(args.output_manifest)
        if args.output_manifest is not None
        else Path(str(output_path) + ".manifest.json")
    )
    if output_path.resolve() == manifest_path.resolve():
        raise CaptureContractError("observation and manifest output paths must differ")
    if output_path.exists() or manifest_path.exists():
        raise CaptureContractError("capture outputs already exist; refusing to overwrite them")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fa4_source = _configure_capture_environment(
        args.plan,
        args.fa4_source,
        args.fa4_source_manifest,
        args.fa4_source_manifest_sha256,
        args.fa4_commit,
        checkpoint.sha256,
        validate_dense_output=args.validate_dense_output,
        max_reuse_span=args.max_reuse_span,
    )
    # Import after setting the gate: worker subprocesses inherit the exact
    # capture environment and never enter policy-backed serving mode.
    from vllm import LLM, SamplingParams

    engine_kwargs = _engine_kwargs(args)
    llm = LLM(**engine_kwargs)
    loaded_checkpoint = verify_checkpoint_manifest(args.model, expected_model=model_id)
    if loaded_checkpoint != checkpoint:
        raise CaptureContractError(
            "checkpoint identity changed while vLLM loaded the model; no capture was started"
        )
    loaded_fa4_source = _verify_fa4_source(
        args.fa4_source,
        args.fa4_source_manifest,
        args.fa4_source_manifest_sha256,
        args.fa4_commit,
    )
    if loaded_fa4_source != fa4_source:
        raise CaptureContractError(
            "FA4 source identity changed while vLLM loaded the model; no capture was started"
        )
    statuses = llm.collective_rpc("mask_reuse_capture_status")
    validate_capture_statuses(statuses)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    capture_manifests: list[dict[str, object]] = []
    seen_sources: dict[str, tuple[str, str, tuple[int, int | None]]] = {}
    capture_digest = sha256()
    capture_count = 0
    candidate_cell_count = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for prompt in prompts:
                token_ids = tokenizer.encode(prompt.prompt, add_special_tokens=True)
                if not isinstance(token_ids, list) or any(
                    type(token) is not int for token in token_ids
                ):
                    raise CaptureContractError(
                        "tokenizer.encode must return a list of integer token IDs"
                    )
                if args.max_model_len is not None and len(token_ids) + 1 > args.max_model_len:
                    raise CaptureContractError(
                        f"prompt {prompt.prompt_id!r} plus one output token exceeds --max-model-len"
                    )
                for target in targets:
                    invocation = build_capture_invocation(
                        model=model_id,
                        checkpoint_manifest_sha256=checkpoint.sha256,
                        prompt=prompt,
                        prompt_token_ids=token_ids,
                        target_sparsity=target,
                        threshold_scale_factor=threshold_scale_factor,
                    )
                    fingerprint = str(invocation["source_capture_sha256"])
                    identity = (prompt.split, prompt.prompt_id, prompt.bucket)
                    previous = seen_sources.setdefault(fingerprint, identity)
                    if previous != identity:
                        raise CaptureContractError(
                            "the same tokenized source is assigned to multiple prompt captures: "
                            f"{previous} and {identity}"
                        )
                    acknowledgements = llm.collective_rpc(
                        "mask_reuse_capture_begin", args=(invocation,)
                    )
                    validate_begin_acks(acknowledgements, invocation)
                    # Passing token IDs makes sample_length and source_capture_sha256
                    # identical to the request that reaches the vLLM scheduler.
                    llm.generate(token_ids, sampling, use_tqdm=False)
                    rank_captures = llm.collective_rpc("mask_reuse_capture_drain")
                    merge = (
                        merge_rank_topology_discovery_captures
                        if args.plan.endswith("_topology_discovery")
                        else merge_rank_captures
                    )
                    merged = merge(rank_captures, invocation)
                    line = _canonical_capture_line(merged.capture)
                    temporary.write(line)
                    capture_digest.update(line)
                    capture_count += 1
                    candidate_cell_count += cast("int", merged.manifest["candidate_cell_count"])
                    capture_manifests.append(merged.manifest)
            temporary.flush()
            os.fsync(temporary.fileno())
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    try:
        prompt_final = read_stable_file_snapshot(args.prompts_jsonl, label="prompt plan")
        vanilla_final = read_stable_file_snapshot(args.vanilla_config, label="vanilla config")
        final_fa4_source = _verify_fa4_source(
            args.fa4_source,
            args.fa4_source_manifest,
            args.fa4_source_manifest_sha256,
            args.fa4_commit,
        )
    except BaseException:
        assert temporary_path is not None
        temporary_path.unlink(missing_ok=True)
        raise
    if (
        prompt_final.sha256 != prompt_plan_sha256
        or vanilla_final.sha256 != vanilla_config_sha256
        or final_fa4_source != fa4_source
    ):
        assert temporary_path is not None
        temporary_path.unlink(missing_ok=True)
        raise CaptureContractError(
            "prompt plan, vanilla calibration, or FA4 source changed during capture; "
            "evidence was discarded"
        )

    capture_sha256 = capture_digest.hexdigest()
    topology_discovery = args.plan.endswith("_topology_discovery")
    manifest = {
        "capture_manifest_schema_version": 5 if topology_discovery else 4,
        "capture_protocol": (
            "modelopt_vllm_mask_reuse_topology_discovery_v1"
            if topology_discovery
            else "modelopt_vllm_mask_reuse_target_sparsity_v4"
        ),
        "model": model_id,
        "checkpoint_manifest_sha256": checkpoint.sha256,
        "checkpoint_manifest_path": str(checkpoint.manifest_path),
        "checkpoint_file_count": checkpoint.file_count,
        "checkpoint_total_size_bytes": checkpoint.total_size_bytes,
        "plan": args.plan,
        "fa4_source": str(fa4_source.source_root),
        "fa4_source_commit": fa4_source.git_commit,
        "fa4_source_git_tree": fa4_source.git_tree,
        "fa4_source_git_archive_sha256": fa4_source.git_archive_sha256,
        "fa4_source_manifest_path": str(fa4_source.manifest_path),
        "fa4_source_manifest_sha256": fa4_source.manifest_sha256,
        "fa4_source_directory_count": fa4_source.directory_count,
        "fa4_source_file_count": fa4_source.file_count,
        "fa4_source_total_size_bytes": fa4_source.total_size_bytes,
        "engine_kwargs": engine_kwargs,
        "dense_shadow_validation_requested": args.validate_dense_output,
        "target_sparsity_hex": [target.hex() for target in targets],
        "vanilla_threshold_scale_factor": threshold_scale_factor,
        "vanilla_fit_sha256": canonical_json_sha256(threshold_scale_factor),
        "vanilla_config_file_sha256": vanilla_config_sha256,
        "prompt_plan_file_sha256": prompt_plan_sha256,
        "capture_count": capture_count,
        "candidate_cell_count": candidate_cell_count,
        "captures": capture_manifests,
    }
    if topology_discovery:
        manifest["capture_mode"] = "topology_discovery"
        manifest["max_reuse_span"] = args.max_reuse_span
        manifest["topology_discovery_capture_file_sha256"] = capture_sha256
    else:
        manifest["compact_capture_file_sha256"] = capture_sha256
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()
    manifest_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            manifest_temporary = Path(temporary.name)
            temporary.write(manifest_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
        assert temporary_path is not None and manifest_temporary is not None
        capture_identity = _publish_no_clobber(temporary_path, output_path)
        temporary_path = None
        try:
            _publish_no_clobber(manifest_temporary, manifest_path)
            manifest_temporary = None
        except BaseException:
            _unlink_if_identity(output_path, capture_identity)
            raise
    except FileExistsError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)
        raise CaptureContractError("capture destination appeared during publication") from error
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)
        raise
    print(
        f"[ModelOpt] Wrote {capture_count} compact captures "
        f"({candidate_cell_count} candidate cells) to {output_path.resolve()}"
    )
    print(f"[ModelOpt] compact_capture_file_sha256={capture_sha256}")
    print(f"[ModelOpt] Wrote capture manifest to {manifest_path.resolve()}")
    return output_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

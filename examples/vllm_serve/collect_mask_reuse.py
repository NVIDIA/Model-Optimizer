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
import importlib.util
import json
import os
import sqlite3
import stat
import sys
import tempfile
import time
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    read_stable_file_snapshot,
    stable_file_sha256,
    verify_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.source_manifest import (
    SourceManifestError,
    VerifiedSourceManifest,
    verify_source_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.staged_capture import (
    CaptureStageFiles,
    load_stage_receipt,
    verify_controller_bridge,
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
    source_capture_sha256,
    validate_begin_acks,
    validate_capture_statuses,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_stream import (
    WHOLE_PREFILL_RANGES,
    build_analytical_dense_capture,
    build_capture_stream,
    capture_stream_work,
    read_capture_stream_spools,
    stream_chunk_identity,
    validate_stream_acknowledgements,
)

CAPTURE_ENV = "MASK_REUSE_FA4_CALIBRATION_CAPTURE"
PLAN_ENV = "MASK_REUSE_FA4_PLAN"
CHECKPOINT_ENV = "MASK_REUSE_FA4_CHECKPOINT_MANIFEST_SHA256"
DENSE_SHADOW_ENV = "MASK_REUSE_FA4_CAPTURE_DENSE_SHADOW"
TOPOLOGY_MAX_REUSE_SPAN_ENV = "MASK_REUSE_FA4_TOPOLOGY_MAX_REUSE_SPAN"
PROVIDER_ENV = "MASK_REUSE_CALIBRATION_PROVIDER"
_POLICY_ENVS = (
    "MASK_REUSE_FA4_POLICY",
    "MASK_REUSE_FA4_POLICY_SHA256",
)
_ALLOWED_ENGINE_KWARGS = frozenset(
    {
        "gpu_memory_utilization",
        "swap_space",
        "cpu_offload_gb",
        "disable_log_stats",
        "distributed_executor_backend",
        "enable_expert_parallel",
        "disable_custom_all_reduce",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect ModelOpt mask-reuse sufficient statistics through vLLM V1"
    )
    parser.add_argument("model", help="HF checkpoint path loaded by vLLM")
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        help="External content manifest for a read-only checkpoint (all files are still hashed)",
    )
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
        "--capture-provider",
        choices=("fa4", "reference"),
        default="reference",
        help="Reference is accuracy-only and does not establish FA4 parity or performance",
    )
    parser.add_argument(
        "--capture-scope",
        choices=("final-chunk", "whole-prefill"),
        default="whole-prefill",
        help="Whole-prefill streams every actual chunk through the reference provider",
    )
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
        help="FlashAttention source tree extracted from the exact witnessed git archive",
    )
    parser.add_argument(
        "--fa4-source-manifest",
        help="Canonical full-tree witness generated with create_fa4_source_witness.py",
    )
    parser.add_argument(
        "--fa4-source-manifest-sha256",
        help="Separately pinned SHA256 of --fa4-source-manifest",
    )
    parser.add_argument(
        "--fa4-commit",
        help="Separately pinned 40-hex FlashAttention Git commit",
    )
    parser.add_argument("--provider-source", help="Witnessed reference backend source tree")
    parser.add_argument("--provider-source-manifest", help="Reference full-tree source witness")
    parser.add_argument("--provider-source-manifest-sha256", help="Pinned reference witness hash")
    parser.add_argument("--provider-commit", help="Pinned reference source Git commit")
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
        "--phase",
        choices=("monolithic", "prepare", "collect", "finalize"),
        default="monolithic",
        help="Opt-in whole-prefill host stages; no runtime policy is exported",
    )
    parser.add_argument("--stage-receipt", type=Path, help="Trusted prior-stage receipt")
    parser.add_argument(
        "--stage-receipt-sha256", help="Separately pinned prior-stage receipt SHA256"
    )
    parser.add_argument("--source-bridge", type=Path, help="Reviewed old/new controller bridge")
    parser.add_argument("--source-bridge-sha256", help="Separately pinned source-bridge SHA256")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Reuse a verified journal from a previous output directory into a NEW directory",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        help="Stop after a completed request and preserve a resumable journal; excludes final verification",
    )
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
        conflicts = raw.keys() - _ALLOWED_ENGINE_KWARGS
        if conflicts:
            raise CaptureContractError(
                f"--engine-kwargs unsupported keys (model and capture identity are fixed): {sorted(conflicts)}"
            )
        extra.update(raw)
    extra.update(
        {
            "model": args.model,
            "worker_cls": (
                "modelopt.torch.sparsity.attention_sparsity.plugins."
                "vllm_mask_reuse_capture.MaskReuseCaptureWorker"
            ),
            "attention_backend": (
                "FLASH_ATTN"
                if getattr(args, "capture_provider", "fa4") == "reference"
                else "CUSTOM"
            ),
            "dtype": "bfloat16",
            # "auto" may inherit FP8 KV-cache metadata from an NVFP4 checkpoint.
            "kv_cache_dtype": "bfloat16",
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
    provider: str = "fa4",
) -> VerifiedSourceManifest:
    verified = _verify_provider_source(
        fa4_source,
        fa4_source_manifest,
        fa4_source_manifest_sha256,
        fa4_commit,
        provider=provider,
    )
    if provider == "fa4":
        plugins = [
            entry
            for entry in importlib.metadata.entry_points(group="vllm.general_plugins")
            if entry.name == "mask_reuse_fa4"
        ]
        if len(plugins) != 1 or plugins[0].value != "mask_reuse_vllm.plugin:register":
            raise CaptureContractError(
                "the mask_reuse_fa4 vLLM plugin entry point is missing or ambiguous"
            )
    elif validate_dense_output:
        raise CaptureContractError(
            "--validate-dense-output is a FA4-only replay check; qualify reference accuracy separately"
        )
    os.environ[CAPTURE_ENV] = "1"
    os.environ[DENSE_SHADOW_ENV] = "1" if validate_dense_output else "0"
    os.environ[PLAN_ENV] = plan
    os.environ[CHECKPOINT_ENV] = checkpoint_manifest_sha256
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
    os.environ[PROVIDER_ENV] = provider
    if provider == "fa4":
        os.environ["MASK_REUSE_FA4_SOURCE"] = str(verified.source_root)
        os.environ.pop("MASK_REUSE_CALIBRATION_SOURCE_ROOT", None)
        os.environ["VLLM_PLUGINS"] = "mask_reuse_fa4"
    else:
        os.environ.pop("MASK_REUSE_FA4_SOURCE", None)
        os.environ["MASK_REUSE_CALIBRATION_SOURCE_ROOT"] = str(verified.source_root)
        os.environ["VLLM_PLUGINS"] = ""
        import_root = str(verified.source_root / "src/attention_backend/src")
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
        previous = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = (
            os.pathsep.join([import_root, previous]) if previous else import_root
        )
    os.environ["MASK_REUSE_FA4_FORCE_DENSE"] = "0"
    # Collection is deliberately policy-free.  Remove inherited serving
    # settings so a stale deployment policy cannot become threshold authority.
    for name in _POLICY_ENVS:
        os.environ.pop(name, None)
    return verified


def _verify_provider_source(
    source: str, manifest: str, manifest_sha256: str, commit: str, *, provider: str
) -> VerifiedSourceManifest:
    if provider == "fa4":
        return _verify_fa4_source(source, manifest, manifest_sha256, commit)
    if provider != "reference":
        raise CaptureContractError("capture provider must be fa4 or reference")
    try:
        verified = verify_source_manifest(
            source,
            manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_commit=commit,
            expected_source_kind="mask-reuse-reference",
        )
    except SourceManifestError as error:
        raise CaptureContractError(f"reference source verification failed: {error}") from error
    for relative in (
        "src/attention_backend/src/mask_reuse_vllm/reference/capture.py",
        "src/attention_backend/src/mask_reuse_vllm/reference/vendor/triton_fa.py",
    ):
        path = verified.source_root / relative
        try:
            regular = stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
        except OSError:
            regular = False
        if not regular:
            raise CaptureContractError(f"reference source must contain {relative}")
    return verified


def _source_arguments(args: argparse.Namespace) -> tuple[str, str, str, str]:
    provider = getattr(args, "capture_provider", "fa4")
    prefix, other = ("provider", "fa4") if provider == "reference" else ("fa4", "provider")
    fields = ("source", "source_manifest", "source_manifest_sha256", "commit")
    selected = tuple(getattr(args, f"{prefix}_{field}", None) for field in fields)
    if not all(isinstance(value, str) and value for value in selected):
        raise CaptureContractError(
            f"--capture-provider {provider} requires all --{prefix}-source/commit arguments"
        )
    if any(getattr(args, f"{other}_{field}", None) is not None for field in fields):
        raise CaptureContractError("FA4 and reference source arguments cannot be mixed")
    return cast("tuple[str, str, str, str]", selected)


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


class _CapturePauseError(Exception):
    pass


def _controller_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), *sorted((root / "modelopt").rglob("*.py"))]
    versions = {}
    for name in ("vllm", "torch", "triton", "transformers", "nvidia-modelopt"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "files": {
            str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "versions": versions,
    }


def _strict_progress_json(payload: bytes) -> dict[str, object]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CaptureContractError("duplicate progress JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise CaptureContractError("nonfinite progress JSON value")

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise CaptureContractError("progress JSON must contain an object")
    return value


class _CaptureJournal:
    """Durable, source-bound completed-request prefix; never a candidate input."""

    def __init__(self, directory, request, invocations, resume_from=None):
        self.directory = directory.resolve()
        self.request = request
        self.request_sha = sha256(_canonical_capture_line(request)).hexdigest()
        self.invocations = invocations
        self.count = 0
        self.previous = self.request_sha
        self.path = self.directory / "capture.progress.jsonl"
        self.resume_path = None
        self.resume_count = 0
        self.resume_digest = None
        self.resume_spool_digests = {}
        if resume_from is not None:
            prior = resume_from.resolve()
            if self.directory == prior or self.directory.is_relative_to(prior):
                raise CaptureContractError("resume output must be outside the original directory")
            self._load_resume(prior)
        for name in (
            "capture.request.json",
            "capture.progress.jsonl",
            "capture.partial.json",
            "capture.failure.json",
            "capture.complete.json",
            "capture.runtime.json",
        ):
            if (self.directory / name).exists():
                raise CaptureContractError("capture progress output already exists")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write("capture.request.json", request)
        with self.path.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        self.replayed = iter(self._records(self.resume_path)) if self.resume_path else iter(())

    def _load_resume(self, prior, *, on_record=None):
        if _strict_progress_json((prior / "capture.request.json").read_bytes()) != self.request:
            raise CaptureContractError("resume identity differs: inputs/source/checkpoint/runtime")
        self.resume_path = prior / "capture.progress.jsonl"
        # Only an uncommitted trailing write may be ignored. A receipt-bound
        # journal must still match every byte and its complete hash chain.
        self._allow_torn_resume = not any(
            (prior / name).exists()
            for name in (
                "capture.partial.json",
                "capture.complete.json",
                "capture.failure.json",
                "capture.collected.json",
            )
        )
        if self._allow_torn_resume:
            self._resume_file_sha = stable_file_sha256(self.resume_path, label="resume journal")
        self.resume_count, self.resume_digest = self._validate_prefix(
            self.resume_path, on_record=on_record
        )
        for name in ("capture.partial.json", "capture.complete.json"):
            receipt = prior / name
            if receipt.exists():
                evidence = _strict_progress_json(receipt.read_bytes())
                if (
                    evidence.get("request_sha256") != self.request_sha
                    or evidence.get("journal_sha256") != self.resume_digest
                    or evidence.get("completed_capture_count") != self.resume_count
                    or evidence.get("identity_verification") != "passed"
                ):
                    raise CaptureContractError("capture prefix differs from its completion receipt")
        failed = prior / "capture.failure.json"
        if failed.exists():
            failure = _strict_progress_json(failed.read_bytes())
            if (
                failure.get("resumable_prefix") is not True
                or failure.get("identity_verification") != "passed"
                or failure.get("request_sha256") != self.request_sha
                or failure.get("journal_sha256") != self.resume_digest
                or failure.get("completed_capture_count") != self.resume_count
            ):
                raise CaptureContractError("prior failure invalidated this capture prefix")

    @classmethod
    def existing(cls, directory, request, invocations, *, prepared=None, on_record=None):
        """Read a prior journal or adopt a separately verified prepared prefix."""
        self = cls.__new__(cls)
        self.directory = Path(directory).resolve()
        self.request, self.invocations = request, invocations
        self.request_sha = sha256(_canonical_capture_line(request)).hexdigest()
        self.path = self.directory / "capture.progress.jsonl"
        self.resume_path, self.resume_digest = None, None
        self.resume_spool_digests = {}
        self.count = self.resume_count = 0
        self.previous = self.request_sha
        if prepared is None:
            self._load_resume(self.directory, on_record=on_record)
            self.replayed = iter(self._records(self.path))
        else:
            self.count = self.resume_count = prepared["completed_capture_count"]
            self.previous = prepared["chain_tail_sha256"]
            self.replayed = iter(())
            self.expected_runtime = prepared.get("expected_runtime")
        return self

    def bind_runtime(self, geometry):
        if getattr(self, "expected_runtime", None) not in (None, geometry):
            raise CaptureContractError("prepared actual runtime geometry differs")
        evidence = {"request_sha256": self.request_sha, "geometry": geometry}
        if self.resume_path is not None:
            prior = self.resume_path.parent / "capture.runtime.json"
            if _strict_progress_json(prior.read_bytes()) != evidence:
                raise CaptureContractError("resume actual runtime geometry differs")
        self._write("capture.runtime.json", evidence)

    def _records(self, path):
        with path.open("rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    if path == self.resume_path and getattr(self, "_allow_torn_resume", False):
                        return
                    raise CaptureContractError("capture journal has a partial final line")
                yield line, _strict_progress_json(line)

    def _validate_prefix(self, path, *, on_record=None):
        self._validated_prefix = None
        digest = sha256()
        previous = self.request_sha
        count = 0
        for line, record in self._records(path):
            if set(record) != {"capture", "manifest", "previous_sha256", "record_sha256"}:
                raise CaptureContractError("capture journal record schema differs")
            if (
                count >= len(self.invocations)
                or record["capture"]["invocation"] != self.invocations[count]
            ):
                raise CaptureContractError("capture journal is not the exact planned prefix")
            if record["manifest"]["invocation"] != self.invocations[count]:
                raise CaptureContractError("capture journal manifest invocation differs")
            unsigned = {k: v for k, v in record.items() if k != "record_sha256"}
            actual = sha256(_canonical_capture_line(unsigned)).hexdigest()
            if record["previous_sha256"] != previous or record["record_sha256"] != actual:
                raise CaptureContractError("capture journal hash chain differs")
            previous = actual
            digest.update(line)
            count += 1
            if on_record is not None:
                on_record(record)
        digest = digest.hexdigest()
        self._validated_prefix = (
            path,
            self.request_sha,
            sha256(_canonical_capture_line(self.invocations)).hexdigest(),
            count,
            digest,
            previous,
        )
        return count, digest

    def _recheck_validated_prefix(self):
        """Recheck exact bytes against a full scan retained only in this process."""
        validated = getattr(self, "_validated_prefix", None)
        current = (
            self.path,
            sha256(_canonical_capture_line(self.request)).hexdigest(),
            sha256(_canonical_capture_line(self.invocations)).hexdigest(),
            self.resume_count,
            self.resume_digest,
        )
        if validated is None or validated[:-1] != current or self.request_sha != current[1]:
            raise CaptureContractError("journal has no matching in-process prefix validation")
        expected_bytes = (
            self._resume_file_sha if getattr(self, "_allow_torn_resume", False) else validated[4]
        )
        if stable_file_sha256(self.path, label="validated capture journal") != expected_bytes:
            raise CaptureContractError("validated capture journal changed before completion")
        return validated[3:]

    def replay(self):
        _, record = next(self.replayed)
        return record["capture"], record["manifest"]

    def append(self, capture, manifest):
        record = {"capture": capture, "manifest": manifest, "previous_sha256": self.previous}
        self.previous = sha256(_canonical_capture_line(record)).hexdigest()
        record["record_sha256"] = self.previous
        with self.path.open("ab") as handle:
            handle.write(_canonical_capture_line(record))
            handle.flush()
            os.fsync(handle.fileno())
        self.count += 1

    def _write(self, name, value):
        with (self.directory / name).open("xb") as handle:
            handle.write(_canonical_capture_line(value))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.directory)

    def finish(
        self, status, *, identity_verified, stage, error=None, write=None, _validated_bytes=False
    ):
        if _validated_bytes:
            count, digest, tail = self._recheck_validated_prefix()
            if (self.count, self.previous) != (count, tail):
                raise CaptureContractError("journal count/tail differs from its validated prefix")
        else:
            count, digest = self._validate_prefix(self.path)
        (self._write if write is None else write)(
            f"capture.{status}.json",
            {
                "schema_version": 1,
                "status": status,
                "request_sha256": self.request_sha,
                "journal_sha256": digest,
                "completed_capture_count": count,
                "planned_capture_count": len(self.invocations),
                "identity_verification": "passed" if identity_verified else "failed",
                "resumable_prefix": identity_verified
                and (
                    stage == "publication"
                    or (stage == "collection" and not isinstance(error, CaptureContractError))
                ),
                "failure_stage": stage,
                "error_type": type(error).__name__ if error else None,
                "eligible_as_capture_manifest": False,
                **(
                    {
                        "recovered_rank_spools": [
                            {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
                            for path, digest in sorted(self.resume_spool_digests.items())
                        ],
                        "uncommitted_spool_suffix_preserved": True,
                    }
                    if self.resume_spool_digests
                    else {}
                ),
            },
        )


def run(args: argparse.Namespace) -> tuple[Path, Path] | None:
    """Collect complete evidence, or preserve a separately typed resumable prefix."""
    if getattr(args, "phase", "monolithic") != "monolithic":
        return _run_staged(args)
    if any(
        getattr(args, field, None) is not None
        for field in (
            "stage_receipt",
            "stage_receipt_sha256",
            "source_bridge",
            "source_bridge_sha256",
        )
    ):
        raise CaptureContractError("stage arguments require an explicit staged phase")
    state = {}
    try:
        result = _run(args, state)
    except BaseException as error:
        journal = state.get("journal")
        if journal is None:
            raise
        verified = False
        if state.get("stage") != "identity_verification":
            try:
                state["verify"]()
                verified = True
            except BaseException:
                pass
        status = "partial" if isinstance(error, _CapturePauseError) and verified else "failure"
        journal.finish(status, identity_verified=verified, stage=state.get("stage"), error=error)
        if status == "partial":
            print(
                f"[ModelOpt] Partial capture: {journal.count}/{len(journal.invocations)} requests; "
                "no complete manifest published"
            )
            return None
        raise
    journal = state.get("journal")
    if journal is not None:
        journal.finish("complete", identity_verified=True, stage="publication")
    return result


def _stream_records(
    llm,
    sampling,
    prompts,
    requests,
    journal,
    output_directory,
    *,
    world_size,
    topology,
    replay_prefix=True,
    recover_only=False,
):
    """Replay validated progress and merge only one complete TP chunk at a time."""
    prompts_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    position = 0
    merge = merge_rank_topology_discovery_captures if topology else merge_rank_captures

    def merged_record(request, index, payloads):
        chunk = request["chunks"][index]
        if chunk["invocation"] is None:
            identity = stream_chunk_identity(request, chunk)
            return (
                {"invocation": identity, "dense_exclusion": True},
                {"invocation": identity, "candidate_cell_count": 0},
            )
        merged = merge(payloads, chunk["invocation"], streamed_chunk=True)
        return merged.capture, merged.manifest

    for request in requests:
        chunks = request["chunks"]
        skipped = min(max(0, journal.resume_count - position), len(chunks)) if journal else 0
        for _ in range(skipped):
            if replay_prefix:
                yield journal.replay()
            position += 1
        if skipped == len(chunks):
            continue
        request_sha = canonical_json_sha256(request)
        if journal is not None and journal.resume_path is not None:
            prior_spool = journal.resume_path.parent / "capture-spools" / request_sha
            first_rank = prior_spool / "rank-0.jsonl"
            recover_spool = first_rank.exists() and first_rank.stat().st_size > 0
            if recover_spool:
                with first_rank.open("rb") as handle:
                    first_line = handle.readline()
                start = (
                    _strict_progress_json(first_line).get("chunk_index")
                    if first_line.endswith(b"\n")
                    else skipped
                )
                if type(start) is not int or not 0 <= start <= skipped:
                    raise CaptureContractError("prior stream spool omits a planned chunk prefix")
                for rank in range(world_size):
                    path = prior_spool / f"rank-{rank}.jsonl"
                    if path.exists():
                        journal.resume_spool_digests[path] = _spool_sha256(path)
                for index, payloads in read_capture_stream_spools(
                    request,
                    prior_spool,
                    world_size=world_size,
                    completed_chunk_count=start,
                    allow_partial=True,
                ):
                    if index < skipped:
                        continue
                    yield merged_record(request, index, payloads)
                    position += 1
                    skipped = index + 1
                if skipped == len(chunks):
                    continue
        if recover_only:
            return
        spool = output_directory / "capture-spools" / request_sha
        spool.mkdir(parents=True, exist_ok=False)
        acknowledgements = llm.collective_rpc(
            "mask_reuse_capture_stream_begin", args=(request, str(spool.resolve()), skipped)
        )
        validate_stream_acknowledgements(acknowledgements, request, expected_world_size=world_size)
        failure = None
        try:
            prompt = prompts_by_id[request["parent_invocation"]["prompt_id"]]
            outputs = llm.generate(
                {"prompt_token_ids": list(prompt.prompt_token_ids)}, sampling, use_tqdm=False
            )
            if (
                not isinstance(outputs, list)
                or len(outputs) != 1
                or getattr(outputs[0], "prompt_token_ids", None) != list(prompt.prompt_token_ids)
            ):
                raise CaptureContractError(
                    "stream output does not confirm the exact input token IDs"
                )
            receipts = llm.collective_rpc("mask_reuse_capture_stream_drain")
            if (
                not isinstance(receipts, list)
                or len(receipts) != world_size
                or any(
                    not isinstance(value, dict) or type(value.get("rank")) is not int
                    for value in receipts
                )
            ):
                raise CaptureContractError("stream completion does not cover every TP rank")
            expected_fields = {
                "stream_schema_version",
                "rank",
                "world_size",
                "request_sha256",
                "complete",
                "captured_chunk_indices",
                "spool_sha256",
                "completed_chunk_count",
                "observed_chunk_count",
                "native_dense_chunk_indices",
            }
            for rank, receipt in enumerate(sorted(receipts, key=lambda value: value["rank"])):
                if (
                    set(receipt) != expected_fields
                    or receipt["stream_schema_version"] != 1
                    or receipt["rank"] != rank
                    or receipt["world_size"] != world_size
                    or receipt["request_sha256"] != request_sha
                    or receipt["complete"] is not True
                    or receipt["completed_chunk_count"] != len(chunks)
                    or receipt["observed_chunk_count"] != len(chunks)
                    or receipt["captured_chunk_indices"]
                    != [
                        chunk["chunk_index"]
                        for chunk in chunks[skipped:]
                        if chunk["invocation"] is not None
                    ]
                    or receipt["native_dense_chunk_indices"]
                    != [chunk["chunk_index"] for chunk in chunks if chunk["invocation"] is None]
                    or receipt["spool_sha256"] != _spool_sha256(spool / f"rank-{rank}.jsonl")
                ):
                    raise CaptureContractError(
                        "stream completion receipt differs from the full request"
                    )
        except CaptureContractError:
            # A failed completion contract must not become a reusable all-complete prefix.
            raise
        except BaseException as error:
            failure = error
        for index, payloads in read_capture_stream_spools(
            request,
            spool,
            world_size=world_size,
            completed_chunk_count=skipped,
            allow_partial=failure is not None,
        ):
            yield merged_record(request, index, payloads)
            position += 1
        if failure is not None:
            raise failure


def _spool_sha256(path):
    if path.is_symlink():
        raise CaptureContractError("stream spool must not be a symlink")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_capture_inputs(args, *, prepared_request=None):
    """Prepare source-bound inputs without loading a model."""
    provider = getattr(args, "capture_provider", "fa4")
    whole_prefill = getattr(args, "capture_scope", "final-chunk") == "whole-prefill"
    source_arguments = _source_arguments(args)
    prompt_snapshot = read_stable_file_snapshot(args.prompts_jsonl, label="prompt plan")
    prompt_plan_sha256 = prompt_snapshot.sha256
    prompts = parse_prompt_specs_jsonl(prompt_snapshot.payload)
    if any(prompt.max_kv_tokens is None for prompt in prompts):
        raise CaptureContractError("calibration context ranges require finite maximum lengths")
    vanilla_snapshot = read_stable_file_snapshot(args.vanilla_config, label="vanilla config")
    vanilla_config_sha256 = vanilla_snapshot.sha256
    threshold_scale_factor = parse_vanilla_prefill_fit(vanilla_snapshot.payload)
    fit = cast("dict[str, float]", threshold_scale_factor["prefill"])
    if "min_observed_sparsity" not in fit or "max_observed_sparsity" not in fit:
        raise CaptureContractError("vanilla calibration must include observed sparsity bounds")
    targets = _target_menu(args.target_sparsities)
    if any(
        not fit["min_observed_sparsity"] <= target <= fit["max_observed_sparsity"]
        for target in targets
    ):
        raise CaptureContractError("target sparsity is outside observed vanilla calibration bounds")
    checkpoint = verify_checkpoint_manifest(
        args.model,
        expected_model=args.model_id,
        manifest_path=getattr(args, "checkpoint_manifest", None),
    )
    if checkpoint.sha256 != args.checkpoint_manifest_sha256:
        raise CaptureContractError(
            "checkpoint manifest does not match --checkpoint-manifest-sha256"
        )
    model_id = checkpoint.model
    if any(prompt.prompt_token_ids is None for prompt in prompts):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=args.trust_remote_code
        )
        frozen = []
        for prompt in prompts:
            if prompt.prompt_token_ids is None:
                tokens = tuple(tokenizer.encode(prompt.prompt, add_special_tokens=True))
                prompt = replace(
                    prompt,
                    prompt_token_ids=tokens,
                    prompt_token_ids_sha256=source_capture_sha256(tokens),
                )
            frozen.append(prompt)
        prompts = frozen
    seen_sources = {}
    for prompt in prompts:
        tokens = prompt.prompt_token_ids
        if args.max_model_len is not None and len(tokens) + 1 > args.max_model_len:
            raise CaptureContractError(f"prompt {prompt.prompt_id!r} exceeds --max-model-len")
        fingerprint = source_capture_sha256(tokens)
        owner = (prompt.split, prompt.prompt_id, prompt.bucket)
        if seen_sources.setdefault(fingerprint, owner) != owner:
            raise CaptureContractError(
                "the same tokenized source is assigned to multiple prompt captures"
            )
        if not whole_prefill:
            for target in targets:
                build_capture_invocation(
                    model=model_id,
                    checkpoint_manifest_sha256=checkpoint.sha256,
                    prompt=prompt,
                    prompt_token_ids=tokens,
                    target_sparsity=target,
                    threshold_scale_factor=threshold_scale_factor,
                )
    streams = (
        prepared_request["stream_requests"]
        if prepared_request is not None
        else (
            [
                build_capture_stream(
                    model=model_id,
                    checkpoint_manifest_sha256=checkpoint.sha256,
                    prompt=prompt,
                    target_sparsity=target,
                    threshold_scale_factor=threshold_scale_factor,
                    coverage_ranges=sorted({(p.min_kv_tokens, p.max_kv_tokens) for p in prompts}),
                )
                for prompt in prompts
                for target in targets
            ]
            if whole_prefill
            else []
        )
    )
    if prepared_request is not None and [
        (
            stream["parent_invocation"]["prompt_id"],
            stream["parent_invocation"]["target_sparsity_hex"],
        )
        for stream in streams
    ] != [(prompt.prompt_id, target.hex()) for prompt in prompts for target in targets]:
        raise CaptureContractError("prepared stream targets or parent order differ from CLI inputs")
    if whole_prefill and not any(
        chunk["dense_reason"] is None for request in streams for chunk in request["chunks"]
    ):
        raise CaptureContractError(
            "capture plan has no sparse invocation to establish provider head/layer geometry; "
            "include a longer prompt or a lower target sparsity"
        )
    output_path = Path(args.output)
    manifest_path = (
        Path(args.output_manifest)
        if args.output_manifest is not None
        else Path(str(output_path) + ".manifest.json")
    )
    if output_path.resolve() == manifest_path.resolve():
        raise CaptureContractError("observation and manifest output paths must differ")
    if getattr(args, "phase", "monolithic") != "finalize" and (
        output_path.exists()
        or output_path.is_symlink()
        or manifest_path.exists()
        or manifest_path.is_symlink()
    ):
        raise CaptureContractError("capture outputs already exist; refusing to overwrite them")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["MASK_REUSE_CHECKPOINT_MANIFEST"] = str(checkpoint.manifest_path)
    fa4_source = _configure_capture_environment(
        args.plan,
        *source_arguments,
        checkpoint.sha256,
        validate_dense_output=args.validate_dense_output,
        max_reuse_span=args.max_reuse_span,
        provider=provider,
    )
    engine_kwargs = _engine_kwargs(args)

    inputs = SimpleNamespace(
        provider=provider,
        whole_prefill=whole_prefill,
        source_arguments=source_arguments,
        prompt_plan_sha256=prompt_plan_sha256,
        prompts=prompts,
        vanilla_config_sha256=vanilla_config_sha256,
        threshold_scale_factor=threshold_scale_factor,
        targets=targets,
        checkpoint=checkpoint,
        model_id=model_id,
        streams=streams,
        output_path=output_path,
        manifest_path=manifest_path,
        fa4_source=fa4_source,
        engine_kwargs=engine_kwargs,
    )
    return inputs


def _capture_request(args, inputs, controller):
    prompts = inputs.prompts
    model_id = inputs.model_id
    checkpoint = inputs.checkpoint
    threshold_scale_factor = inputs.threshold_scale_factor
    targets = inputs.targets
    whole_prefill = inputs.whole_prefill
    streams = inputs.streams
    provider = inputs.provider
    prompt_plan_sha256 = inputs.prompt_plan_sha256
    vanilla_config_sha256 = inputs.vanilla_config_sha256
    fa4_source = inputs.fa4_source
    engine_kwargs = inputs.engine_kwargs
    if any(prompt.prompt_token_ids is None for prompt in prompts):
        raise CaptureContractError("resumable capture requires frozen exact prompt token IDs")
    invocations = (
        [
            build_capture_invocation(
                model=model_id,
                checkpoint_manifest_sha256=checkpoint.sha256,
                prompt=prompt,
                prompt_token_ids=prompt.prompt_token_ids,
                target_sparsity=target,
                threshold_scale_factor=threshold_scale_factor,
            )
            for prompt in prompts
            for target in targets
        ]
        if not whole_prefill
        else [
            stream_chunk_identity(request, chunk)
            for request in streams
            for chunk in request["chunks"]
        ]
    )
    request = {
        "schema_version": 1,
        "model": model_id,
        "checkpoint_manifest_sha256": checkpoint.sha256,
        "prompt_plan_sha256": prompt_plan_sha256,
        "vanilla_config_sha256": vanilla_config_sha256,
        "provider": provider,
        "provider_manifest_sha256": fa4_source.manifest_sha256,
        "provider_commit": fa4_source.git_commit,
        "provider_archive_sha256": fa4_source.git_archive_sha256,
        "controller": controller,
        "engine_kwargs": engine_kwargs,
        "plan": args.plan,
        "max_reuse_span": args.max_reuse_span,
        "validate_dense_output": args.validate_dense_output,
        "kv_cache_layout": os.environ.get("VLLM_KV_CACHE_LAYOUT"),
        "invocations": invocations,
    }
    if whole_prefill:
        request["capture_scope"] = "whole-prefill"
        request["stream_requests"] = streams
        request["coverage_ranges"] = streams[0].get(
            "coverage_ranges", [list(b) for b in WHOLE_PREFILL_RANGES]
        )
    return request, invocations


def _load_capture_engine(args, inputs, journal):
    engine_kwargs = inputs.engine_kwargs
    model_id = inputs.model_id
    checkpoint = inputs.checkpoint
    fa4_source = inputs.fa4_source
    source_arguments = inputs.source_arguments
    provider = inputs.provider
    # Import after setting the gate: worker subprocesses inherit the exact
    # capture environment and never enter policy-backed serving mode.
    from vllm import LLM, SamplingParams

    llm = LLM(**engine_kwargs)
    loaded_checkpoint = verify_checkpoint_manifest(
        args.model, expected_model=model_id, manifest_path=checkpoint.manifest_path
    )
    if loaded_checkpoint != checkpoint:
        raise CaptureContractError(
            "checkpoint identity changed while vLLM loaded the model; no capture was started"
        )
    loaded_fa4_source = _verify_provider_source(
        *source_arguments,
        provider=provider,
    )
    if loaded_fa4_source != fa4_source:
        raise CaptureContractError(
            "provider source identity changed while vLLM loaded the model; no capture was started"
        )
    statuses = llm.collective_rpc("mask_reuse_capture_status")
    validate_capture_statuses(statuses)
    geometries = llm.collective_rpc("mask_reuse_capture_runtime_geometry")
    if (
        not isinstance(geometries, list)
        or len(geometries) != args.tensor_parallel_size
        or any(
            not isinstance(value, dict) or set(value) != {"kv_page_tokens"} for value in geometries
        )
        or any(type(value["kv_page_tokens"]) is not int for value in geometries)
        or any(value != geometries[0] for value in geometries)
    ):
        raise CaptureContractError("capture ranks disagree on actual KV page geometry")
    observed_page = geometries[0]["kv_page_tokens"]
    if observed_page not in ((16, 4160) if provider == "reference" else (16,)):
        raise CaptureContractError("capture provider does not support the actual KV page size")
    if journal is not None:
        journal.bind_runtime({"kv_page_tokens": observed_page})
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    return llm, tokenizer, sampling, observed_page


def _normalize_stream_file(path, streams, fit, manifests, *, destination=None):
    """Sort on disk and insert explicitly analytical all-keep rows before publication."""
    digest = sha256()
    ordered_manifests = []
    with (
        tempfile.TemporaryDirectory(dir=path.parent) as directory,
        closing(sqlite3.connect(str(Path(directory) / "sort.sqlite"))) as db,
    ):
        db.execute(
            "CREATE TABLE captures (lo INTEGER, hi INTEGER, split TEXT, prompt TEXT, "
            "target REAL, payload BLOB, metadata BLOB)"
        )

        def insert(row, metadata):
            inv = row["invocation"]
            db.execute(
                "INSERT INTO captures VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    inv["min_kv_tokens"],
                    inv["max_kv_tokens"],
                    inv["split"],
                    inv["prompt_id"],
                    float.fromhex(inv["target_sparsity_hex"]),
                    _canonical_capture_line(row),
                    _canonical_capture_line(metadata),
                ),
            )

        template = None
        with path.open("rb") as source:
            for line, metadata in zip(source, manifests, strict=True):
                template = json.loads(line)
                insert(template, metadata)
        if template is None:
            raise CaptureContractError("no measured capture supplies head/layer geometry")
        for request in streams:
            for chunk in request["chunks"]:
                if chunk["dense_reason"] != "threshold_disabled":
                    continue
                row = build_analytical_dense_capture(
                    request, chunk, fit=fit, measured_template=template
                )
                cells = template["global_num_heads"] ** 2 * (
                    sum(len(v) for v in template["consumer_candidates_by_layer"].values())
                    if "consumer_candidates_by_layer" in template
                    else len(template["consumer_layers"])
                )
                insert(
                    row,
                    {
                        "candidate_cell_count": cells,
                        "analytical_dense": row["analytical_dense"],
                    },
                )
        ordered = Path(directory) / "captures.jsonl"
        with ordered.open("wb") as target:
            for payload, metadata in db.execute(
                "SELECT payload, metadata FROM captures ORDER BY lo, hi, split, prompt, target"
            ):
                target.write(payload)
                digest.update(payload)
                ordered_manifests.append(json.loads(metadata))
            target.flush()
            os.fsync(target.fileno())
        if destination is None:
            os.replace(ordered, path)
        elif destination.exists():
            if _spool_sha256(destination) != digest.hexdigest():
                raise CaptureContractError("existing normalized solver input differs")
        else:
            os.link(ordered, destination, follow_symlinks=False)
    return digest.hexdigest(), ordered_manifests


def _capture_manifest(
    args,
    inputs,
    observed_page,
    capture_count,
    candidate_cell_count,
    capture_manifests,
    capture_sha256,
):
    model_id = inputs.model_id
    checkpoint = inputs.checkpoint
    fa4_source = inputs.fa4_source
    engine_kwargs = inputs.engine_kwargs
    targets = inputs.targets
    threshold_scale_factor = inputs.threshold_scale_factor
    vanilla_config_sha256 = inputs.vanilla_config_sha256
    prompt_plan_sha256 = inputs.prompt_plan_sha256
    provider = inputs.provider
    whole_prefill = inputs.whole_prefill
    streams = inputs.streams
    topology_discovery = args.plan.endswith("_topology_discovery")
    if args.validate_dense_output:
        for row in capture_manifests:
            if row.get("analytical_dense") is not None:
                continue
            shadow = row.get("dense_shadow_evidence", {})
            if shadow.get("enabled") is not True or not shadow.get("validated_layer_indices"):
                raise CaptureContractError(
                    "requested dense-output validation is missing from captured evidence"
                )
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
    if provider == "reference":
        manifest["capture_manifest_schema_version"] = 7 if topology_discovery else 6
        manifest["capture_protocol"] = (
            "modelopt_vllm_mask_reuse_reference_topology_discovery_v1"
            if topology_discovery
            else "modelopt_vllm_mask_reuse_reference_target_sparsity_v1"
        )
        for field in tuple(manifest):
            if field.startswith("fa4_source"):
                manifest[field.replace("fa4_source", "provider_source", 1)] = manifest.pop(field)
        manifest["provider_kind"] = "triton-reference"
        manifest["evidence_level"] = "accuracy-reference-only"
        if observed_page != 16:
            manifest["capture_manifest_schema_version"] = 9 if topology_discovery else 8
            manifest["observed_kv_page_tokens"] = observed_page
        if whole_prefill:
            manifest["capture_manifest_schema_version"] = 11 if topology_discovery else 10
            manifest["observed_kv_page_tokens"] = observed_page
            manifest["whole_prefill"] = {
                "schema_version": 1,
                "coverage_ranges": streams[0].get(
                    "coverage_ranges", [list(b) for b in WHOLE_PREFILL_RANGES]
                ),
                "requests": streams,
                "dense_exclusion_count": sum(
                    chunk["invocation"] is None
                    for request in streams
                    for chunk in request["chunks"]
                ),
                "measured_capture_count": sum(
                    c["invocation"] is not None for r in streams for c in r["chunks"]
                ),
                "analytical_dense_capture_count": sum(
                    c["dense_reason"] == "threshold_disabled" for r in streams for c in r["chunks"]
                ),
                "work_by_request": [capture_stream_work(request) for request in streams],
                "target_scope": (
                    "per-range selector constraint; whole-workload target requires measured execution accounting"
                ),
            }
    return manifest


def _verify_capture_inputs(args, inputs, controller):
    if (
        read_stable_file_snapshot(args.prompts_jsonl, label="prompt plan").sha256
        != inputs.prompt_plan_sha256
        or read_stable_file_snapshot(args.vanilla_config, label="vanilla config").sha256
        != inputs.vanilla_config_sha256
        or _verify_provider_source(*inputs.source_arguments, provider=inputs.provider)
        != inputs.fa4_source
        or verify_checkpoint_manifest(
            args.model,
            expected_model=inputs.model_id,
            manifest_path=inputs.checkpoint.manifest_path,
        )
        != inputs.checkpoint
        or (controller is not None and _controller_identity() != controller)
    ):
        raise CaptureContractError("capture input/checkpoint/provider/controller identity changed")


def _prepare_stage(args, inputs, request, invocations, files):
    prior = None
    bridge = None
    runtime = None
    prior_receipts = {}
    if args.resume_from is not None:
        old = _strict_progress_json((args.resume_from / "capture.request.json").read_bytes())
        bridge = verify_controller_bridge(
            old,
            request,
            args.source_bridge,
            args.source_bridge_sha256,
            parse=_strict_progress_json,
            serialize=_canonical_capture_line,
        )
        if files.directory == args.resume_from.resolve() or files.directory.is_relative_to(
            args.resume_from.resolve()
        ):
            raise CaptureContractError("prepared output must be outside the prior directory")
        if (args.resume_from / "capture.collected.json").exists() and not any(
            (args.resume_from / f"capture.{name}.json").exists() for name in ("partial", "complete")
        ):
            raise CaptureContractError("staged collection requires CPU finalization before resume")
    elif args.source_bridge is not None or args.source_bridge_sha256 is not None:
        raise CaptureContractError("source bridge requires a prior journal")
    journal = _CaptureJournal(files.directory, request, invocations)
    with files.normalized.open("xb") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    if args.resume_from is not None:

        def inherit_record(record):
            journal.append(record["capture"], record["manifest"])
            files.append_normalized(record["capture"])

        # Provisional writes become prepared only after the full scan and late byte proof.
        prior = _CaptureJournal.existing(
            args.resume_from, old, invocations, on_record=inherit_record
        )
        runtime_receipt = _strict_progress_json(
            (args.resume_from / "capture.runtime.json").read_bytes()
        )
        if runtime_receipt.get("request_sha256") != prior.request_sha:
            raise CaptureContractError("prior runtime request differs")
        runtime = runtime_receipt["geometry"]
        for name in (
            "request",
            "runtime",
            "prepared",
            "collected",
            "partial",
            "complete",
            "finalized",
        ):
            path = args.resume_from / f"capture.{name}.json"
            if path.exists():
                prior_receipts[path.name] = read_stable_file_snapshot(
                    path, label="prior stage receipt"
                ).sha256
        if "capture.collected.json" in prior_receipts:
            collected = _strict_progress_json(
                (args.resume_from / "capture.collected.json").read_bytes()
            )
            if collected.get("prepared_receipt_sha256") != prior_receipts.get(
                "capture.prepared.json"
            ):
                raise CaptureContractError("prior collection lost its prepared provenance")
        if "capture.finalized.json" in prior_receipts:
            finalized = _strict_progress_json(
                (args.resume_from / "capture.finalized.json").read_bytes()
            )
            if (
                finalized.get("prepared_receipt_sha256")
                != prior_receipts.get("capture.prepared.json")
                or finalized.get("collected_receipt_sha256")
                != prior_receipts.get("capture.collected.json")
                or finalized.get("status") not in ("partial", "complete")
                or finalized.get("typed_receipt_sha256")
                != prior_receipts.get(f"capture.{finalized.get('status')}.json")
            ):
                raise CaptureContractError("prior finalized provenance chain differs")
    if prior is not None:
        for capture, manifest in _stream_records(
            None,
            None,
            inputs.prompts,
            inputs.streams,
            prior,
            files.directory,
            world_size=args.tensor_parallel_size,
            topology=args.plan.endswith("_topology_discovery"),
            replay_prefix=False,
            recover_only=True,
        ):
            journal.append(capture, manifest)
            files.append_normalized(capture)
        prior._recheck_validated_prefix()
        if any(
            _spool_sha256(path) != digest for path, digest in prior.resume_spool_digests.items()
        ):
            raise CaptureContractError("prior spool changed during preparation")
        if any(
            read_stable_file_snapshot(args.resume_from / name, label="prior stage receipt").sha256
            != digest
            for name, digest in prior_receipts.items()
        ):
            raise CaptureContractError("prior stage provenance changed during preparation")
    _verify_capture_inputs(args, inputs, request["controller"])
    receipt = files.seal(
        journal,
        "prepared",
        runtime=False,
        expected_runtime=runtime,
        migration=bridge,
        inherited_from=None
        if prior is None
        else {
            "directory": str(args.resume_from.resolve()),
            "request_sha256": prior.request_sha,
            "journal_sha256": prior.resume_digest,
            "journal_count": prior.resume_count,
            "inherited_count": journal.count,
            "receipt_sha256": prior_receipts,
            "recovered_rank_spools": {
                str(path): digest for path, digest in prior.resume_spool_digests.items()
            },
        },
    )
    return files.normalized, receipt


def _collect_stage(args, inputs, request, invocations, files, receipt, started):
    if args.max_wall_seconds is None or not 0 < args.max_wall_seconds < float("inf"):
        raise CaptureContractError("staged collection requires a finite positive wall budget")
    if time.monotonic() - started >= args.max_wall_seconds:
        raise CaptureContractError(
            "collection budget exhausted before engine load; prepared state unchanged"
        )
    journal = _CaptureJournal.existing(files.directory, request, invocations, prepared=receipt)
    if (files.directory / "capture.collection-started.json").exists():
        raise CaptureContractError("collection was already attempted; finalize or recover on CPU")
    journal._write(
        "capture.collection-started.json", {"prepared_sha256": args.stage_receipt_sha256}
    )
    error = None
    identity_verified = False
    llm = None
    try:
        llm, _, sampling, _ = _load_capture_engine(args, inputs, journal)
        for capture, manifest in _stream_records(
            llm,
            sampling,
            inputs.prompts,
            inputs.streams,
            journal,
            files.directory,
            world_size=args.tensor_parallel_size,
            topology=args.plan.endswith("_topology_discovery"),
            replay_prefix=False,
        ):
            journal.append(capture, manifest)
            files.append_normalized(capture)
            if time.monotonic() - started >= args.max_wall_seconds and journal.count < len(
                invocations
            ):
                raise _CapturePauseError()
    except BaseException as failure:
        error = failure
    try:
        _verify_capture_inputs(args, inputs, request["controller"])
        files.verify_inherited_bytes(receipt)
        identity_verified = True
    except BaseException as failure:
        error = failure
    # A failed model/runtime contract is sticky; restoring files cannot qualify it.
    runtime_bound = (files.directory / "capture.runtime.json").exists()
    path = files.seal(
        journal,
        "collected",
        runtime=runtime_bound,
        prepared_receipt_sha256=args.stage_receipt_sha256,
        identity_verification="passed" if identity_verified else "failed",
        runtime_bound=runtime_bound,
        resumable_prefix=runtime_bound
        and identity_verified
        and not isinstance(error, CaptureContractError),
        error_type=None if error is None else type(error).__name__,
    )
    if error is not None and not isinstance(error, _CapturePauseError):
        raise error
    return files.normalized, path


def _finalize_stage(args, inputs, request, invocations, files, receipt):
    if receipt.get("resumable_prefix") is not True:
        raise CaptureContractError("collection failed its runtime contract; prefix cannot qualify")
    prepared = load_stage_receipt(
        files.directory / "capture.prepared.json",
        receipt.get("prepared_receipt_sha256"),
        _strict_progress_json,
    )
    files.verify_inherited_bytes(prepared)
    digest, manifests, cells = sha256(), [], 0
    chain_tail = None

    def summarize_record(record):
        nonlocal cells, chain_tail
        chain_tail = record["record_sha256"]
        if not record["capture"].get("dense_exclusion"):
            digest.update(_canonical_capture_line(record["capture"]))
            manifests.append(record["manifest"])
            cells += record["manifest"]["candidate_cell_count"]

    # Summarize validated records; finish() checks their exact bytes again.
    journal = _CaptureJournal.existing(
        files.directory, request, invocations, on_record=summarize_record
    )
    journal.count = journal.resume_count
    if chain_tail is not None:
        journal.previous = chain_tail
    if (
        journal.count != receipt["completed_capture_count"]
        or journal.resume_digest != receipt["files"]["capture.progress.jsonl"]
    ):
        raise CaptureContractError("collection count differs from validated journal")
    if journal.previous != receipt["chain_tail_sha256"]:
        raise CaptureContractError("collection chain tail differs from validated journal")
    normalized = files.normalized
    if digest.hexdigest() != receipt["files"][files.NORMALIZED]:
        if receipt.get("error_type") is None:
            raise CaptureContractError(
                "normalized staged prefix differs from the validated journal"
            )
        # The journal is authoritative; an interrupted derived append must not strand it.
        normalized = files.directory / "capture.normalized.recovered.jsonl"
        files.publish_derived(
            normalized,
            (
                _canonical_capture_line(record["capture"])
                for _, record in journal._records(journal.path)
                if not record["capture"].get("dense_exclusion")
            ),
            digest.hexdigest(),
        )
        files.write_derived(
            "capture.normalization-recovery.json",
            {
                "collected_receipt_sha256": args.stage_receipt_sha256,
                "journal_sha256": journal.resume_digest,
                "original_normalized_sha256": receipt["files"][files.NORMALIZED],
                "recovered_normalized_sha256": digest.hexdigest(),
            },
        )
    _verify_capture_inputs(args, inputs, request["controller"])
    files.verify(receipt, request, stage="collected")

    def finish(status, *, stage, error=None):
        journal.finish(
            status,
            identity_verified=True,
            stage=stage,
            error=error,
            write=files.write_derived,
            _validated_bytes=True,
        )
        files.write_derived(
            "capture.finalized.json",
            {
                "stage_schema_version": 1,
                "prepared_receipt_sha256": receipt["prepared_receipt_sha256"],
                "collected_receipt_sha256": args.stage_receipt_sha256,
                "typed_receipt_sha256": _spool_sha256(files.directory / f"capture.{status}.json"),
                "normalized_file": normalized.name,
                "normalized_sha256": _spool_sha256(normalized),
                "status": status,
            },
        )

    if journal.count != len(invocations):
        finish("partial", stage="collection", error=_CapturePauseError())
        return normalized, files.directory / "capture.partial.json"
    runtime = _strict_progress_json((files.directory / "capture.runtime.json").read_bytes())
    if runtime.get("request_sha256") != journal.request_sha:
        raise CaptureContractError("collected runtime request differs")
    geometry = runtime["geometry"]
    if not isinstance(geometry, dict):
        raise CaptureContractError("collected runtime geometry must be an object")
    # Preserve the journal-normalized receipt bytes; this is a derived solver input.
    calibrated = files.directory / "capture.calibration.jsonl"
    capture_sha, manifests = _normalize_stream_file(
        normalized, inputs.streams, inputs.threshold_scale_factor, manifests, destination=calibrated
    )
    normalized = calibrated
    cells = sum(m["candidate_cell_count"] for m in manifests)
    manifest = _capture_manifest(
        args,
        inputs,
        geometry["kv_page_tokens"],
        len(manifests),
        cells,
        manifests,
        capture_sha,
    )
    manifest_stage = files.directory / "capture.manifest.staged.json"
    files.write_derived(manifest_stage.name, manifest)
    # Keep staged evidence, even after publication; never move or replace inherited files.
    for source, destination in (
        (normalized, inputs.output_path),
        (manifest_stage, inputs.manifest_path),
    ):
        if destination.exists() or destination.is_symlink():
            if _spool_sha256(destination) != _spool_sha256(source):
                raise CaptureContractError("existing published capture bytes differ")
        else:
            os.link(source, destination, follow_symlinks=False)
    _fsync_directory(inputs.output_path.parent)
    _fsync_directory(inputs.manifest_path.parent)
    finish("complete", stage="publication")
    return inputs.output_path, inputs.manifest_path


def _run_staged(args):
    started = time.monotonic()
    if args.capture_scope != "whole-prefill" or args.capture_provider != "reference":
        raise CaptureContractError("staged capture supports reference whole-prefill only")
    files = CaptureStageFiles(
        Path(args.output).parent, serialize=_canonical_capture_line, parse=_strict_progress_json
    )
    files.directory.mkdir(parents=True, exist_ok=True)
    with files.exclusive():
        receipt = prepared_request = None
        if args.phase == "prepare":
            if args.stage_receipt is not None or args.stage_receipt_sha256 is not None:
                raise CaptureContractError("preparation uses --resume-from, not a stage receipt")
        else:
            if (
                args.resume_from is not None
                or args.source_bridge is not None
                or args.source_bridge_sha256 is not None
            ):
                raise CaptureContractError("collection/finalization cannot change resume lineage")
            if args.stage_receipt is None or args.stage_receipt.resolve().parent != files.directory:
                raise CaptureContractError("stage receipt must belong to this output directory")
            receipt = load_stage_receipt(
                args.stage_receipt, args.stage_receipt_sha256, _strict_progress_json
            )
            snapshot = read_stable_file_snapshot(
                files.directory / "capture.request.json", label="prepared request"
            )
            if snapshot.sha256 != receipt.get("request_sha256"):
                raise CaptureContractError("prepared request SHA differs")
            prepared_request = _strict_progress_json(snapshot.payload)
        inputs = _prepare_capture_inputs(args, prepared_request=prepared_request)
        request, invocations = _capture_request(args, inputs, _controller_identity())
        if args.phase == "prepare":
            return _prepare_stage(args, inputs, request, invocations, files)
        files.verify(receipt, request, stage="prepared" if args.phase == "collect" else "collected")
        if args.phase == "collect":
            return _collect_stage(args, inputs, request, invocations, files, receipt, started)
        return _finalize_stage(args, inputs, request, invocations, files, receipt)


def _run(args: argparse.Namespace, state: dict) -> tuple[Path, Path]:
    started = time.monotonic()
    budget = getattr(args, "max_wall_seconds", None)
    resume_from = getattr(args, "resume_from", None)
    if budget is not None and (not 0 < budget < float("inf")):
        raise CaptureContractError("--max-wall-seconds must be finite and positive")
    # Always journal progress; a later invalid prompt must not erase earlier captures.
    resumable = True
    provider = getattr(args, "capture_provider", "fa4")
    whole_prefill = getattr(args, "capture_scope", "final-chunk") == "whole-prefill"
    if whole_prefill and provider != "reference":
        raise CaptureContractError("whole-prefill capture requires --capture-provider reference")
    inputs = _prepare_capture_inputs(args)
    provider = inputs.provider
    whole_prefill = inputs.whole_prefill
    prompts = inputs.prompts
    threshold_scale_factor = inputs.threshold_scale_factor
    targets = inputs.targets
    checkpoint = inputs.checkpoint
    model_id = inputs.model_id
    streams = inputs.streams
    output_path = inputs.output_path
    manifest_path = inputs.manifest_path
    journal = None
    controller = _controller_identity() if resumable else None

    def verify_identity():
        _verify_capture_inputs(args, inputs, controller)
        if journal is not None and journal.resume_path is not None:
            if journal._validate_prefix(journal.resume_path) != (
                journal.resume_count,
                journal.resume_digest,
            ):
                raise CaptureContractError("prior capture journal changed during replay")
            if any(
                _spool_sha256(path) != digest
                for path, digest in journal.resume_spool_digests.items()
            ):
                raise CaptureContractError("prior stream spool changed during recovery")

    if resumable:
        request, invocations = _capture_request(args, inputs, controller)
        journal = _CaptureJournal(output_path.parent, request, invocations, resume_from)
        state.update(journal=journal, verify=verify_identity, stage="load")
    assert journal is not None  # Every capture now has a resumable journal.
    llm, tokenizer, sampling, observed_page = _load_capture_engine(args, inputs, journal)

    capture_manifests: list[dict[str, object]] = []
    seen_sources: dict[str, tuple[str, str, tuple[int, int | None]]] = {}
    capture_digest = sha256()
    capture_count = 0
    candidate_cell_count = 0
    temporary_path: Path | None = None
    state["stage"] = "collection"
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            if whole_prefill:
                for capture, capture_manifest in _stream_records(
                    llm,
                    sampling,
                    prompts,
                    streams,
                    journal,
                    output_path.parent,
                    world_size=args.tensor_parallel_size,
                    topology=args.plan.endswith("_topology_discovery"),
                ):
                    if journal is not None:
                        journal.append(capture, capture_manifest)
                    if not capture.get("dense_exclusion"):
                        line = _canonical_capture_line(capture)
                        temporary.write(line)
                        capture_digest.update(line)
                        capture_count += 1
                        candidate_cell_count += cast(
                            "int", capture_manifest["candidate_cell_count"]
                        )
                        capture_manifests.append(capture_manifest)
                    if (
                        budget is not None
                        and time.monotonic() - started >= budget
                        and journal.count < len(journal.invocations)
                    ):
                        raise _CapturePauseError()
            for prompt in () if whole_prefill else prompts:
                token_ids = (
                    list(prompt.prompt_token_ids)
                    if prompt.prompt_token_ids is not None
                    else tokenizer.encode(prompt.prompt, add_special_tokens=True)
                )
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
                    if journal is not None and capture_count < journal.resume_count:
                        capture, capture_manifest = journal.replay()
                    else:
                        acknowledgements = llm.collective_rpc(
                            "mask_reuse_capture_begin", args=(invocation,)
                        )
                        validate_begin_acks(acknowledgements, invocation)
                        # Exact TokensPrompt binds scheduler input to the frozen invocation.
                        llm.generate({"prompt_token_ids": token_ids}, sampling, use_tqdm=False)
                        rank_captures = llm.collective_rpc("mask_reuse_capture_drain")
                        merge = (
                            merge_rank_topology_discovery_captures
                            if args.plan.endswith("_topology_discovery")
                            else merge_rank_captures
                        )
                        merged = merge(rank_captures, invocation)
                        capture, capture_manifest = merged.capture, merged.manifest
                    if journal is not None:
                        journal.append(capture, capture_manifest)
                    line = _canonical_capture_line(capture)
                    temporary.write(line)
                    capture_digest.update(line)
                    capture_count += 1
                    candidate_cell_count += cast("int", capture_manifest["candidate_cell_count"])
                    capture_manifests.append(capture_manifest)
                    if (
                        budget is not None
                        and time.monotonic() - started >= budget
                        and capture_count >= journal.resume_count
                        and capture_count < len(prompts) * len(targets)
                    ):
                        raise _CapturePauseError()
            temporary.flush()
            os.fsync(temporary.fileno())
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    state["stage"] = "identity_verification"
    try:
        verify_identity()
    except BaseException:
        assert temporary_path is not None
        temporary_path.unlink(missing_ok=True)
        raise
    capture_sha256 = capture_digest.hexdigest()
    if whole_prefill:
        capture_sha256, capture_manifests = _normalize_stream_file(
            temporary_path, streams, threshold_scale_factor, capture_manifests
        )
        capture_count = len(capture_manifests)
        candidate_cell_count = sum(
            cast("int", m["candidate_cell_count"]) for m in capture_manifests
        )
    state["stage"] = "publication"
    manifest = _capture_manifest(
        args,
        inputs,
        observed_page,
        capture_count,
        candidate_cell_count,
        capture_manifests,
        capture_sha256,
    )
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
    return 0 if run(args) is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly REMOTE_STUDY_ROOT="${STUDY_ROOT:-/study}"
readonly SHARED_PYTHON_DIR="${REMOTE_STUDY_ROOT}/cache/python"
readonly SHARED_HF_HOME="${REMOTE_STUDY_ROOT}/cache/huggingface"
readonly SHARED_DATASETS_CACHE="${REMOTE_STUDY_ROOT}/cache/datasets"
readonly DATASET_ID="abisee/cnn_dailymail"
readonly DATASET_CONFIG="3.0.0"
readonly DATASET_SPLIT="train"
readonly DATASET_REQUESTED_REVISION="main"
readonly STUDY_CONTAINER_IMAGE="${STUDY_CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/weimingc/vllm_container_images/qwen36_pr_stack_latest_runtime_only_20260519_234147/images/vllm_qwen36_pr_stack_latest_runtime_only.sqsh}"
# Packed calibration requests a conservative 128 * 8 raw-document prefix (packing
# may fill its rows before every requested document contributes). Unpacked evaluation
# selects rows [1024, 1056), so the two source pools are guaranteed disjoint.
readonly DATASET_SOURCE_ROW_COUNT=1056

# The launcher injects a per-experiment HF_HOME after task-specific environment
# variables. Override it here so model downloads survive across all five jobs and
# both model pipelines.
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_CACHE="${SHARED_HF_HOME}/hub"
export HF_DATASETS_CACHE="${SHARED_DATASETS_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export PYTHONPATH="${SHARED_PYTHON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export STUDY_CONTAINER_IMAGE
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

usage() {
    cat >&2 <<'EOF'
Usage:
  run_study.sh stage --model <Qwen model id>
  run_study.sh run --model <Qwen model id> --candidates <candidate[,candidate...]>

Launcher candidate names:
  per_tensor_fp8
  per_tensor_fp8_weight_only_control
  block128_static_weight_only
  block128_dynamic_w8a8_research
  block128_dynamic_weight_only_control
  mxfp8
  mxfp8_weight_only_control

The run mode accepts a comma-separated list so a future controlled submatrix can
run inside one Nemo task without increasing the five-task pipeline limit.
EOF
    exit 2
}

MODE="${1:-}"
[[ -n "${MODE}" ]] || usage
shift

MODEL_ID=""
CANDIDATES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || usage
            MODEL_ID="$2"
            shift 2
            ;;
        --candidate|--candidates)
            [[ $# -ge 2 ]] || usage
            CANDIDATES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

case "${MODEL_ID}" in
    Qwen/Qwen3.6-35B-A3B)
        MODEL_SLUG="qwen3.6-35b-a3b"
        ;;
    Qwen/Qwen3.6-27B)
        MODEL_SLUG="qwen3.6-27b"
        ;;
    *)
        echo "Unsupported model for this study: ${MODEL_ID:-<empty>}" >&2
        usage
        ;;
esac
readonly MODEL_ID MODEL_SLUG
export MODEL_ID MODEL_SLUG

PYTHON_BIN="$(command -v python3 || command -v python || true)"
readonly PYTHON_BIN
[[ -n "${PYTHON_BIN}" ]] || {
    echo "No Python interpreter found in ${STUDY_CONTAINER_IMAGE}" >&2
    exit 2
}

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PACKAGED_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PACKAGED_REPO_ROOT}:${PYTHONPATH}"

# ModelOpt's FP8 and MXFP8 fake-quantizers JIT-compile CUDA extensions. Pyxis
# disables the home mount, so pin all compiler caches to the writable study mount.
export TORCH_EXTENSIONS_DIR="${REMOTE_STUDY_ROOT}/cache/torch_extensions/${MODEL_SLUG}"
export CUDA_CACHE_PATH="${REMOTE_STUDY_ROOT}/cache/cuda/${MODEL_SLUG}"
export TRITON_CACHE_DIR="${REMOTE_STUDY_ROOT}/cache/triton/${MODEL_SLUG}"

mkdir -p \
    "${REMOTE_STUDY_ROOT}/cache" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "${TRITON_CACHE_DIR}" \
    "${REMOTE_STUDY_ROOT}/manifests/${MODEL_SLUG}" \
    "${REMOTE_STUDY_ROOT}/results/${MODEL_SLUG}"

validate_runtime() {
    "${PYTHON_BIN}" - "${PACKAGED_REPO_ROOT}" <<'PY'
import json
import os
import pathlib
import platform
import shutil
import sys

import accelerate
import modelopt
import torch
import transformers
from torch.utils.cpp_extension import CUDA_HOME

repo_root = pathlib.Path(sys.argv[1]).resolve()
modelopt_file = pathlib.Path(modelopt.__file__).resolve()
expected_modelopt_root = (repo_root / "modelopt").resolve()
if not modelopt_file.is_relative_to(expected_modelopt_root):
    raise RuntimeError(
        f"Imported ModelOpt from {modelopt_file}, expected packaged source under "
        f"{expected_modelopt_root}"
    )

architecture_by_model = {
    "Qwen/Qwen3.6-35B-A3B": "Qwen3_5MoeForConditionalGeneration",
    "Qwen/Qwen3.6-27B": "Qwen3_5ForConditionalGeneration",
}
architecture = architecture_by_model[os.environ["MODEL_ID"]]
model_class = getattr(transformers, architecture, None)
if model_class is None:
    raise RuntimeError(
        f"transformers {transformers.__version__} does not expose {architecture}; "
        "this container cannot load the requested Qwen3.6 model"
    )

cuda_home = pathlib.Path(CUDA_HOME) if CUDA_HOME else None
nvcc = cuda_home / "bin" / "nvcc" if cuda_home else None
missing_build_tools = []
if shutil.which("c++") is None:
    missing_build_tools.append("c++")
if shutil.which("ninja") is None:
    missing_build_tools.append("ninja")
if nvcc is None or not nvcc.is_file():
    missing_build_tools.append("nvcc")
if missing_build_tools:
    raise RuntimeError(
        "Container cannot JIT-build ModelOpt FP8/MXFP8 extensions; missing: "
        + ", ".join(missing_build_tools)
    )

payload = {
    "architecture": platform.machine(),
    "container_image": os.environ["STUDY_CONTAINER_IMAGE"],
    "model_class": f"{model_class.__module__}.{model_class.__name__}",
    "modelopt": str(modelopt_file),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "cuda_home": str(cuda_home),
    "torch_extensions_dir": os.environ["TORCH_EXTENSIONS_DIR"],
}
print(json.dumps(payload, indent=2, sort_keys=True))

if torch.cuda.is_available():
    from modelopt.torch.quantization.extensions import get_cuda_ext_fp8, get_cuda_ext_mx

    # Compile before loading tens of billions of parameters. Later sequential
    # candidates reuse these binaries from the shared per-model cache.
    get_cuda_ext_fp8(raise_if_failed=True)
    get_cuda_ext_mx(raise_if_failed=True)
PY
}

ensure_staging_dependencies() {
    if "${PYTHON_BIN}" -c 'import datasets, huggingface_hub' >/dev/null 2>&1; then
        return
    fi

    mkdir -p "${SHARED_PYTHON_DIR}"
    # The two model pipelines may stage concurrently. Serialize the uncommon
    # install path so they cannot partially overwrite the shared target.
    (
        flock 9
        if ! "${PYTHON_BIN}" -c 'import datasets, huggingface_hub' >/dev/null 2>&1; then
            "${PYTHON_BIN}" -m pip install \
                --disable-pip-version-check \
                --target "${SHARED_PYTHON_DIR}" \
                'datasets>=3.1,<5' \
                'huggingface_hub>=0.34,<2'
        fi
    ) 9>"${REMOTE_STUDY_ROOT}/cache/python-dependencies.lock"
}

stage_inputs() {
    ensure_staging_dependencies
    export MODEL_ID MODEL_SLUG REMOTE_STUDY_ROOT
    export DATASET_ID DATASET_CONFIG DATASET_SPLIT DATASET_REQUESTED_REVISION
    export DATASET_SOURCE_ROW_COUNT
    "${PYTHON_BIN}" - <<'PY'
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

model_id = os.environ["MODEL_ID"]
model_slug = os.environ["MODEL_SLUG"]
study_root = Path(os.environ["REMOTE_STUDY_ROOT"])
manifest_path = study_root / "manifests" / model_slug / "staging.json"
dataset_id = os.environ["DATASET_ID"]
dataset_config = os.environ["DATASET_CONFIG"]
dataset_split = os.environ["DATASET_SPLIT"]
dataset_requested_revision = os.environ["DATASET_REQUESTED_REVISION"]
dataset_source_row_count = int(os.environ["DATASET_SOURCE_ROW_COUNT"])
dataset_path = (
    study_root
    / "cache"
    / "datasets"
    / model_slug
    / "cnn_dailymail_train_first_1056.jsonl"
)

last_error = None
for attempt in range(1, 4):
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=model_id,
                revision="main",
                cache_dir=os.environ["HF_HUB_CACHE"],
                max_workers=4,
            )
        )
        break
    except Exception as error:
        last_error = error
        if attempt == 3:
            raise
        time.sleep(15 * attempt)
else:  # pragma: no cover - defensive; the loop either breaks or raises.
    raise last_error  # type: ignore[misc]

required = [snapshot_path / "config.json"]
if not all(path.is_file() for path in required):
    raise RuntimeError(f"Incomplete snapshot at {snapshot_path}: config.json is missing")
if not any(snapshot_path.glob("*.safetensors")):
    raise RuntimeError(f"Incomplete snapshot at {snapshot_path}: no safetensors files found")
if not any(
    (snapshot_path / name).is_file()
    for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
):
    raise RuntimeError(f"Incomplete snapshot at {snapshot_path}: tokenizer assets are missing")

# Resolve the mutable dataset branch once, then load the exact repository commit.
# This keeps reruns deterministic even if the upstream dataset branch advances.
last_error = None
for attempt in range(1, 4):
    try:
        dataset_resolved_revision = HfApi().dataset_info(
            repo_id=dataset_id,
            revision=dataset_requested_revision,
        ).sha
        if not dataset_resolved_revision:
            raise RuntimeError(f"Could not resolve {dataset_id}@{dataset_requested_revision}")
        dataset = load_dataset(
            path=dataset_id,
            name=dataset_config,
            split=dataset_split,
            revision=dataset_resolved_revision,
            cache_dir=os.environ["HF_DATASETS_CACHE"],
        )
        break
    except Exception as error:
        last_error = error
        if attempt == 3:
            raise
        time.sleep(15 * attempt)
else:  # pragma: no cover - defensive; the loop either breaks or raises.
    raise last_error  # type: ignore[misc]

if len(dataset) < dataset_source_row_count:
    raise RuntimeError(
        f"Dataset {dataset_id}/{dataset_config}:{dataset_split} has only {len(dataset)} rows; "
        f"need {dataset_source_row_count}"
    )

# Materialize exactly the required source prefix as a generic text JSONL. The
# temporary file is model-specific and renamed atomically only after all rows
# have been serialized successfully.
dataset_path.parent.mkdir(parents=True, exist_ok=True)
dataset_temporary = dataset_path.with_name(f".{dataset_path.name}.{os.getpid()}.tmp")
dataset_hash = hashlib.sha256()
try:
    with dataset_temporary.open("wb") as output:
        for source_row_index in range(dataset_source_row_count):
            article = dataset[source_row_index]["article"]
            if not isinstance(article, str):
                raise TypeError(
                    f"Dataset row {source_row_index} has non-string article: "
                    f"{type(article).__name__}"
                )
            encoded = (
                json.dumps({"text": article}, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            output.write(encoded)
            dataset_hash.update(encoded)
        output.flush()
        os.fsync(output.fileno())
    dataset_temporary.replace(dataset_path)
finally:
    dataset_temporary.unlink(missing_ok=True)

payload = {
    "schema_version": 2,
    "model_id": model_id,
    "requested_revision": "main",
    "resolved_revision": snapshot_path.name,
    "snapshot_path": str(snapshot_path),
    "dataset": {
        "dataset_id": dataset_id,
        "requested_revision": dataset_requested_revision,
        "resolved_revision": dataset_resolved_revision,
        "config": dataset_config,
        "split": dataset_split,
        "datasets_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_version": str(dataset.info.version) if dataset.info.version else None,
        "source_row_start": 0,
        "source_row_count": dataset_source_row_count,
        "text_source_column": "article",
        "local_format": "jsonl",
        "local_text_column": "text",
        "local_path": str(dataset_path),
        "sha256": dataset_hash.hexdigest(),
    },
    "staged_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "architecture": platform.machine(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
temporary = manifest_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(manifest_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

driver_recipe() {
    case "$1" in
        per_tensor_fp8|per_tensor_fp8_weight_only_control|\
        block128_static_weight_only|block128_dynamic_w8a8_research|\
        block128_dynamic_weight_only_control|mxfp8|mxfp8_weight_only_control)
            echo "$1"
            ;;
        *)
            # Permit future driver recipe names without changing this wrapper.
            # The study.py argument parser remains the authoritative validator.
            [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]] || {
                echo "Unsafe candidate name: $1" >&2
                return 2
            }
            echo "$1"
            ;;
    esac
}

run_candidate() {
    local candidate="$1"
    local recipe
    recipe="$(driver_recipe "${candidate}")"

    local repo_root driver staging_manifest resolved_revision dataset_path dataset_sha256
    local output_dir reference_cache launch_manifest
    repo_root="${PACKAGED_REPO_ROOT}"
    driver="${repo_root}/experimental/qwen36_fp8_granularity_study/study.py"
    staging_manifest="${REMOTE_STUDY_ROOT}/manifests/${MODEL_SLUG}/staging.json"
    output_dir="${REMOTE_STUDY_ROOT}/results/${MODEL_SLUG}/${candidate}"
    reference_cache="${REMOTE_STUDY_ROOT}/cache/reference/${MODEL_SLUG}"
    launch_manifest="${REMOTE_STUDY_ROOT}/manifests/${MODEL_SLUG}/${candidate}.json"
    mkdir -p "${output_dir}" "${reference_cache}"

    # Invalidate any result from an older attempt before checks that can fail outside
    # study.py. The report will show this schema-valid placeholder as pending rather
    # than silently ranking a stale complete artifact.
    export CANDIDATE="${candidate}" DRIVER_RECIPE="${recipe}" OUTPUT_DIR="${output_dir}"
    "${PYTHON_BIN}" - <<'PY'
import datetime as dt
import json
import os
from pathlib import Path

path = Path(os.environ["OUTPUT_DIR"]) / "results.json"
payload = {
    "schema_version": "qwen36-fp8-granularity-study-v1",
    "status": "launcher_preflight",
    "model": os.environ["MODEL_ID"],
    "recipe": os.environ["DRIVER_RECIPE"],
    "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "launcher_note": "study.py has not started; prior candidate results were invalidated",
}
temporary = path.with_suffix(".json.preflight.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

    # Resolve ModelOpt from the exact packaged checkout that contains study.py,
    # ahead of both shared helper dependencies and the container installation.
    export PYTHONPATH="${repo_root}:${PYTHONPATH}"

    [[ -f "${driver}" ]] || {
        echo "Study driver not found in packaged checkout: ${driver}" >&2
        return 2
    }
    [[ -f "${staging_manifest}" ]] || {
        echo "Staging manifest not found: ${staging_manifest}" >&2
        return 2
    }
    local staging_values
    staging_values="$("${PYTHON_BIN}" - \
        "${staging_manifest}" \
        "${REMOTE_STUDY_ROOT}" \
        "${MODEL_ID}" \
        "${MODEL_SLUG}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
study_root = Path(sys.argv[2])
expected_model_id = sys.argv[3]
model_slug = sys.argv[4]
payload = json.loads(manifest_path.read_text())

if payload.get("schema_version") != 2:
    raise RuntimeError(
        f"Unsupported staging manifest schema {payload.get('schema_version')!r}; expected 2"
    )
if payload.get("model_id") != expected_model_id:
    raise RuntimeError(
        f"Staging manifest model {payload.get('model_id')!r} does not match {expected_model_id!r}"
    )
resolved_revision = payload.get("resolved_revision")
if not isinstance(resolved_revision, str) or not resolved_revision:
    raise RuntimeError("Staging manifest has no resolved model revision")

dataset = payload.get("dataset")
if not isinstance(dataset, dict):
    raise RuntimeError("Staging manifest has no dataset snapshot")
expected_metadata = {
    "dataset_id": "abisee/cnn_dailymail",
    "config": "3.0.0",
    "split": "train",
    "source_row_start": 0,
    "source_row_count": 1056,
    "text_source_column": "article",
    "local_format": "jsonl",
    "local_text_column": "text",
}
for key, expected in expected_metadata.items():
    if dataset.get(key) != expected:
        raise RuntimeError(
            f"Staging dataset metadata {key}={dataset.get(key)!r}; expected {expected!r}"
        )
if not dataset.get("resolved_revision"):
    raise RuntimeError("Staging dataset metadata has no resolved repository revision")

dataset_path = Path(dataset.get("local_path", ""))
expected_path = (
    study_root
    / "cache"
    / "datasets"
    / model_slug
    / "cnn_dailymail_train_first_1056.jsonl"
)
if dataset_path != expected_path:
    raise RuntimeError(f"Staged dataset path {dataset_path} does not match {expected_path}")
if not dataset_path.is_file():
    raise RuntimeError(f"Staged dataset file is missing: {dataset_path}")

expected_sha256 = dataset.get("sha256")
if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
    raise RuntimeError("Staging dataset metadata has an invalid SHA-256")
digest = hashlib.sha256()
row_count = 0
with dataset_path.open("rb") as source:
    for raw_line in source:
        digest.update(raw_line)
        row_count += 1
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid JSONL at source row {row_count - 1}") from error
        if set(row) != {"text"} or not isinstance(row["text"], str):
            raise RuntimeError(
                f"Invalid staged dataset schema at source row {row_count - 1}: "
                "expected exactly one string 'text' field"
            )
if row_count != 1056:
    raise RuntimeError(f"Staged dataset has {row_count} rows; expected 1056")
if digest.hexdigest() != expected_sha256:
    raise RuntimeError(
        f"Staged dataset SHA-256 {digest.hexdigest()} does not match {expected_sha256}"
    )

print("\t".join((resolved_revision, str(dataset_path), expected_sha256)))
PY
    )"
    IFS=$'\t' read -r resolved_revision dataset_path dataset_sha256 <<< "${staging_values}"
    [[ -n "${resolved_revision}" && -n "${dataset_path}" && -n "${dataset_sha256}" ]] || {
        echo "Staging validation returned incomplete values" >&2
        return 2
    }

    # Everything needed by this GPU task is now present and content-verified.
    # Enforce offline loading only after CPU staging has completed.
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    export CANDIDATE="${candidate}" DRIVER_RECIPE="${recipe}" OUTPUT_DIR="${output_dir}"
    export REFERENCE_CACHE="${reference_cache}" RESOLVED_REVISION="${resolved_revision}"
    export DATASET_PATH="${dataset_path}" DATASET_SHA256="${dataset_sha256}"
    export LAUNCH_MANIFEST="${launch_manifest}" STAGING_MANIFEST="${staging_manifest}"
    "${PYTHON_BIN}" - <<'PY'
import datetime as dt
import json
import os
import platform
import socket
from pathlib import Path

path = Path(os.environ["LAUNCH_MANIFEST"])
payload = {
    "schema_version": 1,
    "status": "running",
    "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "model_id": os.environ["MODEL_ID"],
    "resolved_revision": os.environ["RESOLVED_REVISION"],
    "candidate": os.environ["CANDIDATE"],
    "driver_recipe": os.environ["DRIVER_RECIPE"],
    "output_dir": os.environ["OUTPUT_DIR"],
    "reference_cache": os.environ["REFERENCE_CACHE"],
    "staging_manifest": os.environ["STAGING_MANIFEST"],
    "dataset_snapshot": {
        "path": os.environ["DATASET_PATH"],
        "sha256": os.environ["DATASET_SHA256"],
    },
    "hostname": socket.gethostname(),
    "architecture": platform.machine(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
}
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY

    echo "===== Qwen3.6 FP8 study preflight ====="
    uname -a
    nvidia-smi
    "${PYTHON_BIN}" - "${repo_root}" <<'PY'
import pathlib
import sys

import modelopt
import torch
import transformers

repo_root = pathlib.Path(sys.argv[1]).resolve()
modelopt_file = pathlib.Path(modelopt.__file__).resolve()
expected_root = (repo_root / "modelopt").resolve()
if not modelopt_file.is_relative_to(expected_root):
    raise RuntimeError(
        f"Imported ModelOpt from {modelopt_file}, expected packaged source under {expected_root}"
    )
print(f"modelopt={modelopt_file}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
PY
    echo "model=${MODEL_ID} revision=${resolved_revision} candidate=${candidate} recipe=${recipe}"
    echo "dataset=${dataset_path} sha256=${dataset_sha256} rows=1056 offline=true"

    local exit_code
    set +e
    "${PYTHON_BIN}" "${driver}" \
        --model "${MODEL_ID}" \
        --revision "${resolved_revision}" \
        --recipe "${recipe}" \
        --output-dir "${output_dir}" \
        --reference-cache "${reference_cache}" \
        --calib-dataset "${dataset_path}" \
        --eval-dataset "${dataset_path}" \
        --calib-size 128 \
        --eval-size 32 \
        --activation-mse-size 32 \
        --calib-seq-len 512 \
        --eval-seq-len 512 \
        --calib-batch-size 1 \
        --eval-batch-size 1 \
        --dtype bfloat16 \
        --seed 1234 \
        --device-map auto \
        --local-files-only
    exit_code=$?
    set -e

    export EXIT_CODE="${exit_code}"
    "${PYTHON_BIN}" - <<'PY'
import datetime as dt
import json
import os
from pathlib import Path

path = Path(os.environ["LAUNCH_MANIFEST"])
payload = json.loads(path.read_text())
exit_code = int(os.environ["EXIT_CODE"])
payload.update(
    {
        "status": "complete" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
)
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY
    return "${exit_code}"
}

case "${MODE}" in
    stage)
        [[ -z "${CANDIDATES}" ]] || usage
        validate_runtime
        stage_inputs
        ;;
    run)
        [[ -n "${CANDIDATES}" ]] || usage
        validate_runtime
        IFS=',' read -r -a candidate_list <<< "${CANDIDATES}"
        overall_exit_code=0
        for candidate in "${candidate_list[@]}"; do
            [[ -n "${candidate}" ]] || {
                echo "Candidate list contains an empty item: ${CANDIDATES}" >&2
                overall_exit_code=2
                continue
            }
            # Invoke in a strict subshell, then collect its status with errexit
            # disabled only in the parent. Calling a function directly in an `if`
            # condition would suppress `set -e` throughout that function.
            set +e
            (
                set -e
                run_candidate "${candidate}"
            )
            candidate_exit_code=$?
            set -e
            if [[ "${candidate_exit_code}" -ne 0 ]]; then
                overall_exit_code=1
            fi
        done
        exit "${overall_exit_code}"
        ;;
    *)
        usage
        ;;
esac

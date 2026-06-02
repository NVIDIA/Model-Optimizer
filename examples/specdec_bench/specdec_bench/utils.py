# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

from . import __version__ as specdec_bench_version

_SENSITIVE_SUBSTRINGS = ("token", "key", "secret", "password")
# Keys whose names contain a sensitive substring but are NOT actually secrets.
# Without this allowlist `tokenizer` redacts the model path because it contains
# `token`, losing meaningful provenance.
_SENSITIVE_KEY_ALLOWLIST = frozenset(
    {"tokenizer", "tokenizer_path", "tokenizer_mode", "tokenizer_revision"}
)


def get_tokenizer(path, trust_remote_code=False):
    return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)


def encode_chat(tokenizer, messages, chat_template_args={}, completions=False):
    if completions:
        return tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    return tokenizer.encode(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_args
        ),
        add_special_tokens=False,
    )


def decode_chat(tokenizer, out_tokens):
    return tokenizer.decode(out_tokens)


def read_json(path):
    if path is not None:
        with open(path) as f:
            data = json.load(f)
        return data
    return {}


def postprocess_base(text):
    return text


def postprocess_gptoss(text):
    final_message = text.split("<|channel|>final<|message|>")[-1]
    if "<|end|>" in final_message:
        final_message = final_message.split("<|end|>")[0]
    if "<|return|>" in final_message:
        final_message = final_message.split("<|return|>")[0]
    if "<|channel|>" in final_message:
        final_message = final_message.split("<|channel|>")[0]
    return final_message


def _get_engine_version(engine):
    """Return the engine package's __version__, or None on failure."""
    try:
        if engine in ("TRTLLM", "AUTO_DEPLOY"):
            import tensorrt_llm

            return tensorrt_llm.__version__
        elif engine == "VLLM":
            import vllm

            return vllm.__version__
        elif engine == "SGLANG":
            import sglang

            return sglang.__version__
    except Exception:
        pass
    return None


def _get_gpu_name():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _get_modelopt_version():
    try:
        import modelopt

        return getattr(modelopt, "__version__", None)
    except Exception:
        return None


def _git_sha(path):
    """git rev-parse HEAD inside `path`. Returns None if not a repo or git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _shard_files_from_index(index_path):
    """Return the set of shard filenames referenced by a safetensors index JSON."""
    try:
        with open(index_path) as f:
            wm = json.load(f).get("weight_map", {}) or {}
        return set(wm.values())
    except Exception:
        return set()


def _checkpoint_provenance(model_dir):
    """Cheap reproducibility fingerprint for a HuggingFace checkpoint directory.

    Returns {path, size_bytes, index_sha256, index_source}:
      - index_sha256 hashes model.safetensors.index.json (or config.json fallback)
        so it changes whenever the shard set or model config changes.
      - size_bytes sums only the index-listed shards + config.json. For a
        sharded 70B+ checkpoint this avoids a full rglob walk over hundreds
        of tokenizer/cache files. Falls back to rglob when no index exists.
    """
    if model_dir is None:
        return None
    try:
        p = Path(model_dir)
        if not p.is_dir():
            return {"path": str(model_dir)}
        hash_target = None
        for name in ("model.safetensors.index.json", "config.json"):
            candidate = p / name
            if candidate.is_file():
                hash_target = candidate
                break
        index_sha256 = None
        if hash_target is not None:
            h = hashlib.sha256()
            with open(hash_target, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            index_sha256 = h.hexdigest()
        # Size: shards listed in the safetensors index + the index/config file
        # itself. Avoids walking the entire model directory (which can be huge
        # for sharded multi-100B checkpoints).
        size_bytes = 0
        if hash_target is not None and hash_target.name == "model.safetensors.index.json":
            for shard_name in _shard_files_from_index(hash_target):
                shard_path = p / shard_name
                if shard_path.is_file():
                    size_bytes += shard_path.stat().st_size
            size_bytes += hash_target.stat().st_size
        else:
            # No shard index — fall back to summing every file under the dir.
            size_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return {
            "path": str(model_dir),
            "size_bytes": size_bytes,
            "index_sha256": index_sha256,
            "index_source": hash_target.name if hash_target is not None else None,
        }
    except Exception:
        return {"path": str(model_dir)}


def _is_sensitive_key(key):
    # Engine configs can carry non-string dict keys (e.g. int layer ids in a
    # serving_config); those are never sensitive field *names*, so skip them.
    if not isinstance(key, str):
        return False
    klow = key.lower()
    if klow in _SENSITIVE_KEY_ALLOWLIST:
        return False
    return any(s in klow for s in _SENSITIVE_SUBSTRINGS)


def _redact_value(value):
    """Recursively redact secrets in nested dict/list values.

    The top-level `_redact_config` walks one level of keys, but engine configs
    (serving_config from VLLMModel/SGLANGModel) and user-supplied runtime_params
    are nested arbitrarily — fields like `hf_token`, `tokenizer_revision`, or
    `aws_secret_access_key` need to be redacted at any depth.
    """
    if isinstance(value, dict):
        return {
            k: ("***REDACTED***" if _is_sensitive_key(k) else _redact_value(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v) for v in value)
    return value


def _redact_config(config):
    return _redact_value(config)


def _redact_argv(argv):
    """Mask values that follow a sensitive flag name (e.g. --hf_token VALUE).

    Conservative: only masks when the previous element looks like a flag whose
    bare name (sans leading dashes) trips _is_sensitive_key. Also handles the
    --flag=VALUE form.
    """
    redacted = []
    prev_is_sensitive = False
    for tok in argv:
        s = str(tok)
        if prev_is_sensitive:
            redacted.append("***REDACTED***")
            prev_is_sensitive = False
            continue
        if s.startswith("--"):
            name, sep, _val = s[2:].partition("=")
            if _is_sensitive_key(name):
                if sep:
                    redacted.append(f"--{name}=***REDACTED***")
                    prev_is_sensitive = False
                else:
                    redacted.append(s)
                    prev_is_sensitive = True
                continue
        redacted.append(s)
        prev_is_sensitive = False
    return redacted


def dump_env(args, save_dir, overrides=None):
    """Write configuration.json to save_dir capturing run args, engine version, and provenance.

    `overrides` is merged in last and is the channel for runtime-only fields
    (e.g. the live engine's serving_config dict from runner.get_serving_config()).
    """
    config = _redact_config(vars(args).copy())
    if overrides:
        config.update(_redact_config(overrides))

    config["engine_version"] = _get_engine_version(config.get("engine"))
    config["gpu"] = _get_gpu_name()
    config["python_version"] = sys.version
    config["argv"] = _redact_argv(sys.argv[:])

    # Provenance for reproducibility / apple-to-orange guarding.
    # Each *_sha and modelopt_version prefers an env var set by the harness
    # (because git/.git is typically not present inside the runtime container),
    # then falls back to runtime detection for standalone usage outside the
    # harness. container_image and nmm_sandbox_sha are env-only — there is no
    # reasonable in-process way to know them.
    config["specdec_bench_version"] = specdec_bench_version
    specdec_bench_dir = Path(__file__).resolve().parent
    config["specdec_bench_sha"] = os.environ.get("SPECDEC_BENCH_SHA") or _git_sha(specdec_bench_dir)
    config["modelopt_version"] = os.environ.get("MODELOPT_VERSION") or _get_modelopt_version()
    # Fallback assumes the in-tree layout examples/specdec_bench/specdec_bench/.
    # parents[2] reaches the modelopt repo root in that case. When vendored
    # elsewhere this would `git rev-parse` an unrelated repo; rely on the env
    # var path instead for non-in-tree deployments.
    config["modelopt_sha"] = os.environ.get("MODELOPT_SHA") or _git_sha(
        specdec_bench_dir.parents[2]
    )
    config["nmm_sandbox_sha"] = os.environ.get("NMM_SANDBOX_SHA") or None
    config["container_image"] = os.environ.get("CONTAINER_IMAGE") or None
    # Checkpoint fingerprint.
    config["checkpoint"] = _checkpoint_provenance(getattr(args, "model_dir", None))
    # UTC timestamp.
    config["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Attestation fields used by the visualizer to distinguish JIRA-tracked
    # official runs (must be on HuggingFace Hub) from community-contributed
    # runs. The harness sets JIRA_TICKET only after verifying the checkpoint
    # resolves on Hub; standalone runs leave both empty.
    config["jira_ticket"] = os.environ.get("JIRA_TICKET") or None
    config["huggingface_model_id"] = os.environ.get("HUGGINGFACE_MODEL_ID") or None

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "configuration.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

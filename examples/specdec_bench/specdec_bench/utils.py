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
import re
import subprocess
import sys
from pathlib import Path
from warnings import warn as _warn

from transformers import AutoTokenizer

from . import __version__ as specdec_bench_version

_SENSITIVE_SUBSTRINGS = ("token", "key", "secret", "password")
# Keys whose names contain a sensitive substring but are NOT actually secrets.
# Without this allowlist `tokenizer` redacts the model path because it contains
# `token`, losing meaningful provenance.
_SENSITIVE_KEY_ALLOWLIST = frozenset(
    {"tokenizer", "tokenizer_path", "tokenizer_mode", "tokenizer_revision"}
)
# Names that are credentials outright, whatever they hold. These redact on the
# name alone: a numeric or boolean value is no proof of safety, and treating one
# as such would persist e.g. an all-digit token.
_ALWAYS_SENSITIVE_SUBSTRINGS = ("hf_token", "api_key", "access_key", "secret", "password")
# The remaining substrings are ambiguous rather than damning: a bare match over
# `token` also swallows engine config, where the serving config alone
# contributes num_speculative_tokens, max_num_batched_tokens,
# skip_tokenizer_init and a dozen similar knobs, none of them credentials.
# Enumerating those doesn't hold — the set grows with every engine release — so
# for the ambiguous names only, a scalar value (int / bool / None) is taken as
# evidence the field is a knob and kept. A string still redacts, because that is
# the shape a credential takes.
_NON_SECRET_VALUE_TYPES = (bool, int, float, type(None))


def get_tokenizer(path, trust_remote_code=False):
    extra_special_tokens = None
    tokenizer_config_path = os.path.join(path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        extra_special_tokens = tokenizer_config.get("extra_special_tokens")

    kwargs = {"trust_remote_code": trust_remote_code}
    if isinstance(extra_special_tokens, list):
        kwargs["extra_special_tokens"] = {
            token.strip("<|>").replace("|", "_") + "_token": token for token in extra_special_tokens
        }

    return AutoTokenizer.from_pretrained(path, **kwargs)


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


_UNSET = object()

# `<org>/<name>`, the shape of a Hub repo id. Deliberately excludes `:@?#` and
# whitespace so a URL with embedded credentials (`https://user:tok@host/...`)
# can't reach configuration.json through the environment.
_HUB_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def _hub_model_id(env_var):
    """Return `$env_var` if it looks like a Hub repo id, else None.

    These ids are copied verbatim into configuration.json and published, so an
    unset or malformed value is dropped rather than recorded. A rejected value
    warns instead of raising: provenance metadata should never fail a benchmark
    that has already run.
    """
    raw = (os.environ.get(env_var) or "").strip()
    if not raw:
        return None
    if not _HUB_MODEL_ID_RE.match(raw):
        _warn(f"{env_var} is not an <org>/<name> Hub id; omitting it from configuration.json")
        return None
    return raw


def _is_sensitive_key(key, value=_UNSET):
    """Whether `key` names a secret.

    Names in `_ALWAYS_SENSITIVE_SUBSTRINGS` redact unconditionally. For the
    merely ambiguous ones, pass `value` when it is available: a scalar rules the
    field an engine knob rather than a credential. Callers that only have the
    name (e.g. argv scanning) omit it and every match redacts.
    """
    # Engine configs can carry non-string dict keys (e.g. int layer ids in a
    # serving_config); those are never sensitive field *names*, so skip them.
    if not isinstance(key, str):
        return False
    klow = key.lower()
    if any(s in klow for s in _ALWAYS_SENSITIVE_SUBSTRINGS):
        return True
    if klow in _SENSITIVE_KEY_ALLOWLIST:
        return False
    if not any(s in klow for s in _SENSITIVE_SUBSTRINGS):
        return False
    if value is _UNSET:
        return True
    return not isinstance(value, _NON_SECRET_VALUE_TYPES)


def _redact_value(value):
    """Recursively redact secrets in nested dict/list values.

    The top-level `_redact_config` walks one level of keys, but engine configs
    (serving_config from VLLMModel/SGLANGModel) and user-supplied runtime_params
    are nested arbitrarily — fields like `hf_token`, `tokenizer_revision`, or
    `aws_secret_access_key` need to be redacted at any depth.
    """
    if isinstance(value, dict):
        return {
            k: ("***REDACTED***" if _is_sensitive_key(k, v) else _redact_value(v))
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

    # The speculation width the engine was actually given, resolved from
    # whichever flag the algorithm reads. DFLASH is configured by --block_size
    # and ignores --draft_length, which stays at its default, so recording the
    # raw args makes a block_size=8 DFLASH run look like draft_length=3.
    #
    # The value is the engine's `num_speculative_tokens` verbatim, not the
    # count of draft tokens: models/vllm.py passes block_size straight through
    # for DFLASH, and draft_length straight through otherwise. Consumers should
    # read this field rather than re-deriving it from the two flags.
    block_size = getattr(args, "block_size", None)
    config["num_speculative_tokens"] = (
        block_size if block_size is not None else getattr(args, "draft_length", None)
    )

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
    config["huggingface_model_id"] = _hub_model_id("HUGGINGFACE_MODEL_ID")
    # Hub id of the external draft checkpoint, for algorithms that use one
    # (EAGLE3 / DRAFT_TARGET / DFLASH / DSPARK). `draft_model_dir` only records
    # a local path, which says nothing about which published drafter ran — two
    # drafters for the same verifier are otherwise indistinguishable. Empty for
    # in-model drafts (MTP heads).
    config["draft_huggingface_model_id"] = _hub_model_id("DRAFT_HUGGINGFACE_MODEL_ID")

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "configuration.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

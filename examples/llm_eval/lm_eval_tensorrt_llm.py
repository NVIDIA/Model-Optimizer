# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""[Deprecated] ModelOpt's TensorRT-LLM backend for lm-evaluation-harness.

lm-eval >= 0.4.12 ships its own TensorRT-LLM backend
(``lm_eval.models.trtllm_causallms``, registered as ``trtllm``), so this example no
longer maintains one. Migrate to:

    python lm_eval_hf.py --model trtllm \
        --model_args model=<quantized checkpoint dir>,tokenizer=<HF model folder>,\
tensor_parallel_size=<tp>,max_batch_size=<max batch size> \
        --tasks <comma separated tasks> --batch_size <max batch size>

This shim keeps legacy ``--model trt-llm`` commands working by translating them to the
upstream backend and forwarding to ``lm_eval_hf.py``. It will be removed in a future
release.
"""

import os
import sys
from pathlib import Path

# Legacy --model_args key -> upstream `trtllm` key. `tokenizer` and `trust_remote_code`
# carry over unchanged.
_RENAMED_ARGS = {"checkpoint_dir": "model"}

# The old wrapper sized the TensorRT-LLM engine from `max_length` (total sequence
# length), defaulting to max_gen_toks + 4096. Upstream splits that into an input and an
# output budget: max_seq_len = max_input_len + max_output_len.
_DEFAULT_MAX_GEN_TOKS = 256
_DEFAULT_MAX_INPUT_LEN = 4096


def _parse_model_args(raw: str) -> dict[str, str]:
    """Parse lm-eval's comma-separated ``key=value`` --model_args string."""
    args = {}
    for field in raw.split(","):
        if not field:
            continue
        key, _, value = field.partition("=")
        args[key.strip()] = value.strip()
    return args


def _translate_model_args(raw: str, batch_size: str | None) -> str:
    """Rewrite legacy `trt-llm` --model_args into upstream `trtllm` --model_args."""
    args = _parse_model_args(raw)
    args = {_RENAMED_ARGS.get(k, k): v for k, v in args.items()}

    max_gen_toks = int(args.get("max_gen_toks", _DEFAULT_MAX_GEN_TOKS))
    # Upstream defaults to max_input_len=2048, short enough to silently left-truncate
    # 5-shot prompts, so always derive it from the legacy sizing instead.
    args.setdefault("max_output_len", str(max_gen_toks))
    max_length = args.pop("max_length", None)
    args.setdefault(
        "max_input_len",
        str(int(max_length) - max_gen_toks) if max_length else str(_DEFAULT_MAX_INPUT_LEN),
    )

    # The old wrapper built the engine for --batch_size and used every visible GPU.
    if batch_size and batch_size.isdigit():
        args.setdefault("max_batch_size", batch_size)
    if "tensor_parallel_size" not in args:
        import torch

        args.setdefault("tensor_parallel_size", str(max(torch.cuda.device_count(), 1)))

    return ",".join(f"{k}={v}" for k, v in args.items())


def _find_option(argv: list[str], name: str) -> tuple[int, str] | None:
    """Locate ``--name <value>`` or ``--name=<value>``; return (index, value)."""
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return i + 1, argv[i + 1]
        if arg.startswith(f"{name}="):
            return i, arg[len(name) + 1 :]
    return None


def _set_option(argv: list[str], name: str, index: int, value: str) -> None:
    """Write *value* back at *index*, preserving the ``--name=<value>`` form."""
    argv[index] = f"{name}={value}" if argv[index].startswith(f"{name}=") else value


def _translate(argv: list[str]) -> list[str]:
    """Rewrite a legacy argv so it targets the upstream `trtllm` backend."""
    out = list(argv)
    # --batch_size sized the engine in the old wrapper; read it before rewriting.
    found = _find_option(out, "--batch_size")
    batch_size = found[1] if found else None

    found = _find_option(out, "--model")
    if found and found[1] in ("trt-llm", "trt_llm"):
        _set_option(out, "--model", found[0], "trtllm")

    found = _find_option(out, "--model_args")
    if found:
        _set_option(out, "--model_args", found[0], _translate_model_args(found[1], batch_size))
    return out


if __name__ == "__main__":
    forwarded = _translate(sys.argv[1:])
    target = Path(__file__).resolve().parent / "lm_eval_hf.py"
    print(
        "WARNING: examples/llm_eval/lm_eval_tensorrt_llm.py is deprecated and will be "
        "removed in a future release.\n"
        "         lm-eval >= 0.4.12 provides the `trtllm` backend; forwarding to:\n"
        f"         python {target.name} {' '.join(forwarded)}",
        file=sys.stderr,
    )
    os.execv(sys.executable, [sys.executable, str(target), *forwarded])

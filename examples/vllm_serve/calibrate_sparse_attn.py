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

"""Calibrate skip-softmax thresholds *through vLLM* and write the serving config.

Runs calibration prompts through a vLLM ``LLM`` whose attention layers use
``ModelOptSparseAttentionImpl`` in calibration mode (see
``sparse_attn_calib_worker.py``). The ModelOpt Triton calibration kernel
measures, per candidate threshold, how many KV tiles would be skipped — over
the paged KV cache, for both prefill and decode — then fits the exponential
model ``scale_factor = a * exp(b * sparsity)``.

The fitted ``(a, b)`` per phase are written as a ``sparse_attention_config``
block, the same one ``hf_sa.py`` produces, so ``vllm_serve_sparse_attn.py`` can
serve the calibrated model directly.

Usage:
    python calibrate_sparse_attn.py <ckpt> \
        --prompts_file prompts.txt \
        --target_sparse_ratio 0.5 \
        --decode_tokens 32 \
        --update_checkpoint_config

``--prompts_file`` is one prompt per line; longer, varied-length prompts give a
better fit. With no file, a tiny built-in demo set is used (fine for a smoke
test, not for a real fit).
"""

import argparse
import json
import os
import sys
from pathlib import Path

_DEMO_PROMPTS = [
    "Summarize the history of computing in a few paragraphs. " * 40,
    "Explain how attention works in transformer models. " * 60,
    "Write a detailed essay about renewable energy sources. " * 80,
]


def _load_prompts(prompts_file: str | None) -> list[str]:
    if prompts_file is None:
        print(
            "[ModelOpt] No --prompts_file given; using a tiny built-in demo set. "
            "Pass real, varied-length prompts for a usable fit."
        )
        return _DEMO_PROMPTS
    lines = [ln.strip() for ln in Path(prompts_file).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No prompts found in {prompts_file}")
    print(f"[ModelOpt] Loaded {len(lines)} calibration prompts from {prompts_file}")
    return lines


def _write_config(ckpt: str, sparse_config: dict, update_checkpoint: bool) -> None:
    """Dump the sparse_attention_config and optionally merge into config.json."""
    out_path = Path("sparse_attention_config.json")
    out_path.write_text(json.dumps(sparse_config, indent=2))
    print(f"[ModelOpt] Wrote calibrated config to {out_path.resolve()}")

    if not update_checkpoint:
        print(
            "[ModelOpt] Re-run with --update_checkpoint_config to merge this into "
            f"{ckpt}/config.json (required for vllm_serve_sparse_attn.py to pick it up)."
        )
        return

    config_json = Path(ckpt) / "config.json"
    config = json.loads(config_json.read_text())
    config["sparse_attention_config"] = sparse_config
    config_json.write_text(json.dumps(config, indent=2))
    print(f"[ModelOpt] Merged sparse_attention_config into {config_json}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate skip-softmax thresholds via vLLM")
    parser.add_argument("model", type=str, help="Path to the HF checkpoint to calibrate")
    parser.add_argument("--prompts_file", type=str, default=None, help="One prompt per line")
    parser.add_argument(
        "--target_sparse_ratio",
        type=float,
        default=0.5,
        help="Target sparsity baked into the exported config (applied to both phases)",
    )
    parser.add_argument(
        "--decode_tokens",
        type=int,
        default=32,
        help="Decode tokens to generate per prompt (drives decode-phase calibration)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None, help="vLLM max_model_len override"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="vLLM tensor-parallel size"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization fraction",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for custom model classes (e.g. NemotronH)",
    )
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype, e.g. bfloat16")
    parser.add_argument(
        "--fit_logspace",
        action="store_true",
        help="Fit the exponential model in log space (wide scale_factor ranges)",
    )
    parser.add_argument(
        "--update_checkpoint_config",
        action="store_true",
        help="Merge the calibrated config into <ckpt>/config.json in place",
    )
    args = parser.parse_args()

    # Workers run in separate processes and must import the calibration worker.
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    current = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([current, repo_root]) if current else repo_root

    from vllm import LLM, SamplingParams

    prompts = _load_prompts(args.prompts_file)

    llm_kwargs = {
        "model": args.model,
        "worker_cls": "sparse_attn_calib_worker.SparseAttnCalibWorker",
        # Calibration swaps the attention impl per layer; eager avoids CUDA-graph
        # capture of the (now Triton) attention path.
        "enforce_eager": True,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.tensor_parallel_size and args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True
    if args.dtype is not None:
        llm_kwargs["dtype"] = args.dtype
    # NOTE: no attention_backend override — the calib worker auto-detects the
    # layer backend (FlashAttention / FlashInfer) and swaps in the matching
    # sparse impl. NemotronH uses FlashInfer by default, which the worker
    # supports via patch_flashinfer_metadata_builder().
    llm = LLM(**llm_kwargs)

    n_layers = llm.collective_rpc("sparse_calib_enable")[0]
    print(f"[ModelOpt] Calibration enabled on {n_layers} attention layers")

    # generate() drives prefill (prefill-phase stats) then decode_tokens decode
    # steps (decode-phase stats). The calibration kernel computes full attention,
    # so the generated text is unaffected — only tile-skip counts are recorded.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens)
    llm.generate(prompts, sampling)

    sparse_config = llm.collective_rpc(
        "sparse_calib_fit",
        args=({"prefill": args.target_sparse_ratio, "decode": args.target_sparse_ratio},),
        kwargs={"fit_logspace": args.fit_logspace},
    )[0]

    if sparse_config is None:
        print(
            "[ModelOpt] Calibration produced no valid fit — try more/longer prompts "
            "so observed sparsity spans the (10%, 90%) fitting window."
        )
        return

    print("[ModelOpt] Calibrated threshold_scale_factor:")
    print(json.dumps(sparse_config["threshold_scale_factor"], indent=2))
    _write_config(args.model, sparse_config, args.update_checkpoint_config)


if __name__ == "__main__":
    main()

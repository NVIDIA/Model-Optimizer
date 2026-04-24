# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""DeepSeek-V4 PTQ — NVFP4 on the routed experts only, everything else native.

Design notes (in contrast to ``examples/deepseek/ptq.py`` which covers V3):

  * We do **not** register Quant wrappers against DeepSeek-V4's ``Linear`` /
    ``ColumnParallelLinear`` / ``RowParallelLinear`` classes. Wrapping those
    globally would force every linear (attention projections, gate, shared
    expert, lm_head) through a BF16 dequant + ``F.linear`` path with
    pass-through quantizers attached — that changes the compute kernels for
    layers the user explicitly asked to leave untouched.

  * Instead we register a wrapper **only** against the routed-expert module
    (``deekseep_v4_model.Expert``). That wrapper installs per-weight and per-
    input ``TensorQuantizer`` pairs for ``w1``, ``w2``, ``w3`` and redefines
    ``forward`` to dequantize each MXFP4-packed expert weight to BF16 on the
    fly (via ``MXFP4QTensor.dequantize_packed``) before the ``F.linear`` call
    that the quantizers hook into.

  * The shared expert (``MoE.shared_experts``) is also an ``Expert`` instance,
    so it gets the wrapper too — but we disable its quantizers by config
    pattern (``*shared_experts*``) so no amax gets collected and its output
    remains numerically equivalent to the un-wrapped path. (It still pays the
    BF16 dequant cost during calibration; the savings show up at inference
    time where shared_experts' weights on disk are unchanged and downstream
    inference uses the native FP4/FP8 GEMMs.)

  * Router gate weights, attention, dense projections, lm_head, embeddings —
    untouched. Their forward path uses V4's native ``linear()`` dispatch
    which routes to ``fp4_gemm`` / ``fp8_gemm`` / ``F.linear`` based on the
    weight dtype on disk.

  * A ``CalibMoE`` wrapper overrides ``MoE.forward`` during calibration to
    route every token through every local routed expert (top_k =
    n_routed_experts), so every expert's quantizers see calibration data.
    It then re-runs with the real top_k for downstream outputs.

Usage (single node, 4 GPUs, MP=4):

    torchrun --nproc-per-node 4 --master_port 12346 ptq.py \\
        --model_path  /path/to/DeepSeek-V4-Pro-mp4-mxfp4 \\
        --config      /path/to/DeepSeek-V4-Pro/inference/config.json \\
        --output_path /path/to/amax_dump

For MP=8 across two nodes use torchrun's ``--nnodes=2 --node_rank=<i>
--master_addr=<ip>`` flags.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model
from tqdm import tqdm
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor.mxfp4_tensor import MXFP4QTensor
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader


_DEFAULT_V4_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "models/DeepSeek-V4-Pro/inference"
)


def _inject_v4_module(v4_inference_dir: Path) -> None:
    assert v4_inference_dir.exists(), (
        f"DeepSeek-V4 inference dir not found at {v4_inference_dir}; "
        "pass --dsv4_inference_dir"
    )
    sys.path.insert(0, str(v4_inference_dir))


# Populated by ``install_quant_registry`` once DS-V4's ``model`` is importable.
deekseep_v4_model = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# BF16 dequant of a single DeepSeek-V4 Linear's weight
# ---------------------------------------------------------------------------


def _dequantize_linear_weight(linear_module) -> torch.Tensor:
    """Materialize a BF16 copy of a DS-V4 ``Linear``'s weight.

    Routed experts are the only path we actually hit here:
      * ``float4_e2m1fn_x2`` + paired UE8M0 1x32 scale → MXFP4 dequant.
      * Anything else                                    → passthrough (shared
        experts are allocated as BF16 by ``MoE.__init__`` because no dtype is
        passed to their ``Expert`` constructor, so they land here already BF16).

    FP8 E4M3 + UE8M0 128x128 is intentionally *not* handled — ModelOpt's
    ``weight_dequant`` expects FP32 scales, not UE8M0, and that path is dead
    code for this wrapper (non-expert Linears are never swapped to QuantExpert).
    """
    w = linear_module.weight
    if w.dtype == torch.float4_e2m1fn_x2:
        # DS-V4's Linear attaches ``scale`` as both ``self.scale`` and
        # ``self.weight.scale`` — they're the same nn.Parameter object.
        return MXFP4QTensor.dequantize_packed(
            w, w.scale, block_size=32, dtype=torch.bfloat16
        )
    return w


# ---------------------------------------------------------------------------
# Quant wrappers: Expert + CalibMoE
# ---------------------------------------------------------------------------


def install_quant_registry() -> None:
    """Import DS-V4's ``model`` module and register minimal Quant wrappers."""
    global deekseep_v4_model
    import model as _m  # noqa: PLC0415

    deekseep_v4_model = _m

    class QuantExpert(deekseep_v4_model.Expert):
        """Routed expert with per-``w{1,2,3}`` input & weight quantizers.

        Forward mirrors ``Expert.forward`` (SwiGLU with optional clipping and
        optional per-token weight), but each ``w{1,2,3}`` call goes through
        our ``_qlinear`` which dequantizes on the fly and exposes the hook
        points the quantizers need.

        ``TensorQuantizer`` instances are installed in ``_setup`` (not
        ``__init__``) because ModelOpt's ``DynamicModule.convert`` patches
        ``__class__`` in place and calls ``_setup(**setup_kwargs)`` — it does
        not invoke ``__init__``.
        """

        def _setup(self):
            self.w1_input_quantizer = TensorQuantizer()
            self.w1_weight_quantizer = TensorQuantizer()
            self.w2_input_quantizer = TensorQuantizer()
            self.w2_weight_quantizer = TensorQuantizer()
            self.w3_input_quantizer = TensorQuantizer()
            self.w3_weight_quantizer = TensorQuantizer()

        @staticmethod
        def _qlinear(
            x: torch.Tensor,
            linear_module,
            input_quantizer: TensorQuantizer,
            weight_quantizer: TensorQuantizer,
        ) -> torch.Tensor:
            w = _dequantize_linear_weight(linear_module)
            x = input_quantizer(x)
            w = weight_quantizer(w)
            return F.linear(x, w, linear_module.bias)

        def forward(self, x, weights=None):  # type: ignore[override]
            dtype = x.dtype
            gate = self._qlinear(
                x, self.w1, self.w1_input_quantizer, self.w1_weight_quantizer
            ).float()
            up = self._qlinear(
                x, self.w3, self.w3_input_quantizer, self.w3_weight_quantizer
            ).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            y = F.silu(gate) * up
            if weights is not None:
                y = weights * y
            return self._qlinear(
                y.to(dtype), self.w2, self.w2_input_quantizer, self.w2_weight_quantizer
            )

    class CalibMoE(deekseep_v4_model.MoE):
        """During calibration, route every token through every local routed
        expert so all ``w*_{input,weight}_quantizer`` instances see data.
        Then run the real MoE forward so the model output is correct.

        Empty ``_setup`` because this wrapper installs no quantizer state;
        it exists solely to override ``forward``. ``DynamicModule.convert``
        still requires the method to exist."""

        def _setup(self):
            pass

        def forward(self, x, input_ids):  # type: ignore[override]
            gate = self.gate
            orig_topk = gate.topk
            try:
                gate.topk = self.n_routed_experts
                super().forward(x, input_ids)
            finally:
                gate.topk = orig_topk
            return super().forward(x, input_ids)

    mtq.register(original_cls=deekseep_v4_model.Expert, quantized_cls=QuantExpert)
    mtq.register(original_cls=deekseep_v4_model.MoE, quantized_cls=CalibMoE)


# ---------------------------------------------------------------------------
# Model load + calibration
# ---------------------------------------------------------------------------


def load_deepseek_v4(model_config: str, model_path: str, batch_size: int):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    with open(model_config) as f:
        margs = deekseep_v4_model.ModelArgs(**json.load(f))
        margs.max_batch_size = max(batch_size, margs.max_batch_size)
    with torch.device("cuda"):
        model = deekseep_v4_model.Transformer(margs)
    ckpt = os.path.join(model_path, f"model{rank}-mp{world_size}.safetensors")
    print(f"[rank {rank}] loading {ckpt}")
    load_model(model, ckpt, strict=False)
    print(f"[rank {rank}] loaded")
    return model


def _build_nvfp4_experts_cfg() -> dict:
    """Quant config: NVFP4 weight + NVFP4 input, routed experts only.

    Routed experts live at ``model.layers.<i>.ffn.experts.<j>.w{1,2,3}``. The
    shared expert lives at ``model.layers.<i>.ffn.shared_experts.w{1,2,3}``
    — same ``Expert`` class, different MoE attribute name — and we disable
    its quantizers explicitly.
    """
    nvfp4 = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    }
    return {
        "quant_cfg": [
            # Start with everything disabled so non-Expert modules are never
            # touched — since we only registered Quant wrappers for Expert
            # and MoE, there are no other quantizer instances anyway, but
            # keep the explicit baseline as a safety rail.
            {"quantizer_name": "*input_quantizer", "enable": False},
            {"quantizer_name": "*weight_quantizer", "enable": False},
            # Re-enable only routed experts (``ffn.experts.<idx>.w*``).
            {
                "quantizer_name": "*ffn.experts.*.w*_weight_quantizer",
                "enable": True,
                "cfg": copy.deepcopy(nvfp4),
            },
            {
                "quantizer_name": "*ffn.experts.*.w*_input_quantizer",
                "enable": True,
                "cfg": copy.deepcopy(nvfp4),
            },
            # Belt-and-suspenders: shared expert lives under the same Expert
            # class; make sure it's disabled even if patterns above matched.
            {"quantizer_name": "*shared_experts*", "enable": False},
        ],
        "algorithm": "max",
    }


def ptq(model, tokenizer, batch_size: int, calib_size: int):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    device = next(model.parameters()).device
    calib_dataset = get_dataset_dataloader(
        dataset_name=["cnn_dailymail", "nemotron-post-training-dataset-v2"],
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=[calib_size, calib_size],
        device=device,
    )

    def calibrate_loop(model):
        for data in tqdm(calib_dataset, disable=(world_size > 1 and rank != 0)):
            model(data["input_ids"])

    if world_size > 1:
        dist.barrier()

    mtq_cfg = _build_nvfp4_experts_cfg()
    model = mtq.quantize(model, mtq_cfg, calibrate_loop)
    if rank == 0:
        mtq.print_quant_summary(model)
    return model


# ---------------------------------------------------------------------------
# amax + hf_quant_config dump
# ---------------------------------------------------------------------------


def save_amax_and_quant_config(model, output_path: str):
    """Save routed-expert quantizer state + a manifest enumerating the
    quantized layer paths. The manifest is built by scanning the model for
    ``TensorQuantizer`` instances whose path contains ``.experts.<n>.w``
    (i.e. routed-expert quantizers only, excluding ``shared_experts``); we do
    *not* rely on ``modelopt.torch.export.quant_utils.get_quant_config``
    because its introspection doesn't see weights stored on nested
    submodules — ``QuantExpert``'s ``w{1,2,3}`` are submodules of the
    container, not direct parameters. The downstream export script
    (``quantize_to_nvfp4.py``) uses this manifest as ground truth for which
    tensor paths to replace with NVFP4 packed weight + scales.
    """
    import re as _re

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    # Dump only routed-expert quantizer state (skip any stray shared_experts
    # or other quantizer state attached by mtq's pattern matcher).
    expert_re = _re.compile(r"\.experts\.\d+\.w[123]_")
    state = {
        k: v
        for k, v in model.state_dict().items()
        if expert_re.search(k) and ("amax" in k or "quant" in k)
    }
    torch.save(state, os.path.join(output_path, f"amax_dict_rank{rank}-mp{world_size}.pt"))

    # Enumerate quantized layer tensor paths so the export script knows
    # exactly which weights to replace with NVFP4 packed + scales.
    quantized_layers: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and expert_re.search(name):
            # Strip trailing ``._weight_quantizer`` / ``._input_quantizer`` and
            # the ``w1_/w2_/w3_`` prefix to get the DS-native Linear path.
            # e.g. ``layers.0.ffn.experts.5.w1_weight_quantizer`` →
            #      ``layers.0.ffn.experts.5.w1``.
            base = name.rsplit(".", 1)[0]
            proj = name.rsplit(".", 1)[1].split("_")[0]  # "w1"|"w2"|"w3"
            quantized_layers.add(f"{base}.{proj}")

    manifest = {
        "quantization_format": "NVFP4_W4A4",
        "quantized_layers": sorted(quantized_layers),
        "world_size": world_size,
        "layer_cfg": {
            "num_bits": [2, 1],
            "block_size": 16,
            "scale_bits": [4, 3],
        },
    }
    if world_size > 1:
        all_manifests: list = [None] * world_size
        dist.all_gather_object(all_manifests, manifest)
    else:
        all_manifests = [manifest]
    if rank == 0:
        merged: set[str] = set()
        for m in all_manifests:
            assert m is not None
            merged.update(m["quantized_layers"])
        manifest["quantized_layers"] = sorted(merged)
        with open(os.path.join(output_path, "quantized_layers_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model_path", required=True, help="MP-sharded DS-V4 checkpoint dir")
    p.add_argument("--config", required=True, help="DS-V4 ModelArgs JSON")
    p.add_argument(
        "--dsv4_inference_dir",
        type=Path,
        default=_DEFAULT_V4_DIR,
        help="dir containing DS-V4 inference/ model.py + kernel.py",
    )
    p.add_argument("--output_path", required=True, help="where to dump amax + hf_quant_config.json")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--calib_size", type=int, default=64)
    p.add_argument("--trust_remote_code", action="store_true")
    args = p.parse_args()

    _inject_v4_module(args.dsv4_inference_dir)
    install_quant_registry()
    model = load_deepseek_v4(args.config, args.model_path, args.batch_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = ptq(model, tokenizer, args.batch_size, args.calib_size)
    save_amax_and_quant_config(model, args.output_path)


if __name__ == "__main__":
    main()

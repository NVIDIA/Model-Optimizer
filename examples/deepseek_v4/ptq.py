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
from modelopt.torch.utils.distributed import ParallelState


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


def _fp8_ue8m0_blockwise_to_bf16(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """FP8 E4M3 × UE8M0 128x128 block-scale → BF16 (V4 native FP8 layout).

    Same math as ``scripts/convert_dsv4_to_bf16.py``'s helper; ModelOpt's
    triton ``weight_dequant`` expects FP32 scales and cannot consume UE8M0
    directly, so we dequant inline.
    """
    M, N = weight.shape
    assert M % block == 0 and N % block == 0, f"FP8 weight shape {(M, N)} not divisible by {block}"
    assert scale.shape == (M // block, N // block), (
        f"FP8 scale shape {tuple(scale.shape)} != expected ({M // block}, {N // block})"
    )
    exp = scale.contiguous().view(torch.uint8).to(torch.int32) - 127
    exp = exp.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return torch.ldexp(weight.to(torch.float32), exp).to(torch.bfloat16)


def _dequantize_linear_weight(linear_module) -> torch.Tensor:
    """Materialize a BF16 copy of a DS-V4 ``Linear``'s weight.

    Three on-disk formats are possible here. Both routed and shared experts
    are wrapped (both are ``Expert`` instances); their weight dtype depends
    on V4's global default (``set_dtype`` typically sets it to fp8) and on
    the optional ``dtype=`` kwarg passed to ``Expert.__init__``:
      * ``float4_e2m1fn_x2`` + UE8M0 1x32 scale   → MXFP4 dequant.
      * ``float8_e4m3fn``    + UE8M0 128x128 scale → FP8 dequant (inline).
      * any float dtype without a ``.scale``       → passthrough (already BF16).
    """
    w = linear_module.weight
    if w.dtype == torch.float4_e2m1fn_x2:
        return MXFP4QTensor.dequantize_packed(w, w.scale, block_size=32, dtype=torch.bfloat16)
    if w.dtype == torch.float8_e4m3fn:
        return _fp8_ue8m0_blockwise_to_bf16(w, w.scale, block=128)
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
            # Tell modelopts post-calibration amax sync not to do any
            # cross-rank all_reduce on these quantizers. Routed experts
            # are sharded across ranks (rank i has experts i*n_local ..
            # (i+1)*n_local-1), so the same quantizer path exists on
            # only one rank. The default parallel_state treats all ranks
            # as one group, which deadlocks when some ranks have amax
            # populated (routed) and others do not.
            self._parallel_state = ParallelState(
                data_parallel_group=-1,
                tensor_parallel_group=-1,
            )

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
        """During calibration, force ``gate.topk = n_routed_experts`` so every
        token routes to every local routed expert and all
        ``w*_{input,weight}_quantizer`` instances see data.

        We do NOT do a second "real" forward pass — MoE has an internal
        ``dist.all_reduce(y)`` and the double-forward form kept collectives
        misaligned when TileLang JIT compile times varied across ranks,
        producing 10-minute NCCL timeouts. Calibration output is discarded
        by ``calibrate_loop`` anyway so a single all-expert forward is
        sufficient.

        Empty ``_setup`` because this wrapper installs no quantizer state;
        it exists solely to override ``forward``. ``DynamicModule.convert``
        still requires the method to exist."""

        def _setup(self):
            pass

        def forward(self, x, input_ids):  # type: ignore[override]
            gate = self.gate
            orig_topk = gate.topk
            gate.topk = self.n_routed_experts
            try:
                return super().forward(x, input_ids)
            finally:
                gate.topk = orig_topk

    mtq.register(original_cls=deekseep_v4_model.Expert, quantized_cls=QuantExpert)
    # CalibMoE registration is skipped for first-sanity runs — see
    # --calib_all_experts flag below. Registering it forces every token
    # through every local expert which populates amax on all experts but
    # multiplies per-batch work ~64x and surfaces NCCL-timing issues on
    # multi-node runs. Default behavior: natural top-k routing, fewer
    # experts observed per batch.
    if os.environ.get("CALIB_ALL_EXPERTS", "0") == "1":
        mtq.register(original_cls=deekseep_v4_model.MoE, quantized_cls=CalibMoE)


# ---------------------------------------------------------------------------
# Model load + calibration
# ---------------------------------------------------------------------------


def load_deepseek_v4(model_config: str, model_path: str, batch_size: int, dummy_weights: bool = False):
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

    if dummy_weights:
        print(f"[rank {rank}] --dummy-weights: skipping load_model; params remain at torch.empty values")
    else:
        ckpt = os.path.join(model_path, f"model{rank}-mp{world_size}.safetensors")
        print(f"[rank {rank}] loading {ckpt}")
        load_model(model, ckpt, strict=False)
        print(f"[rank {rank}] loaded")
    # Match generate.py: default device to this rank's CUDA so that any
    # torch.empty(...) / torch.zeros(...) inside the V4 forward (TileLang
    # kernels verify device_type==cuda and device_id==local_rank on their
    # inputs) lands on the right GPU.
    torch.set_default_device("cuda")
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

    def _trace(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    _trace("ptq() entered")
    device = next(model.parameters()).device
    _trace(f"device={device}, building calib dataloader")
    calib_dataset = get_dataset_dataloader(
        dataset_name=["cnn_dailymail", "nemotron-post-training-dataset-v2"],
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=[calib_size, calib_size],
        device=device,
    )
    _trace("calib dataloader ready")

    def calibrate_loop(model):
        import time as _time
        _trace("calibrate_loop: entering")
        t_loop = _time.time()
        for i, data in enumerate(calib_dataset):
            t0 = _time.time()
            model(data["input_ids"])
            dt = _time.time() - t0
            _trace(f"calibrate_loop: batch {i} shape={tuple(data['input_ids'].shape)} dt={dt:.2f}s")
        _trace(f"calibrate_loop: exited after {_time.time() - t_loop:.1f}s")

    if world_size > 1:
        _trace("pre-calib barrier")
        dist.barrier()
        _trace("pre-calib barrier done")

    mtq_cfg = _build_nvfp4_experts_cfg()
    _trace("calling mtq.quantize")
    model = mtq.quantize(model, mtq_cfg, calibrate_loop)
    _trace("mtq.quantize returned")
    if rank == 0:
        mtq.print_quant_summary(model)
    _trace("ptq() returning")
    return model


# ---------------------------------------------------------------------------
# amax + hf_quant_config dump
# ---------------------------------------------------------------------------


def save_amax_and_quant_config(model, output_path: str):
    import sys as _sys
    def _trace(msg):
        print(f"[rank {os.getenv('RANK','0')}] SAVE: {msg}", flush=True)
    _trace("entered")
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
        _trace("pre-save barrier")
        dist.barrier()
        _trace("pre-save barrier done")

    # Dump only routed-expert quantizer state (skip any stray shared_experts
    # or other quantizer state attached by mtq's pattern matcher).
    _trace("building filtered state dict")
    expert_re = _re.compile(r"\.experts\.\d+\.w[123]_")
    full_sd = model.state_dict()
    _trace(f"full state_dict size={len(full_sd)}")
    state = {
        k: v
        for k, v in full_sd.items()
        if expert_re.search(k) and ("amax" in k or "quant" in k)
    }
    _trace(f"filtered state size={len(state)}, saving")
    torch.save(state, os.path.join(output_path, f"amax_dict_rank{rank}-mp{world_size}.pt"))
    _trace("torch.save done")

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
    p.add_argument(
        "--dummy_weights",
        action="store_true",
        help="skip load_model (fast iteration on wiring; weights are uninitialized)",
    )
    p.add_argument(
        "--run_generate",
        type=str,
        default=None,
        help="after calibration, generate from this prompt string (rank 0 prints) "
        "to smoke-test that the quantized forward produces coherent text.",
    )
    p.add_argument("--max_new_tokens", type=int, default=16)
    args = p.parse_args()

    _inject_v4_module(args.dsv4_inference_dir)
    install_quant_registry()
    model = load_deepseek_v4(
        args.config, args.model_path, args.batch_size, dummy_weights=args.dummy_weights
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = ptq(model, tokenizer, args.batch_size, args.calib_size)
    save_amax_and_quant_config(model, args.output_path)

    if args.run_generate is not None:
        _run_quantized_generate(model, tokenizer, args.run_generate, args.max_new_tokens)


def _run_quantized_generate(model, tokenizer, prompt: str, max_new_tokens: int):
    """Smoke-test: generate tokens from a prompt through the calibrated+quantized
    model. If the output decodes to coherent text, the quantized forward works
    end-to-end."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    tokens = torch.full((1, input_ids.shape[1] + max_new_tokens), -1, dtype=torch.long, device="cuda")
    tokens[0, : input_ids.shape[1]] = input_ids[0]
    prev_pos = 0
    with torch.inference_mode():
        for cur_pos in range(input_ids.shape[1], tokens.shape[1]):
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            next_token = logits.argmax(dim=-1)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
    if rank == 0:
        out = tokens[0, input_ids.shape[1]:].tolist()
        text = tokenizer.decode(out, skip_special_tokens=False)
        print(f"\n[generate] prompt: {prompt!r}\n[generate] completion: {text!r}\n", flush=True)


if __name__ == "__main__":
    main()

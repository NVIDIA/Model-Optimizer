#!/usr/bin/env python
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

"""End-to-end pipeline for Qwen2.5-1.5B-Instruct:

    [2:4 weight sparsity]  ->  INT8 W8A8 SmoothQuant PTQ  ->  [QAT]
      ->  finalize  ->  torch->ONNX export (opset 20)
      ->  TensorRT engine build (trtexec, structured sparsity enabled)
      ->  validate sparse INT8 kernels  ->  real inference (text-in -> text-out).

Sparsity (--sparsity) and QAT (--qat) are OPTIONAL and OFF by default; the default run is
plain INT8 W8A8 SmoothQuant, which preserves accuracy (coherent generation).

WARNING: one-shot 2:4 magnitude sparsity zeros half the weights and causes SEVERE accuracy
degradation by itself -- the model produces gibberish until recovered with QAT/SAT fine-tuning.
So --sparsity is only useful together with --qat (and realistically a longer recovery run than
this smoke-level QAT). When sparsity is on, ordering is the proven one
(examples/llm_sparsity/weight_sparsity): sparsify FIRST, then quantize so SmoothQuant calibrates
on the sparse weights, then QAT.

Tested with (Docker container + library versions/commits):
  - Docker container: nvcr.io/nvidia/pytorch:26.01-py3
  - PyTorch:          2.10.0a0+a36e1d39eb  (git commit a36e1d39eb)
  - ONNX:             1.18.0
  - TensorRT:         10.14.1.48  (trtexec v101401)
  - CUDA:             13.1
  - NVIDIA ModelOpt:  0.45.0rc0  (this repository, installed editable)
  - transformers:     5.9.0  (supported range >=4.56,<5.10)
  - accelerate:       1.13.0
  - Model:            Qwen/Qwen2.5-1.5B-Instruct
  - GPU used:         NVIDIA RTX 6000 Ada Generation (sm_89)

ModelOpt is installed editable in the container; transformers/accelerate and the ONNX-export
helper deps (onnxruntime, onnx-graphsurgeon, onnxconverter-common, onnxslim) are pip-installed
without disturbing the container's torch 2.10 / onnx 1.18.
"""

import argparse
import contextlib
import json
import os
import re
import subprocess
import time

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- helpers
BANNER = "=" * 92


def log(msg: str) -> None:
    print(f"\n{BANNER}\n# {msg}\n{BANNER}", flush=True)


def linear_weight_zero_fraction(model: nn.Module, needle: str = "q_proj"):
    """Return zero-fraction of the first decoder Linear matching `needle` (effective weight)."""
    import modelopt.torch.sparsity as mts

    for name, mod in model.named_modules():
        if (
            needle in name
            and hasattr(mod, "weight")
            and isinstance(getattr(mod, "weight"), torch.Tensor)
        ):
            w = mod.weight
            mask = getattr(mod, "_weight_mask", None)
            if isinstance(mod, getattr(mts, "SparseModule", ())) and mask is not None:
                w = w * mask
            return name, float((w == 0).float().mean().item()), tuple(w.shape)
    return None, None, None


def count_enabled_weight_quantizers(model: nn.Module) -> int:
    n = 0
    for _, mod in model.named_modules():
        q = getattr(mod, "weight_quantizer", None)
        if q is not None and getattr(q, "is_enabled", False):
            n += 1
    return n


# ----------------------------------------------------------------------------- stages
def load_model(model_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp32 for the torch stages: the post-training-sparsified (un-recovered) model has large
    # intermediate activations that overflow to NaN in fp16 during SmoothQuant calibration.
    # fp32 has full dynamic range -> stable calibration/QAT. ONNX export later converts weights
    # to fp16 via weights_dtype="fp16" (matches the proven timm INT8 example). Eager attention
    # keeps the graph plain matmul+softmax so it exports cleanly to ONNX (no flash/sdpa op).
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.float32, attn_implementation="eager"
        )
        .to("cuda")
        .eval()
    )
    model.config.use_cache = False
    return model, tok


def build_calib_forward_loop(model, tok, num_samples, calib_seq, dataset_name, batch_size):
    """SmoothQuant calibration forward_loop, mirroring examples/llm_ptq/hf_ptq.py:

        calib_dataloader = get_dataset_dataloader(dataset_name, tokenizer, batch_size,
                                                  num_samples, max_sample_length=calib_seq, device)
        calibrate_loop    = create_forward_loop(dataloader=calib_dataloader)

    Supports cnn_dailymail / nemotron-* datasets (see get_supported_datasets()). Falls back to
    synthetic prompts only if the dataset can't be fetched.
    """
    from modelopt.torch.utils.dataset_utils import (
        create_forward_loop,
        get_dataset_dataloader,
        get_supported_datasets,
    )

    try:
        supported = get_supported_datasets()
        if dataset_name not in supported:
            print(
                f"[calib] '{dataset_name}' not in supported {supported}; using cnn_dailymail",
                flush=True,
            )
            dataset_name = "cnn_dailymail"
        calib_dataloader = get_dataset_dataloader(
            dataset_name=dataset_name,
            tokenizer=tok,
            batch_size=batch_size,
            num_samples=num_samples,
            max_sample_length=calib_seq,
            device="cuda",
        )
        forward_loop = create_forward_loop(dataloader=calib_dataloader)
        # materialize once so a fetch failure surfaces here (and we can fall back)
        n_batches = len(calib_dataloader)
        print(
            f"[calib] dataset={dataset_name} samples={num_samples} seq={calib_seq} "
            f"batch_size={batch_size} -> {n_batches} batches",
            flush=True,
        )
        return forward_loop, dataset_name
    except Exception as e:  # pragma: no cover - network/dataset fallback
        print(
            f"[calib] dataset '{dataset_name}' unavailable ({e}); using synthetic prompts",
            flush=True,
        )
        prompts = _synthetic_prompts(num_samples)
        enc = tok(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=calib_seq,
        )
        ids = enc["input_ids"]

        def forward_loop(m):
            with torch.no_grad():
                for i in range(0, ids.shape[0], batch_size):
                    m(input_ids=ids[i : i + batch_size].to("cuda"))

        return forward_loop, "synthetic"


def _synthetic_prompts(n):
    base = [
        "The history of artificial intelligence spans many decades of research.",
        "In quantum mechanics, the wave function describes the state of a system.",
        "Climate change is driven by greenhouse gas emissions from human activity.",
        "The recipe calls for flour, sugar, eggs, and a pinch of salt.",
        "Neural networks learn representations from large amounts of data.",
        "The stock market reacted sharply to the central bank's announcement.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "She traveled across the country to visit the ancient ruins.",
    ]
    out = []
    while len(out) < n:
        for i, b in enumerate(base):
            out.append(f"{b} This is calibration example number {len(out)}, paragraph {i}.")
            if len(out) >= n:
                break
    return out


def apply_sparsity(model):
    import modelopt.torch.sparsity as mts

    out = mts.sparsify(model, "sparse_magnitude", config=None)
    model = out[0] if isinstance(out, tuple) else out
    name, zf, shape = linear_weight_zero_fraction(model, "q_proj")
    print(f"[sparsity] sample {name} effective zero-frac={zf:.4f} shape={shape}", flush=True)
    assert zf is not None and 0.45 <= zf <= 0.55, f"2:4 sparsity not applied (zero-frac={zf})"
    return model


def apply_ptq(model, forward_loop, quant_output=False):
    import copy

    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    if quant_output:
        # Enable INT8 output quantizers so every GEMM has an INT8-out epilogue. Without this,
        # each Linear's consumer (attention / SiLU / residual+layernorm) is unquantized, so the
        # GEMM dequantizes to fp32 -> the dense-favorable epilogue. This tests whether the fp32
        # output (not the K=1536 shape) is what keeps TRT from choosing sparse INT8 kernels.
        cfg["quant_cfg"].append(
            {"quantizer_name": "*output_quantizer", "cfg": {"num_bits": 8, "axis": None}}
        )
        print("[ptq] INT8 output quantizers ENABLED (GEMMs INT8-out)", flush=True)
    model = mtq.quantize(model, cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(model)
    nwq = count_enabled_weight_quantizers(model)
    print(f"[ptq] enabled weight quantizers: {nwq}", flush=True)
    assert nwq > 0, "no weight quantizers enabled after PTQ"
    return model


def run_qat(model, tok, steps, seq_len, lr):
    """Minimal EXAMPLE QAT (quantization-aware training) loop -- NOT production training.

    This is a smoke-level placeholder: a few steps of next-token cross-entropy on a handful of
    cnn_dailymail samples, just to exercise the QAT code path (the quantizer amax values stay
    frozen and ModelOpt's fake-quant + the 2:4 sparsity mask are applied on every forward).

    To actually recover accuracy -- which is REQUIRED after 2:4 sparsification -- replace the body
    below with YOUR OWN dataset and training pipeline: your task-appropriate training/instruction
    data, loss, optimizer/scheduler, batch size, sequence length, and a realistic number of
    steps/epochs (and ideally a held-out eval). The surrounding pipeline (mtq.quantize before this,
    finalize/export after) stays the same; only this training loop is meant to be swapped out.
    Distributed training (DDP/FSDP) and HF Trainer also work -- see examples/llm_qat and
    examples/llm_sparsity/weight_sparsity/finetune.py for fuller references.
    """
    from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

    try:
        dl = get_dataset_dataloader(
            dataset_name="cnn_dailymail",
            tokenizer=tok,
            batch_size=1,
            num_samples=max(steps, 8),
            max_sample_length=seq_len,
            include_labels=True,
            device="cuda",
        )
        batches = list(dl)
    except Exception as e:  # pragma: no cover
        print(f"[qat] dataset unavailable ({e}); using synthetic", flush=True)
        enc = tok(
            _synthetic_prompts(max(steps, 8)),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        batches = [
            {"input_ids": enc["input_ids"][i : i + 1]} for i in range(enc["input_ids"].shape[0])
        ]

    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    done = 0
    for batch in batches:
        if done >= steps:
            break
        ids = batch["input_ids"].to("cuda")
        attn = batch.get("attention_mask")
        kwargs = {"input_ids": ids, "labels": ids}
        if attn is not None:
            kwargs["attention_mask"] = attn.to("cuda")
        out = model(**kwargs)
        loss = out.loss
        if not torch.isfinite(loss):
            print(f"[qat] step {done}: non-finite loss, skipping", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        done += 1
        if done % 5 == 0 or done == 1:
            print(f"[qat] step {done}/{steps} loss={loss.item():.4f}", flush=True)
    model.eval()
    print(f"[qat] completed {done} steps", flush=True)
    return model


def finalize_for_export(model, expect_sparse=True):
    """Prepare the sparse+quant model for ONNX export.

    We deliberately do NOT call ``mts.export``: ModelOpt's mode system unwinds modes LIFO, so
    with ``quantize`` applied on top of ``sparse_magnitude`` the sparse export is blocked until
    quantize is exported first — and exporting quantize would strip the fake-quant QDQ we need.
    For the ONNX-QDQ -> TensorRT path we keep both modes live: the SparseModule's dynamic
    ``weight*mask`` getter feeds the 2:4 zeros into the torch.onnx trace, and the TensorQuantizer
    emits QuantizeLinear/DequantizeLinear. As insurance, fold the mask into the underlying weight
    parameter so the structural zeros are guaranteed to land in the ONNX initializers even if the
    exporter reads the raw parameter rather than the dynamic getter.
    """
    import modelopt.torch.sparsity as mts

    baked = 0
    with torch.no_grad():
        for _, mod in model.named_modules():
            if isinstance(mod, getattr(mts, "SparseModule", ())):
                mask = getattr(mod, "_weight_mask", None)
                if mask is None:
                    continue
                # underlying parameter lives in _parameters; the public ``mod.weight`` getter
                # returns weight*mask. Write zeros straight into the stored parameter.
                raw = mod._parameters.get("weight", None)
                if raw is not None:
                    raw.data.mul_(mask.to(raw.dtype))
                    baked += 1
    print(f"[finalize] folded 2:4 mask into {baked} underlying weight params", flush=True)
    name, zf, shape = linear_weight_zero_fraction(model, "q_proj")
    print(f"[finalize] effective {name} weight zero-frac={zf:.4f} shape={shape}", flush=True)
    if expect_sparse:
        assert zf is not None and 0.45 <= zf <= 0.55, (
            f"2:4 sparsity not present pre-export (zf={zf})"
        )
    nwq = count_enabled_weight_quantizers(model)
    print(f"[finalize] enabled weight quantizers (INT8 QDQ source): {nwq}", flush=True)
    assert nwq > 0, "no INT8 quantizers present before ONNX export"
    return model


class PrefillExportWrapper(nn.Module):
    """input_ids -> logits, no KV cache / attention_mask, for a clean prefill ONNX graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, use_cache=False, return_dict=True).logits


@contextlib.contextmanager
def patch_transformers_for_export():
    """Make a single unpadded prefill graph export cleanly to ONNX/TensorRT.

    transformers' ``find_packed_sequence_indices`` skips its ``return None`` early-exit while
    tracing (``is_tracing()`` is True), so it bakes a packed-sequence ``diff``+``cumsum``-on-bool
    into the graph: ``torch.diff`` has no ONNX symbolic, and TRT's CumSum rejects bool inputs.
    For one contiguous sequence there is genuinely no packing, so we force it to return None.
    We also decompose torch.diff as a belt-and-suspenders safeguard.
    """
    import transformers.masking_utils as mu

    orig_fps = mu.find_packed_sequence_indices
    orig_diff = torch.diff

    def _diff(input, n=1, dim=-1, prepend=None, append=None):
        parts = []
        if prepend is not None:
            parts.append(prepend)
        parts.append(input)
        if append is not None:
            parts.append(append)
        x = torch.cat(parts, dim=dim) if len(parts) > 1 else input
        for _ in range(n):
            hi = [slice(None)] * x.dim()
            lo = [slice(None)] * x.dim()
            hi[dim] = slice(1, None)
            lo[dim] = slice(None, -1)
            x = x[tuple(hi)] - x[tuple(lo)]
        return x

    mu.find_packed_sequence_indices = lambda position_ids: None
    torch.diff = _diff
    try:
        yield
    finally:
        mu.find_packed_sequence_indices = orig_fps
        torch.diff = orig_diff


def export_onnx(model, out_dir, model_name, seq_len, vocab_size, weights_dtype="fp16"):
    import glob

    from modelopt.torch._deploy.utils.torch_onnx import OnnxBytes, get_onnx_bytes_and_metadata

    # The output weight dtype is set by the torch MODEL's dtype, not by the helper's weights_dtype
    # arg. For fp16 we cast the model and export NATIVELY (torch emits a self-consistent fp16 graph,
    # with explicit Casts where RMSNorm upcasts to fp32); fp32 keeps the graph fp32. Either way the
    # graph is self-consistently typed, so TensorRT --stronglyTyped can parse it.
    if weights_dtype == "fp16":
        model = model.half()
    # Always pass weights_dtype="fp32" to the helper. Inside get_onnx_bytes_and_metadata that arg
    # ONLY gates onnxconverter's convert_float_to_float16 pass -- which we must NOT run: it
    # block-lists Div and leaves fp32 islands around RMSNorm that strongly-typed REJECTS. So "fp32"
    # here means "helper, don't do your own fp16 conversion" (the model.half() above already did it).
    helper_dtype = "fp32"
    wrapper = PrefillExportWrapper(model).eval()
    dummy = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int64, device="cuda")
    os.makedirs(out_dir, exist_ok=True)
    # onnx 1.18 resolves the external-data `location` against CWD and refuses to overwrite an
    # existing .onnx_data, so clear any stale artifacts from a previous run first.
    for stale in glob.glob(os.path.join(out_dir, f"{model_name}.onnx*")):
        os.remove(stale)
    print(
        f"[onnx] exporting weights_dtype={weights_dtype} "
        f"(native {'fp16' if weights_dtype == 'fp16' else 'fp32'} graph for strongly-typed)",
        flush=True,
    )
    with patch_transformers_for_export():
        onnx_bytes, _ = get_onnx_bytes_and_metadata(
            model=wrapper,
            dummy_input=(dummy,),
            model_name=model_name,
            onnx_opset=20,
            dynamo_export=False,  # TorchScript path (dynamo avoided for quantized graphs on torch 2.10)
            weights_dtype=helper_dtype,
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
        )
    OnnxBytes.from_bytes(onnx_bytes).write_to_disk(out_dir, clean_dir=False)
    onnx_path = os.path.join(out_dir, f"{model_name}.onnx")
    assert os.path.isfile(onnx_path), f"ONNX not written to {onnx_path} (dir={os.listdir(out_dir)})"
    print(f"[onnx] wrote {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)", flush=True)
    return onnx_path


def inspect_onnx(onnx_path, expect_sparse=True):
    import onnx
    from onnx import numpy_helper

    m = onnx.load(onnx_path, load_external_data=True)
    op_counts = {}
    for n in m.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    q = op_counts.get("QuantizeLinear", 0)
    dq = op_counts.get("DequantizeLinear", 0)
    print(
        f"[onnx] QuantizeLinear={q} DequantizeLinear={dq} MatMul={op_counts.get('MatMul', 0)} "
        f"Gemm={op_counts.get('Gemm', 0)} Conv={op_counts.get('Conv', 0)} CumSum={op_counts.get('CumSum', 0)}",
        flush=True,
    )
    assert q + dq > 0, "no QDQ nodes in ONNX -> not an INT8 graph"

    # Verify the 2:4 sparsity zeros reached the ONNX weights. The quantized Linear weights are
    # stored as 2D tensors feeding QuantizeLinear (as initializers or Constant-node values). Count
    # how many large 2D weight tensors are ~50% zero (the 2:4 sparsified Linears) vs dense ones
    # (embedding / lm_head are intentionally NOT sparsified).
    def _tensors():
        yield from m.graph.initializer
        for node in m.graph.node:
            if node.op_type == "Constant":
                for a in node.attribute:
                    if a.name == "value" and a.t.dims:
                        yield a.t

    sparse_2to4 = dense = 0
    example = None
    for t in _tensors():
        if len(t.dims) == 2 and min(t.dims) >= 256:
            arr = numpy_helper.to_array(t)
            zf = float((arr == 0).mean())
            if 0.45 <= zf <= 0.55:
                sparse_2to4 += 1
                if example is None:
                    example = (t.name, tuple(t.dims), zf)
            elif zf < 0.05:
                dense += 1
    ex = f" e.g. {example[0]} dims={example[1]} zf={example[2]:.4f}" if example else ""
    print(
        f"[onnx] 2D weights: {sparse_2to4} are 2:4-sparse (~50% zero), {dense} dense.{ex}",
        flush=True,
    )
    if expect_sparse:
        assert sparse_2to4 >= 50, (
            f"2:4 zeros did not reach ONNX weights (only {sparse_2to4} sparse)"
        )
    return q, dq


def build_trt(
    onnx_path,
    engine_path,
    log_path,
    layer_info_path,
    seq_len,
    timeout_s,
    strongly_typed=True,
    sparsity="enable",
):
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.isfile(trtexec):
        trtexec = (
            subprocess.run(
                ["bash", "-lc", "command -v trtexec"], capture_output=True, text=True
            ).stdout.strip()
            or "trtexec"
        )
    prec = ["--stronglyTyped"] if strongly_typed else ["--int8", "--fp16"]
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        *prec,
        f"--sparsity={sparsity}",
        f"--saveEngine={engine_path}",
        "--minShapes=input_ids:1x1",
        f"--optShapes=input_ids:1x{seq_len}",
        f"--maxShapes=input_ids:1x{max(seq_len, 512)}",
        "--builderOptimizationLevel=4",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info_path}",
        "--verbose",
    ]
    print(f"[trt] {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s)
    print(f"[trt] trtexec returncode={proc.returncode}; log -> {log_path}", flush=True)
    return proc.returncode


def validate_sparse(log_path, layer_info_path):
    with open(log_path, errors="ignore") as f:
        text = f.read()
    spars_lines = [ln for ln in text.splitlines() if "(Sparsity)" in ln]
    print("[validate] trtexec sparsity report lines:", flush=True)
    for ln in spars_lines:
        print("   " + ln.split("] ")[-1].strip(), flush=True)
    found = chose = 0
    for ln in spars_lines:
        mf = re.search(r"Found (\d+) layer", ln)
        mc = re.search(r"Chose (\d+) layer", ln)
        # TRT emits several (Sparsity) lines (per myelin foreign node); take the max, not the last.
        if mf:
            found = max(found, int(mf.group(1)))
        if mc:
            chose = max(chose, int(mc.group(1)))
    # secondary signal: sparse kernel markers in layer-info tactic metadata
    sparse_tactics = 0
    try:
        info = json.load(open(layer_info_path))
        for layer in info.get("Layers", []):
            blob = json.dumps(layer).lower()
            if any(k in blob for k in ("spars", "_2_1", "_4_2", "sp_mma", "spmma")):
                sparse_tactics += 1
    except Exception as e:
        print(f"[validate] layer_info parse skipped: {e}", flush=True)
    print(
        f"[validate] eligible(Found)={found}  chosen(Chose)={chose}  "
        f"layer_info_sparse_markers={sparse_tactics}",
        flush=True,
    )
    return found, chose, sparse_tactics


_TRT_TO_TORCH = None


def _trt_dtype_map():
    import tensorrt as trt

    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }


def run_inference(engine_path, model_dir, prompt, max_new_tokens):
    """Real text-in -> text-out through the built engine (greedy decode).

    The engine is a PREFILL graph (input_ids -> logits, no KV cache), so we generate greedily by
    re-running the full prefill each step (fine for a short demo). Mirrors generate.py. Also serves
    as the engine sanity check: it deserializes, runs, and must produce finite logits.
    """
    import tensorrt as trt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    # instruct model -> render the chat template to text, then tokenize (robust across versions)
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    ids = list(tok(text)["input_ids"])
    n_prompt = len(ids)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    assert engine is not None, "failed to deserialize engine"
    ctx = engine.create_execution_context()
    dmap = _trt_dtype_map()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    in_dtype = dmap[engine.get_tensor_dtype(in_name)]
    out_dtype = dmap[engine.get_tensor_dtype(out_name)]
    max_seq = engine.get_tensor_profile_shape(in_name, 0)[2][1]  # max seq from the build profile
    stream = torch.cuda.Stream()
    eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    print(
        f"[infer] prompt={prompt!r} (prompt_tokens={n_prompt}, engine max_seq={max_seq})",
        flush=True,
    )

    for step in range(max_new_tokens):
        cur_len = len(ids)
        if cur_len >= max_seq:
            print(f"[infer] reached engine max_seq={max_seq}; stopping", flush=True)
            break
        inp = torch.tensor([ids], dtype=in_dtype, device="cuda")
        ctx.set_input_shape(in_name, (1, cur_len))
        ctx.set_tensor_address(in_name, inp.data_ptr())
        out = torch.empty(tuple(ctx.get_tensor_shape(out_name)), dtype=out_dtype, device="cuda")
        ctx.set_tensor_address(out_name, out.data_ptr())
        ok = ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        assert ok, "execute_async_v3 returned False"
        last = out[0, cur_len - 1].float()
        if step == 0:
            assert torch.isfinite(last).all(), "engine produced non-finite logits"
        ids.append(int(last.argmax()))
        if ids[-1] in eos:
            break

    gen = tok.decode(ids[n_prompt:], skip_special_tokens=True)
    print(f"[infer] generated={gen!r}", flush=True)
    print("[infer] inference OK", flush=True)
    return gen


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out-dir", default="/workspace/out")
    ap.add_argument("--model-name", default="qwen2_5_1_5b_int8_sparse")
    # calibration (mirrors examples/llm_ptq/hf_ptq.py defaults: 1024 samples, calib_seq 512)
    ap.add_argument(
        "--calib-samples",
        type=int,
        default=1024,
        help="number of calibration samples (hf_ptq.py default 1024)",
    )
    ap.add_argument(
        "--calib-seq",
        type=int,
        default=512,
        help="max sequence length for calibration samples (hf_ptq.py default 512)",
    )
    ap.add_argument(
        "--calib-dataset",
        default="cnn_dailymail",
        help="calibration dataset, e.g. cnn_dailymail or a nemotron-* dataset",
    )
    ap.add_argument("--calib-batch-size", type=int, default=4, help="calibration batch size")
    # 2:4 sparsity is optional and OFF by default. WARNING: one-shot 2:4 magnitude pruning zeros
    # half the weights and causes SEVERE accuracy degradation on its own; recovering quality
    # requires QAT/SAT fine-tuning (--qat). Without recovery the model produces gibberish.
    ap.add_argument(
        "--sparsity",
        action="store_true",
        help="apply 2:4 structured sparsity before PTQ. WARNING: severe accuracy "
        "degradation -- requires QAT (--qat) to recover quality.",
    )
    # QAT is optional. The built-in loop is only a minimal EXAMPLE -- integrate your own dataset
    # and training pipeline (see run_qat docstring) for real accuracy recovery.
    ap.add_argument(
        "--qat",
        action="store_true",
        help="run the EXAMPLE QAT fine-tune after PTQ (smoke-level placeholder; "
        "swap in your own dataset + training loop for real recovery)",
    )
    ap.add_argument("--qat-steps", type=int, default=20)
    ap.add_argument("--qat-lr", type=float, default=1e-5)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--trt-timeout", type=int, default=1800)
    ap.add_argument(
        "--weights-dtype",
        choices=["fp32", "fp16"],
        default="fp16",
        help="ONNX/engine weight dtype. fp16 exports a NATIVE fp16 graph; fp32 keeps "
        "it fp32. Both are self-consistently typed and build with TRT "
        "--stronglyTyped (INT8 GEMMs via QDQ either way).",
    )
    ap.add_argument(
        "--reuse-onnx",
        action="store_true",
        help="skip torch stages 1-7 and build TRT from an already-exported ONNX",
    )
    # final-stage real inference (text-in -> text-out) through the built engine
    ap.add_argument(
        "--prompt",
        default="What is the capital of France? Answer in one word.",
        help="prompt for the final real-inference step",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="number of tokens to greedily generate in the final inference step",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    print(
        f"torch={torch.__version__} cuda_cc={torch.cuda.get_device_capability()} "
        f"device={torch.cuda.get_device_name()}",
        flush=True,
    )

    onnx_path = os.path.join(args.out_dir, f"{args.model_name}.onnx")
    if args.reuse_onnx:
        log("REUSE: skipping stages 1-7, building TRT from existing ONNX")
        assert os.path.isfile(onnx_path), f"--reuse-onnx but {onnx_path} not found"
        return _build_validate_infer(args, onnx_path, t0)

    log("STAGE 1/8: load Qwen2.5-1.5B-Instruct (fp32, eager attention)")
    model, tok = load_model(args.model_dir)
    vocab = model.config.vocab_size
    print(f"vocab_size={vocab} layers={model.config.num_hidden_layers}", flush=True)

    log("STAGE 2/8: build SmoothQuant calibration forward_loop (hf_ptq.py style)")
    forward_loop, src = build_calib_forward_loop(
        model, tok, args.calib_samples, args.calib_seq, args.calib_dataset, args.calib_batch_size
    )
    print(f"calibration source: {src}", flush=True)

    sparsity_on = args.sparsity
    if sparsity_on:
        log("STAGE 3/8: apply 2:4 structured sparsity (sparse_magnitude)")
        if not args.qat:
            print(
                "[warn] 2:4 sparsity causes SEVERE accuracy degradation without recovery; "
                "the model will likely produce gibberish. Add --qat (or do real SAT/QAT) to "
                "recover quality.",
                flush=True,
            )
        model = apply_sparsity(model)
    else:
        log("STAGE 3/8: sparsity SKIPPED (default; pass --sparsity to enable -- INT8-only run)")

    # Enable INT8 output quantizers when sparsity is on: an INT8-out epilogue is what lets
    # TensorRT actually CHOOSE the structured-sparse INT8 GEMM kernels (a dequant-to-fp32 epilogue
    # makes dense faster). For the dense INT8-only path we leave outputs in fp16/fp32 for accuracy.
    log(
        "STAGE 4/8: INT8 W8A8 SmoothQuant PTQ"
        + (" (+INT8 output quantizers)" if sparsity_on else "")
    )
    model = apply_ptq(model, forward_loop, quant_output=sparsity_on)

    if args.qat:
        log(f"STAGE 5/8: EXAMPLE QAT fine-tune ({args.qat_steps} steps, lr={args.qat_lr})")
        print(
            "[qat] NOTE: this is a minimal example loop -- integrate your own dataset and "
            "training pipeline here for real accuracy recovery (see run_qat docstring).",
            flush=True,
        )
        model = run_qat(model, tok, args.qat_steps, args.seq_len, args.qat_lr)
    else:
        log("STAGE 5/8: QAT skipped (pass --qat to enable)")

    log("STAGE 6/8: finalize model for export (fold 2:4 zeros, keep INT8 QDQ)")
    model = finalize_for_export(model, expect_sparse=sparsity_on)

    log("STAGE 7/8: export to ONNX (INT8 QDQ, opset 20)")
    onnx_path = export_onnx(
        model, args.out_dir, args.model_name, args.seq_len, vocab, weights_dtype=args.weights_dtype
    )
    inspect_onnx(onnx_path, expect_sparse=sparsity_on)
    del model
    torch.cuda.empty_cache()

    return _build_validate_infer(args, onnx_path, t0)


def _build_validate_infer(args, onnx_path, t0):
    # trtexec --sparsity follows the model: enable sparse tactics only when 2:4 sparsity was
    # applied (--sparsity), otherwise disable (the weights are dense, so there's nothing to gain).
    trt_sparsity = "enable" if args.sparsity else "disable"
    log(f"STAGE 8/8: TensorRT engine build (--sparsity={trt_sparsity}) + validate + infer")
    engine_path = os.path.join(args.out_dir, f"{args.model_name}.engine")
    build_log = os.path.join(args.out_dir, "trtexec_build.log")
    layer_info = os.path.join(args.out_dir, "layer_info.json")
    # Always build a strongly-typed engine. Both --weights-dtype fp16 (native fp16 export) and
    # fp32 produce a self-consistently typed ONNX that --stronglyTyped can parse, so no
    # --int8/--fp16 fallback is needed.
    mode = "stronglyTyped"
    rc = build_trt(
        onnx_path,
        engine_path,
        build_log,
        layer_info,
        args.seq_len,
        args.trt_timeout,
        strongly_typed=True,
        sparsity=trt_sparsity,
    )
    assert rc == 0, (
        f"strongly-typed trtexec build failed (rc={rc}); see {build_log}. "
        f"The ONNX must be self-consistently typed for --stronglyTyped."
    )
    print(f"[trt] engine built OK in mode: {mode}", flush=True)

    found, chose, markers = validate_sparse(build_log, layer_info)

    run_inference(engine_path, args.model_dir, args.prompt, args.max_new_tokens)

    log("PIPELINE SUMMARY")
    print(f"ONNX:    {onnx_path}", flush=True)
    print(f"engine:  {engine_path}  (build mode: {mode})", flush=True)
    print(
        f"sparse INT8 kernels -> eligible(Found)={found} chosen(Chose)={chose} markers={markers}",
        flush=True,
    )
    if chose > 0:
        print("RESULT: PASS - TensorRT selected structured-sparse INT8 kernels.", flush=True)
    elif found > 0:
        print(
            "RESULT: PASS(weak) - 2:4 pattern detected & eligible; dense tactic was faster "
            "for some/all GEMMs.",
            flush=True,
        )
    else:
        print(
            "RESULT: WARN - no sparse-eligible layers reported; check ONNX weight zero-fraction.",
            flush=True,
        )
    print(f"total wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

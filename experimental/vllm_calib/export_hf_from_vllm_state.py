#!/usr/bin/env python
"""Restore a vLLM-calibrated quantizer state into a HuggingFace model and export
a unified-HF NVFP4 checkpoint.

Pipeline:
  1. Load source HF model via examples/llm_ptq/example_utils.get_model
     (slow ~28 min for K2; pack-quantized loader keeps INT4 experts packed in
     HBM via _QuantCompressedLinear). gpu_mem_pct is cranked high to keep
     accelerate from offloading onto CPU/meta — that triggers an
     ``AttributeError: weight is not a valid attribute`` inside
     replace_quant_module.
  2. mtq.quantize(model, NVFP4_EXPERTS_ONLY_CFG, forward_loop=None)
     to wrap with HF-side QuantModules. We deliberately avoid
     ``mto.restore_from_modelopt_state`` because the saved modelopt_state's
     metadata carries vLLM-side fused names (``experts.w13_*``,
     ``mla_attn.mla_attn.*``) that don't match HF's per-expert naming.
  3. Per-quantizer non-strict ``load_state_dict`` of our merged state. Strict
     load fails on dynamic NVFP4 quantizers that don't carry an ``_amax``
     buffer; non-strict tolerates the mismatch and applies what fits.
  4. export_hf_checkpoint to write the unified-HF NVFP4 checkpoint.
"""

import argparse
import os
import sys
import time

ROOT = "/lustre/fs1/portfolios/adlr/projects/adlr_psx_numerics/users/jingyux/kimi-k2"
sys.path.insert(0, f"{ROOT}/source/Model-Optimizer")
sys.path.insert(0, f"{ROOT}/source/Model-Optimizer/examples/llm_ptq")

import torch  # noqa: E402

import modelopt.torch.quantization as mtq  # noqa: E402
from modelopt.torch.export import export_hf_checkpoint  # noqa: E402
from modelopt.torch.quantization.nn import TensorQuantizer  # noqa: E402
from modelopt.torch.utils import get_unwrapped_name, safe_load  # noqa: E402

from example_utils import get_model  # noqa: E402


def _ts(stage, t0):
    print(f"[export] {stage}: {time.perf_counter() - t0:.2f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True, help="Source HF checkpoint dir")
    ap.add_argument(
        "--state-path",
        required=True,
        help="Merged HF-named state file produced by merge_and_convert_vllm_state.py",
    )
    ap.add_argument("--export-dir", required=True, help="Output dir for unified-HF NVFP4 ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--gpu-mem-pct",
        type=float,
        default=0.95,
        help="Crank high to avoid accelerate CPU-offloading parts of the model "
             "as meta tensors (which breaks the modelopt wrap step).",
    )
    ap.add_argument("--attn", default=None, help="attn_implementation override")
    ap.add_argument(
        "--save-modelopt-state",
        action="store_true",
        help="Also save modelopt_state.pth alongside the unified checkpoint",
    )
    args = ap.parse_args()

    print(f"[export] python : {sys.executable}", flush=True)
    print(f"[export] torch  : {torch.__version__}", flush=True)
    print(f"[export] cuda   : avail={torch.cuda.is_available()} devices={torch.cuda.device_count()}", flush=True)
    print(f"[export] ckpt   : {args.ckpt_path}", flush=True)
    print(f"[export] state  : {args.state_path}", flush=True)
    print(f"[export] out    : {args.export_dir}", flush=True)
    os.makedirs(args.export_dir, exist_ok=True)

    # 1. Load the source HF model (real weights). gpu_mem_pct=0.95 keeps the
    #    pack-quantized model fully in HBM on 8x80GB, avoiding offload.
    t = time.perf_counter()
    model = get_model(
        args.ckpt_path,
        device=args.device,
        gpu_mem_percentage=args.gpu_mem_pct,
        trust_remote_code=True,
        attn_implementation=args.attn,
    )
    _ts("get_model (HF load)", t)

    # 2. Read saved state (modelopt_state metadata + per-quantizer state).
    t = time.perf_counter()
    saved = safe_load(args.state_path, map_location="cpu")
    state_weights = saved["modelopt_state_weights"]
    src = saved.get("source") or {}
    sm = src.get("quant_config_summary") or {}
    qformat = sm.get("qformat", "NVFP4_EXPERTS_ONLY_CFG")
    print(
        f"[export] saved state: {len(state_weights)} entries  "
        f"source_tag={src.get('tag')}  num_ranks={src.get('num_ranks')}  "
        f"num_experts={src.get('num_experts')}  qformat={qformat}",
        flush=True,
    )
    _ts("safe_load(state)", t)

    # 3. Wrap with modelopt using the SAME quant config the calibration used.
    #    forward_loop=None skips re-running calibration; we'll inject the
    #    previously-collected amax via load_state_dict below.
    t = time.perf_counter()
    quant_cfg = getattr(mtq, qformat)
    print(f"[export] mtq.quantize with {qformat} (forward_loop=None)")
    mtq.quantize(model, quant_cfg, forward_loop=None)
    _ts("mtq.quantize (wrap)", t)

    # 4. Non-strict per-quantizer load. Tolerates dynamic-quantizer modules
    #    that don't register ``_amax`` (calibration over them is best-effort).
    t = time.perf_counter()
    state_weights = {k: v for k, v in state_weights.items() if v}
    print(f"[export] non-empty state entries: {len(state_weights)}")
    n_loaded = n_skipped = 0
    sample_skipped: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        key = get_unwrapped_name(name, model)
        v = state_weights.get(key)
        if not v:
            continue
        try:
            module.load_state_dict(v, strict=False)
            n_loaded += 1
        except Exception as e:
            n_skipped += 1
            if len(sample_skipped) < 5:
                sample_skipped.append(f"{key}: {type(e).__name__}: {str(e)[:200]}")
    print(f"[export] non-strict load: applied to {n_loaded} quantizers (skipped {n_skipped})")
    for s in sample_skipped:
        print(f"  {s}")
    _ts("set_quantizer_state_dict (non-strict)", t)

    # 5. Export the unified-HF NVFP4 checkpoint.
    t = time.perf_counter()
    export_hf_checkpoint(
        model,
        torch.bfloat16,
        export_dir=args.export_dir,
        save_modelopt_state=args.save_modelopt_state,
    )
    _ts("export_hf_checkpoint", t)

    print(f"[export] DONE -> {args.export_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

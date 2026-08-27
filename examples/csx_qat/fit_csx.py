#!/usr/bin/env python
"""Fit CSX PTQ artifacts (per-PE codebook + per-block qscale) for GPT-OSS.

These are the artifacts the QAT run trains under and the deploy export ships.
The model is loaded on CPU and one expert slice at a time moves to the GPU, so
peak GPU memory is one expert tensor rather than the whole model.

    python fit_csx.py --model openai/gpt-oss-20b --num-bits 3 \
        --out artifacts/gptoss20b_csx3bit.pt
"""
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from qlab.qat import CSX_GEOMETRY, fit_csx_artifacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--num-bits", type=int, default=3, choices=(3, 4))
    ap.add_argument("--out", required=True)
    ap.add_argument("--outer-iters", type=int, default=3,
                    help="joint codebook/scale rounds. Warm-started, so >1 helps: "
                         "3 warm rounds gain ~2 dB, 3 COLD rounds lose ~1 dB.")
    ap.add_argument("--lloyd-steps", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    kw = dict(dtype=torch.bfloat16)
    qc = getattr(AutoConfig.from_pretrained(args.model), "quantization_config", None)
    if qc and qc.get("quant_method") == "mxfp4":
        # GPT-OSS ships MXFP4; dequantize so the CSX fit sees real weights.
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)

    artifacts, sqnrs = fit_csx_artifacts(
        model, CSX_GEOMETRY, num_bits=args.num_bits,
        outer_iters=args.outer_iters, lloyd_steps=args.lloyd_steps,
        device=args.device, progress=False,
    )
    torch.save(
        {"artifacts": artifacts, "num_bits": args.num_bits,
         "geometry": CSX_GEOMETRY, "sqnr_db": sqnrs,
         "outer_iters": args.outer_iters, "lloyd_steps": args.lloyd_steps},
        args.out,
    )
    by_proj: dict = {}
    for k, v in sqnrs.items():
        by_proj.setdefault(k.rsplit(".", 1)[-1], []).append(v)
    print(f"[fit] {len(artifacts)} expert tensors -> {args.out}")
    for p, v in sorted(by_proj.items()):
        print(f"[fit]   {p:<14} mean SQNR {sum(v)/len(v):.2f} dB  (n={len(v)})")


if __name__ == "__main__":
    main()

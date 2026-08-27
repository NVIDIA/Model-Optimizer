#!/usr/bin/env python
"""Put a post-QAT checkpoint into QLab deploy format.

    python export_qlab.py --model <qat_ckpt_dir> \
        --artifacts artifacts/csx_4bit.pt --out <deploy_dir>

`--model` must be a loadable checkpoint. A sharded training run has to be
consolidated first (for DeepSpeed, `checkpoint-N/zero_to_fp32.py`); note that
transformers <=5.14 silently corrupts `save_pretrained` for MXFP4-native models,
so prefer reconstructing from the shards over trusting the trainer's own save.
"""
import argparse
import json

import torch

from qlab.qat import export_qat_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="QAT-trained checkpoint dir")
    ap.add_argument("--artifacts", required=True, help=".pt from fit_artifacts.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bake-weights", action="store_true",
                    help="also write fake-quantized weight values (default: keep "
                         "the trained weights and let the runtime apply the LUT)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    blob = torch.load(args.artifacts, map_location="cpu", weights_only=False)
    summary = export_qat_model(
        args.model, blob["artifacts"], blob["geometry"], args.out,
        num_bits=blob["num_bits"], bake_weights=args.bake_weights,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

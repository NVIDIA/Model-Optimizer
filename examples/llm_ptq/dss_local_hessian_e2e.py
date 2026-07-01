# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""End-to-end NVFP4 local-Hessian PTQ with optional Decoupled Scale Search (DSS).

Quantizes a HF model with the ``local_hessian`` algorithm (NVFP4 W4A4), optionally enabling DSS
(``--dss``), prints a short generation preview for a sanity check, and optionally exports a
HuggingFace checkpoint. Intended to validate Stage 1 (calibration + fake-quant) and Stage 2
(export) of DSS on real models.
"""

import argparse
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils.dataset_utils import create_forward_loop

PROMPT = "The key idea behind post-training quantization is"


@torch.no_grad()
def preview(model, tokenizer, max_new_tokens=48):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dss", action="store_true", help="enable Decoupled Scale Search")
    ap.add_argument("--dss-beta-step", type=float, default=0.05)
    ap.add_argument("--calib-size", type=int, default=128)
    ap.add_argument("--calib-seq", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dataset", default="cnn_dailymail")
    ap.add_argument("--export-dir", default=None, help="if set, export an HF checkpoint here")
    args = ap.parse_args()

    print(f"[dss-e2e] loading {args.model} (dss={args.dss}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model.eval()

    print("[dss-e2e] baseline (unquantized) preview:\n  " + preview(model, tokenizer), flush=True)

    cfg = copy.deepcopy(mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG)
    cfg["algorithm"]["decoupled_scale_search"] = args.dss
    cfg["algorithm"]["dss_beta_step"] = args.dss_beta_step

    forward_loop = create_forward_loop(
        model=model,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_size,
        max_sample_length=args.calib_seq,
    )
    print("[dss-e2e] quantizing (NVFP4 local_hessian) ...", flush=True)
    mtq.quantize(model, cfg, forward_loop=forward_loop)

    n_dss = sum(
        1
        for _, m in model.named_modules()
        if isinstance(m, TensorQuantizer) and getattr(m, "quant_amax", None) is not None
    )
    print(f"[dss-e2e] weight quantizers carrying a DSS quant_amax: {n_dss}", flush=True)
    print("[dss-e2e] quantized preview:\n  " + preview(model, tokenizer), flush=True)

    if args.export_dir:
        print(f"[dss-e2e] exporting HF checkpoint to {args.export_dir} ...", flush=True)
        export_hf_checkpoint(model, export_dir=args.export_dir)
        print("[dss-e2e] export complete.", flush=True)


if __name__ == "__main__":
    main()

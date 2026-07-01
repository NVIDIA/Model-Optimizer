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

"""Is DSS's beta!=1 driven by rounding ties or by saturation/clipping?

For each NVFP4 block we already know DSS never reduces the L2 loss beyond fp32 noise. This
script asks *why* DSS still picks beta!=1, separating the two candidate mechanisms:

  * rounding ties     : a value sits on an FP4 rounding boundary, so a range of s_q yields the
                        same codes -> same loss -> argmin lands on some beta!=1 arbitrarily.
  * saturation/clip   : with the coupled s_d, some |w| > 6*s_d clips to code 6, making Q no
                        longer a plain nearest map. A different s_q could move values in/out of
                        saturation -- the case where the per-element-optimal proof might break.

A block is "saturated" under the coupled optimum iff max|w| > coupled_amax (since the FP4 max
code is 6 and recon_max = 6*(amax/6) = amax). We cross-tabulate saturated x (beta!=1) and report
the MAX and mean relative L2 gain in each cell. If the max gain among *saturated* blocks is still
fp-noise, saturation does not open a real gap and the proof holds there too.

Reuses the production coupled sweep (``nvfp4_fp8_scale_sweep``) and the diagnostic
(``nvfp4_dss_diag_sweep``); no kernel changes.
"""

import argparse
import glob
import json
import os
import time

import torch

from modelopt.torch.kernels.quantization.gemm import nvfp4_dss_diag_sweep, nvfp4_fp8_scale_sweep

BLOCK_SIZE = 16
TIE_REL = 1e-7  # rel gain below this is numerically a tie (no real improvement)
SKIP_KEYS = ("embed", "lm_head", "norm", "bias", "rotary", "router", "gate.weight")


def build_betas(step, device):
    n = round(0.5 / step)
    return 1.0 + torch.arange(-n, n + 1, device=device, dtype=torch.float32) * step


class Acc:
    """Scalar cross-tab accumulators (saturated x beta!=1) over per-block stats."""

    def __init__(self):
        self.keys = [
            "n_blocks",
            "n_sat",
            "n_bne1",
            "n_sat_bne1",
            "n_notsat_bne1",
            "n_sat_bne1_tie",
            "n_sat_bne1_real",  # real = rel > 1e-5
            "sum_rel_sat",
            "sum_rel_notsat",
            "sum_beta_sat_bne1",
            "sum_beta_notsat_bne1",
        ]
        self.s = dict.fromkeys(self.keys, 0.0)
        self.max_rel_sat = 0.0
        self.max_rel_notsat = 0.0

    @torch.no_grad()
    def update(self, w, global_amax, betas):
        nb = w.numel() // BLOCK_SIZE
        coupled_amax = nvfp4_fp8_scale_sweep(w, global_amax, block_size=BLOCK_SIZE).reshape(nb)
        coupled, dss, beta = nvfp4_dss_diag_sweep(w, global_amax, betas, BLOCK_SIZE)
        block_max = w.reshape(nb, BLOCK_SIZE).abs().amax(dim=1)

        sat = block_max > coupled_amax
        notsat = ~sat
        rel = (coupled - dss) / coupled.clamp_min(1e-12)
        bne1 = beta != 1.0

        self.s["n_blocks"] += nb
        self.s["n_sat"] += int(sat.sum())
        self.s["n_bne1"] += int(bne1.sum())
        self.s["n_sat_bne1"] += int((sat & bne1).sum())
        self.s["n_notsat_bne1"] += int((notsat & bne1).sum())
        self.s["n_sat_bne1_tie"] += int((sat & bne1 & (rel < TIE_REL)).sum())
        self.s["n_sat_bne1_real"] += int((sat & bne1 & (rel > 1e-5)).sum())
        self.s["sum_rel_sat"] += float(rel[sat].sum()) if sat.any() else 0.0
        self.s["sum_rel_notsat"] += float(rel[notsat].sum()) if notsat.any() else 0.0
        self.s["sum_beta_sat_bne1"] += float(beta[sat & bne1].sum()) if (sat & bne1).any() else 0.0
        self.s["sum_beta_notsat_bne1"] += (
            float(beta[notsat & bne1].sum()) if (notsat & bne1).any() else 0.0
        )
        if sat.any():
            self.max_rel_sat = max(self.max_rel_sat, float(rel[sat].max()))
        if notsat.any():
            self.max_rel_notsat = max(self.max_rel_notsat, float(rel[notsat].max()))

    def report(self) -> dict:
        s = self.s
        nb = s["n_blocks"] or 1
        nsat = s["n_sat"] or 1
        nnotsat = (s["n_blocks"] - s["n_sat"]) or 1
        return {
            "n_blocks": s["n_blocks"],
            "frac_saturated": s["n_sat"] / nb,
            "frac_beta_ne1": s["n_bne1"] / nb,
            "saturated": {
                "max_rel_gain": self.max_rel_sat,
                "mean_rel_gain": s["sum_rel_sat"] / nsat,
                "frac_with_beta_ne1": s["n_sat_bne1"] / nsat,
                "of_those_beta_ne1_fraction_ties": (s["n_sat_bne1_tie"] / (s["n_sat_bne1"] or 1)),
                "of_those_beta_ne1_count_real_gain_gt_1e-5": int(s["n_sat_bne1_real"]),
                "mean_beta_when_ne1": (s["sum_beta_sat_bne1"] / (s["n_sat_bne1"] or 1)),
            },
            "not_saturated": {
                "max_rel_gain": self.max_rel_notsat,
                "mean_rel_gain": s["sum_rel_notsat"] / nnotsat,
                "frac_with_beta_ne1": s["n_notsat_bne1"] / nnotsat,
                "mean_beta_when_ne1": (s["sum_beta_notsat_bne1"] / (s["n_notsat_bne1"] or 1)),
            },
        }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-shards", type=int, default=6, help="cap shards (0 = all)")
    ap.add_argument("--beta-step", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(args.model, "*.safetensors")))
    if args.max_shards:
        shards = shards[: args.max_shards]
    betas = build_betas(args.beta_step, args.device)
    acc = Acc()
    t0 = time.perf_counter()
    for si, shard in enumerate(shards):
        print(f"[sat] shard {si + 1}/{len(shards)}: {os.path.basename(shard)}", flush=True)
        with safe_open(shard, framework="pt", device=args.device) as f:
            for key in f.keys():  # noqa: SIM118  (safetensors handle, not a dict)
                shape = f.get_slice(key).get_shape()
                if len(shape) not in (2, 3) or shape[-1] % BLOCK_SIZE:
                    continue
                if any(k in key.lower() for k in SKIP_KEYS):
                    continue
                t = f.get_tensor(key)
                slices = t.unsqueeze(0) if t.ndim == 2 else t
                for w in slices:
                    wf = w.to(torch.float32)
                    acc.update(wf, wf.abs().max(), betas)
                    del wf
                del t
    rep = {
        "model": args.model,
        "shards": len(shards),
        "beta_step": args.beta_step,
        "elapsed_s": time.perf_counter() - t0,
        **acc.report(),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "dss_saturation_check.json")
    with open(out, "w") as fp:
        json.dump(rep, fp, indent=2)
    print(json.dumps(rep, indent=2), flush=True)
    print(f"[sat] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

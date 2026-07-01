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

"""Diagnostic: does Decoupled Scale Search (DSS) help the NVFP4 weight-MSE FP8 sweep?

For every NVFP4 weight block this measures the best reconstruction loss of the *coupled*
sweep (today's behavior, ``s_q == s_d``) against DSS's decoupled ``(s_q, s_d)`` search, and
reports how often DSS picks ``beta != 1`` and by how much it reduces the L2 loss. It runs the
search only — no two-scale export plumbing — so it is a cheap probe before committing to that.

Two modes:
  * ``model``         : load the model and run ``mtq.quantize`` with the real NVFP4 weight-MSE
                        FP8-sweep config so each weight quantizer's ``global_amax`` is the
                        grouping-synced production value (dense + fused-MoE). Fits in GPU.
  * ``weights-only``  : stream safetensors shards and use per-tensor ``global_amax`` (no model
                        instantiation). For checkpoints too large to load (e.g. 397B). The
                        sibling global_amax grouping is approximated as per-tensor.

Example::

    CUDA_VISIBLE_DEVICES=2,3 python examples/llm_ptq/dss_diagnostic.py \
        --model /path/to/Qwen3-8B --output-dir /path/to/logs/Qwen3-8B
"""

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from modelopt.torch.kernels.quantization.gemm import nvfp4_dss_diag_sweep

# Relative-improvement thresholds used to bucket "did DSS actually help this block".
REL_THRESHOLDS = (1e-6, 1e-4, 1e-2)
BLOCK_SIZE = 16


def build_betas(step: float, device="cpu") -> torch.Tensor:
    """Symmetric quant-scale grid over [0.5, 1.5] with exact 1.0 at the center (coupled ref).

    ``step=0.01`` reproduces SOAR's 101-point grid; a coarser step trades search density for
    speed on huge checkpoints (parity already established with the full grid).
    """
    n = round(0.5 / step)
    k = torch.arange(-n, n + 1, device=device, dtype=torch.float32)
    return 1.0 + k * step


def classify_layer(name: str) -> str:
    """Coarse layer-type bucket from a module/weight name."""
    n = name.lower()
    if "expert" in n:
        return "experts"
    if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "self_attn", "attn")):
        return "attn"
    if "router" in n or "gate.weight" in n or n.endswith(".gate"):
        return "router"
    if any(k in n for k in ("gate_proj", "up_proj", "down_proj", "gate_up", "mlp", "fc")):
        return "mlp"
    return "other"


@dataclass
class Bucket:
    """Streaming aggregate of per-block DSS-vs-coupled stats for one layer type."""

    n_weights: int = 0
    n_blocks: int = 0
    n_beta_ne1: int = 0
    sum_rel: float = 0.0
    max_rel: float = 0.0
    rel_over: dict[float, int] = field(default_factory=lambda: dict.fromkeys(REL_THRESHOLDS, 0))
    beta_hist: torch.Tensor = field(default_factory=lambda: torch.zeros(101, dtype=torch.int64))

    def update(self, coupled, dss, best_beta):
        rel = (coupled - dss) / coupled.clamp_min(1e-12)
        self.n_weights += 1
        self.n_blocks += rel.numel()
        self.n_beta_ne1 += int((best_beta != 1.0).sum())
        self.sum_rel += float(rel.sum())
        self.max_rel = max(self.max_rel, float(rel.max()))
        for t in REL_THRESHOLDS:
            self.rel_over[t] += int((rel > t).sum())
        # beta in [0.50, 1.50] step 0.01 -> bin index round((beta-0.5)*100).
        idx = ((best_beta - 0.5) * 100.0).round().clamp(0, 100).to(torch.int64)
        self.beta_hist += torch.bincount(idx.flatten(), minlength=101).cpu()

    def to_raw(self) -> dict:
        """Picklable additive state, for merging buckets across worker processes."""
        return {
            "n_weights": self.n_weights,
            "n_blocks": self.n_blocks,
            "n_beta_ne1": self.n_beta_ne1,
            "sum_rel": self.sum_rel,
            "max_rel": self.max_rel,
            "rel_over": {f"{t:.0e}": self.rel_over[t] for t in REL_THRESHOLDS},
            "beta_hist": self.beta_hist.tolist(),
        }

    def merge_raw(self, raw: dict):
        self.n_weights += raw["n_weights"]
        self.n_blocks += raw["n_blocks"]
        self.n_beta_ne1 += raw["n_beta_ne1"]
        self.sum_rel += raw["sum_rel"]
        self.max_rel = max(self.max_rel, raw["max_rel"])
        for t in REL_THRESHOLDS:
            self.rel_over[t] += raw["rel_over"][f"{t:.0e}"]
        self.beta_hist += torch.tensor(raw["beta_hist"], dtype=torch.int64)

    def as_dict(self) -> dict:
        mean_rel = self.sum_rel / self.n_blocks if self.n_blocks else 0.0
        return {
            "n_weights": self.n_weights,
            "n_blocks": self.n_blocks,
            "frac_beta_ne1": self.n_beta_ne1 / self.n_blocks if self.n_blocks else 0.0,
            "mean_rel_improvement": mean_rel,
            "max_rel_improvement": self.max_rel,
            "frac_blocks_rel_over": {
                f"{t:.0e}": (self.rel_over[t] / self.n_blocks if self.n_blocks else 0.0)
                for t in REL_THRESHOLDS
            },
        }


class Report:
    """Per-layer-type + overall accumulator and Markdown/JSON writer."""

    def __init__(self):
        self.buckets: dict[str, Bucket] = defaultdict(Bucket)
        self.overall = Bucket()

    def add(self, name: str, coupled, dss, best_beta):
        self.buckets[classify_layer(name)].update(coupled, dss, best_beta)
        self.overall.update(coupled, dss, best_beta)

    def to_raw(self) -> dict:
        return {"buckets": {k: v.to_raw() for k, v in self.buckets.items()}}

    def merge_raw(self, raw: dict):
        for k, b in raw["buckets"].items():
            self.buckets[k].merge_raw(b)
            self.overall.merge_raw(b)

    def to_json(self, meta: dict) -> dict:
        return {
            "meta": meta,
            "overall": self.overall.as_dict(),
            "by_layer_type": {k: v.as_dict() for k, v in sorted(self.buckets.items())},
        }

    def to_markdown(self, meta: dict) -> str:
        o = self.overall.as_dict()
        verdict = (
            "PARITY (DSS does not help the L2 objective)"
            if o["max_rel_improvement"] < 1e-4
            else "DSS SHOWS A MEASURABLE GAIN — investigate"
        )
        lines = [
            f"# DSS diagnostic — {meta.get('model', '?')}",
            "",
            f"- mode: `{meta.get('mode')}`  block_size: {meta.get('block_size')}  "
            f"betas: {meta.get('num_betas')} in [0.50, 1.50]",
            f"- weights analyzed: {self.overall.n_weights}  blocks: {self.overall.n_blocks:,}",
            f"- elapsed: {meta.get('elapsed_s', 0):.1f}s",
            "",
            f"## Verdict: {verdict}",
            "",
            f"- fraction of blocks where DSS picks beta != 1: **{o['frac_beta_ne1']:.3%}**",
            f"- mean relative L2 improvement (coupled->dss): **{o['mean_rel_improvement']:.3e}**",
            f"- max  relative L2 improvement over any block: **{o['max_rel_improvement']:.3e}**",
            "- fraction of blocks with relative improvement over threshold:",
        ]
        lines += [
            f"    - > {t:.0e}: {o['frac_blocks_rel_over'][f'{t:.0e}']:.3%}" for t in REL_THRESHOLDS
        ]
        lines += [
            "",
            "## By layer type",
            "",
            "| type | weights | blocks | beta!=1 | mean rel | max rel |",
            "|---|---|---|---|---|---|",
        ]
        for k, v in sorted(self.buckets.items()):
            d = v.as_dict()
            lines.append(
                f"| {k} | {d['n_weights']} | {d['n_blocks']:,} | {d['frac_beta_ne1']:.2%} | "
                f"{d['mean_rel_improvement']:.2e} | {d['max_rel_improvement']:.2e} |"
            )
        return "\n".join(lines) + "\n"


@torch.no_grad()
def analyze_model_mode(args, report: Report) -> int:
    """Faithful path: load the model, calibrate, iterate production weight quantizers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.model_calib import _uses_modelopt_fp8_weight_scales
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModule
    from modelopt.torch.quantization.utils import enable_weight_access_and_writeback
    from modelopt.torch.utils.dataset_utils import create_forward_loop

    print(f"[dss] loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model.eval()

    forward_loop = create_forward_loop(
        model=model,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_size,
        max_sample_length=args.calib_seq,
    )
    print("[dss] calibrating (NVFP4 weight-MSE FP8 sweep) to populate global_amax ...", flush=True)
    mtq.quantize(model, mtq.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG, forward_loop=forward_loop)

    print("[dss] running DSS-vs-coupled search over weight quantizers ...", flush=True)
    name_to_module = dict(model.named_modules())
    seen: set[int] = set()
    n_done = 0
    for name, module in name_to_module.items():
        if id(module) in seen or not isinstance(module, QuantModule):
            continue
        seen.add(id(module))
        with enable_weight_access_and_writeback(module, model, name_to_module):
            for sub_idx, (weight, wq) in enumerate(module.iter_weights_for_calibration()):
                # global_amax exists only after NVFP4-static promotion; the format-only
                # is_nvfp4_static check can be true before that, so fetch defensively.
                global_amax = getattr(wq, "global_amax", None)
                if not _uses_modelopt_fp8_weight_scales(wq) or global_amax is None:
                    continue
                if weight is None or weight.dim() < 2 or weight.numel() % args.block_size:
                    continue
                betas = build_betas(args.beta_step, weight.device)
                coupled, dss, best_beta = nvfp4_dss_diag_sweep(
                    weight, global_amax, betas, args.block_size
                )
                report.add(f"{name}.{sub_idx}", coupled, dss, best_beta)
                n_done += 1
                if n_done % 50 == 0:
                    print(f"[dss]   {n_done} weights analyzed", flush=True)
                if args.max_weights and n_done >= args.max_weights:
                    return n_done
    return n_done


_SKIP_KEYS = ("embed", "lm_head", "norm", "bias", "rotary", "router", "gate.weight")


def _is_quantizable_tensor(name: str, shape, block_size: int) -> bool:
    """Raw-checkpoint linear/expert weights the NVFP4 weight sweep would touch.

    Accepts 2-D linear weights (``...proj.weight``) and 3-D fused-expert stacks
    (``...experts.gate_up_proj`` / ``down_proj`` of shape ``[n_experts, out, in]``); both
    quantize along the last (contraction) dim, which must be block-divisible.
    """
    if len(shape) not in (2, 3) or shape[-1] % block_size:
        return False
    return not any(k in name.lower() for k in _SKIP_KEYS)


@torch.no_grad()
def _process_shards(shards, device, block_size, max_weights, beta_step, log_prefix=""):
    """Run the DSS-vs-coupled search over one worker's shards; return raw report state.

    Per-expert ``global_amax = max|slice|`` is faithful for fused experts (per-expert weight
    quantizers); for grouped linears (qkv/gate_up siblings) it is a per-tensor approximation.
    """
    from safetensors import safe_open

    report = Report()
    betas = build_betas(beta_step, device)
    n_done = 0
    for si, shard in enumerate(shards):
        print(
            f"[dss]{log_prefix} shard {si + 1}/{len(shards)}: {os.path.basename(shard)}", flush=True
        )
        with safe_open(shard, framework="pt", device=device) as f:
            for key in f.keys():  # noqa: SIM118  (safetensors handle, not a dict)
                shape = f.get_slice(key).get_shape()
                if not _is_quantizable_tensor(key, shape, block_size):
                    continue
                t = f.get_tensor(key)
                # 2-D linear -> one weight; 3-D expert stack -> one weight per expert slice.
                slices = t.unsqueeze(0) if t.ndim == 2 else t
                for w in slices:
                    wf = w.to(torch.float32)
                    coupled, dss, best_beta = nvfp4_dss_diag_sweep(
                        wf, wf.abs().max(), betas, block_size
                    )
                    report.add(key, coupled, dss, best_beta)
                    n_done += 1
                    del wf, coupled, dss, best_beta
                del t
                if n_done % 200 == 0:
                    print(f"[dss]{log_prefix}   {n_done} weights analyzed", flush=True)
                if max_weights and n_done >= max_weights:
                    return report.to_raw(), n_done
    return report.to_raw(), n_done


def _worker(arg):
    """ProcessPool entry: pin to the assigned GPU and process a shard subset."""
    shards, gpu_id, block_size, max_weights, beta_step = arg
    import torch as _torch

    _torch.cuda.set_device(gpu_id)
    return _process_shards(
        shards, f"cuda:{gpu_id}", block_size, max_weights, beta_step, f"[gpu{gpu_id}]"
    )


@torch.no_grad()
def analyze_weights_only_mode(args, report: Report) -> int:
    """Streaming path for checkpoints too large to instantiate, parallelized across GPUs."""
    shards = sorted(glob.glob(os.path.join(args.model, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no .safetensors shards under {args.model}")

    n_gpus = args.num_gpus or torch.cuda.device_count()
    if n_gpus <= 1:
        raw, n = _process_shards(
            shards, "cuda:0", args.block_size, args.max_weights, args.beta_step
        )
        report.merge_raw(raw)
        return n

    # One worker per GPU; round-robin shards so disk I/O + compute overlap across devices.
    import torch.multiprocessing as mp

    tasks = [
        (
            shards[i::n_gpus],
            i,
            args.block_size,
            args.max_weights // n_gpus if args.max_weights else 0,
            args.beta_step,
        )
        for i in range(n_gpus)
    ]
    print(f"[dss] launching {n_gpus} GPU workers over {len(shards)} shards", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_gpus) as pool:
        results = pool.map(_worker, tasks)
    n_done = 0
    for raw, n in results:
        report.merge_raw(raw)
        n_done += n
    return n_done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model path")
    ap.add_argument("--mode", choices=["auto", "model", "weights-only"], default="auto")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset", default="cnn_dailymail")
    ap.add_argument("--calib-size", type=int, default=128)
    ap.add_argument("--calib-seq", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--max-weights", type=int, default=0, help="cap analyzed weights (0 = all)")
    ap.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="GPU workers for weights-only mode (0 = all visible). Each handles a shard subset.",
    )
    ap.add_argument(
        "--beta-step",
        type=float,
        default=0.01,
        help="Quant-scale grid step over [0.5, 1.5] (0.01 = SOAR's 101 pts; coarser = faster).",
    )
    args = ap.parse_args()

    if args.mode == "auto":
        # Pick streaming for checkpoints that clearly won't fit; else load the model.
        total_bytes = sum(
            os.path.getsize(p) for p in glob.glob(os.path.join(args.model, "*.safetensors"))
        )
        args.mode = "weights-only" if total_bytes > 150 * 2**30 else "model"
    print(f"[dss] mode = {args.mode}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    report = Report()
    t0 = time.perf_counter()
    if args.mode == "model":
        n = analyze_model_mode(args, report)
    else:
        n = analyze_weights_only_mode(args, report)
    elapsed = time.perf_counter() - t0

    meta = {
        "model": args.model,
        "mode": args.mode,
        "block_size": args.block_size,
        "num_betas": int(build_betas(args.beta_step).numel()),
        "n_weights": n,
        "elapsed_s": elapsed,
    }
    base = os.path.join(args.output_dir, "dss_diagnostic")
    with open(base + ".json", "w") as f:
        json.dump(report.to_json(meta), f, indent=2)
    md = report.to_markdown(meta)
    with open(base + ".md", "w") as f:
        f.write(md)
    print("\n" + md, flush=True)
    print(f"[dss] wrote {base}.json and {base}.md", flush=True)


if __name__ == "__main__":
    main()

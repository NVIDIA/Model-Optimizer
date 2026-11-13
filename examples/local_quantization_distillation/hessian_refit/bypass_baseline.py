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

"""Baseline driver: GPTQ PTQ -> held-out KLD -> Bypass scale-only repair -> held-out KLD.

Single process on one GPU; run as a module from the repository root. Experimental.
"""

import argparse
import copy
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import transformers

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import (
    TransformersLocalDistillationAdapter,
    transformers_family,
)

from ..evaluate import evaluate_pair
from ..prepare_data import PackedBatchSequence

# The IQ1 formats need a heavier GPTQ damping than the 0.01 used elsewhere.
DEFAULT_DAMP = {"iq1_s": 0.3, "iq1_m": 0.3}


class Head(Sequence):
    """``count`` consecutive batches of a re-iterable batch sequence, starting at ``start``."""

    def __init__(self, inner, count, start=0):
        self.inner, self.start = inner, start % max(1, len(inner))
        self.count = min(count, len(inner) - self.start)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError(index)
        return self.inner[self.start + index]


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="prepare_data output directory")
    p.add_argument("--format", required=True, help="ggml format, e.g. iq2_xxs")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-class", default="auto", choices=["auto", "causal", "image-text"])
    p.add_argument("--unit-types", nargs="+", default=["mlp", "moe"])
    p.add_argument("--mode", default="scale_only")
    p.add_argument("--updates", type=int, default=512)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=32768)
    p.add_argument("--scale-lr", type=float, default=0.01)
    p.add_argument("--optimizer", default="adagrad", choices=["adagrad", "adamw"])
    p.add_argument("--selection", default="final")
    p.add_argument("--validation-interval", type=int, default=64)
    p.add_argument("--report-interval", type=int, default=16)
    p.add_argument("--perc-damp", type=float, default=None)
    p.add_argument("--gptq-block-size", type=int, default=256)
    p.add_argument("--no-importance-weighted", action="store_true")
    p.add_argument("--calib-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--validation-packs", type=int, default=None)
    p.add_argument("--diagnostic-packs", type=int, default=None)
    p.add_argument(
        "--ptq-checkpoint", default=None, help="mto.save state to restore instead of GPTQ"
    )
    p.add_argument("--save-ptq", action="store_true")
    p.add_argument("--save-repaired", action="store_true")
    p.add_argument("--skip-before-eval", action="store_true")
    p.add_argument("--skip-repair", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(path, device, model_class):
    config = transformers.AutoConfig.from_pretrained(path)
    if model_class == "auto":
        model_class = (
            "image-text" if getattr(config, "vision_config", None) is not None else "causal"
        )
    cls = (
        transformers.AutoModelForImageTextToText
        if model_class == "image-text"
        else transformers.AutoModelForCausalLM
    )
    model = cls.from_pretrained(path, dtype=torch.bfloat16)
    return model.to(device).eval().requires_grad_(False)


def gpu_stats(device):
    torch.cuda.synchronize(device)
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}


def main():
    args = parse()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    log = {"args": vars(args), "started_unix": time.time()}

    def dump():
        (out / "metrics.json").write_text(json.dumps(log, indent=2, default=str) + "\n")

    repair = mtd.LocalDistillationConfig(
        unit_types=tuple(args.unit_types),
        mode=args.mode,
        updates=args.updates,
        sequence_length=args.sequence_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        scale_lr=args.scale_lr,
        optimizer=args.optimizer,
        selection=args.selection,
        validation_interval=args.validation_interval,
        report_interval=args.report_interval,
        seed=args.seed,
    )
    log["repair_config"] = json.loads(repair.model_dump_json())

    t0 = time.perf_counter()
    teacher = load_model(args.model, device, args.model_class)
    student = copy.deepcopy(teacher)
    adapter = TransformersLocalDistillationAdapter()
    log["model_class"] = type(teacher).__name__
    log["load_seconds"] = time.perf_counter() - t0

    train = PackedBatchSequence(
        args.data, "train", global_batch_size=args.global_batch_size, device=device
    )
    calibration = PackedBatchSequence(
        args.data, "calibration", global_batch_size=args.calib_batch_size, device=device
    )
    validation = PackedBatchSequence(
        args.data, "validation", global_batch_size=args.eval_batch_size, device=device
    )
    diagnostic = PackedBatchSequence(
        args.data, "diagnostic", global_batch_size=args.eval_batch_size, device=device
    )
    if args.validation_packs:
        validation = Head(validation, args.validation_packs // args.eval_batch_size)
    if args.diagnostic_packs:
        diagnostic = Head(diagnostic, args.diagnostic_packs // args.eval_batch_size)
    log["data"] = {
        "train_batches": len(train),
        "train_tokens": len(train) * args.global_batch_size * train.sequence_length,
        "calibration_batches": len(calibration),
        "validation_batches": len(validation),
        "diagnostic_batches": len(diagnostic),
        "manifest_fingerprint": train.fingerprint,
    }

    # ---- PTQ -------------------------------------------------------------------------------
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    patterns = transformers_family(teacher).weight_patterns(teacher, repair)
    if not patterns:
        raise ValueError("No selected projection weights; inspect the family and unit types.")
    log["ptq_patterns"] = patterns
    if args.ptq_checkpoint:
        mto.restore(student, args.ptq_checkpoint)
        log["ptq_source"] = args.ptq_checkpoint
    else:
        damp = args.perc_damp if args.perc_damp is not None else DEFAULT_DAMP.get(args.format, 0.01)
        recipe = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                *[
                    {
                        "quantizer_name": pattern,
                        "enable": True,
                        "cfg": {"num_bits": args.format, "backend": "ggml"},
                    }
                    for pattern in patterns
                ],
            ],
            "algorithm": {
                "method": "gptq",
                "block_size": args.gptq_block_size,
                "importance_weighted": not args.no_importance_weighted,
                "perc_damp": damp,
                "fused": False,
            },
        }
        log["ptq_recipe_algorithm"] = recipe["algorithm"]

        @torch.no_grad()
        def calibrate(model):
            for batch in calibration:
                adapter.teacher_microbatch(model, [], batch, lambda *_: None)

        mtq.quantize(student, recipe, forward_loop=calibrate)
        if args.save_ptq:
            mto.save(student, out / "ptq.pt")
    log["ptq_seconds"] = time.perf_counter() - t0
    log["ptq_gpu"] = gpu_stats(device)
    dump()

    # ---- Before ----------------------------------------------------------------------------
    if not args.skip_before_eval:
        torch.cuda.reset_peak_memory_stats(device)
        log["before"] = evaluate_pair(teacher, student, diagnostic, adapter=adapter)
        print(json.dumps({"before": log["before"]}), flush=True)
        dump()
    if args.skip_repair:
        return

    # ---- Repair ----------------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)
    result = mtd.local_distill(
        student, teacher, train, repair, validation_dataloader=validation, adapter=adapter
    )
    log["repair"] = result.metrics
    dump()

    # ---- After -----------------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)
    log["after"] = evaluate_pair(teacher, result.student, diagnostic, adapter=adapter)
    if args.save_repaired:
        mto.save(result.student, out / "repaired.pt")
    log["finished_unix"] = time.time()
    dump()

    units = result.metrics["units"]
    summary = {
        "format": args.format,
        "updates": result.metrics["updates"],
        "kld_before": log.get("before", {}).get("kld_teacher_candidate"),
        "kld_after": log["after"]["kld_teacher_candidate"],
        "ppl_teacher": log["after"].get("teacher_ppl"),
        "ppl_before": log.get("before", {}).get("ppl"),
        "ppl_after": log["after"].get("ppl"),
        "mean_unit_mse_baseline": sum(u.get("baseline_validation_mse", 0) for u in units.values())
        / max(1, len(units)),
        "mean_unit_mse_final": sum(u.get("validation_mse", 0) for u in units.values())
        / max(1, len(units)),
        "repair_elapsed_seconds": result.metrics.get("elapsed_seconds"),
        "repair_gpu_hours": result.metrics.get("process_gpu_hours"),
        "repair_peak_gib": result.metrics.get("peak_allocated_bytes_max", 0) / 2**30,
    }
    log["summary"] = summary
    dump()
    print(json.dumps({"summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()

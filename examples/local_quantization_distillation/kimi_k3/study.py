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


"""Compare IQ1_S PTQ and independent scale repair on real first-MoE activations."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import modelopt.torch.distill as mtd
from examples.local_quantization_distillation.repair import quantize_student
from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter
from modelopt.torch.quantization.utils import (
    disable_weight_cache,
    enable_weight_cache,
    weight_cache_scope,
)

from .source import load_moe

METHODS = ("unweighted", "imatrix", "gptq", "gptq_importance")


class ActivationBatches(Sequence):
    """Expose exact BF16 inputs without repeating or padding a rank's token stream."""

    def __init__(self, directory, partition, rank, size, *, chunk=2048, global_batch=8):
        """Read rank-owned BF16 rows with explicit sample and update boundaries."""
        self.array = np.load(directory / f"{partition}-rank{rank}.npy", mmap_mode="r")
        if self.array.dtype != np.uint16 or self.array.ndim != 3 or self.array.shape[1] != 8192:
            raise ValueError("Activation captures must contain exact BF16 bits in 8K rows.")
        self.local_batch, remainder = divmod(global_batch, size)
        if remainder or not self.local_batch or self.array.shape[1] % chunk:
            raise ValueError("Activation samples must divide the update batch and capture context.")
        self.chunk = chunk
        self.per_row = self.array.shape[1] // chunk
        self.samples = len(self.array) * self.per_row
        if self.samples % self.local_batch:
            raise ValueError("Activation samples must exactly fill update batches.")

    def __len__(self):
        return self.samples // self.local_batch

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        samples = []
        for i in range(index * self.local_batch, (index + 1) * self.local_batch):
            row, piece = divmod(i, self.per_row)
            words = self.array[row, piece * self.chunk : (piece + 1) * self.chunk].copy()
            samples.append(torch.from_numpy(words).view(torch.bfloat16))
        states = torch.stack(samples).cuda()
        valid = torch.ones(states.shape[:2], device=states.device, dtype=torch.bool)
        return {"hidden_states": states, "padding_mask": ~valid, "valid_mask": valid}


@torch.no_grad()
def measure(teacher, candidate, data):
    """Measure paired full-MoE MSE across independent token/data lanes exactly once."""
    totals = torch.zeros(3, device="cuda", dtype=torch.float64)
    enable_weight_cache(candidate)
    try:
        # Expert views become fresh contiguous buffers inside quantization.
        # Weights are frozen throughout this diagnostic scope, so reuse DQ once.
        with weight_cache_scope(candidate) as cache:
            for batch in data:
                target = teacher(batch["hidden_states"], batch["padding_mask"])
                actual = candidate(batch["hidden_states"], batch["padding_mask"])
                totals[0] += (actual.float() - target.float()).square().sum(dtype=torch.float64)
                totals[1] += target.numel()
                totals[2] += target.float().square().sum(dtype=torch.float64)
    finally:
        disable_weight_cache(candidate)
    dist.all_reduce(totals)
    error, count, energy = totals.tolist()
    if count <= 0 or energy <= 0 or not all(math.isfinite(x) for x in (error, count, energy)):
        raise ValueError("Held-out MoE diagnostics require finite outputs and nonempty targets.")
    return {
        "cache_hits_local": cache.hits,
        "cache_misses_local": cache.misses,
        "cache_peak_bytes_local": cache.peak_bytes,
        "mse": error / count,
        "normalized_mse": error / energy,
        "elements": int(count),
        "tokens": int(count // target.shape[-1]),
    }


def _write(root, rows, metadata):
    payload = {"metadata": metadata, "results": rows}
    temporary = root / "results.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(root / "results.json")
    lines = [
        "| IQ1_S initialization | PTQ held-out MSE | Repaired held-out MSE | Reduction |",
        "|---|---:|---:|---:|",
    ]
    for method, row in rows.items():
        lines.append(
            f"| {method} | {row['ptq']['mse']:.8g} | {row['repair']['mse']:.8g} | "
            f"{row['reduction_pct']:.2f}% |"
        )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    """Run each PTQ/repair pair sequentially, retaining metrics instead of checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--keep-activations", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    source = json.loads((root / "source.json").read_text())
    output = root / "smoke" if args.smoke else root
    output.mkdir(exist_ok=True)
    previous = (
        json.loads((output / "results.json").read_text())
        if (output / "results.json").is_file()
        else None
    )
    fingerprint = hashlib.sha256((root / "data/manifest.json").read_bytes()).hexdigest()
    if previous is not None:
        if (
            previous["metadata"]["source"]["revision"] != source["revision"]
            or previous["metadata"]["capture"]["data_fingerprint"] != fingerprint
        ):
            raise ValueError("Existing results belong to different source or diagnostic data.")
        if set(args.methods) <= previous["results"].keys():
            if rank == 0:
                _write(output, previous["results"], previous["metadata"])
                print((output / "RESULTS.md").read_text(), flush=True)
                cache = root / "activations"
                if (
                    not args.smoke
                    and not args.keep_activations
                    and set(previous["results"]) == set(METHODS)
                    and cache.is_dir()
                ):
                    shutil.rmtree(cache)
            return
    # Completed reports can be read without a compatible AutoModel GPU runtime.
    from nemo_automodel.components.distributed.mesh import MeshContext

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    directory = root / ("smoke-activations" if args.smoke else "activations")
    receipts = [
        json.loads((directory / f"rank{i}-COMMITTED.json").read_text()) for i in range(size)
    ]
    if any(
        r["data_fingerprint"] != fingerprint
        or r["world_size"] != size
        or r["model_revision"] != source["revision"]
        or r["rank"] != i
        for i, r in enumerate(receipts)
    ):
        raise ValueError("Activation captures must share their exact data and rank topology.")
    ep = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
    mesh = init_device_mesh(
        "cuda", (1, 1, size, 1, 1), mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp")
    )
    mesh["dp_replicate", "dp_shard"]._flatten("dp")
    mesh["dp_replicate", "dp_shard", "cp"]._flatten("dp_cp")
    context = MeshContext.from_meshes(mesh, moe_mesh=ep)
    teacher = load_moe(source["source"], ep)
    calibration = ActivationBatches(
        directory, "calibration", rank, size, chunk=8192, global_batch=size
    )
    train = ActivationBatches(directory, "train", rank, size)
    validation = ActivationBatches(
        directory, "validation", rank, size, chunk=8192, global_batch=size
    )
    heldout = ActivationBatches(directory, "diagnostic", rank, size, chunk=8192, global_batch=size)
    rows = previous["results"] if previous is not None else {}
    metadata = {
        "source": source,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capture": receipts[0],
        "ep": size,
        "objective": "full native MoE output; routed expert scales only",
        "updates": 2 if args.smoke else 512,
        "global_batch": 8,
        "repair_sequence_length": 2048,
        "heldout_tokens": len(heldout) * size * 8192,
    }
    for method in args.methods:
        if method in rows:
            continue
        student = copy.deepcopy(teacher)
        adapter = AutoModelLocalDistillationAdapter(context)
        algorithm = (
            {"method": "max", "skip_forward_without_activation_calib": True}
            if method == "unweighted"
            else {"method": "imatrix"}
        )
        if method.startswith("gptq"):
            algorithm = {
                "method": "gptq",
                "block_size": 256,
                "perc_damp": 0.01,
                "importance_weighted": method == "gptq_importance",
                "fused": False,
            }
        settings = {
            "quantization": {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "experts.*weight_quantizers.*",
                        "cfg": {"num_bits": "iq1_s", "backend": "ggml"},
                    },
                ],
                "algorithm": algorithm,
            },
            "repair": {"units": [""]},
        }
        ptq = quantize_student(student, calibration, adapter, settings)
        before = measure(teacher, student, heldout)
        if rank == 0:
            print(json.dumps({"method": method, "ptq_heldout": before}), flush=True)
        started = time.perf_counter()
        result = mtd.local_distill(
            student,
            teacher,
            train,
            {
                "units": [""],
                "updates": metadata["updates"],
                "global_batch_size": 8,
                "micro_batch_size": 2,
                "sequence_length": 2048,
                "scale_lr": 0.01,
                "optimizer": "adagrad",
                "scheduler": "cosine",
                "warmup_fraction": 0.05,
                "selection": "final",
                "seed": 42,
            },
            validation_dataloader=validation,
            adapter=adapter,
        )
        after = measure(teacher, result.student, heldout)
        row = {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "ptq": before,
            "repair": after,
            "reduction_pct": 100 * (1 - after["mse"] / before["mse"]),
            "ptq_seconds_local": ptq["ptq_elapsed_seconds_local"],
            "repair_and_diagnostic_seconds_local": time.perf_counter() - started,
            "training": result.metrics,
        }
        rows[method] = row
        if rank == 0:
            _write(output, rows, metadata)
            print(
                json.dumps(
                    {
                        "method": method,
                        "ptq": before,
                        "repair": after,
                        "reduction_pct": row["reduction_pct"],
                    }
                ),
                flush=True,
            )
        del student, result, adapter
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    dist.barrier()
    if rank == 0 and not args.smoke and not args.keep_activations and set(rows) == set(METHODS):
        shutil.rmtree(directory)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

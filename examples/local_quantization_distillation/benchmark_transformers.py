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


"""Random-weight MoE prefix performance study; never writes model weights."""

import argparse
import importlib
import json
import os
import statistics
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq

from .benchmark_models import build_unpartitioned_models
from .benchmark_topology import data_lane, parameter_layouts, partition_models

__all__ = ["main"]


def build_models(args, device):
    student, teacher, config = build_unpartitioned_models(args, device)
    cfg = {
        "iq2_s": {"num_bits": "iq2_s", "backend": "ggml"},
        "iq2_xxs": {"num_bits": "iq2_xxs", "backend": "ggml"},
        "fp8": {"num_bits": (4, 3), "axis": None},
        "nvfp4": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        },
    }[args.format]
    mtq.quantize(
        student,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*mlp.experts.*weight_quantizer*", "cfg": cfg},
            ],
            "algorithm": "max",
        },
    )
    student, teacher, adapter = partition_models(args, student, teacher)
    return student, teacher, adapter, config


@contextmanager
def step_measurements(warmup, total, device):
    module = importlib.import_module("modelopt.torch.distill.local_distillation")
    original = module.print_rank_0
    values = {"seconds": [], "peak_allocated": 0, "peak_reserved": 0, "last": None}

    def progress(message, *args, **kwargs):
        try:
            data = json.loads(message)
        except (ValueError, TypeError):
            return original(message, *args, **kwargs)
        if "repair_step" not in data:
            return original(message, *args, **kwargs)
        torch.cuda.synchronize(device)
        now = time.perf_counter()
        step = data["repair_step"]
        if step > warmup and values["last"] is not None:
            values["seconds"].append(now - values["last"])
        values["last"] = now
        if step == warmup:
            torch.cuda.reset_peak_memory_stats(device)
        if step == total:
            values["peak_allocated"] = torch.cuda.max_memory_allocated(device)
            values["peak_reserved"] = torch.cuda.max_memory_reserved(device)
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {"step": step, "total": total, "elapsed_seconds": data["elapsed_seconds"]}
                ),
                flush=True,
            )

    module.print_rank_0 = progress
    try:
        yield values
    finally:
        module.print_rank_0 = original


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure random-weight Transformers prefixes; no model checkpoints are written.",
        epilog="WORLD_SIZE = pp * cp * dp * tp. EP overlaps dp * tp; efsdp = dp * tp / ep.",
    )
    p.add_argument("--family", choices=["qwen3", "moonlight"], default="qwen3")
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--init-device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", required=True)
    p.add_argument("--format", choices=["iq2_s", "iq2_xxs", "fp8", "nvfp4"], default="iq2_s")
    p.add_argument(
        "--dp", type=int, default=1, help="Unique-data lanes; dense FSDP size in fsdp mode."
    )
    p.add_argument(
        "--parameter-sharding",
        choices=["replicated", "fsdp"],
        default="fsdp",
        help="Replicated control supports pure DP only; fsdp uses native HF plans.",
    )
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1, help="Expert owners overlapping the dp * tp ranks.")
    p.add_argument(
        "--extend-context",
        action="store_true",
        help="Extend random-prefix position range; no quality claim.",
    )
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--hf-experts", choices=["eager", "grouped_mm"], default="grouped_mm")
    args = p.parse_args(argv)
    if (
        min(
            args.dp,
            args.cp,
            args.ep,
            args.pp,
            args.tp,
            args.layers,
            args.batch,
            args.micro_batch,
            args.seq_len,
            args.warmup,
            args.steps,
        )
        < 1
    ):
        p.error("Topology, batch, lengths and measurement counts must be positive.")
    if args.batch % (args.dp * args.micro_batch) or args.layers % args.pp:
        p.error("Batch must divide evenly across DP microbatches, and prefix depth across PP.")
    if args.parameter_sharding == "replicated":
        if args.pp * args.cp * args.ep * args.tp != 1:
            p.error("The replicated-parameter control requires pure DP (pp=cp=ep=tp=1).")
    else:
        if args.ep > 1 and (args.ep % args.tp or (args.dp * args.tp) % args.ep):
            p.error("Native EP must be a multiple of TP and divide dp * tp.")
        if args.pp > 1 and args.dp > 1:
            p.error("Native Transformers does not support combining FSDP with PP.")
        if args.pp > 1 and args.ep > 1:
            p.error("Native Transformers token-dispatch EP does not support PP.")
    world_size = args.dp * args.cp * args.pp * args.tp
    if int(os.environ.get("WORLD_SIZE", "1")) != world_size:
        p.error(f"WORLD_SIZE must equal pp * cp * dp * tp = {world_size}; EP overlaps DP/TP.")
    return args


def main():
    """Measure local repair and write a single timing and placement JSON report."""
    args = _parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=240))
    try:
        torch.set_float32_matmul_precision("highest")
        started = time.perf_counter()
        # Avoid compiling small Python wrappers; real native kernels remain enabled.
        with torch.compiler.set_stance("force_eager"):
            student, teacher, adapter, config = build_models(args, device)
            torch.manual_seed(39)
            ids = torch.randint(3, config.vocab_size, (args.batch, args.seq_len), device=device)
            lanes, lane = data_lane(adapter)
            ids = ids.chunk(lanes, dim=0)[lane]
            layouts = parameter_layouts(student)
            docs = torch.ones_like(ids)
            batch = {
                "input_ids": ids,
                "attention_mask": docs,
                "_packed_seq_ids": docs,
                "valid_mask": docs.bool(),
                "position_ids": torch.arange(args.seq_len, device=device).expand_as(ids),
            }
            total = args.warmup + args.steps
            torch.cuda.synchronize(device)
            setup_seconds = time.perf_counter() - started
            with step_measurements(args.warmup, total, device) as measurements:
                result = mtd.local_distill(
                    student,
                    teacher,
                    [batch] * total,
                    {
                        "updates": total,
                        "global_batch_size": args.batch,
                        "micro_batch_size": args.micro_batch,
                        "sequence_length": args.seq_len,
                        "warmup_fraction": 0,
                        "scheduler": "constant",
                        "report_interval": 1,
                    },
                    adapter=adapter,
                )
            local = {key: value for key, value in measurements.items() if key != "last"}
            assert len(local["seconds"]) == args.steps
            local["setup_seconds"] = setup_seconds
            local["parameter_layouts"] = layouts
            local["execution"] = adapter.execution_metadata()
            local["data_lanes"] = lanes
            local["data_lane"] = lane
            local["rank"] = dist.get_rank()
            local["selected_units"] = list(result.metrics["units"])
            all_values: list[dict] = [{} for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_values, local)
            if dist.get_rank() == 0:
                step_times = [
                    max(row["seconds"][i] for row in all_values) for i in range(args.steps)
                ]
                mean_seconds = max(sum(row["seconds"]) for row in all_values) / args.steps
                output = {
                    "args": vars(args),
                    "model_config": {
                        key: getattr(config, key, None)
                        for key in (
                            "model_type",
                            "hidden_size",
                            "intermediate_size",
                            "moe_intermediate_size",
                            "n_routed_experts",
                            "num_experts",
                            "num_experts_per_tok",
                            "num_hidden_layers",
                            "num_attention_heads",
                            "qk_nope_head_dim",
                            "qk_rope_head_dim",
                            "v_head_dim",
                            "first_k_dense_replace",
                        )
                    },
                    "seeds": {"weights": 42, "tokens": 39},
                    "unit_paths_global": sorted(
                        set().union(*(set(row["selected_units"]) for row in all_values))
                    ),
                    "gpu": torch.cuda.get_device_name(device),
                    "gpu_memory_total": torch.cuda.get_device_properties(device).total_memory,
                    "torch": torch.__version__,
                    "transformers": importlib.import_module("transformers").__version__,
                    "median_step_seconds": statistics.median(step_times),
                    "min_step_seconds": min(step_times),
                    "max_step_seconds": max(step_times),
                    "tokens_per_second": args.batch * args.seq_len / mean_seconds,
                    "peak_allocated_max": max(row["peak_allocated"] for row in all_values),
                    "peak_allocated_sum": sum(row["peak_allocated"] for row in all_values),
                    "peak_reserved_max": max(row["peak_reserved"] for row in all_values),
                    "ranks": all_values,
                }
                destination = Path(args.output)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_text(json.dumps(output, indent=2) + "\n")
                temporary.replace(destination)
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in output.items()
                            if key not in ("ranks", "args")
                        }
                    ),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

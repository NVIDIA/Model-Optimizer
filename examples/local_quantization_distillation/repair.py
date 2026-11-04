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

"""AutoModel GPTQ → independent local repair → native ModelOpt checkpoint."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter


def load_models(config_path, *, checkpoint=None, model_path=None):
    """Load frozen original/ModelOpt student through the same AutoModel topology."""
    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.distributed.config import DistributedSetup
    from nemo_automodel.components.distributed.mesh import ParallelismSizes

    settings = yaml.safe_load(Path(config_path).read_text())
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.manual_seed(settings.get("repair", {}).get("seed", 42))
    setup = DistributedSetup.build(
        "fsdp2",
        ParallelismSizes(**settings.get("parallelism", {})),
        pipeline_config=settings.get("pipeline"),
        moe_parallel_config=settings.get("moe_parallel"),
    )
    if setup.mesh_context.pp_size > 1:
        raise ValueError(
            "This full-model GPTQ/diagnostic CLI uses PP=1. For existing PP PTQ stages, "
            "call local_distill with the teacher AutoPipeline and its matching local model part."
        )
    options = dict(settings.get("model", {}))
    source = model_path or options.pop("source", None)
    if not source:
        raise ValueError("Set model.source in the YAML or pass --model.")
    options.pop("source", None)
    teacher = (
        NeMoAutoModelForCausalLM.from_pretrained(
            source, distributed_setup=setup, dtype=torch.bfloat16, **options
        )
        .eval()
        .requires_grad_(False)
    )
    student = (
        NeMoAutoModelForCausalLM.from_pretrained(
            source, distributed_setup=setup, dtype=torch.bfloat16, **options
        )
        .eval()
        .requires_grad_(False)
    )
    if dist.get_rank() == 0:
        dtypes = {}
        for name, parameter in student.named_parameters():
            if ".experts." in name or ".mlp." in name:
                key = str(parameter.dtype)
                dtypes[key] = dtypes.get(key, 0) + parameter.numel()
        print(json.dumps({"loaded_mlp_parameter_dtypes": dtypes}), flush=True)
    adapter = AutoModelLocalDistillationAdapter(setup.mesh_context)
    adapter._configure_native_attention(teacher)
    adapter._configure_native_attention(student)
    if checkpoint:
        directory = Path(checkpoint)
        if (directory / "COMMITTED.json").is_file():
            student = adapter.restore(student, directory)
        else:
            student = mto.restore(
                student, directory / "modelopt.pt" if directory.is_dir() else directory
            )
    return student, teacher, adapter, settings


def quantize_student(student, calibration, adapter, settings):
    """Calibrate an explicit ModelOpt PTQ recipe, preserving its native format."""
    recipe = settings["quantization"]
    algorithm = recipe["algorithm"]
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(json.dumps({"ptq": recipe}, indent=2), flush=True)
    started = time.perf_counter()

    def forward_loop(model):
        for batch in calibration:
            for microbatch in adapter.microbatches(
                batch, settings.get("calibration_micro_batch_size", 1)
            ):
                adapter.teacher_microbatch(model, [], microbatch, lambda unit, record: None)

    mtq.quantize(student, {**recipe, "algorithm": None})
    adapter.configure_quantization(student)
    inventory = {}
    selected = settings.get("repair", {}).get("units")
    if selected:
        for path in selected:
            quantizers = [
                (name, module)
                for name, module in student.get_submodule(path).named_modules()
                if isinstance(module, mtq.nn.TensorQuantizer) and module.is_enabled
            ]
            weights = [name for name, _ in quantizers if "weight_quantizer" in name]
            if not weights:
                raise ValueError(f"PTQ selected no weights in repair unit {path}")
            inventory[path] = {
                "weight_quantizers": len(weights),
                "input_quantizers": sum("input_quantizer" in name for name, _ in quantizers),
                "formats": sorted(
                    {f"{module.backend}:{module.num_bits}" for _, module in quantizers}
                ),
            }
    # GPTQ needs per-expert GEMM inputs. Keep the tutorial's fast grouped
    # backend for teacher execution and repair, using native eager experts
    # only during the existing PTQ calibration pass.
    from modelopt.torch.quantization.plugins.automodel import _QuantAutoModelExperts

    experts = [
        (module, module.use_torch_mm)
        for module in student.modules()
        if isinstance(module, _QuantAutoModelExperts) and hasattr(module, "use_torch_mm")
    ]
    try:
        for module, _ in experts:
            module.use_torch_mm = False
        mtq.calibrate(student, algorithm, forward_loop)
    finally:
        for module, grouped in experts:
            module.use_torch_mm = grouped
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "ptq_elapsed_seconds_local": time.perf_counter() - started,
        "ptq": recipe,
        "quantizer_inventory": inventory,
    }


def main():
    """Run GPTQ, repair an existing native checkpoint, or perform both stages."""
    from examples.local_quantization_distillation.prepare_data import PackedBatchSequence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model", help="Original teacher source; overrides model.source in the YAML."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", help="Existing ModelOpt PTQ checkpoint; skips GPTQ.")
    parser.add_argument("--ptq-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    student, teacher, adapter, settings = load_models(
        args.config, checkpoint=args.checkpoint, model_path=args.model
    )
    mesh = adapter.mesh_context
    config = mtd.LocalDistillationConfig(**settings.get("repair", {}))
    text = getattr(teacher.config, "text_config", teacher.config)
    maximum = getattr(text, "max_position_embeddings", None)
    if maximum is not None and config.sequence_length > maximum:
        raise ValueError("Repair context exceeds the source model's supported context length.")
    data_options = {
        "global_batch_size": config.global_batch_size,
        "dp_rank": mesh.device_mesh["dp"].get_local_rank(),
        "dp_size": mesh.dp_size or 1,
        "device": next(student.parameters()).device,
    }
    ptq_metrics = {}
    if not args.checkpoint:
        calibration = PackedBatchSequence(
            args.data,
            "calibration",
            **data_options,
            sequence_length=settings.get("calibration_sequence_length", 4096),
        )
        ptq_metrics = quantize_student(student, calibration, adapter, settings)
    if args.ptq_only:
        paths = adapter.publish(
            student, config.model_copy(update={"output_dir": args.output}), ptq_metrics
        )
    else:
        train = PackedBatchSequence(
            args.data, "train", **data_options, sequence_length=config.sequence_length
        )
        validation = PackedBatchSequence(
            args.data, "validation", **data_options, sequence_length=config.sequence_length
        )
        result = mtd.local_distill(
            student,
            teacher,
            train,
            config.model_copy(
                update={
                    "output_dir": args.output,
                    "checkpoint_dir": args.checkpoint_dir,
                    "resume_from": args.resume_from,
                }
            ),
            validation_dataloader=validation,
            adapter=adapter,
        )
        paths = result.report_paths
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {"artifacts": paths, "data_manifest": str(Path(args.data) / "manifest.json")},
                indent=2,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

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

"""PTQ, scale-only IQ Bypass, ordinary ModelOpt saving, and held-out diagnostics."""

import argparse
import copy
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, PreTrainedConfig

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter
from modelopt.torch.models import get_spec
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY

from .evaluate import evaluate_pair
from .prepare_data import PackedBatchSequence


def _load_teacher(args):
    raw, _ = PreTrainedConfig.get_config_dict(args.model, revision=args.revision)
    spec = get_spec(raw.get("model_type"))
    if spec is not None and spec.modeling_source == "modelopt":
        package = importlib.import_module("modelopt.torch.models." + spec.model_type)
        names = [name for name in package.__all__ if name.endswith("ForCausalLM")]
        if len(names) != 1:
            raise ValueError("Local text ports must export one unambiguous ForCausalLM task.")
        cls = getattr(package, names[0])
    else:
        architectures = raw.get("architectures") or []
        cls = (
            getattr(transformers, architectures[0], AutoModelForCausalLM)
            if architectures
            else AutoModelForCausalLM
        )
    model, info = cls.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=getattr(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise ValueError(
            "Load/convert this checkpoint with its native model/source importer first; "
            "the tutorial requires a complete floating teacher."
        )
    return model.to(args.device).eval()


def main(argv=None):
    """Run one unsharded model; ADVANCED.md shows caller-owned native parallel placement."""
    defaults = mtd.BypassConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Complete floating HF model or converted native text port."
    )
    parser.add_argument("--revision")
    parser.add_argument(
        "--data", required=True, help="Directory created by examples.bypass.prepare_data."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=sorted(IQ_FORMAT_REGISTRY), default="iq2_xxs")
    parser.add_argument(
        "--quant-config", help="Optional ordinary ModelOpt quantization JSON config."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    for field, kind in (
        ("num_steps", int),
        ("global_batch_size", int),
        ("sequence_length", int),
        ("max_lr", float),
        ("warmup_fraction", float),
        ("checkpoint_interval", int),
    ):
        parser.add_argument(
            "--" + field.replace("_", "-"), type=kind, default=getattr(defaults, field)
        )
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--layer-indices", type=int, nargs="+")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume-from")
    args = parser.parse_args(argv)
    config = mtd.BypassConfig(
        **{
            field: getattr(args, field)
            for field in (
                "num_steps",
                "global_batch_size",
                "sequence_length",
                "max_lr",
                "warmup_fraction",
                "checkpoint_interval",
                "modules",
                "layer_indices",
                "checkpoint_dir",
                "resume_from",
            )
        }
    )
    adapter = TransformersBypassAdapter()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "Use the explicit native mesh recipe in ADVANCED.md for multi-process placement."
        )
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output must be a new directory.")
    data = {
        partition: PackedBatchSequence(
            args.data,
            partition,
            device=args.device,
            global_batch_size=config.global_batch_size if partition == "train" else 1,
        )
        for partition in ("train", "calibration", "validation", "diagnostic")
    }
    if len(data["train"]) < config.num_steps:
        raise ValueError("Prepare enough unique training packs for the requested update budget.")
    if data["train"].sequence_length != config.sequence_length:
        raise ValueError("Training context must match the prepared pack length.")
    teacher = _load_teacher(args)
    student = copy.deepcopy(teacher)
    quant_config = (
        json.loads(Path(args.quant_config).read_text())
        if args.quant_config
        else {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                *[
                    {"quantizer_name": path, "cfg": {"num_bits": args.format, "backend": "ggml"}}
                    for path in adapter.weight_patterns(student, config)
                ],
            ],
            "algorithm": "max",
        }
    )

    def calibrate(model):
        for batch in data["calibration"]:
            adapter.teacher_microbatch(model, [], batch, lambda *_: None)

    mtq.quantize(student, quant_config, calibrate)
    mtd.bypass(student, teacher, data["train"], config, validation_dataloader=data["validation"])
    diagnostics = evaluate_pair(teacher, student, data["diagnostic"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        mto.save(student, pending / "student.pt")
        (pending / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        os.replace(pending, output)
    except BaseException:
        shutil.rmtree(pending)
        raise
    print(json.dumps(diagnostics, indent=2))
    return student


if __name__ == "__main__":
    main()

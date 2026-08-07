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

"""Calibration-free W4A16 weight-only PTQ for a speculative-decoding drafter.

Block-wise ``max`` quantization derives every scale from the weight tensor itself, so
this needs neither a dataset nor a forward pass -- only the drafter's safetensors file.
That in turn means we never have to import the drafter's modeling code: each 2-D weight
is wrapped in a throwaway ``nn.Linear`` under its checkpoint name, and ModelOpt's normal
``quantizer_name`` patterns select over those names exactly as they would on the real
module tree. Works for any drafter layout (DSpark / DFlash / EAGLE3 / Medusa).

The format is fixed to ModelOpt's ``w4a16_nvfp4`` preset (E2M1 values, block 16, FP8 E4M3
scales). It is calibration-free because the block scales are dynamic; AWQ formats are
deliberately not offered, since ``awq_lite`` silently degrades to plain RTN when no
``forward_loop`` is supplied.

Example:
    python quantize_drafter_w4a16.py \
        --drafter_path nvidia/MiniMax-M3-DSpark \
        --export_path ./MiniMax-M3-DSpark-W4A16
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

import modelopt.torch.quantization as mtq
from modelopt.recipe.presets import QUANT_CFG_CHOICES
from modelopt.torch.export.quant_utils import (
    get_quant_config,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    to_quantized_weight,
)
from modelopt.torch.quantization.utils import is_quantized_linear

# NVFP4 weight-only: block-16 E2M1 values with dynamic FP8 E4M3 scales, so every scale is
# derived from the weight block itself and no calibration data is involved.
QFORMAT = "w4a16_nvfp4"

# The Markov head writes straight into the draft logits and is only a few percent of the
# drafter, so the bits it would save are not worth the acceptance-rate risk. It is also not
# a plain GEMM in the real module tree -- markov_w1 is an nn.Embedding, which ModelOpt's
# stock presets already exclude via `parent_class: nn.Embedding`; the flat-linear view here
# has lost the original module classes, so the exclusion has to be restated by name.
# lm_head is excluded by the preset itself (see --quantize_lm_head).
DEFAULT_EXCLUDE = ["*markov_head*"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--drafter_path", required=True, help="HF repo id or local dir of the drafter checkpoint."
    )
    parser.add_argument("--export_path", required=True, help="Output directory.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype the weights are cast to before quantizing.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="PATTERN",
        help="Extra fnmatch patterns to leave unquantized, in `quantizer_name` form. "
        f"Appended to the defaults ({' '.join(DEFAULT_EXCLUDE)}), which always apply.",
    )
    parser.add_argument(
        "--quantize_lm_head",
        action="store_true",
        help="Also quantize lm_head. It is the single largest drafter tensor, but it feeds "
        "the acceptance test directly -- measure AL before shipping this.",
    )
    return parser.parse_args()


def load_drafter(drafter_path: str) -> tuple[Path, dict[str, torch.Tensor]]:
    """Resolve a local dir or HF repo id to (dir, state_dict)."""
    local_dir = Path(drafter_path)
    if not local_dir.is_dir():
        from huggingface_hub import snapshot_download

        local_dir = Path(snapshot_download(drafter_path))

    shards = sorted(local_dir.glob("*.safetensors"))
    assert shards, f"No .safetensors found under {local_dir}"
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        state_dict.update(load_file(shard))
    return local_dir, state_dict


def build_linear_view(state_dict: dict[str, torch.Tensor], dtype: torch.dtype) -> nn.Module:
    """Expose every 2-D weight as an nn.Linear whose module name is its checkpoint key.

    Nested ModuleDicts are used so that ``named_modules()`` reproduces the dotted keys
    (``layers.0.self_attn.q_proj``), which is what the preset's ``quantizer_name``
    patterns match against.
    """
    root = nn.ModuleDict()
    for key, weight in state_dict.items():
        if weight.dim() != 2 or not key.endswith(".weight"):
            continue
        *parents, leaf = key[: -len(".weight")].split(".")
        node = root
        for part in parents:
            if part not in node:
                node[part] = nn.ModuleDict()
            node = node[part]
        out_features, in_features = weight.shape
        linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
        linear.weight.data = weight.to(dtype)
        node[leaf] = linear
    return root


def build_quant_cfg(exclude: list[str], quantize_lm_head: bool) -> dict:
    """Take the shipped preset and layer the drafter-specific exclusions on top."""
    quant_cfg = copy.deepcopy(QUANT_CFG_CHOICES[QFORMAT])
    if quantize_lm_head:
        quant_cfg["quant_cfg"].append(
            {"quantizer_name": "*lm_head*weight_quantizer", "enable": True}
        )
    for pattern in DEFAULT_EXCLUDE + exclude:
        quant_cfg["quant_cfg"].append({"quantizer_name": pattern, "enable": False})
    return quant_cfg


def export_quantized_state_dict(
    root: nn.Module, state_dict: dict[str, torch.Tensor], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Pack each quantized weight and emit it alongside its scales.

    Follows the unified-HF naming convention: ``w.weight`` / ``w.weight_scale`` /
    ``w.weight_scale_2``. Untouched tensors are carried through in ``dtype``.
    """
    export_sd = {k: v.to(dtype) for k, v in state_dict.items()}
    for name, module in root.named_modules():
        if not is_quantized_linear(module) or not module.weight_quantizer.is_enabled:
            continue
        quantization = get_quantization_format(module)
        assert quantization is not None, f"{name}: enabled quantizer resolved to no format"
        weight_scale = get_weight_scaling_factor(module)
        weight_scale_2 = get_weight_scaling_factor_2(module)
        export_sd[f"{name}.weight"] = to_quantized_weight(
            module.weight,
            weight_scale,
            quantization,
            weight_scale_2,
            get_weight_block_size(module),
        )
        export_sd[f"{name}.weight_scale"] = weight_scale
        if weight_scale_2 is not None:
            export_sd[f"{name}.weight_scale_2"] = weight_scale_2
    return export_sd


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    source_dir, state_dict = load_drafter(args.drafter_path)
    root = build_linear_view(state_dict, dtype)

    quant_cfg = build_quant_cfg(args.exclude, args.quantize_lm_head)
    mtq.quantize(root, quant_cfg)  # no forward_loop: scales come from the weights
    mtq.print_quant_summary(root)

    export_sd = export_quantized_state_dict(root, state_dict, dtype)

    export_dir = Path(args.export_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    save_file(export_sd, export_dir / "model.safetensors", metadata={"format": "pt"})

    config = json.loads((source_dir / "config.json").read_text())
    hf_quant_config = get_quant_config(root)
    config["quantization_config"] = hf_quant_config["quantization"]
    config["torch_dtype"] = args.dtype
    (export_dir / "config.json").write_text(json.dumps(config, indent=2))
    (export_dir / "hf_quant_config.json").write_text(json.dumps(hf_quant_config, indent=2))

    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (source_dir / extra).is_file():
            shutil.copy2(source_dir / extra, export_dir / extra)

    before = sum(v.numel() * v.element_size() for v in state_dict.values())
    after = sum(v.numel() * v.element_size() for v in export_sd.values())
    print(f"\n{QFORMAT}: {before / 2**30:.2f} GiB -> {after / 2**30:.2f} GiB")
    print(f"Exported to {export_dir}")


if __name__ == "__main__":
    main()

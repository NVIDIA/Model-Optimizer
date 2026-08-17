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

"""Calibration-free PTQ for a speculative-decoding drafter.

Every scale is derived from the weights, so this needs no dataset and no forward pass --
only the drafter's safetensors file. That also means the drafter's modeling code is never
imported: each 2-D weight is wrapped in a throwaway ``nn.Linear`` under its checkpoint
name, and ModelOpt's usual ``quantizer_name`` patterns select over those names. Works for
any drafter layout (DSpark / DFlash / EAGLE3 / Medusa).

Avoiding the modeling code is what makes this work for DFlash-family drafters at all: an
exported drafter declares ``architectures: ["DFlashDraftModel"]`` but ships no importable
class, so ``AutoModelForCausalLM`` silently loads it as a plain Qwen3 and drops the draft
tensors, and the real draft ``forward`` takes hidden states rather than ``input_ids``, so
the stock ``hf_ptq.py`` calibration loop cannot drive it either.

``fp8`` and ``nvfp4`` quantize activations against a static amax that is normally measured
on calibration data; a fixed ``input_scale`` of 1.0 is applied instead. That holds up
because acceptance length is governed by clipping rather than resolution: on Qwen3-8B +
a DSpark drafter (MT-Bench), AL falls off a cliff below input_scale ~0.03 but sits on a
flat plateau from ~0.3 to 4.0 with no drop-off at the top, so the scale only has to be big
enough. 1.0 is the middle of that plateau, and measures +0.1% AL for FP8 and -3.9% for
NVFP4 (that gap is the 4-bit resolution cost; no scale recovers it). Deriving the amax
from the weights instead clips badly and measures -31% to -46%.

AWQ formats are deliberately not offered: ``awq_lite`` silently degrades to plain RTN when
no ``forward_loop`` is supplied.

Example:
    python quantize_drafter.py \
        --drafter_path nvidia/MiniMax-M3-DSpark \
        --qformat fp8 \
        --export_path ./MiniMax-M3-DSpark-FP8
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
    get_activation_scaling_factor,
    get_quant_config,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    to_quantized_weight,
)
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.quantization.utils import is_quantized_linear

# INT8/INT4 weight-only formats are deliberately absent: vLLM's ModelOpt backend accepts
# only FP8, FP8_PER_CHANNEL_PER_TOKEN, FP8_PB_WO, NVFP4, W4A16_NVFP4, MXFP8 and
# MIXED_PRECISION, so an INT checkpoint quantizes cleanly but cannot be served.
SUPPORTED_QFORMATS = [
    "w4a16_nvfp4",
    "nvfp4",
    "fp8",
    "fp8_pc_pt",
    "fp8_pb_wo",
]

# These are 2-D, so the flat-linear view treats them as GEMMs, but none of them is one:
# markov_w1 and embed_tokens are embeddings (row lookups), which ModelOpt's presets
# normally skip via `parent_class: nn.Embedding` -- a class the flat view cannot see, so
# the exclusion is restated by name. Quantizing embed_tokens also breaks deployment, since
# a drafter inherits it from the target (vLLM: KeyError: 'embed_tokens.weight_scale').
# confidence_head projects to a single output, whose per-channel scale collapses to a
# 0-dim tensor. All are tiny; nothing is lost by leaving them alone.
# Not excluded: `fc` (the presets decide; use --exclude '*fc*' to compare) and `lm_head`
# (the preset already excludes it -- see --quantize_lm_head).
DEFAULT_EXCLUDE = ["*markov_head*", "*confidence_head*", "*embed_tokens*"]

# The amax that yields input_scale 1.0 for FP8. NVFP4 divides by 6*448 instead, so the same
# amax records as input_scale 0.1667 there; both mean the same activation range.
FP8_E4M3_MAX = 448.0
STATIC_ACT_AMAX = FP8_E4M3_MAX


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--drafter_path", required=True, help="HF repo id or local dir of the drafter checkpoint."
    )
    parser.add_argument("--export_path", required=True, help="Output directory.")
    parser.add_argument(
        "--qformat",
        default="w4a16_nvfp4",
        choices=SUPPORTED_QFORMATS,
        help="Quantization format. All are calibration-free: weight-only and "
        "dynamic-activation formats derive every scale from the weights or per token at "
        "runtime, and the static-activation formats (fp8, nvfp4) use a fixed input_scale "
        "of 1.0.",
    )
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


def set_static_activation_amax(root: nn.Module, amax: float = STATIC_ACT_AMAX) -> int:
    """Give every static ``input_quantizer`` the same fixed amax. Returns how many were set.

    Skips quantizers that are dynamic (scale computed per token at runtime) or that already
    have an amax, so this composes as a fallback rather than an overwrite.
    """
    count = 0
    for _, module in root.named_modules():
        if not is_quantized_linear(module):
            continue
        input_quantizer = getattr(module, "input_quantizer", None)
        if input_quantizer is None or not input_quantizer.is_enabled:
            continue
        # A dynamic quantizer derives its scale at runtime; leave it untouched.
        if getattr(input_quantizer, "_dynamic", False):
            continue
        if getattr(input_quantizer, "amax", None) is not None:
            continue
        input_quantizer.amax = torch.tensor(amax, dtype=torch.float32).to(module.weight.dtype)
        count += 1
    return count


def resolve_activation_scales(root: nn.Module, quant_cfg: dict) -> None:
    """Establish activation scales for a format that quantizes activations statically.

    The single place deciding where a static amax comes from. Real calibration would slot
    in here ahead of the fixed fallback, leaving the CLI and call site unchanged::

        if calib_forward_loop is not None:
            mtq.calibrate(root, quant_cfg["algorithm"], forward_loop=calib_forward_loop)
        set_static_activation_amax(root)  # fills in what calibration did not reach
    """
    if not need_calibration(quant_cfg):
        return
    n = set_static_activation_amax(root)
    print(f"Set {n} static activation amax values (fixed, input_scale 1.0) -- not calibrated.")


def build_quant_cfg(qformat: str, exclude: list[str], quantize_lm_head: bool) -> dict:
    """Take the shipped preset and layer the drafter-specific exclusions on top."""
    quant_cfg = copy.deepcopy(QUANT_CFG_CHOICES[qformat])
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
        # A per-channel scale over a single-output projection collapses to a 0-dim tensor
        # that the packing helpers index as ``scale[:, None]``. Skip rather than crash --
        # it is one row of weights and stays in ``dtype``.
        if weight_scale is not None and weight_scale.dim() == 0 and module.weight.shape[0] == 1:
            print(f"Skipping {name}: single-output projection, per-channel scale is scalar")
            continue
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
        # Static-activation formats need the input scale in the checkpoint too; without it
        # the runtime has no activation scale to apply and the format silently degrades.
        activation_scale = get_activation_scaling_factor(module)
        if activation_scale is not None:
            export_sd[f"{name}.input_scale"] = activation_scale
    return export_sd


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    source_dir, state_dict = load_drafter(args.drafter_path)
    root = build_linear_view(state_dict, dtype)

    quant_cfg = build_quant_cfg(args.qformat, args.exclude, args.quantize_lm_head)

    mtq.quantize(root, quant_cfg)  # no forward_loop: scales come from the weights
    resolve_activation_scales(root, quant_cfg)

    mtq.print_quant_summary(root)

    export_sd = export_quantized_state_dict(root, state_dict, dtype)

    export_dir = Path(args.export_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    save_file(export_sd, export_dir / "model.safetensors", metadata={"format": "pt"})

    config = json.loads((source_dir / "config.json").read_text())
    hf_quant_config = get_quant_config(root)
    # ``get_quant_config`` only knows the modules in the linear view, so anything it never
    # saw (norms, 1-D weights) is missing from ``exclude_modules`` and a loader walking the
    # checkpoint expects a ``weight_scale`` for it. List every unquantized weight instead.
    quantized = {
        name
        for name, module in root.named_modules()
        if is_quantized_linear(module)
        and f"{name}.weight" in export_sd
        and f"{name}.weight_scale" in export_sd
    }
    unquantized = sorted(
        key[: -len(".weight")]
        for key in state_dict
        if key.endswith(".weight") and key[: -len(".weight")] not in quantized
    )
    exclude_modules = hf_quant_config["quantization"].get("exclude_modules", [])
    for name in unquantized:
        if name not in exclude_modules:
            exclude_modules.append(name)
        # A runtime matches this against its own module prefix, which is nested relative to
        # the checkpoint key (vLLM builds the draft's ``fc`` at ``model.fc``). Add a suffix
        # wildcard so the exclusion matches at any depth.
        wildcard = f"*{name}"
        if wildcard not in exclude_modules:
            exclude_modules.append(wildcard)
    # Runtimes fuse sibling projections into one layer whose name appears in no checkpoint
    # key (q/k/v -> ``qkv_proj``, gate/up -> ``gate_up_proj``), so excluding only the parts
    # leaves the fused layer quantized. Emit the alias once every component is excluded.
    for fused, parts in (
        ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
        ("gate_up_proj", ("gate_proj", "up_proj")),
    ):
        if all(any(p in name for name in exclude_modules) for p in parts):
            alias = f"*{fused}"
            if alias not in exclude_modules:
                exclude_modules.append(alias)
    hf_quant_config["quantization"]["exclude_modules"] = exclude_modules
    config["quantization_config"] = dict(hf_quant_config["quantization"])
    # ModelOpt names the format ``quant_algo``; vLLM reads ``quant_method`` and treats its
    # absence as unquantized. Emit both. vLLM splits ModelOpt checkpoints across two
    # backends: ``modelopt_fp4`` for block-scaled NVFP4, ``modelopt`` for the rest.
    quant_algo = str(hf_quant_config["quantization"].get("quant_algo") or "")
    config["quantization_config"].setdefault(
        "quant_method", "modelopt_fp4" if "NVFP4" in quant_algo.upper() else "modelopt"
    )
    # The exclusion list is read under two different keys: ModelOpt nests it under
    # ``quantization.exclude_modules``, but a runtime parsing the flat
    # ``quantization_config`` in config.json looks for ``ignore``. Emit both.
    config["quantization_config"]["ignore"] = list(exclude_modules)
    config["torch_dtype"] = args.dtype
    (export_dir / "config.json").write_text(json.dumps(config, indent=2))
    (export_dir / "hf_quant_config.json").write_text(json.dumps(hf_quant_config, indent=2))

    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (source_dir / extra).is_file():
            shutil.copy2(source_dir / extra, export_dir / extra)

    before = sum(v.numel() * v.element_size() for v in state_dict.values())
    after = sum(v.numel() * v.element_size() for v in export_sd.values())
    print(f"\n{args.qformat}: {before / 2**30:.2f} GiB -> {after / 2**30:.2f} GiB")
    print(f"Exported to {export_dir}")


if __name__ == "__main__":
    main()

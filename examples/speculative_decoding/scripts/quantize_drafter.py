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

Every scale is derived from the weights, so this needs no dataset and no forward pass, and
never imports the drafter's modeling code: each 2-D weight is wrapped in a throwaway
``nn.Linear`` under its checkpoint name, and ModelOpt's usual ``quantizer_name`` patterns
select over those names. Works for any drafter layout (DSpark / DFlash / EAGLE3 / Medusa),
including exported ones that ship no importable model class.

``fp8`` and ``nvfp4`` need a static activation amax, normally measured on calibration data;
a fixed ``input_scale`` of 1.0 is applied instead. Acceptance length is governed by
clipping rather than resolution, and AL sits on a flat plateau from input_scale ~0.3 to 4.0
(Qwen3-8B + DSpark, MT-Bench: +0.1% for FP8, -3.9% for NVFP4), so the scale only has to be
big enough. AWQ is not offered: ``awq_lite`` silently degrades to RTN without a
``forward_loop``.

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

# INT8/INT4 are absent on purpose: they quantize cleanly but vLLM's ModelOpt backend
# cannot serve them.
SUPPORTED_QFORMATS = [
    "w4a16_nvfp4",
    "nvfp4",
    "fp8",
    "fp8_pc_pt",
]

# All 2-D, so the flat view treats them as GEMMs, but none is one: markov_w1/embed_tokens
# are embeddings (ModelOpt's presets skip these via `parent_class`, which the flat view
# cannot see), and confidence_head has a single output whose per-channel scale is 0-dim.
# All tiny. `fc` is left to the presets; `lm_head` is excluded by the preset itself.
DEFAULT_EXCLUDE = ["*markov_head*", "*confidence_head*", "*embed_tokens*"]

# Sidecars carried over to the export, and -- with the weights and config.json -- the only
# files fetched when --drafter_path is a repo id rather than a local directory.
SIDECAR_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")

# The amax that yields input_scale 1.0 for FP8. NVFP4 divides by 6*448, so the same amax
# records as 0.1667 there; both mean the same activation range.
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
        help="Quantization format. All are calibration-free; fp8 and nvfp4 additionally "
        "quantize activations, using a fixed input_scale of 1.0.",
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
        help="Also quantize lm_head -- the largest drafter tensor, but it feeds the "
        "acceptance test directly, so measure AL first.",
    )
    return parser.parse_args()


def auto_map_modules(config: dict, source_dir: Path) -> set[Path]:
    """Module files an ``auto_map`` points at, relative to ``source_dir``.

    Values are either ``"modeling_x.XModel"`` or, for tokenizers, a list whose entries may
    be null (``[null, "tokenization_x.XTokenizerFast"]``); a ``repo--`` prefix points at
    another repository and is not a local file. A value may also carry a package path
    (``"pkg/modeling_x.XModel"``), which is preserved so the exported reference resolves.

    The config is untrusted -- it may come straight off the Hub -- so references that
    escape ``source_dir`` are dropped rather than followed.
    """
    modules = set()
    for value in (config.get("auto_map") or {}).values():
        for ref in value if isinstance(value, list) else [value]:
            if not isinstance(ref, str) or "." not in ref:
                continue
            relative = Path(ref.split("--")[-1].rsplit(".", 1)[0] + ".py")
            candidate = (source_dir / relative).resolve()
            if relative.is_absolute() or not candidate.is_relative_to(source_dir.resolve()):
                print(f"Skipping auto_map entry outside the checkpoint: {ref}")
                continue
            modules.add(relative)
    return modules


def load_drafter(drafter_path: str) -> tuple[Path, dict[str, torch.Tensor]]:
    """Resolve a local dir or HF repo id to (dir, state_dict)."""
    local_dir = Path(drafter_path)
    if not local_dir.is_dir():
        from huggingface_hub import snapshot_download

        # A drafter that ships custom modeling code references it from auto_map, and the
        # export carries that config verbatim -- so those .py files have to be fetched too
        # or the exported references dangle.
        local_dir = Path(
            snapshot_download(
                drafter_path,
                allow_patterns=["*.safetensors", "config.json", "*.py", *SIDECAR_FILES],
            )
        )

    shards = sorted(local_dir.glob("*.safetensors"))
    assert shards, f"No .safetensors found under {local_dir}"
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        state_dict.update(load_file(shard))
    return local_dir, state_dict


def build_linear_view(state_dict: dict[str, torch.Tensor], dtype: torch.dtype) -> nn.Module:
    """Expose every 2-D weight as an nn.Linear whose module name is its checkpoint key.

    Nested ModuleDicts so ``named_modules()`` reproduces the dotted checkpoint keys, which
    is what ``quantizer_name`` patterns match against.
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
        # On meta, so nn.Linear skips allocating and randomly initializing a weight that
        # the next line replaces anyway.
        with torch.device("meta"):
            linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
        linear.weight = nn.Parameter(weight.to(dtype), requires_grad=False)
        node[leaf] = linear
    return root


def set_static_activation_amax(root: nn.Module, amax: float = STATIC_ACT_AMAX) -> int:
    """Give every static ``input_quantizer`` the same fixed amax. Returns how many were set.

    Skips dynamic quantizers and any that already have an amax, so it composes as a
    fallback rather than an overwrite.
    """
    count = 0
    for _, module in root.named_modules():
        if not is_quantized_linear(module):
            continue
        input_quantizer = getattr(module, "input_quantizer", None)
        if input_quantizer is None or not input_quantizer.is_enabled:
            continue
        if getattr(input_quantizer, "_dynamic", False):
            continue
        if getattr(input_quantizer, "amax", None) is not None:
            continue
        # Keep amax in fp32, as ModelOpt does everywhere else -- casting to the weight
        # dtype would round a measured amax through bf16's 8-bit mantissa.
        input_quantizer.amax = torch.tensor(amax, dtype=torch.float32)
        count += 1
    return count


def resolve_activation_scales(root: nn.Module, quant_cfg: dict) -> None:
    """Establish activation scales for a format that quantizes activations statically.

    The single place deciding where a static amax comes from: real calibration would call
    ``mtq.calibrate`` here, ahead of the fixed fallback.
    """
    if not need_calibration(quant_cfg):
        return
    n = set_static_activation_amax(root)
    print(f"Set {n} static activation amax values (fixed, input_scale 1.0) -- not calibrated.")


def build_quant_cfg(qformat: str, exclude: list[str], quantize_lm_head: bool) -> dict:
    """Take the shipped preset and layer the drafter-specific exclusions on top."""
    quant_cfg = copy.deepcopy(QUANT_CFG_CHOICES[qformat])
    if quantize_lm_head:
        # The preset disables *all* of lm_head's quantizers. Re-enabling only the weight
        # one would leave a W+A format exporting lm_head with no input_scale while the
        # config still advertises it as fully quantized, which a runtime fails to load.
        for quantizer in ("weight_quantizer", "input_quantizer"):
            quant_cfg["quant_cfg"].append(
                {"quantizer_name": f"*lm_head*{quantizer}", "enable": True}
            )
    for pattern in DEFAULT_EXCLUDE + exclude:
        quant_cfg["quant_cfg"].append({"quantizer_name": pattern, "enable": False})
    return quant_cfg


def export_quantized_state_dict(
    root: nn.Module, state_dict: dict[str, torch.Tensor], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Pack each quantized weight and emit it alongside its scales.

    Unified-HF naming (``w.weight_scale`` etc). Untouched tensors carry through in ``dtype``.
    """
    export_sd = {k: v.to(dtype) for k, v in state_dict.items()}
    for name, module in root.named_modules():
        if not is_quantized_linear(module) or not module.weight_quantizer.is_enabled:
            continue
        quantization = get_quantization_format(module)
        assert quantization is not None, f"{name}: enabled quantizer resolved to no format"
        weight_scale = get_weight_scaling_factor(module)
        weight_scale_2 = get_weight_scaling_factor_2(module)
        # The packing helpers index the scale as ``scale[:, None]``, which a 0-dim scale
        # cannot satisfy. One row of weights, so leave it in ``dtype``.
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
        # Without this the runtime has no activation scale and the format silently degrades.
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
    # ``get_quant_config`` only knows the linear view, so tensors it never saw (norms, 1-D
    # weights) are missing and a loader walking the checkpoint expects a scale for them.
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
        # Runtimes match against their own module prefix, which is nested relative to the
        # checkpoint key (vLLM builds the draft's ``fc`` at ``model.fc``).
        wildcard = f"*{name}"
        if wildcard not in exclude_modules:
            exclude_modules.append(wildcard)
    # Runtimes fuse sibling projections into one layer whose name is in no checkpoint key,
    # so excluding only the parts would leave the fused layer quantized.
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
    # absence as unquantized, splitting NVFP4 off into its own backend. Emit both.
    quant_algo = str(hf_quant_config["quantization"].get("quant_algo") or "")
    config["quantization_config"].setdefault(
        "quant_method", "modelopt_fp4" if "NVFP4" in quant_algo.upper() else "modelopt"
    )
    # Same list, second key: the flat ``quantization_config`` in config.json is read for
    # ``ignore``, not ``exclude_modules``.
    config["quantization_config"]["ignore"] = list(exclude_modules)
    config["torch_dtype"] = args.dtype
    (export_dir / "config.json").write_text(json.dumps(config, indent=2))
    (export_dir / "hf_quant_config.json").write_text(json.dumps(hf_quant_config, indent=2))

    for extra in SIDECAR_FILES:
        if (source_dir / extra).is_file():
            shutil.copy2(source_dir / extra, export_dir / extra)

    # A drafter that ships custom modeling code points at it from auto_map; the export
    # carries that config verbatim, so the .py files have to come along or the reference
    # dangles. (The DFlash/DSpark exports have no auto_map -- this is for the ones that do.)
    for relative in auto_map_modules(config, source_dir):
        source_py = source_dir / relative
        if source_py.is_file():
            target = export_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_py, target)

    before = sum(v.numel() * v.element_size() for v in state_dict.values())
    after = sum(v.numel() * v.element_size() for v in export_sd.values())
    print(f"\n{args.qformat}: {before / 2**30:.2f} GiB -> {after / 2**30:.2f} GiB")
    print(f"Exported to {export_dir}")


if __name__ == "__main__":
    main()

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

Block-wise ``max`` quantization derives every scale from the weight tensor itself, so
this needs neither a dataset nor a forward pass -- only the drafter's safetensors file.
That in turn means we never have to import the drafter's modeling code: each 2-D weight
is wrapped in a throwaway ``nn.Linear`` under its checkpoint name, and ModelOpt's normal
``quantizer_name`` patterns select over those names exactly as they would on the real
module tree. Works for any drafter layout (DSpark / DFlash / EAGLE3 / Medusa).

Avoiding the modeling code is what makes this work at all for DFlash-family drafters. An
exported drafter declares ``architectures: ["DFlashDraftModel"]`` but ships no importable
class, so ``AutoModelForCausalLM`` silently loads it as a plain Qwen3 and drops ``fc`` /
``hidden_norm`` / the DSpark heads; and the real draft module's ``forward`` takes
``(noise_embedding, target_hidden, ...)``, not ``input_ids``, so the stock ``hf_ptq.py``
calibration loop cannot drive it either. The flat-linear view sidesteps both problems.

Two families of format are supported:

* **Weight-only / dynamic-activation** (``w4a16_nvfp4``, ``fp8_pc_pt``, ``fp8_pb_wo``) --
  every scale is either derived from the weight or computed at runtime per token.
* **Static-activation** (``fp8``, ``nvfp4``) -- these need an ``input_quantizer`` amax,
  normally measured with calibration data. Instead a fixed ``input_scale`` of 1.0 is
  applied to every layer.

A fixed activation scale sounds crude but measures well, because acceptance length is
governed almost entirely by *clipping* rather than by resolution. Sweeping input_scale over
three decades on Qwen3-8B + a DSpark drafter (MT-Bench, 80 questions):

===========  ==============  ==============
input_scale  FP8 AL          NVFP4 AL
===========  ==============  ==============
0.003        2.220 (-29.3%)  2.208 (-29.8%)
0.03         2.975  (-5.3%)  2.926  (-6.9%)
0.3          3.137  (-0.2%)  3.022  (-3.8%)
1.0          3.146  (+0.1%)  3.019  (-3.9%)
4.0          3.125  (-0.6%)  3.003  (-4.4%)
===========  ==============  ==============

(bf16 baseline 3.142.) Both formats fall off a cliff below ~0.03, where the declared range
is far under the activations' true magnitude and most of the tensor is clipped, and both
sit on a flat plateau from ~0.3 to 4.0 with no drop-off at the top -- so the scale needs to
be big enough, and little else. 1.0 sits in the middle of that plateau, which is why it is
fixed rather than exposed as a knob. NVFP4 trails FP8 by a roughly constant 3.5% across the
plateau: that gap is the 4-bit resolution cost itself and no choice of scale recovers it.

Estimating the amax from the weights instead was tried and does not work: max|W| averages
0.79 while a RMSNorm'd activation is O(1) with outlier channels in the tens, so the range
lands 1-2 orders of magnitude low and clips. That measured -31% to -46% AL.

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

# Formats this script offers, in the order they are most likely to be wanted.
#   w4a16_nvfp4 -- block-16 E2M1 weights, dynamic FP8 E4M3 scales. Smallest, weight-only.
#   nvfp4       -- same weights plus NVFP4 activations (static amax -> needs a scale).
#   fp8         -- E4M3 weights and activations, per-tensor (static amax -> needs a scale).
#   fp8_pc_pt   -- E4M3 per-channel weights, DYNAMIC per-token activations. The
#                  calibration-free way to get FP8 activations.
#   fp8_pb_wo   -- E4M3 per-block weight-only fallback.
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

# The Markov head writes straight into the draft logits and is only a few percent of the
# drafter, so the bits it would save are not worth the acceptance-rate risk. It is also not
# a plain GEMM in the real module tree -- markov_w1 is an nn.Embedding, which ModelOpt's
# stock presets already exclude via `parent_class: nn.Embedding`; the flat-linear view here
# has lost the original module classes, so the exclusion has to be restated by name.
#
# The confidence head is excluded for the same reason plus a mechanical one: it projects to
# a single output (``[1, hidden]``), so a per-channel scale collapses to a 0-dim tensor and
# the per-channel export path (fp8_pc_pt) indexes it as ``scale[:, None]`` and raises. It is
# one row of weights, so there is nothing to gain by quantizing it.
#
# ``fc`` is deliberately NOT excluded here: it is the single largest non-lm_head tensor and
# the formats' own presets decide its fate. Exclude it explicitly with --exclude '*fc*' when
# comparing acceptance length.
# lm_head is excluded by the preset itself (see --quantize_lm_head).
#
# ``embed_tokens`` must be excluded too. It is 2-D, so the flat-linear view happily treats
# it as a GEMM, but it is an ``nn.Embedding``: a row lookup, not a matmul. ModelOpt's stock
# presets skip embeddings via ``parent_class: nn.Embedding``, which the flat view cannot
# see. Quantizing it also breaks deployment -- a drafter inherits ``embed_tokens`` from the
# target, and vLLM's loader then fails with ``KeyError: 'embed_tokens.weight_scale'``.
DEFAULT_EXCLUDE = ["*markov_head*", "*confidence_head*", "*embed_tokens*"]

# Static activation amax applied to every layer of a static-activation format, expressed as
# the amax that yields input_scale 1.0. The exported input_scale is amax/448 for FP8 but
# amax/(6*448) for NVFP4, so this is the FP8 convention; an NVFP4 checkpoint built from the
# same amax records input_scale 0.1667. See the module docstring for why 1.0 and why this is
# not a CLI knob: the usable plateau spans ~0.3 to 4.0, so the value only has to be big
# enough, and 1.0 sits in the middle of it.
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

    Static-activation formats (fp8, nvfp4) need an ``input_quantizer`` amax, which is
    normally *measured* by pushing calibration data through the model. A drafter cannot be
    driven that way here: it has no importable modeling code, and its real forward takes
    hidden states rather than token ids. See the module docstring for why one fixed value
    stands in for that measurement, and why 1.0.

    Only quantizers that are enabled and still missing an amax are touched, so
    dynamic-activation formats -- whose scales are computed per token at runtime -- are left
    alone, and any amax already established (by a future calibration pass, see
    ``resolve_activation_scales``) wins over the fixed default.
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

    The single place that decides *where* a static activation amax comes from. Today there
    is one source -- a fixed value applied uniformly -- because the drafter cannot be run
    forward without its modeling code. Real calibration would slot in here as a second
    source ahead of the fixed fallback::

        if calib_forward_loop is not None:
            mtq.calibrate(root, quant_cfg["algorithm"], forward_loop=calib_forward_loop)
        set_static_activation_amax(root)  # fills in whatever calibration did not reach

    ``set_static_activation_amax`` deliberately skips quantizers that already have an amax,
    so it composes as a fallback rather than overwriting measured values, and the callers
    below do not change when that day comes.
    """
    if not need_calibration(quant_cfg):
        # Weight-only, or activations scaled per token at runtime -- nothing to establish.
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
        # A per-channel scale over a single-output projection (``[1, hidden]``) collapses to
        # a 0-dim tensor that the packing helpers index as ``scale[:, None]``. Such a layer
        # is one row of weights, so skip it rather than crash -- it stays in ``dtype``.
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
        # Static-activation formats also need the input scale in the checkpoint. Without
        # it the runtime has no activation scale to apply, so every heuristic produces a
        # byte-identical export and the format silently degrades.
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
    # ``get_quant_config`` only knows about the modules in the linear view, so anything the
    # view never saw -- embeddings, norms, and any 1-D weight -- is absent from
    # ``exclude_modules``. A loader that walks the checkpoint (vLLM does) then expects a
    # ``weight_scale`` for those too and dies with e.g. KeyError: 'embed_tokens.weight_scale'.
    # List every weight that was not quantized so the exclusion set is complete.
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
        # A runtime matches this list against its own module prefix, which is usually
        # nested relative to the checkpoint key (vLLM builds the draft's ``fc`` at
        # ``model.fc`` via ``maybe_prefix``). A bare ``fc`` then fails to match, the layer
        # is built quantized, and loading the bf16 weight into a packed parameter raises a
        # size assertion. Add a suffix wildcard so the exclusion matches at any depth.
        wildcard = f"*{name}"
        if wildcard not in exclude_modules:
            exclude_modules.append(wildcard)
    # Runtimes fuse sibling projections into one layer whose name appears in no checkpoint
    # key: q/k/v -> ``qkv_proj``, gate/up -> ``gate_up_proj``. Excluding only the individual
    # names leaves the fused layer quantized, and its merged shard shapes then disagree with
    # the unpacked weights being loaded into it. Emit the fused aliases whenever every
    # component of a fusion group was excluded.
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
    # ModelOpt names the format ``quant_algo``; vLLM's ModelConfig reads
    # ``quant_cfg["quant_method"]`` and treats a config without that key as unquantized,
    # then dies on the packed weight shapes. Emit both so either loader is satisfied.
    # vLLM splits ModelOpt checkpoints into two backends: ``modelopt_fp4`` for the
    # block-scaled NVFP4 layouts and ``modelopt`` for the per-tensor/per-channel ones.
    quant_algo = str(hf_quant_config["quantization"].get("quant_algo") or "")
    config["quantization_config"].setdefault(
        "quant_method", "modelopt_fp4" if "NVFP4" in quant_algo.upper() else "modelopt"
    )
    # The exclusion list is read under two different keys. ModelOpt's own
    # ``hf_quant_config.json`` nests it under ``quantization.exclude_modules``, but when a
    # runtime parses the flat ``quantization_config`` block inside ``config.json`` it looks
    # for ``ignore`` (vLLM's ModelOptQuantConfigBase.from_config). Emitting only
    # ``exclude_modules`` there yields an empty exclusion set, every layer is built
    # quantized, and loading an untouched bf16 weight into a packed parameter raises.
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

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

"""Export MiniMax-M3 as vendor MXFP8 base + NVFP4 routed experts via ``hf_ptq.py``.

This wrapper keeps ``hf_ptq.py`` as the owner of loading, quantization, and HF
checkpoint export for the BF16 source checkpoint, then post-processes that
intermediate export into the MiniMax mixed checkpoint:

* non-routed-expert tensors come unchanged from the vendor MXFP8 checkpoint
* routed-expert tensors come from the NVFP4 checkpoint exported by ``hf_ptq.py``
* routed-expert ``input_scale`` tensors are forced to 1.0 by default

Any unrecognized arguments are forwarded to ``examples/llm_ptq/hf_ptq.py``.

Usage:
    python hf_ptq_mixed_mxfp8_nvfp4.py \\
        --mxfp8_ckpt /models/minimax-m3-mxfp8 \\
        --bf16_ckpt /models/minimax-m3-bf16 \\
        --recipe /workspace/quant/recipe.yaml \\
        --output_ckpt /workspace/quant/minimax-m3-mxfp8-nvfp4-mixed \\
        --device cuda --trust_remote_code
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_LLM_PTQ_DIR = _THIS_DIR.parent / "llm_ptq"
for _path in (str(_THIS_DIR), str(_LLM_PTQ_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import hf_ptq  # noqa: E402

_BLOCK = 16
_EXPERT_TENSOR_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<L>\d+)\.block_sparse_moe\.experts\.\d+\.w[123]\."
)


_CONTROLLED_HF_PTQ_ARGS = {
    "--model",
    "--pyt_ckpt_path",
    "--qformat",
    "--recipe",
    "--export_path",
}


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _validate_passthrough_args(args: list[str]) -> None:
    controlled = [
        arg.split("=", 1)[0]
        for arg in args
        if arg.split("=", 1)[0] in _CONTROLLED_HF_PTQ_ARGS
    ]
    if controlled:
        joined = ", ".join(sorted(set(controlled)))
        raise ValueError(f"These hf_ptq arguments are controlled by the wrapper: {joined}")


def _default_nvfp4_export_path(output_ckpt: Path) -> Path:
    return output_ckpt.with_name(f"{output_ckpt.name}.hf_ptq_nvfp4")


def _hf_ptq_argv(
    args: argparse.Namespace, nvfp4_export_path: Path, passthrough: list[str]
) -> list[str]:
    argv = [
        "--pyt_ckpt_path",
        str(args.bf16_ckpt),
        "--recipe",
        str(args.recipe),
        "--qformat",
        "nvfp4",
        "--export_path",
        str(nvfp4_export_path),
        "--skip_generate",
    ]
    if not _has_option(passthrough, "--batch_size"):
        argv.extend(["--batch_size", "1"])
    if not _has_option(passthrough, "--calib_size"):
        argv.extend(["--calib_size", "1"])
    argv.extend(passthrough)
    return argv


def _load_index(src: Path) -> dict[str, str]:
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    return idx["weight_map"]


def _is_routed_expert_tensor(key: str) -> bool:
    return bool(_EXPERT_TENSOR_RE.match(key))


def _selected_layers(layer_keys: dict[int, list[str]], layers_arg: str | None) -> list[int]:
    layers = sorted(layer_keys)
    if layers_arg:
        want = {int(x) for x in layers_arg.split(",")}
        layers = [layer for layer in layers if layer in want]
    return layers


def _build_quantized_layers(
    mxfp8_map: dict[str, str], nvfp4_expert_mods: list[str]
) -> dict[str, dict[str, Any]]:
    ql: dict[str, dict[str, Any]] = {}
    for key in mxfp8_map:
        if key.endswith(".weight_scale_inv"):
            mod = key[: -len(".weight_scale_inv")]
            if "block_sparse_moe.experts." in mod:
                continue
            ql[mod] = {"quant_algo": "MXFP8"}
    for mod in nvfp4_expert_mods:
        ql[mod] = {"quant_algo": "NVFP4", "group_size": _BLOCK}
    return ql


def _build_quant_config(
    quantized_layers: dict[str, dict[str, Any]],
    kv_cache_quant_algo: str | None,
    exclude_modules: list[str],
) -> dict[str, Any]:
    return {
        "producer": {"name": "modelopt", "version": "minimax-m3-mxfp8-nvfp4-mixed"},
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": kv_cache_quant_algo,
            "exclude_modules": exclude_modules,
            "quantized_layers": quantized_layers,
        },
    }


def _copy_experts_from_nvfp4_export(
    nvfp4: Path,
    dst: Path,
    layers_arg: str | None,
    force_input_scale_one: bool,
) -> tuple[dict[str, str], list[str]]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    nvfp4_map = _load_index(nvfp4)
    layer_keys: dict[int, list[str]] = defaultdict(list)
    for key in nvfp4_map:
        match = _EXPERT_TENSOR_RE.match(key)
        if match:
            layer_keys[int(match.group("L"))].append(key)
    layers = _selected_layers(layer_keys, layers_arg)
    print(f"[mixed] {len(layers)} MoE layers; experts NVFP4<-{nvfp4}, base MXFP8<-vendor")

    new_index: dict[str, str] = {}
    nvfp4_expert_mods: list[str] = []
    for layer_idx, layer in enumerate(layers):
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key in sorted(layer_keys[layer]):
            by_shard[nvfp4_map[key]].append(key)
        shard_name = f"experts-layer-{layer:03d}.safetensors"
        full = {}
        for shard, keys in by_shard.items():
            with safe_open(str(nvfp4 / shard), framework="pt", device="cpu") as handle:
                for key in keys:
                    full[key] = handle.get_tensor(key)
        if force_input_scale_one:
            for key in list(full):
                if key.endswith(".input_scale"):
                    full[key] = torch.tensor(1.0, dtype=torch.float32).reshape(())
            for key in list(full):
                if key.endswith(".weight"):
                    full.setdefault(
                        f"{key[: -len('.weight')]}.input_scale",
                        torch.tensor(1.0, dtype=torch.float32).reshape(()),
                    )
        save_file(full, str(dst / shard_name))
        for key in full:
            new_index[key] = shard_name
            if key.endswith(".weight"):
                nvfp4_expert_mods.append(key[: -len(".weight")])
        print(
            f"[mixed] layer {layer} ({layer_idx + 1}/{len(layers)}): "
            f"copied {len(full)} NVFP4 expert tensors"
        )
    return new_index, nvfp4_expert_mods


def _copy_mxfp8_bf16_from_base(
    mxfp8: Path, dst: Path, mxfp8_map: dict[str, str], new_index: dict[str, str]
) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_shards = sorted({mxfp8_map[key] for key in mxfp8_map})
    for shard_idx, shard in enumerate(src_shards):
        passthrough = {}
        with safe_open(str(mxfp8 / shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                if _is_routed_expert_tensor(key):
                    continue
                passthrough[key] = handle.get_tensor(key)
        if not passthrough:
            continue
        out_name = f"base-mxfp8-{shard_idx:05d}.safetensors"
        save_file(passthrough, str(dst / out_name))
        for key in passthrough:
            new_index[key] = out_name
        print(f"[mixed] base shard {shard} -> {out_name}: {len(passthrough)} tensors")


def _write_mixed_config(
    mxfp8: Path,
    dst: Path,
    mxfp8_map: dict[str, str],
    nvfp4_expert_mods: list[str],
) -> None:
    mxfp8_cfg = json.loads((mxfp8 / "config.json").read_text())
    vendor_q = mxfp8_cfg.get("quantization_config", {})
    exclude_modules = list(vendor_q.get("ignored_layers", []) or [])
    kv_cache = vendor_q.get("kv_cache_quant_algo")

    quantized_layers = _build_quantized_layers(mxfp8_map, nvfp4_expert_mods)
    num_mxfp8 = sum(1 for value in quantized_layers.values() if value["quant_algo"] == "MXFP8")
    num_nvfp4 = sum(1 for value in quantized_layers.values() if value["quant_algo"] == "NVFP4")
    print(f"[mixed] quantized_layers: {num_mxfp8} MXFP8 + {num_nvfp4} NVFP4")

    hf_quant = _build_quant_config(quantized_layers, kv_cache, exclude_modules)
    mxfp8_cfg["quantization_config"] = hf_quant["quantization"]
    mxfp8_cfg["quantization_config"]["quant_method"] = "modelopt"
    (dst / "config.json").write_text(json.dumps(mxfp8_cfg, indent=2))
    (dst / "hf_quant_config.json").write_text(json.dumps(hf_quant, indent=2))


def _export_mixed_mxfp8_nvfp4(
    mxfp8_ckpt: Path,
    nvfp4_ckpt: Path,
    output_ckpt: Path,
    layers: str | None,
    force_input_scale_one: bool,
) -> None:
    mxfp8 = Path(mxfp8_ckpt)
    nvfp4 = Path(nvfp4_ckpt)
    dst = Path(output_ckpt)
    dst.mkdir(parents=True, exist_ok=True)

    mxfp8_map = _load_index(mxfp8)
    new_index, nvfp4_expert_mods = _copy_experts_from_nvfp4_export(
        nvfp4, dst, layers, force_input_scale_one
    )
    _copy_mxfp8_bf16_from_base(mxfp8, dst, mxfp8_map, new_index)
    _write_mixed_config(mxfp8, dst, mxfp8_map, nvfp4_expert_mods)

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "pt"}, "weight_map": new_index}, indent=2)
    )

    for item in mxfp8.iterdir():
        if item.name in {"config.json", "model.safetensors.index.json"} or item.name.endswith(
            ".safetensors"
        ):
            continue
        if item.is_file():
            shutil.copy2(item, dst / item.name)
    print(f"[mixed] done -> {dst}")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mxfp8_ckpt", required=True, help="vendor MiniMax-M3-MXFP8 checkpoint")
    parser.add_argument("--bf16_ckpt", required=True, help="BF16 source checkpoint for hf_ptq")
    parser.add_argument("--recipe", required=True, help="expert-only NVFP4 PTQ recipe")
    parser.add_argument("--output_ckpt", required=True, help="final mixed checkpoint output")
    parser.add_argument(
        "--nvfp4_export_path",
        default=None,
        help="intermediate hf_ptq NVFP4 export path; defaults next to --output_ckpt",
    )
    parser.add_argument("--layers", default=None, help="comma list of layer indices (debug subset)")
    parser.add_argument(
        "--preserve_nvfp4_input_scale",
        action="store_true",
        help="preserve hf_ptq expert input_scale tensors instead of forcing final values to 1.0",
    )
    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    _validate_passthrough_args(passthrough)
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()

    output_ckpt = Path(args.output_ckpt)
    nvfp4_export_path = (
        Path(args.nvfp4_export_path)
        if args.nvfp4_export_path is not None
        else _default_nvfp4_export_path(output_ckpt)
    )

    if nvfp4_export_path.exists():
        raise FileExistsError(
            f"{nvfp4_export_path} already exists; choose a different --nvfp4_export_path."
        )
    hf_ptq.main(hf_ptq.prepare_args(hf_ptq.parse_args(_hf_ptq_argv(args, nvfp4_export_path, passthrough))))

    _export_mixed_mxfp8_nvfp4(
        Path(args.mxfp8_ckpt),
        nvfp4_export_path,
        output_ckpt,
        layers=args.layers,
        force_input_scale_one=not args.preserve_nvfp4_input_scale,
    )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Content checks for an exported unified HuggingFace checkpoint.

Existence checks pass even when the exporter silently drops a whole module family, so compare
the exported tensors against the HuggingFace checkpoint the model came from.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

__all__ = [
    "assert_expert_scales_match_source",
    "assert_exported_checkpoint_matches",
    "assert_per_expert_experts_complete",
    "assert_safetensors_index_consistent",
    "fused_experts_source_weights",
    "load_safetensors_dir",
]

# Per-tensor / per-channel scales the exporter adds for quantized weights. These have no
# counterpart in the (unquantized) reference checkpoint.
QUANT_SUFFIXES = (
    "weight_scale",
    "weight_scale_2",
    "input_scale",
    "output_scale",
    "k_scale",
    "v_scale",
)


def load_safetensors_dir(path: Path | str) -> dict[str, torch.Tensor]:
    """Load every safetensors shard under ``path`` into one dict."""
    state_dict: dict[str, torch.Tensor] = {}
    for shard in sorted(Path(path).glob("*.safetensors")):
        state_dict.update(load_file(str(shard)))
    assert state_dict, f"No safetensors tensors found in {path}"
    return state_dict


def assert_safetensors_index_consistent(export_dir: Path | str) -> None:
    """Assert ``model.safetensors.index.json`` matches the shards actually written."""
    export_dir = Path(export_dir)
    index_file = export_dir / "model.safetensors.index.json"
    if not index_file.exists():  # single unsharded file: nothing to cross-check
        return
    weight_map = json.loads(index_file.read_text())["weight_map"]
    missing_files = {f for f in set(weight_map.values()) if not (export_dir / f).exists()}
    assert not missing_files, f"index.json references missing shards: {sorted(missing_files)}"
    exported = set(load_safetensors_dir(export_dir))
    assert set(weight_map) == exported, (
        f"index.json disagrees with the shards: {sorted(set(weight_map) - exported)[:10]} only in "
        f"index, {sorted(exported - set(weight_map))[:10]} only in shards"
    )


def _is_packed(tensor: torch.Tensor) -> bool:
    """Whether a tensor holds sub-byte weights packed two-per-``uint8`` (NVFP4 / INT4)."""
    return tensor.dtype == torch.uint8


def _expected_shape(exported: torch.Tensor, reference: torch.Tensor) -> tuple[int, ...]:
    """Reference shape, with the last dim halved when the export packs two 4-bit values per byte."""
    shape = tuple(reference.shape)
    if _is_packed(exported) and shape:
        return (*shape[:-1], shape[-1] // 2)
    return shape


def _unpack_nvfp4(packed: torch.Tensor) -> torch.Tensor:
    """Expand two E2M1 values per ``uint8`` back into a float tensor (low nibble first)."""
    codes = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.long, device=packed.device
    )
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    return NVFP4QTensor.get_e2m1_values(packed.device).to(torch.float32)[codes]


def _scale_key(key: str, suffix: str) -> str:
    """Scale key for the dotted (``...proj.weight``) and packed (``...gate_up_proj``) layouts."""
    return key.replace(".weight", f".{suffix}") if key.endswith(".weight") else f"{key}_{suffix}"


def _dequantize(key: str, exported: dict[str, torch.Tensor]) -> torch.Tensor:
    """Undo the exporter's weight scaling so the value can be compared to the source."""
    scale = exported.get(_scale_key(key, "weight_scale"))
    if _is_packed(exported[key]):
        # NVFP4: E2M1 codes, an E4M3 scale per block along the last dim, and a global FP32 scale.
        weight = _unpack_nvfp4(exported[key])
        block_size = weight.shape[-1] // scale.shape[-1]
        weight = weight * scale.to(torch.float32).repeat_interleave(block_size, dim=-1)
        scale_2 = exported.get(_scale_key(key, "weight_scale_2"))
        return weight if scale_2 is None else weight * scale_2.to(torch.float32)
    weight = exported[key].to(torch.float32)
    if scale is None:
        return weight
    scale = scale.to(torch.float32)
    return weight * scale.reshape(*scale.shape, *([1] * (weight.ndim - scale.ndim)))


def assert_exported_checkpoint_matches(
    export_dir: Path | str,
    ref_hf_dir: Path | str,
    *,
    allow_missing: tuple[str, ...] = (),
    allow_unexpected: tuple[str, ...] = (),
    check_values: bool = True,
    bit_exact_prefixes: tuple[str, ...] = (),
    rtol: float = 0.15,
    packed_rtol: float = 0.3,
) -> None:
    """Assert an exported unified HF checkpoint reproduces the reference model it came from.

    Args:
        export_dir: Exported unified HF checkpoint.
        ref_hf_dir: HuggingFace checkpoint the Megatron model was built from.
        allow_missing: Substrings of reference keys the export may omit.
        allow_unexpected: Substrings of extra exported keys, beyond quantization scales.
        check_values: Compare values too; ``False`` when the Megatron weights are random.
        bit_exact_prefixes: Keys that must match exactly even when ``check_values`` is off.
        rtol: Max relative error for quantized tensors.
        packed_rtol: Same for sub-byte formats, whose grid is coarser.
    """
    exported = load_safetensors_dir(export_dir)
    reference = load_safetensors_dir(ref_hf_dir)
    assert_safetensors_index_consistent(export_dir)

    missing = {k for k in set(reference) - set(exported) if not any(a in k for a in allow_missing)}
    assert not missing, (
        f"{len(missing)} reference tensor(s) absent from the export, e.g. {sorted(missing)[:8]}"
    )

    # Anything extra must be a quantization scale; a stray weight means a mis-named rule.
    unexpected = {
        k
        for k in set(exported) - set(reference)
        if not k.endswith(QUANT_SUFFIXES) and not any(a in k for a in allow_unexpected)
    }
    assert not unexpected, f"Export produced unexpected tensors: {sorted(unexpected)[:8]}"

    shared = sorted(set(exported) & set(reference))
    mismatched = [
        (k, tuple(exported[k].shape), tuple(reference[k].shape))
        for k in shared
        if tuple(exported[k].shape) != _expected_shape(exported[k], reference[k])
    ]
    assert not mismatched, f"Shape mismatches (key, exported, reference): {mismatched[:8]}"

    if bit_exact_prefixes:
        drifted = [
            k
            for k in shared
            if k.startswith(bit_exact_prefixes) and not torch.equal(exported[k], reference[k])
        ]
        assert not drifted, (
            f"{len(drifted)} copied-through tensor(s) differ from the reference: {drifted[:8]}"
        )
        checked = [k for k in shared if k.startswith(bit_exact_prefixes)]
        assert checked, f"bit_exact_prefixes {bit_exact_prefixes} matched no exported tensor"

    if not check_values:
        return

    wrong = []
    for key in shared:
        got, want = _dequantize(key, exported), reference[key].to(torch.float32)
        if (
            exported[key].dtype == reference[key].dtype
            and _scale_key(key, "weight_scale") not in exported
        ):
            # Copied through untouched (norms, router, vision tower): must be bit-exact.
            if not torch.equal(exported[key], reference[key]):
                wrong.append((key, "not bit-exact"))
        else:
            denom = want.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
            rel = ((got - want).abs().max() / denom).item()
            tol = packed_rtol if _is_packed(exported[key]) else rtol
            if rel > tol:
                wrong.append((key, f"max_rel_err={rel:.4f} > {tol}"))
    assert not wrong, f"{len(wrong)} tensor(s) differ from the reference, e.g. {wrong[:8]}"


def assert_per_expert_experts_complete(export_dir: Path | str) -> None:
    """Every routed expert must export all three projections.

    Callers waive the packed-vs-per-expert name difference wholesale via ``allow_unexpected``, so
    without this a rule that emitted only gate/up would still pass the reference comparison.
    """
    pattern = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    found: dict[tuple[str, int], set[str]] = {}
    for key in load_safetensors_dir(export_dir):
        m = pattern.match(key)
        if m:
            found.setdefault((m.group(1), int(m.group(2))), set()).add(m.group(3))
    assert found, "no per-expert routed-expert tensors in the export"
    incomplete = {k: sorted(v) for k, v in found.items() if len(v) != 3}
    assert not incomplete, f"experts missing projections: {sorted(incomplete.items())[:4]}"


def fused_experts_source_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Weights of every gated fused-experts module, keyed like the per-expert export names them."""
    source = {}
    for name, module in model.named_modules():
        if hasattr(module, "gate_up_proj") and hasattr(module, "num_experts"):
            gate, up = module.gate_up_proj.detach().clone().chunk(2, dim=1)
            down = module.down_proj.detach().clone()
            for idx in range(module.num_experts):
                source[f"{name}.{idx}.gate_proj.weight"] = gate[idx]
                source[f"{name}.{idx}.up_proj.weight"] = up[idx]
                source[f"{name}.{idx}.down_proj.weight"] = down[idx]
    return source


_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
# Projections that share one weight quantizer, hence one scale: vLLM fuses gate and up at load.
_SCALE_SHARING_PROJECTIONS = (("gate_proj", "up_proj"), ("down_proj",))


def assert_expert_scales_match_source(
    exported: Mapping[str, torch.Tensor], source: Mapping[str, torch.Tensor], *, rtol: float = 1e-5
) -> int:
    """Assert every routed expert's weight scales come from that expert's own source weights.

    A global scale borrowed from a larger expert moves the dequantized values by a few percent, inside
    the tolerance of :func:`assert_exported_checkpoint_matches`, so compare the scales themselves:
    NVFP4 ``weight_scale_2 == amax / (6 * 448)`` with the expert's largest block scale at 448, and
    per-tensor FP8 ``weight_scale == amax / 448``, where ``amax`` is taken over the projections that
    share a quantizer. Experts with equal amax cannot tell a borrowed scale from their own, so the
    caller must make them differ.

    Args:
        exported: Exported tensors keyed ``<prefix><expert>.<proj>.<weight_scale | weight_scale_2>``.
        source: Unquantized weights keyed ``<prefix><expert>.<proj>.weight``.

    Returns:
        The number of experts checked, for the caller to assert the check was not vacuous.
    """
    pattern = re.compile(r"^(.*?)(\d+)\.(gate_proj|up_proj|down_proj)\.weight_scale(?:_2)?$")
    exported_projections: dict[tuple[str, str], set[str]] = {}
    for key in exported:
        match = pattern.match(key)
        if match:
            exported_projections.setdefault((match[1], match[2]), set()).add(match[3])

    wrong, checked = [], set()
    for (prefix, expert), present in sorted(exported_projections.items()):
        for group in _SCALE_SHARING_PROJECTIONS:
            names = [f"{prefix}{expert}.{proj}" for proj in group if proj in present]
            if not names:
                continue
            nvfp4 = f"{names[0]}.weight_scale_2" in exported
            if not nvfp4 and exported[f"{names[0]}.weight_scale"].numel() != 1:
                continue  # per-channel / per-block FP8: no single scale to compare
            amax = max(source[f"{name}.weight"].abs().amax().float() for name in names)
            want = amax / (_E2M1_MAX * _E4M3_MAX if nvfp4 else _E4M3_MAX)
            scale_name = "weight_scale_2" if nvfp4 else "weight_scale"
            for name in names:
                got = exported[f"{name}.{scale_name}"].float()
                if not torch.isclose(got, want, rtol=rtol, atol=0.0).all():
                    wrong.append(f"{name}.{scale_name}={got.flatten()[0]:.6g}, want {want:.6g}")
            if nvfp4:
                largest_block = max(
                    exported[f"{name}.weight_scale"].float().max() for name in names
                )
                if largest_block != _E4M3_MAX:
                    wrong.append(f"{names[0]}: largest block scale {largest_block}, want 448")
            checked.add((prefix, expert))
    assert not wrong, (
        f"{len(wrong)} scale(s) not derived from the expert's own weights: {wrong[:6]}"
    )
    return len(checked)

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

Existence checks (``config.json`` is there, some safetensors were written) pass even when the
exporter silently drops a whole module family, so these helpers compare the exported tensors
against the Hugging Face checkpoint the model came from.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

__all__ = [
    "assert_exported_checkpoint_matches",
    "assert_safetensors_index_consistent",
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


def _dequantize(key: str, exported: dict[str, torch.Tensor]) -> torch.Tensor:
    """Undo per-tensor / per-channel weight scaling so the value can be compared to the source."""
    weight = exported[key].to(torch.float32)
    scale = exported.get(key.replace(".weight", ".weight_scale"))
    if scale is None:
        return weight
    scale = scale.to(torch.float32)
    return weight * (scale if scale.ndim == 0 else scale.reshape(-1, *([1] * (weight.ndim - 1))))


def assert_exported_checkpoint_matches(
    export_dir: Path | str,
    ref_hf_dir: Path | str,
    *,
    allow_missing: tuple[str, ...] = (),
    allow_unexpected: tuple[str, ...] = (),
    check_values: bool = True,
    rtol: float = 0.15,
) -> None:
    """Assert an exported unified HF checkpoint reproduces the reference model it came from.

    Args:
        export_dir: Directory holding the exported unified HF checkpoint.
        ref_hf_dir: The HuggingFace checkpoint the Megatron model was built from.
        allow_missing: Substrings of reference keys the export is expected to omit.
        allow_unexpected: Substrings of exported keys with no reference counterpart, beyond the
            quantization scales that are always allowed.
        check_values: Compare tensor values, not just names and shapes. Set ``False`` when the
            Megatron weights are random rather than loaded from ``ref_hf_dir``.
        rtol: Max relative error for quantized tensors (per-tensor FP8 lands well inside 0.15).
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

    if not check_values:
        return

    wrong = []
    for key in shared:
        if _is_packed(exported[key]):
            continue  # sub-byte weights need format-specific unpacking to compare
        got, want = _dequantize(key, exported), reference[key].to(torch.float32)
        if exported[key].dtype == reference[key].dtype and key + "_scale" not in exported:
            # Copied through untouched (norms, router, vision tower): must be bit-exact.
            if not torch.equal(exported[key], reference[key]):
                wrong.append((key, "not bit-exact"))
        else:
            denom = want.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
            rel = ((got - want).abs().max() / denom).item()
            if rel > rtol:
                wrong.append((key, f"max_rel_err={rel:.4f} > {rtol}"))
    assert not wrong, f"{len(wrong)} tensor(s) differ from the reference, e.g. {wrong[:8]}"

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

"""Compare the quantization scales of two PTQ runs.

Export both runs as safetensors checkpoints and diff every scale tensor::

    python -m modelopt.torch.quantization.utils.scale_compare ckpt_a ckpt_b --rtol 1e-3

The exit status is 0 when all scales agree within the tolerance, so the command doubles as a
reproducibility check for repeated runs or different parallel layouts. :func:`compare_scales`
takes any two ``{name: tensor}`` mappings, e.g. flattened ``get_quantizer_state_dict`` outputs.
"""

import argparse
import re
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

__all__ = ["ScaleDiff", "TensorDiff", "compare_scales", "load_checkpoint_scales", "main"]

# Exported scales (input_scale, weight_scale, weight_scale_2, weight_scale_inv, pre_quant_scale,
# k_scale, v_scale) and the quantizer buffers (_amax) they come from.
_SCALE_NAME = re.compile(r"(_scale(_2|_inv)?|_amax)$")


@dataclass
class TensorDiff:
    """Difference between one scale tensor of the two runs."""

    name: str
    elements: int
    outside_tolerance: int
    max_rel: float  # largest |a - b| / max(|a|, |b|) over the elements

    @property
    def kind(self) -> str:
        """Last component of the name, e.g. ``input_scale``."""
        return self.name.rsplit(".", 1)[-1]


@dataclass
class ScaleDiff:
    """Result of :func:`compare_scales`."""

    tensors: list[TensorDiff] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    shape_mismatch: list[str] = field(default_factory=list)

    @property
    def within_tolerance(self) -> bool:
        """Whether tensors were compared and none is missing, reshaped or outside the tolerance."""
        return (
            bool(self.tensors)
            and not (self.only_in_a or self.only_in_b or self.shape_mismatch)
            and all(t.outside_tolerance == 0 for t in self.tensors)
        )

    def summary(self, top: int = 5) -> str:
        """Per-kind table (kind = last component of the tensor name) and the worst tensors.

        ``differing`` counts the tensors outside the tolerance and ``median rel`` is the median of
        their largest relative differences; ``max rel`` is the largest over all tensors of the
        kind, including those inside the tolerance.
        """
        by_kind: dict[str, list[TensorDiff]] = {}
        for t in self.tensors:
            by_kind.setdefault(t.kind, []).append(t)
        lines = [
            f"{'kind':<16}{'tensors':>9}{'differing':>11}{'diff elems':>12}"
            f"{'median rel':>12}{'max rel':>10}"
        ]
        for kind, ts in sorted(by_kind.items()):
            differing = [t.max_rel for t in ts if t.outside_tolerance]
            elements = sum(t.elements for t in ts)
            outside = sum(t.outside_tolerance for t in ts)
            median = statistics.median(differing) if differing else 0.0
            lines.append(
                f"{kind:<16}{len(ts):>9}{len(differing):>11}{outside / max(elements, 1):>12.2%}"
                f"{median:>12.2e}{max(t.max_rel for t in ts):>10.2e}"
            )
        worst = sorted((t for t in self.tensors if t.outside_tolerance), key=lambda t: -t.max_rel)
        if worst:
            lines.append("worst:")
            lines += [f"  {t.max_rel:.2e}  {t.name}" for t in worst[:top]]
        for label, names in (
            ("only in A", self.only_in_a),
            ("only in B", self.only_in_b),
            ("shape mismatch", self.shape_mismatch),
        ):
            if names:
                lines.append(f"{label}: {len(names)} tensors, e.g. {names[0]}")
        lines.append("result: " + ("within tolerance" if self.within_tolerance else "DIFFERENT"))
        return "\n".join(lines)


def _diff(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> tuple[int, float]:
    """Return the number of elements outside the tolerance and the largest relative difference."""
    dtype = torch.float64 if torch.float64 in (a.dtype, b.dtype) else torch.float32
    a, b = a.to(dtype), b.to(dtype)  # fp8 and bf16 scales have no arithmetic of their own
    same = (a == b) | (a.isnan() & b.isnan())
    delta = (a - b).abs()
    magnitude = torch.maximum(a.abs(), b.abs())
    rel = torch.nan_to_num(delta / magnitude, nan=torch.inf).masked_fill(same, 0.0)
    close = same | (a.isfinite() & b.isfinite() & (delta <= atol + rtol * magnitude))
    return int((~close).sum()), float(rel.max()) if rel.numel() else 0.0


def compare_scales(
    a: Mapping[str, torch.Tensor],
    b: Mapping[str, torch.Tensor],
    rtol: float = 0.0,
    atol: float = 0.0,
) -> ScaleDiff:
    """Diff the tensors of ``a`` and ``b`` by name.

    An element is outside the tolerance when ``|a - b| > atol + rtol * max(|a|, |b|)``. The default
    tolerance of zero asks for identical scales.
    """
    # Plain name sets: a lazy mapping's `in` would load the tensor.
    names_a, names_b = set(a), set(b)
    result = ScaleDiff(only_in_a=sorted(names_a - names_b), only_in_b=sorted(names_b - names_a))
    for name in sorted(names_a & names_b):
        ta, tb = a[name], b[name]
        if ta.shape != tb.shape:
            result.shape_mismatch.append(name)
            continue
        outside, max_rel = _diff(ta, tb, rtol, atol)
        result.tensors.append(TensorDiff(name, ta.numel(), outside, max_rel))
    return result


class _CheckpointScales(Mapping[str, torch.Tensor]):
    """Lazy ``name -> tensor`` view of the scale tensors of a safetensors checkpoint."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No .safetensors files in {path}")
        self._handles: dict[str, safe_open] = {}
        for file in files:
            handle = safe_open(file, framework="pt")
            names = list(filter(_SCALE_NAME.search, handle.keys()))
            if duplicates := self._handles.keys() & set(names):
                raise ValueError(f"{sorted(duplicates)[0]} is in more than one file of {path}")
            self._handles.update(dict.fromkeys(names, handle))
        if not self._handles:
            raise ValueError(f"No quantization scale tensors in {path}")

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._handles[name].get_tensor(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._handles)

    def __len__(self) -> int:
        return len(self._handles)


def load_checkpoint_scales(path: str | Path) -> Mapping[str, torch.Tensor]:
    """Scale tensors of a safetensors file or of every ``*.safetensors`` file in a directory."""
    return _CheckpointScales(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Compare two checkpoints; the exit status is 0 when their scales agree within tolerance."""
    parser = argparse.ArgumentParser(description="Compare the quantization scales of two PTQ runs.")
    parser.add_argument("checkpoint_a", help="Safetensors file or checkpoint directory.")
    parser.add_argument("checkpoint_b", help="Safetensors file or checkpoint directory.")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance (default 0).")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance (default 0).")
    parser.add_argument("--top", type=int, default=5, help="Number of worst tensors to list.")
    args = parser.parse_args(argv)
    result = compare_scales(
        load_checkpoint_scales(args.checkpoint_a),
        load_checkpoint_scales(args.checkpoint_b),
        args.rtol,
        args.atol,
    )
    print(result.summary(args.top))
    return 0 if result.within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())

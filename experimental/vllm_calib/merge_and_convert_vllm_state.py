#!/usr/bin/env python
"""Merge per-rank vLLM-calibrated quantizer state files and convert vLLM
fused-MoE quantizer key names to HuggingFace per-expert names.

Input  : <state_dir>/<tag>-rank<NN>-of<MM>.pth produced by
         examples/vllm_serve/fakequant_worker.py:_save_vllm_calibrated_state
Output : <state_dir>/<tag>-merged-hf.pth, ready to be restored into an HF model
         for unified-HF NVFP4 export via hf_ptq.py --restore_from_modelopt_state.

Conversions performed (NVFP4_EXPERTS_ONLY scope):
  - Across-rank merge for *_amax tensors: element-wise max if shapes match,
    otherwise concat-along-dim-0 (TP shard reassembly). Reuses
    modelopt.torch.export.plugins.vllm_fakequant_hf.merge_amax_tensors_for_group.
  - vLLM fused-MoE key names → HF per-expert names:
      model.layers.<L>.mlp.experts.w13_input_quantizer.<field>
        → model.layers.<L>.mlp.experts.<i>.gate_proj.input_quantizer.<field>
        → model.layers.<L>.mlp.experts.<i>.up_proj.input_quantizer.<field>
      model.layers.<L>.mlp.experts.w2_input_quantizer.<field>
        → model.layers.<L>.mlp.experts.<i>.down_proj.input_quantizer.<field>
      (same for *_weight_quantizer / *_output_quantizer when populated)
    The same amax is duplicated across the per-expert HF keys — this matches the
    vLLM Marlin MoE assumption that all experts in a layer share an input scale.

The output preserves modelopt_state (one shared modelopt mode chain) and folds
the per-rank quantizer_state_dicts into one HF-keyed dict.
"""

import argparse
import glob
import os
import re
import sys
from typing import Any

ROOT = "/lustre/fs1/portfolios/adlr/projects/adlr_psx_numerics/users/jingyux/kimi-k2"
sys.path.insert(0, f"{ROOT}/source/Model-Optimizer")

import torch  # noqa: E402

from modelopt.torch.utils import safe_load, safe_save  # noqa: E402
from modelopt.torch.export.plugins.vllm_fakequant_hf import (  # noqa: E402
    merge_amax_tensors_for_group,
)


# vLLM fused-MoE quantizer keys → HF per-expert mapping.
# Matches keys like: model.layers.<L>.mlp.experts.<w13|w2>_<input|weight|output>_quantizer[.<field>]
_VLLM_FUSED_MOE_KEY = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts)\."
    r"(?P<wbranch>w13|w2)_(?P<qkind>input|weight|output)_quantizer"
    r"(?P<suffix>\..+)?$"
)

# Maps the vLLM fused-branch name to the per-expert HF projection names.
_W13_TO_HF = ("gate_proj", "up_proj")
_W2_TO_HF = ("down_proj",)


def _is_amax_field(field: str) -> bool:
    """Return True for fields whose values should be merged via max/concat."""
    return field.endswith("_amax")


def _is_pre_quant_scale(field: str) -> bool:
    """Per-input-channel scale; identical across TP, take any."""
    return field.endswith("_pre_quant_scale")


def merge_field_across_ranks(field: str, values: list[torch.Tensor]) -> torch.Tensor:
    """Combine the same field's values from N ranks into one canonical tensor."""
    if not values:
        raise ValueError(f"empty values for field {field!r}")
    if len(values) == 1:
        return values[0]
    # All-non-tensor: pick first
    if not all(isinstance(v, torch.Tensor) for v in values):
        return values[0]
    if _is_pre_quant_scale(field):
        return values[0]
    if _is_amax_field(field):
        return merge_amax_tensors_for_group(values)
    # Fallback: try concat along dim 0; on failure, return first.
    try:
        return torch.cat(values, dim=0)
    except RuntimeError:
        return values[0]


def merge_quantizer_state_dicts(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-rank quantizer_state_dicts into a single dict.

    For each top-level quantizer key, merge each inner field across ranks.
    Empty dicts (disabled quantizers) are passed through.
    """
    merged: dict[str, Any] = {}
    all_keys: set[str] = set()
    for s in states:
        all_keys.update(s.keys())

    for k in sorted(all_keys):
        per_rank = [s.get(k) for s in states]
        non_none = [v for v in per_rank if v is not None]
        if not non_none:
            continue
        if all(isinstance(v, dict) and len(v) == 0 for v in non_none):
            merged[k] = {}
            continue
        # Collect the union of inner fields across ranks
        if not all(isinstance(v, dict) for v in non_none):
            # Mixed: take the first dict-shaped one
            merged[k] = next((v for v in non_none if isinstance(v, dict)), non_none[0])
            continue
        all_fields: set[str] = set()
        for v in non_none:
            all_fields.update(v.keys())
        merged_v: dict[str, Any] = {}
        for f in sorted(all_fields):
            field_values = [v[f] for v in non_none if f in v]
            if not field_values:
                continue
            merged_v[f] = merge_field_across_ranks(f, field_values)
        merged[k] = merged_v
    return merged


def vllm_to_hf_quantizer_keys(
    qsd: dict[str, Any],
    num_experts: int,
) -> dict[str, Any]:
    """Rewrite vLLM fused-MoE quantizer key names to HF per-expert names.

    Non-MoE quantizer keys pass through unchanged. The same amax is duplicated
    across each expert × HF projection (Marlin assumes a shared input scale).
    """
    out: dict[str, Any] = {}
    for k, v in qsd.items():
        m = _VLLM_FUSED_MOE_KEY.match(k)
        if not m:
            out[k] = v
            continue
        prefix = m.group("prefix")
        wbranch = m.group("wbranch")
        qkind = m.group("qkind")
        suffix = m.group("suffix") or ""
        hf_projs = _W13_TO_HF if wbranch == "w13" else _W2_TO_HF
        for expert_idx in range(num_experts):
            for proj in hf_projs:
                hf_key = (
                    f"{prefix}.{expert_idx}.{proj}.{qkind}_quantizer{suffix}"
                )
                out[hf_key] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--state-dir", required=True)
    p.add_argument(
        "--tag",
        default=None,
        help="JOBID prefix used at save time. If omitted, infers a single tag.",
    )
    p.add_argument(
        "--num-experts",
        type=int,
        required=True,
        help="Total number of routed experts in the source model "
             "(read from config.json: text_config.n_routed_experts for K2.6, "
             "n_routed_experts for K2-Thinking).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output path (default: <state_dir>/<tag>-merged-hf.pth)",
    )
    args = p.parse_args()

    # Discover rank files
    if args.tag:
        pattern = f"{args.state_dir}/{args.tag}-rank*-of*.pth"
    else:
        pattern = f"{args.state_dir}/*-rank*-of*.pth"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: no rank state files matching {pattern}", file=sys.stderr)
        return 2

    if not args.tag:
        # Infer tag from first file
        first = os.path.basename(files[0])
        args.tag = first.split("-rank")[0]
        # Re-filter to only this tag
        files = [f for f in files if os.path.basename(f).startswith(f"{args.tag}-rank")]

    print(f"[merge] tag={args.tag}  files={len(files)}")
    states = [safe_load(f, map_location="cpu") for f in files]
    n_ranks = states[0].get("world_size", len(files))
    print(f"[merge] world_size={n_ranks}  per-rank dict sizes: "
          f"{[len(s['quantizer_state_dict']) for s in states]}")

    # 1. Merge per-rank quantizer_state_dicts
    qsd_per_rank = [s["quantizer_state_dict"] for s in states]
    merged_qsd = merge_quantizer_state_dicts(qsd_per_rank)
    print(f"[merge] merged_qsd has {len(merged_qsd)} entries")

    # 2. Convert vLLM fused-MoE → HF per-expert
    hf_qsd = vllm_to_hf_quantizer_keys(merged_qsd, num_experts=args.num_experts)
    print(f"[merge] after vLLM→HF rename: {len(hf_qsd)} entries  "
          f"(growth = {len(hf_qsd) - len(merged_qsd)})")

    # 3. Build output
    out = {
        "modelopt_state": states[0]["modelopt_state"],
        "modelopt_state_weights": hf_qsd,
        "source": {
            "tag": args.tag,
            "num_ranks": n_ranks,
            "num_experts": args.num_experts,
            "quant_config_summary": states[0].get("quant_config_summary"),
        },
    }
    out_path = args.out or os.path.join(
        args.state_dir, f"{args.tag}-merged-hf.pth"
    )
    safe_save(out, out_path)
    print(f"[merge] wrote {out_path}")

    # 4. Quick sanity sample
    sample_keys = [
        k for k in hf_qsd
        if "experts.0.gate_proj.input_quantizer" in k
        or "experts.0.down_proj.input_quantizer" in k
        or "shared_experts.gate_up_proj.input_quantizer" in k
    ][:6]
    print(f"[merge] sample HF keys present:")
    for k in sample_keys:
        v = hf_qsd[k]
        if isinstance(v, dict) and v:
            fields = {kk: (tuple(vv.shape) if hasattr(vv, "shape") else type(vv).__name__) for kk, vv in v.items()}
            print(f"  {k}: {fields}")
        else:
            print(f"  {k}: empty/{type(v).__name__}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

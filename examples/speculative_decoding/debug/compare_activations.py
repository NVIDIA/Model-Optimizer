"""Sub-step diff: TRAIN (real ModelOpt) vs EVAL (real vLLM module).

Auto-pairs every tensor present in both dumps and reports them in forward order.
Reports max|d| and RELATIVE error -- cosine over a flattened 3M-element bf16
tensor is unreliable near 1.0 (it read 1.0035 on bit-identical inputs earlier),
so relative error is the column to trust.
"""

import re
import sys

import torch

tr = torch.load(sys.argv[1], map_location="cpu")
ev = torch.load(sys.argv[2], map_location="cpu")

PRE = ["input_ids", "noise_ids", "position_ids", "aux_hidden", "ctx_fc",
       "ctx_hidden_norm", "noise_embedding"]
STEP = ["input_layernorm", "q_proj", "q_norm", "q_rope", "k_proj_block",
        "k_rope_block", "k_proj_ctx", "k_rope_ctx", "attn_out", "o_proj",
        "post_attn_norm", "after_attn_residual", "pre_ffn_norm", "mlp",
        "post_ffn_norm", "out"]

order = [k for k in PRE if k in tr or k in ev]
layers = sorted({int(m.group(1)) for k in list(tr) + list(ev)
                 if (m := re.match(r"L(\d+)_", k))})
for li in layers:
    for st in STEP:
        n = f"L{li}_{st}"
        if n in tr or n in ev:
            order.append(n)
for tail in ("draft_final", "backbone_logits", "top1_backbone"):
    if tail in tr or tail in ev:
        order.append(tail)

print("=" * 96)
print("  TRAIN (ModelOpt) vs EVAL (real vLLM module) -- sub-step diff, forward order")
print("=" * 96)
print(f"  {'tensor':<28}{'shape':>16}{'max|d|':>13}{'rel_max':>10}{'rel_rms':>10}  verdict")
print("-" * 96)

first = None
for n in order:
    if n not in tr or n not in ev:
        print(f"  {n:<28}{'':>16}{'':>13}{'':>10}{'':>10}  "
              f"{'train only' if n in tr else 'eval only'}")
        continue
    a, b = tr[n].float(), ev[n].float()
    if a.shape != b.shape:
        if a.numel() == b.numel():
            b = b.reshape(a.shape)
        else:
            print(f"  {n:<28}{str(tuple(a.shape)):>16}{'':>13}{'':>10}{'':>10}"
                  f"  SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
    if "ids" in n or "top1" in n:
        eq = (a == b).float().mean().item()
        print(f"  {n:<28}{str(tuple(a.shape)):>16}{'':>13}{'':>10}{eq:>9.1%}  "
              f"{'IDENTICAL' if eq == 1.0 else 'DIFFER'}")
        continue
    md = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-9)
    rmsd = (a - b).pow(2).mean().sqrt().item()
    rms = max(a.pow(2).mean().sqrt().item(), 1e-9)
    rel, relr = md / scale, rmsd / rms
    if md == 0.0:
        v = "BIT-EXACT"
    elif rel < 0.02:
        v = "ok (numeric)"
    else:
        v = "<-- MISMATCH"
        if first is None:
            first = n
    print(f"  {n:<28}{str(tuple(a.shape)):>16}{md:>13.4e}{rel:>9.2%}{relr:>9.2%}  {v}")

print("=" * 96)
print(f"  FIRST MISMATCH (>2% rel): {first}" if first
      else "  All paired tensors agree within numeric tolerance.")
print("=" * 96)

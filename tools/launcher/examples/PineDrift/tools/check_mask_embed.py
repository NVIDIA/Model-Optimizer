#!/usr/bin/env python3
"""Check the embedding rows of the mask-token candidates.

DFlash feeds the mask token through the FROZEN base embedding table, so the row's
content matters. A zero row would survive _WeightlessRMSNorm (rsqrt(0+eps) is finite)
but as a degenerate all-zero feature -- exactly the kind of thing that trains without
erroring and produces a weak drafter.
"""
import torch

from modelopt.torch.speculative.plugins.modeling_fakebase import FakeBaseModel

SRC = "/home/haoguo/lustre/hf-local/pinedrift-820b-a42b-nvfp4_vv3"
m = FakeBaseModel.from_source(SRC)
W = m.embed_tokens.weight  # raw table, before the weightless norm

def row(i, label):
    v = W[i].float()
    print(f"  {i:6d}  {label:34s} L2={v.norm():9.4f}  absmax={v.abs().max():8.4f}  "
          f"zeros={int((v == 0).sum())}/{v.numel()}")

print("=== reference rows (tokens the base definitely uses) ===")
row(15339, "a common BPE token")
row(200000, "<|begin_of_text|>")
row(200008, "<|eot|>  (chat template)")

print("\n=== mask-token candidates: the top reserved specials ===")
for i in (201817, 201816, 201815, 201500, 200100):
    row(i, "<|reserved_special_token_*|>")

print("\n=== padding rows the tokenizer can never emit (201818..202047) ===")
for i in (201818, 201900, 202047):
    row(i, "pad row (no token maps here)")

v = W[201817].float()
print("\nVERDICT for 201817:")
print(f"  nonzero row        : {bool(v.norm() > 0)}")
print(f"  L2 vs common token : {v.norm().item():.4f} vs {W[15339].float().norm().item():.4f}")

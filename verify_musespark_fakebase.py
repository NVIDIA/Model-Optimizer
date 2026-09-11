#!/usr/bin/env python3
"""Verify the MuseSpark FakeBase transforms against the real PineDrift checkpoint."""
import math

import torch
from transformers import AutoConfig

from modelopt.torch.speculative.plugins.modeling_fakebase import (
    FakeBaseModel,
    _select_base_transforms,
)

SRC = "/home/haoguo/lustre/hf-local/pinedrift-820b-a42b-nvfp4_vv3"

cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=False)
print("model_type:", cfg.model_type)

t = _select_base_transforms(cfg.model_type, cfg)
print("\n=== derived transforms ===")
for k, v in t.items():
    print(f"  {k}: {v}")

# vLLM: width_mult = hidden/base_width = 8192/256 = 32; metap_mode 'sp' -> 32**-0.5
expected_mult = (8192 / 256.0) ** -0.5
assert abs(t["logits_multiplier"] - expected_mult) < 1e-12, t["logits_multiplier"]
assert t["logits_soft_cap"] == 20.0, t["logits_soft_cap"]
assert t["embed_norm_type"] == "rmsnorm_no_weight", t["embed_norm_type"]
assert t["embed_multiplier"] == 1.0, t["embed_multiplier"]
print(f"\n  multiplier matches vLLM formula: {expected_mult:.6f}  OK")

print("\n=== loading FakeBaseModel from the real checkpoint (embed + lm_head + norm) ===")
m = FakeBaseModel.from_source(SRC)
print("  final_norm_type:", m.config.final_norm_type)
print("  has norm module:", hasattr(m, "norm"))
print("  embed_tokens:", type(m.embed_tokens).__name__, tuple(m.embed_tokens.weight.shape))
print("  lm_head:", type(m.lm_head).__name__, tuple(m.lm_head.weight.shape))
print("  embed_norm:", type(m.embed_tokens.embed_norm).__name__)
print("  lm_head multiplier / soft cap:", m.lm_head.logits_multiplier, m.lm_head.logits_soft_cap)

assert m.config.final_norm_type == "rmsnorm"
assert m.embed_tokens.embed_norm is not None
assert m.lm_head.logits_soft_cap == 20.0

print("\n=== numerical behaviour ===")
ids = torch.tensor([[100, 200, 300, 400]])
with torch.no_grad():
    h = m.embed_tokens(ids)
    raw = torch.nn.functional.linear(h.to(m.lm_head.weight.dtype), m.lm_head.weight)
    capped = m.lm_head(h)
rms = h.float().pow(2).mean(-1).sqrt()
print(f"  embedding RMS after weightless norm: {rms.mean():.4f} (should be ~1.0)")
print(f"  raw logits    min/max: {raw.min():.3f} / {raw.max():.3f}")
print(f"  capped logits min/max: {capped.min():.3f} / {capped.max():.3f}  (|.| < 20)")
assert capped.abs().max() < 20.0, "soft cap not applied"
assert abs(rms.mean().item() - 1.0) < 0.05, f"embed norm not applied (rms={rms.mean()})"

# the cap must be the exact vLLM formula, not merely "something smaller"
ref = 20.0 * torch.tanh(raw.float() * expected_mult / 20.0)
err = (capped.float() - ref).abs().max().item()
print(f"  max |ours - vLLM formula|: {err:.3e}")
assert err < 1e-2, err

print("\nALL CHECKS PASSED")

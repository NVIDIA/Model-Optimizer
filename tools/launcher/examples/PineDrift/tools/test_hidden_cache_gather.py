"""Verify the [B,H,N,C] gather returns [n_tokens, n_planes, hidden] in the right order.

A wrong axis order here does not error -- it trains the drafter on transposed
hidden states -- so check values, not just shapes.
"""
import sys, torch
sys.path.insert(0, "/lustre/modelopt-pinedrift")
from modelopt.torch.speculative.plugins.rdma_hidden_states_connector import (
    extract_from_kv_cache, planes_dim,
)

B, H, N, C = 5, 6, 1, 4          # blocks, planes, tokens/block, hidden
# value = block*1000 + plane*100 + tok*10 + chan, so every element is identifiable
kv = torch.zeros(B, H, N, C)
for b in range(B):
    for h in range(H):
        for n in range(N):
            for c in range(C):
                kv[b, h, n, c] = b * 1000 + h * 100 + n * 10 + c

assert planes_dim(kv, H) == 1, planes_dim(kv, H)
slots = torch.tensor([0, 3, 4])           # global token slots -> blocks 0,3,4 (N=1)
out = extract_from_kv_cache(kv, slots, 3, 1)
assert out.shape == (3, H, C), out.shape
for i, s in enumerate(slots.tolist()):
    b, n = s // N, s % N
    for h in range(H):
        for c in range(C):
            want = b * 1000 + h * 100 + n * 10 + c
            assert out[i, h, c] == want, (i, h, c, out[i, h, c].item(), want)
print(f"[B,H,N,C] gather OK: {tuple(kv.shape)} -> {tuple(out.shape)}")

# and the legacy [B,N,H,C] order still works
B2, N2, H2, C2 = 5, 8, 6, 4
kv2 = torch.zeros(B2, N2, H2, C2)
for b in range(B2):
    for n in range(N2):
        for h in range(H2):
            for c in range(C2):
                kv2[b, n, h, c] = b * 1000 + n * 10 + h * 100 + c
assert planes_dim(kv2, H2) == 2, planes_dim(kv2, H2)
slots2 = torch.tensor([0, 9, 17])
out2 = extract_from_kv_cache(kv2, slots2, 3, 2)
assert out2.shape == (3, H2, C2), out2.shape
for i, s in enumerate(slots2.tolist()):
    b, n = s // N2, s % N2
    for h in range(H2):
        assert out2[i, h, 0] == b * 1000 + n * 10 + h * 100, (i, h)
print(f"[B,N,H,C] gather OK: {tuple(kv2.shape)} -> {tuple(out2.shape)}")

# and the ambiguous case must fail loud, not guess
try:
    planes_dim(torch.zeros(2, 6, 6, 4), 6)
except RuntimeError as e:
    print("ambiguous layout rejected:", str(e)[:70])
else:
    raise SystemExit("FAIL: ambiguous layout was not rejected")

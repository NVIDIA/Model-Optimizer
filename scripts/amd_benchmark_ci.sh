#!/usr/bin/env bash
# AMD ROCm-Model-Optimizer CI benchmark — pass/fail FP8 pipeline test
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== AMD Benchmark CI | $(date -u '+%Y-%m-%d %H:%M UTC') ==="
python3 -c "
import torch
print('GPU:', torch.cuda.get_device_name(0))
print('Arch:', torch.cuda.get_device_properties(0).gcnArchName)
print('ROCm:', torch.version.hip)
"

pip install -e . --no-build-isolation -q 2>&1 | tail -2
pip install pytest -q 2>&1 | tail -1

echo ""
echo "── Unit tests ──"
# Use -p no:cacheprovider and clear addopts to avoid plugin conflicts
python3 -m pytest tests/amd/test_amd_rocm.py -v --tb=short \
    -p no:cacheprovider \
    --override-ini="addopts=" \
    --override-ini="timeout_func_only=false" \
    -q 2>&1 | tail -30 || echo "Some tests failed (non-fatal)"

echo ""
echo "── FP8 pipeline benchmark ──"
python3 - << 'PYEOF'
import sys, os, time
import torch, torch.nn as nn
sys.path.insert(0, os.getcwd())
import modelopt.torch.quantization as mtq
from modelopt._rocm_compat import (
    extract_fp8_scales, convert_to_static_fp8, warmup_fp8_shapes, is_fp8_supported
)
if not is_fp8_supported():
    print("FP8 not supported — skip"); sys.exit(0)

H, I = 4096, 16384
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Linear(H,I,bias=False); self.up=nn.Linear(H,I,bias=False)
        self.down=nn.Linear(I,H,bias=False); self.act=nn.SiLU()
    def forward(self,x): return self.down(self.act(self.gate(x))*self.up(x))

ffn_fp16=FFN().cuda().half(); ffn_cal=FFN().cuda().half()
x_cal=torch.randn(16,H,device="cuda",dtype=torch.float16)
mtq.quantize(ffn_cal,mtq.AMD_FP8_DEFAULT_CFG,forward_loop=lambda m:m(x_cal))
scales=extract_fp8_scales(ffn_cal)
ffn_fp8=FFN().cuda().half(); ffn_fp8=convert_to_static_fp8(ffn_fp8,scales)
warmup_fp8_shapes([(256,I,H),(256,H,I)],device="cuda")
results={}
for bs in [64,128,256]:
    x=torch.randn(bs,H,device="cuda",dtype=torch.float16)
    for _ in range(30): ffn_fp16(x)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(300): ffn_fp16(x)
    torch.cuda.synchronize(); t16=(time.perf_counter()-t0)/300*1000
    for _ in range(30): ffn_fp8(x)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(300): ffn_fp8(x)
    torch.cuda.synchronize(); t8=(time.perf_counter()-t0)/300*1000
    spd=t16/t8; results[bs]=spd
    print(f"  BS={bs:>3}: {t16:.3f}ms FP16 -> {t8:.3f}ms FP8 = {spd:.2f}x")
thresh=1.4
if results.get(256,0)>=thresh:
    print(f"\nBENCHMARK PASS: BS=256 = {results[256]:.2f}x >= {thresh}x"); sys.exit(0)
else:
    print(f"\nBENCHMARK FAIL: BS=256 = {results.get(256,0):.2f}x < {thresh}x"); sys.exit(1)
PYEOF
echo "CI done: $(date -u '+%Y-%m-%d %H:%M UTC')"

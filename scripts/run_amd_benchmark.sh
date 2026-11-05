#!/usr/bin/env bash
# AMD MI300X FP8 Benchmark Suite
# Usage: bash scripts/run_amd_benchmark.sh [--full] [--llama-size 70b] [--batch-sizes 1,64,256]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Defaults
MODEL_SIZE="${MODEL_SIZE:-7b}"
BATCH_SIZES="${BATCH_SIZES:-1,64,128,256}"
FULL="${1:-}"

echo "=== AMD MI300X FP8 Benchmark Suite ==="
echo "Date:  $(date -u '+%Y-%m-%d %H:%M UTC')"
python3 -c "
import torch
print('GPU: ', torch.cuda.get_device_name(0))
print('Arch:', torch.cuda.get_device_properties(0).gcnArchName)
print('ROCm:', torch.version.hip)
"

echo ""
echo "── Installing modelopt ──"
pip install -e . --no-build-isolation -q 2>&1 | tail -2

echo ""
echo "── Running benchmarks (LLaMA-${MODEL_SIZE}, BS=${BATCH_SIZES}) ──"
python3 - << PYEOF
import sys, os, time
sys.path.insert(0, os.getcwd())
import torch, torch.nn as nn
import modelopt.torch.quantization as mtq
from modelopt._rocm_compat import (
    extract_fp8_scales, convert_to_static_fp8, convert_ffn_only_to_fp8,
    warmup_for_llama, get_quantization_strategy,
    profile_amd_model, compare_amd_models, get_llama_ffn_shapes,
    is_fp8_supported, get_gpu_arch
)

MODEL_SIZE = os.environ.get("MODEL_SIZE", "${MODEL_SIZE}")
BATCH_SIZES = [int(b) for b in "${BATCH_SIZES}".split(",")]

print(f"Model: LLaMA-{MODEL_SIZE} | FP8: {is_fp8_supported()} | Arch: {get_gpu_arch()}")

H, I = get_llama_ffn_shapes(MODEL_SIZE)
print(f"FFN: {H} -> {I} -> {H}")

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Linear(H,I,bias=False); self.up=nn.Linear(H,I,bias=False)
        self.down=nn.Linear(I,H,bias=False); self.act=nn.SiLU()
    def forward(self,x): return self.down(self.act(self.gate(x))*self.up(x))

# Calibrate
ffn_cal = FFN().cuda().half()
x_cal = torch.randn(8, H, device="cuda", dtype=torch.float16)
mtq.quantize(ffn_cal, mtq.AMD_FP8_DEFAULT_CFG, forward_loop=lambda m: m(x_cal))
scales = extract_fp8_scales(ffn_cal)
warmup_for_llama(MODEL_SIZE, batch_sizes=BATCH_SIZES[:3])

print(f"\n{'BS':>4}  {'Strategy':15}  {'FP16 ms':>8}  {'FP8 ms':>8}  {'Speedup':>9}  {'TFLOPS':>8}")
print("-"*56)
ITERS = 300

for bs in BATCH_SIZES:
    strategy = get_quantization_strategy(bs, H)
    strat_name = strategy["strategy"]
    
    ffn_fp16 = FFN().cuda().half()
    if strat_name == "ffn_only":
        ffn_fp8 = FFN().cuda().half()
        convert_ffn_only_to_fp8(ffn_fp8, scales)
    else:
        ffn_fp8 = FFN().cuda().half()
        convert_to_static_fp8(ffn_fp8, scales)
    
    x = torch.randn(bs, H, device="cuda", dtype=torch.float16)
    
    for _ in range(30): ffn_fp16(x)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(ITERS): ffn_fp16(x)
    torch.cuda.synchronize()
    t16=(time.perf_counter()-t0)/ITERS*1000
    
    for _ in range(30): ffn_fp8(x)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(ITERS): ffn_fp8(x)
    torch.cuda.synchronize()
    t8=(time.perf_counter()-t0)/ITERS*1000
    
    flops = 2*bs*(H*I+H*I+I*H)
    tfl = flops/(t8/1000)/1e12
    print(f"{bs:>4}  {strat_name:15}  {t16:>8.3f}  {t8:>8.3f}  {t16/t8:>9.2f}x  {tfl:>8.1f}")

print("\n=== Benchmark complete ===")
PYEOF

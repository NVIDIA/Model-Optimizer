# DeepSeek-V4 PTQ — experts-only NVFP4 (W+A)

Post-training quantization recipe for DeepSeek-V4 (e.g. DeepSeek-V4-Pro).
Quantizes the **routed experts only** (W4 NVFP4 + A4 NVFP4). Every other
module — attention projections, the gate, the shared expert, lm_head,
embeddings — is left untouched and keeps using V4's native TileLang
`fp4_gemm` / `fp8_gemm` / BF16 `F.linear` dispatch.

V4-Pro at MP=8 is ~860 GB on disk in the native MXFP4+FP8 layout and fits
across 8× GB200. On this cluster that's two 4-GPU nodes.

## Prerequisites

- V4 checkpoint downloaded locally (e.g. via `hf download deepseek-ai/DeepSeek-V4-Pro --local-dir ...`).
- ModelOpt installed (editable from this repo works).
- Python env with torch (CUDA build), safetensors, transformers, tqdm.

## Step 1 — Reshard to MP with DeepSeek's own `convert.py`

Keep experts in MXFP4 (don't pass `--expert-dtype fp8`):

```bash
export DS_V4=/path/to/DeepSeek-V4-Pro
export MP=8
export MP_CKPT=/path/to/DeepSeek-V4-Pro-mp${MP}-mxfp4

python ${DS_V4}/inference/convert.py \
    --hf-ckpt-path ${DS_V4} \
    --save-path    ${MP_CKPT} \
    --n-experts 384 \
    --model-parallel ${MP}
```

Output: `${MP_CKPT}/model{0..MP-1}-mp${MP}.safetensors`. Experts are
`float4_e2m1fn_x2` with paired `.scale` (UE8M0, 32-wide groups); dense /
attention are `float8_e4m3fn` with paired `.scale` (UE8M0, 128×128 blocks).

## Step 2 — Calibrate + emit amax / `hf_quant_config.json`

```bash
export AMAX=/path/to/amax_dump

torchrun --nproc-per-node ${MP} --master_port 12346 ptq.py \
    --model_path  ${MP_CKPT} \
    --config      ${DS_V4}/inference/config.json \
    --output_path ${AMAX}
```

What the script does:

1. Imports DS-V4's native inference stack (`model.py` + `kernel.py`) via
   `sys.path` — same pattern as the V3 example.
2. Registers **only two** Quant wrappers with ModelOpt:
   - `QuantExpert(Expert)`: installs six `TensorQuantizer` instances
     (`w{1,2,3}_{input,weight}_quantizer`) and redefines `forward` so each
     `w{1,2,3}` call goes through an MXFP4→BF16 dequant
     (`MXFP4QTensor.dequantize_packed`) + ModelOpt hooks + `F.linear`.
   - `CalibMoE(MoE)`: during calibration, first runs with `top_k =
     n_routed_experts` so every expert sees every token (populates amax),
     then reruns with the real top_k for the downstream output.
3. **Does not** register against `Linear` / `ColumnParallelLinear` /
   `RowParallelLinear` / `Attention` / `Gate`. Those classes keep V4's
   native `linear()` dispatch, which routes to `fp4_gemm` / `fp8_gemm` /
   `F.linear` as dictated by the weight dtype on disk.
4. Calibrates on `cnn_dailymail` + `nemotron-post-training-dataset-v2`.
5. Dumps `amax_dict_rank*-mp*.pt` and `hf_quant_config.json`.

Config installed by `_build_nvfp4_experts_cfg`:

```python
{
  "quant_cfg": [
    {"quantizer_name": "*input_quantizer",                         "enable": False},
    {"quantizer_name": "*weight_quantizer",                        "enable": False},
    {"quantizer_name": "*ffn.experts.*.w*_weight_quantizer",       "enable": True, "cfg": NVFP4},
    {"quantizer_name": "*ffn.experts.*.w*_input_quantizer",        "enable": True, "cfg": NVFP4},
    {"quantizer_name": "*shared_experts*",                         "enable": False},
  ],
  "algorithm": "max",
}
```

where `NVFP4 = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}`.

### Multi-node at MP=8

Two 4-GPU nodes, one `torchrun` per node. Standard multi-node flags:

```bash
# on node 0
torchrun --nnodes=2 --node_rank=0 --master_addr=<ip> --master_port=12346 \
         --nproc-per-node 4 ptq.py ...
# on node 1
torchrun --nnodes=2 --node_rank=1 --master_addr=<ip> --master_port=12346 \
         --nproc-per-node 4 ptq.py ...
```

## Shared-expert note

The shared expert is an instance of the same `Expert` class, so registering a
`QuantExpert` wrapper against `Expert` ends up wrapping both routed and shared
experts. We disable the shared expert's quantizers via the `*shared_experts*`
pattern — so no amax is collected for it and its weights on disk are
unchanged. Note that the shared expert still pays a *BF16 dequant cost during
calibration* (its QuantExpert forward dequantizes the weight to BF16 even
though the quantizers pass through). This does not affect downstream
inference: shared_experts weights on disk remain FP8 / MXFP4 and native
kernels can be used.

If you want the shared expert to also use its native `linear()` dispatch
during calibration (no BF16 dequant at all), you can swap the class-level
`mtq.register(Expert, QuantExpert)` for an instance-level swap that only
touches `MoE.experts[i]` (not `MoE.shared_experts`). That's a ~15-line
modification and adds complexity we didn't need for correctness.

## Notes / caveats

- Router gate weights, attention (compressor + indexer included), and the LM
  head path are untouched. If you later want to also quantize those, the V3
  `ptq.py` has reusable patterns for Linear / ColumnParallelLinear /
  RowParallelLinear / MLA — adapt those.
- `CalibMoE` looks up `gate.top_k` (V4's attribute) with a fallback to
  `topk` (V3-style) for safety; restore happens in a `finally` block so
  calibration-mode doesn't leak into downstream output.
- The MXFP4 dequant kernel is covered by the unit tests at
  `tests/unit/torch/quantization/test_mxfp4_dequant.py` (9 tests incl. bit-
  identical cross-validation against `transformers._convert_moe_packed_tensors`
  and DeepSeek's `cast_e2m1fn_to_e4m3fn` → FP8 → BF16 path).

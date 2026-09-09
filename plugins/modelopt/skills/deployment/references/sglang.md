# SGLang Deployment Reference

## Authoritative recipe source — the SGLang cookbook

For any non-trivial model (large MoE, multi-node, Blackwell FP4), **cross-check
the launch command against the SGLang cookbook** before hand-rolling flags. It
is the SGLang analog of `recipes.vllm.ai`: a verified command generator keyed on
a `(hw, variant, quant, strategy, nodes)` tuple.

- **URL:** `docs.sglang.io/cookbook/<category>/<org>/<model>` — e.g.
  `docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4`. Select the
  variant via the URL fragment, e.g.
  `#hw=b200&variant=flash&quant=fp4&strategy=low-latency&nodes=single`.
- **JS-rendered (Mintlify).** A plain fetch usually misses the commands. Fetch
  the **raw markdown** instead:
  `raw.githubusercontent.com/sgl-project/sglang/main/docs_new/cookbook/<category>/<org>/<model>.mdx`
  (the URL path maps directly onto the directory tree). The old
  `sgl-project/sgl-cookbook` repo is archived — the live source is `docs_new/`
  in the main `sgl-project/sglang` repo.
- **Authoritative for:** parallelism layout (`--tp` / `--ep`, plus
  multi-node/data-parallel settings), MoE backends,
  strategy-driven flags (MTP, CUDA-graph batch sizing), Docker image tag, and the
  minimum SGLang version for the chosen variant. Pull the layout/flags from the
  cookbook, then adapt to the GPUs you actually have.

## Requirements

- SGLang >= 0.4.10
- `pip install sglang[all]`

## Server Deployment

### As OpenAI-compatible server

```bash
python -m sglang.launch_server \
    --model-path <checkpoint_path> \
    --quantization modelopt \
    --tp <num_gpus> \
    --host 0.0.0.0 --port 8000
```

For NVFP4 checkpoints, use `--quantization modelopt_fp4`.

### As Python API

```python
import sglang as sgl

llm = sgl.Engine(model_path="<checkpoint_path>", quantization="modelopt")
# For FP4: quantization="modelopt_fp4"

sampling_params = {"temperature": 0.8, "top_p": 0.95}
outputs = llm.generate(["Hello, my name is"], sampling_params)

for output in outputs:
    print(f"Generated: {output['text']}")
```

### From HuggingFace Hub

```python
import sglang as sgl

llm = sgl.Engine(model_path="nvidia/Llama-3.1-8B-Instruct-FP8", quantization="modelopt")
outputs = llm.generate(["What is AI?"], {"temperature": 0.8})
```

## Speculative Decoding

SGLang supports speculative decoding with EAGLE and EAGLE3 models:

```bash
python -m sglang.launch_server \
    --model-path <target_model> \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path <draft_model> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --tp <num_gpus> \
    --host 0.0.0.0 --port 8000
```

Reference: `examples/specdec_bench/specdec_bench/models/sglang.py`

## Key SGLang Flags

| Flag | Description |
|------|-------------|
| `--model-path` | Path to checkpoint or HF model ID |
| `--quantization` | `modelopt` (FP8) or `modelopt_fp4` (FP4) |
| `--tp` | Tensor parallelism size |
| `--ep` | Expert parallelism (for MoE models) |
| `--enable-torch-compile` | Enable torch.compile for better perf |
| `--cuda-graph-max-bs` | Max batch size for CUDA graphs |
| `--attention-backend` | `flashinfer` (default) or `triton` |

## MoE / FP4 backend flags (Blackwell vs Hopper)

For large MoE models the cookbook recipe is authoritative — these are the flags
it selects per hardware. Quantized DeepSeek-style checkpoints are FP4 experts +
FP8 attention/dense.

| Flag | Use |
|------|-----|
| `--moe-runner-backend flashinfer_mxfp4` | Default FP4 MoE runner on Blackwell (SM100/SM103) |
| `--moe-runner-backend marlin` | Hopper W4A16 FP4 MoE runner |
| `--moe-a2a-backend deepep` | Default expert all-to-all backend |
| `--moe-a2a-backend megamoe` | Blackwell-only, high-throughput strategy |
| `--deepep-mode auto\|normal\|low_latency` | DeepEP dispatch mode |

Strategy (low-latency / balanced / high-throughput) tunes
`--cuda-graph-max-bs`, `--max-running-requests`, and MTP (Multi-Token
Prediction) draft steps/tokens — take these from the cookbook variant rather
than guessing.

## Hardware notes

- **Blackwell B200/B300/GB200/GB300 (SM100/SM103):** use the
  `flashinfer_mxfp4` MoE runner; `megamoe` only for the high-throughput
  strategy.
- **Hopper (H100/H200):** FP4 runs via Marlin W4A16 kernels. Converted FP8
  checkpoints (`sgl-project/DeepSeek-V4-*-FP8`) exist for richer parallelism.
- **RTX PRO 6000 (SM120):** `:latest` does **not** support SM120 — use the
  `lmsysorg/sglang:dev` nightly image.

## Common Issues

| Issue | Fix |
|-------|-----|
| `quantization="modelopt"` not recognized | Upgrade SGLang to >= 0.4.10 |
| DeepSeek FP4 not working | Check support matrix — SGLang FP4 support varies by model |
| OOM on startup | Increase `--tp` or reduce `--max-total-tokens` |

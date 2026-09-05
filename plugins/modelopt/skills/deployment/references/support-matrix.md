# Deployment Support Matrix

## Unified HF Checkpoint — Framework Compatibility

**Do not maintain a copy of the matrix here.** The single source of truth is
`docs/source/deployment/3_unified_hf.rst` ("Model Support Matrix"), and every entry in it is drawn
from `tests/examples/hf_ptq/test_deploy.py`.

Read that doc's legend before reporting a model as supported: the cases are marked `release` and do
not run on PR CI, and each is a load-and-generate smoke check on the text path — so an entry is
declared coverage, not proof the combination serves correctly.

To answer "is model X supported on framework Y", read one of those two files — `test_deploy.py` is
the more precise answer, since it also carries the exact checkpoint, tensor-parallel size, and
minimum SM version per entry. It covers language models, VLMs (Qwen2.5-VL, Qwen3-VL,
Nemotron Omni), EAGLE3/Medusa drafters, and diffusion models.

## Supported Quantization Formats

| Format | Description |
|--------|-------------|
| FP8 | 8-bit floating point (E4M3) |
| FP8_PB | 8-bit floating point with per-block scaling |
| NVFP4 | NVIDIA 4-bit floating point |
| NVFP4_AWQ | NVIDIA 4-bit floating point with AWQ optimization |
| INT4_AWQ | 4-bit integer with AWQ (TRT-LLM only) |
| W4A8_AWQ | 4-bit weights, 8-bit activations with AWQ (TRT-LLM only) |

## Minimum Framework Versions

| Framework | Minimum Version |
|-----------|----------------|
| TensorRT-LLM | v0.17.0 |
| vLLM | v0.10.1 |
| SGLang | v0.4.10 |

## Quantization Flag by Framework

| Framework | FP8 flag | FP4 flag |
|-----------|----------|----------|
| vLLM | `quantization="modelopt"` | `quantization="modelopt_fp4"` |
| SGLang | `quantization="modelopt"` | `quantization="modelopt_fp4"` |
| TRT-LLM | auto-detected from checkpoint | auto-detected from checkpoint |

## Models not in the matrix

The matrix covers the combinations modelopt tracks, not the full set of what will run. For unlisted models:

1. **Check the framework's own docs** — vLLM and SGLang support many HuggingFace models natively. Use WebSearch to check `vllm supported models` or `sglang supported models`.
2. **Try it** — if the model uses standard `nn.Linear` layers and has `hf_quant_config.json`, vLLM/SGLang will likely work with `--quantization modelopt`.
3. **Ask the user** — if unsure, ask: "This model isn't in the support matrix. Would you like to try deploying it anyway?"

## Notes

- **NVFP4 inference requires Blackwell GPUs** (B100, B200, B300, GB200, GB300). Hopper can run FP4 calibration but not inference.
  - **B300/GB300 are `sm_103`** and need a **CUDA-13** serving image — from v0.20.0 the unsuffixed tag is CUDA-13 (`-cu129` opts back to CUDA 12); `cu12` images lack the `sm_103` FP4 kernel and serve NVFP4 as gibberish or error out. See the CUDA-13 note in the deployment `SKILL.md`.
  - **Verify the GPU with `nvidia-smi`** before choosing the image — cluster GPU labels can be stale.
- INT4_AWQ and W4A8_AWQ are only supported by TRT-LLM (not vLLM or SGLang).
- For VLMs, only the language model is quantized; the vision encoder stays in high precision, so multimodal serving depends on the framework's own support for that architecture.
- Source: `docs/source/deployment/3_unified_hf.rst` and `tests/examples/hf_ptq/test_deploy.py`

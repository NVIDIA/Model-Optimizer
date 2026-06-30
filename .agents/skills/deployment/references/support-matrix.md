# Deployment Support Matrix

## Unified HF Checkpoint — Framework Compatibility

| Model | Quant Format | TRT-LLM | vLLM | SGLang |
|-------|-------------|---------|------|--------|
| Llama 3.x | FP8 | yes | yes | yes |
| Llama 3.x | FP4 | yes | yes | yes |
| Llama 4 | FP8 | yes | — | yes |
| Llama 4 | FP4 | yes | — | — |
| DeepSeek R1 | FP8 | yes | yes | yes |
| DeepSeek R1 | FP4 | yes | yes | yes |
| DeepSeek V3 | FP8 | yes | yes | yes |
| DeepSeek V3 | FP4 | yes | yes | yes |
| Qwen 3 | FP8 | yes | yes | yes |
| Qwen 3 | FP4 | yes | yes | — |
| Qwen 3 MoE | FP8 | yes | yes | yes |
| Qwen 3 MoE | FP4 | yes | — | — |
| Qwen 2.5 | FP8 | yes | yes | yes |
| Qwen 2.5 | FP4 | yes | yes | — |
| QwQ-32B | FP8 | yes | yes | yes |
| QwQ-32B | FP4 | yes | yes | — |
| Mixtral 8x7B | FP8 | yes | yes | yes |
| Mixtral 8x7B | FP4 | yes | — | — |

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

## Models not in this list

This matrix covers officially validated combinations. For unlisted models:

1. **Check the framework's own docs** — vLLM and SGLang support many HuggingFace models natively. Use WebSearch to check `vllm supported models` or `sglang supported models`.
2. **Try it** — if the model uses standard `nn.Linear` layers and has `hf_quant_config.json`, vLLM/SGLang will likely work with `--quantization modelopt`.
3. **Ask the user** — if unsure, ask: "This model isn't in the validated support matrix. Would you like to try deploying it anyway?"

## Notes

- **NVFP4 inference requires Blackwell GPUs** (B100, B200, B300, GB200, GB300). Hopper can run FP4 calibration but not inference.
  - **B300/GB300 are `sm_103`** and require a **CUDA-13 (`-cu130`) serving image** (e.g. `vllm/vllm-openai:<ver>-cu130`). The default cu12 build has **no `sm_103` FP4 kernel**, so vLLM/SGLang serve NVFP4 as gibberish or error out. See the deployment `SKILL.md` `-cu130` note.
  - **Verify the GPU with `nvidia-smi`** before choosing the image — cluster GPU labels can be stale (in day0 runs, AWS-PDX nodes advertised as B200 were actually B300/`sm_103`).
- INT4_AWQ and W4A8_AWQ are only supported by TRT-LLM (not vLLM or SGLang).
- Source: `examples/hf_ptq/README.md` and `docs/source/deployment/3_unified_hf.rst`

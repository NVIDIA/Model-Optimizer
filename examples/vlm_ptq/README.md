# [Deprecated] Post-training quantization (PTQ) for Vision Language Models

> **This example has been consolidated into [`examples/llm_ptq`](../llm_ptq/README.md) and is
> deprecated.** It will be removed in a future release. VLM PTQ now shares the same entry point
> (`hf_ptq.py`) and shell script as LLM PTQ.

## Migration

Use the `llm_ptq` script with the `--vlm` flag:

```bash
cd examples/llm_ptq
scripts/huggingface_example.sh --model <Hugging Face model card or checkpoint> --quant [fp8|nvfp4|int8_sq|int4_awq|w4a8_awq] --vlm
```

The previous `examples/vlm_ptq/scripts/huggingface_example.sh` entry point still works: it now
prints a deprecation warning and forwards to the command above.

## Where things moved

| Topic | New location |
| :--- | :--- |
| Supported VLMs / support matrix | [llm_ptq/README.md#vision-language-model-vlm-supported-models](../llm_ptq/README.md#vision-language-model-vlm-supported-models) |
| VLM quantization workflow (`--vlm`) | [llm_ptq/README.md#vlm-quantization](../llm_ptq/README.md#vlm-quantization) |
| Image-text calibration (`--calib_with_images`) | [llm_ptq/README.md#vlm-calibration-with-image-text-pairs-eg-nemotron-vl](../llm_ptq/README.md#vlm-calibration-with-image-text-pairs-eg-nemotron-vl) |
| VILA dependencies | [llm_ptq/requirements-vila.txt](../llm_ptq/requirements-vila.txt) |
| Megatron-Bridge VLM PTQ | [examples/megatron_bridge/](../megatron_bridge/README.md) |

## Resources

- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)

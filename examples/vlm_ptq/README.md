# [Deprecated] Post-training quantization (PTQ) for Vision Language Models

> **This example has been consolidated into [`examples/hf_ptq`](../hf_ptq/README.md) and is
> deprecated.** It will be removed in a future release. VLM PTQ now shares the same entry point
> (`hf_ptq.py`) and shell script as LLM PTQ.

## Migration

Use the `hf_ptq` script with the `--vlm` flag:

```bash
cd examples/hf_ptq
scripts/huggingface_example.sh --model <Hugging Face model card or checkpoint> --quant [fp8|nvfp4|int8_sq|int4_awq|w4a8_awq] --vlm
```

The previous `examples/vlm_ptq/scripts/huggingface_example.sh` entry point still works: it now
prints a deprecation warning and forwards to the command above.

## Where things moved

| Topic | New location |
| :--- | :--- |
| Supported VLMs / support matrix | [hf_ptq/README.md#hugging-face-supported-models](../hf_ptq/README.md#hugging-face-supported-models) |
| VLM quantization workflow (`--vlm`) | [hf_ptq/README.md#vlm-quantization](../hf_ptq/README.md#vlm-quantization) |
| Image-text calibration (`--calib_with_images`) | [hf_ptq/README.md#vlm-calibration-with-image-text-pairs-eg-nemotron-vl](../hf_ptq/README.md#vlm-calibration-with-image-text-pairs-eg-nemotron-vl) |
| Megatron-Bridge VLM PTQ | [examples/megatron_bridge/](../megatron_bridge/README.md) |

## Resources

- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)

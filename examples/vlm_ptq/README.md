# Post-training quantization (PTQ) for Vision Language Models

To learn more about the quantization feature, please refer to the [documentation](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html).

Quantization is an effective model optimization technique that compresses your models. Quantization with Model Optimizer can compress model size by 2x-4x, speeding up inference while preserving model quality. \
Model Optimizer enables highly performant quantization formats including NVFP4, FP8, INT8, INT4 and supports advanced algorithms such as SmoothQuant, AWQ, SVDQuant, and Double Quantization with easy-to-use Python APIs.

This section focuses on Post-training quantization for VLM (Vision Language Models), a technique that reduces model precision after training to improve inference efficiency without requiring retraining.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | \[[Link](#pre-requisites)\] | |
| Getting Started | Learn how to optimize your models using PTQ to reduce precision and improve inference efficiency | \[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
| Support Matrix | View the support matrix to see quantization compatibility and feature availability across different models | \[[Link](#support-matrix)\] | |
| Framework Scripts | Example scripts demonstrating quantization techniques for optimizing Hugging Face / NeMo / Megatron-LM models | \[[Link](#framework-scripts)\] | |
| Pre-Quantized Checkpoints | Ready to deploy Hugging Face pre-quantized checkpoints | \[[Link](#pre-quantized-checkpoints)\] | |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

Please refer to the [llm_ptq/README.md](../llm_ptq/README.md#pre-requisites) for the pre-requisites.

## Getting Started

Please refer to the [llm_ptq/README.md](../llm_ptq/README.md#getting-started) for the getting-started.

## Support Matrix

### Supported Models

| Model | fp8 | int8_sq<sup>1</sup> | int4_awq | w4a8_awq<sup>2</sup> | nvfp4<sup>3</sup> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Llava | ✅ | ✅ | ✅ | ✅ | - |
| VILA | ✅ | ✅ | ✅ | ✅ | - |
| Phi-3-vision, Phi-4-multimodal | ✅ | ✅ | ✅ | ✅ | ✅  |
| Qwen2, 2.5-VL | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gemma3 | ✅ | - | - | - | - |

> *<sup>1.</sup>Only TensorRT-LLM checkpoint export is supported. Not compatible with the TensorRT-LLM torch backend* \
> *<sup>2.</sup>The w4a8_awq is an experimental quantization scheme that may result in a higher accuracy penalty.* \
> *<sup>3.</sup>A selective set of the popular models are internally tested. The actual model support list may be longer. NVFP4 inference requires Blackwell GPUs and TensorRT-LLM v0.17 or later.*

> *For detailed TensorRT-LLM torch backend multimodal support, please refer to [this doc](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md#multimodal-feature-support-matrix-pytorch-backend)*

> *The accuracy loss after PTQ may vary depending on the actual model and the quantization method. Different models may have different accuracy loss and usually the accuracy loss is more significant when the base model is small. If the accuracy after PTQ is not meeting the requirement, please try either modifying [hf_ptq.py](../llm_ptq/hf_ptq.py) and disabling the KV cache quantization or using the [QAT](./../llm_qat/README.md) instead.*

## Framework Scripts

Please refer to the [llm_ptq/README.md](../llm_ptq/README.md) about the details of model quantization.

The following scripts provide an all-in-one and step-by-step model quantization example for the supported Hugging Face multi-modal models. The quantization format and the number of GPUs will be supplied as inputs to these scripts.

### Hugging Face Example [Script](./scripts/huggingface_example.sh)

```bash
scripts/huggingface_example.sh --model <Hugging Face model card or checkpoint> --quant [fp8|nvfp4|int8_sq|int4_awq|w4a8_awq]
```

## Pre-Quantized Checkpoints

- Ready-to-deploy checkpoints \[[🤗 Hugging Face - Nvidia Model Optimizer Collection](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)\]
- Deployable on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)
- More models coming soon!

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

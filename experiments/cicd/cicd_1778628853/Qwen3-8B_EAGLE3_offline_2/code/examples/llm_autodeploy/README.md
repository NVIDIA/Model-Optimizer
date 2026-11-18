# Deploy AutoQuant Models with AutoDeploy

This guide demonstrates how to deploy mixed-precision models using ModelOpt's AutoQuant and TRT-LLM's AutoDeploy.

[ModelOpt's AutoQuant](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize) is a post-training quantization (PTQ) algorithm that optimizes model quantization by selecting the best quantization format for each layer while adhering to user-defined compression constraints. This approach allows users to balance model accuracy and performance effectively.

[TRT-LLM's AutoDeploy](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/auto_deploy) is designed to simplify and accelerate the deployment of PyTorch models, including off-the-shelf models like those from Hugging Face, to optimized inference environments with TRT-LLM. It automates graph transformations to integrate inference optimizations such as tensor parallelism, KV-caching and quantization. AutoDeploy supports optimized in-framework deployment, minimizing the amount of manual modification needed.

## Prerequisites

AutoDeploy is available in TensorRT-LLM docker images. Please refer to our [Installation Guide](../../README.md#installation) for more details.

### 1. Quantize and Deploy Model

Run the following command to quantize your model and launch an OpenAI-compatible endpoint:

```bash
./scripts/run_auto_quant_and_deploy.sh \
    --hf_ckpt <path_to_HF_model> \
    --save_quantized_ckpt <path_to_save_quantized_checkpoint> \
    --quant fp8,nvfp4 \
    --effective_bits 4.5
```

Parameters:

- `--hf_ckpt`: Path to the unquantized Hugging Face checkpoint
- `--save_quantized_ckpt`: Output path for the quantized checkpoint
- `--quant`: Quantization formats to use (e.g., `fp8,nvfp4`)
- `--effective_bits`: Target overall precision (higher values preserve accuracy for sensitive layers)
- `--calib_batch_size`: (Optional, default=8) Calibration batch size. Reduce if encountering OOM issues

> **Note**:
>
> - NVFP4 is only available on Blackwell GPUs. For Hopper GPUs:
>   - Remove `nvfp4` from the `--quant` parameter
>   - Increase `--effective_bits` above 8.0 for FP8-only AutoQuant
> - For tensor parallelism, add `--world_size <gpu_num>`
> - Additional generation and sampling configurations can be found in `api_server.py`

### 2. Test the Deployment

Send test prompts to the server:

```bash
python api_client.py --prompt "What is AI?" "What is golf?"
```

This will return generated responses for both prompts from your deployed model.

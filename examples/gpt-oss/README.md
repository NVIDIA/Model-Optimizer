# OpenAI GPT-OSS Quantization Aware Training (QAT) & Quantized Deployment

This folder demonstrates Quantization Aware Training (QAT) and deployment examples for OpenAI's GPT-OSS models (20B and 120B parameters). The GPT-OSS models come natively quantized using [MXFP4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) (Microscaling FP4), a 4-bit floating-point format (E2M1). Thanks to MXFP4, the 20B model fits into a 16 GB GPU and the 120B model fits into a single 80 GB GPU.

Being an open weights model, developers can finetune it to add special abilities or domain knowledge. Native MXFP4 finetuning is challenging since the dynamic range and precision might not be sufficient to handle gradients during backpropagation. Dequantizing the MXFP4 models to BF16 and performing BF16 training is a viable option. However this results in a BF16 weights model which is about 4x of the original model size. Hence the next option is to perform MXFP4 Post Training Quantization (PTQ) on the finetuned model. However, PTQ degrades the finetuned model accuracy.

Performing finetuning with Quantization Aware Training solves these issues. The model after QAT is in MXFP4 precision and can be deployed with smaller memory footprint just like the original GPT-OSS models to performant serving frameworks like
[TensorRTLLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang).

## Table of Contents

1. [Setup](#setup)
1. [Quantization Aware Training from ModelOpt](#quantization-aware-training-from-modelopt)
1. [Deployment](#deployment)
1. [LoRA QAT: low memory footprint alternative for full parameter QAT](#lora-qat-low-memory-footprint-alternative-for-full-parameter-qat)
1. [Quantization Aware Training & Deployment for models beyond GPT-OSS](#quantization-aware-training--deployment-for-models-beyond-gpt-oss)

## Setup

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Quantization Aware Training from ModelOpt

In Quantization Aware Training, the forward computations are performed with 'fake quantized' values and the backward computations are performed with high precision datatype. In 'fake quantization' the numerical equivalent of the quantized value is represented using a high precision datatype such as BF16. Hence, QAT can be integrated to standard training pipeline such as regular BF16 mixed precision training.

During QAT, the model learns to recover the accuracy after quantization. To perform QAT, quantize your model first using ModelOpt's [`mtq.quantize`](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.quantize) API. Then you can train this quantized model with your existing training pipeline.

Here is a code example:

```python
import modelopt.torch.quantization as mtq

# Specify quantization config;
config = mtq.MXFP4_MLP_WEIGHT_ONLY_CFG

# Define forward loop for calibration
def forward_loop(model):
    for data in calib_set:
        model(data)

# quantize the model and prepare for QAT
model = mtq.quantize(model, config, forward_loop)

# QAT with your regular finetuning pipeline
train(model, train_loader, optimizer, scheduler, ...)
```

For an end to end example showcasing the above workflow, checkout [qat-finetune-transformers.ipynb](./qat-finetune-transformers.ipynb).

If you are training Huggingface models with trainer classes from Huggingface such as [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) performing QAT is even easier - simply replace the trainer with its equivalent, `QATSFTTrainer` from ModelOpt and specify additional quantization arguments to it. `QATSFTTrainer` will perform the necessary quantization steps in the backend and train the model just like the original `SFTTrainer`.

A real end-to-end example for this is in `sft.py` in this folder. To perform QAT with full parameter SFT on GPT-OSS 20B model, run:

```sh
# Other supported quantization configs include NVFP4_MLP_WEIGHT_ONLY_CFG, NVFP4_MLP_ONLY_CFG etc.
# [Optional] For faster FlashAttention3, add '--attn_implementation kernels-community/vllm-flash-attn3'
accelerate launch --config_file configs/zero3.yaml sft.py \
    --config configs/sft_full.yaml --model_name_or_path openai/gpt-oss-20b \
    --quant_cfg MXFP4_MLP_WEIGHT_ONLY_CFG \
    --output_dir gpt-oss-20b-qat
```

GPT-OSS 20B full parameter SFT needs one node with 8 x 80 GB GPUs. To change dataset or training hyperparameters, either modify `configs/sft_full.yaml` or pass them as command line arguments.

### Recommended QAT Recipe

For improved accuracy, we recommend the following QAT recipe:

- **Step 1: Fine-tune the model in high precision**

- **Step 2: Apply QAT on the finetuned model (from step 1)**

  - A small learning rate such as 1e-5 with Adam Optimizer works well for QAT after high precision training.
  - QAT usually recovers accuracy within a few million to billion tokens. Evaluate your checkpoints to determine whether accuracy have been recovered.

To perform this recommended QAT recipe, run:

```sh
# Step 1: Perform high precision SFT without quantization
accelerate launch --config_file configs/zero3.yaml sft.py \
  --config configs/sft_full.yaml --model_name_or_path openai/gpt-oss-20b \
  --output_dir gpt-oss-20b-sft

# Step 2: Perform QAT on the high precision SFT checkpoint
accelerate launch --config_file configs/zero3.yaml sft.py \
    --config configs/sft_full.yaml --model_name_or_path gpt-oss-20b-sft \
    --quant_cfg MXFP4_MLP_WEIGHT_ONLY_CFG \
    --output_dir gpt-oss-20b-qat \
```

The final QAT checkpoint is in fake-quantized form. Low memory footprint and speedup comes after [deployment](#deployment) to accelerated runtimes.

Note: For restoring the model checkpoint for Pytorch native evaluation, see [ModelOpt Restore using Huggingface APIs](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html#modelopt-save-restore-using-huggingface-checkpointing-apis).

## Deployment

The GPT-OSS QAT models from above can be deployed in MXFP4 format using performant serving engines like TensorRT-LLM, vLLM, and SGLang. To enable this, we provide a custom conversion script that transforms a Hugging Face–compatible BF16 checkpoint into the same MXFP4 weight-only format used by the original GPT-OSS release. This real MXFP4 quantized checkpoint can be deployed to supported runtimes just like the original GPT-OSS MXFP4 models.

To export the QAT checkpoint to real quantized MXFP4, run:

```bash
python convert_oai_mxfp4_weight_only.py  \
    --model_path gpt-oss-20b-qat \
    --output_path gpt-oss-20b-qat-real-mxfp4
```

Note: Model Optimizer currently exports quantized checkpoints in formats other than MXFP4. Support for ModelOpt-generated MXFP4 checkpoints in vLLM, SGLang, and TensorRT-LLM is planned and actively in development.

<details>
<summary><strong>Deployment on TensorRT-LLM</strong></summary>

To setup TensorRT-LLM, follow the official guide: [Deploying GPT-OSS on TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog9_Deploying_GPT_OSS_on_TRTLLM.md)
Once installed, launch an OpenAI-compatible endpoint using:

```bash
trtllm-serve path/to/quantized/checkpoint --tokenizer /path/to/tokenizer --max_batch_size <max_batch_size> --max_num_tokens <max_num_tokens> --max_seq_len <max_seq_len> --tp_size <tp_size> --pp_size <pp_size> --host <host_ip_address> --port <port> --kv_cache_free_gpu_memory_fraction 0.95

```

</details>

<details>
<summary><strong>Deployment on SGLang</strong></summary>

To setup SGLang, refer to the setup issue: [SGLang Setup Guide](https://github.com/sgl-project/sglang/issues/8833)
Then start the server with:

```bash
python3 -m sglang.launch_server --model <model-path> --tp <tp_size>

```

</details>

<details>
<summary><strong>Deployment on vLLM</strong></summary>

To deploy with vLLM, follow the [OpenAI Cookbook instructions](https://cookbook.openai.com/articles/gpt-oss/run-vllm)
Then start the server with:

```bash
vllm serve <model_path>

```

</details>
<br>

## LoRA QAT: low memory footprint alternative for full parameter QAT

You may run QAT with LoRA to reduce the training GPU memory requirement. Using one node with 8 x 80 GB GPUs, you could perform QAT with LoRA on GPT OSS 120B model.

Here is how to run LoRA QAT for GPT OSS 120B model:

```bash
python sft.py --config configs/sft_lora.yaml \
    --model_name_or_path openai/gpt-oss-120b \
    --quant_cfg MXFP4_MLP_WEIGHT_ONLY_CFG \
    --output_dir gpt-oss-120b-lora-qat
```

The LoRA-QAT adapter weights from the QAT process above need to be merged with the base weights for deployment.
The custom conversion script above performs lora adapter merging before exporting MXFP4 weights. For this, specify the `lora_path` and `base_model_path` to the custom conversion script:

```sh
python convert_oai_mxfp4_weight_only.py  \
    --lora_path gpt-oss-120b-lora-qat \
    --base_path openai/gpt-oss-120b \
    --output_path gpt-oss-120b-lora-qat-merged-real-mxfp4
```

You can deploy this real quantized MXFP4 checkpoint just like the original GPT-OSS MXFP4 model.

## Quantization Aware Training & Deployment for models beyond GPT-OSS

### Easy QAT from ModelOpt using LLaMA-Factory

ModelOpt provides easy end to end QAT via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), an open-source repository for LLM/VLM finetuning. Please refer to [LLaMa-Factory QAT example](../llm_qat/llama_factory) for performing QAT on your favorite models.

### Deployment of ModelOpt QAT/PTQ models beyond GPT-OSS

ModelOpt supports exporting a wide variety of models after QAT/PTQ to TensorRT-LLM, vLLM, SGLang etc. Please refer to [llm_ptq](../llm_ptq).

# Quantization Aware Training and Distillation with LLaMA-Factory

This directory provides integration between [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main) and ModelOpt for **Quantization-Aware Training (QAT)** and **Quantization-Aware Distillation (QAD)**. This enables efficient training of large language models with quantization techniques while maintaining model quality. This README only covers QAT/QAD training with LLaMA-Factory. For more information on setting up other datasets, training different models, or customizing training configurations, please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main).

## Quick Start

### Basic QAT/QAD Training with FSDP

```bash
./launch_llamafactory.sh llama_config.yaml
```

> **_NOTE:_** The `launch_llamafactory.sh` script automatically installs LLaMA Factory if it's not already present in your environment.

In order to train using FSDP2:

```sh
./launch_llamafactory.sh llama_config.yaml --use_fsdp2 true
```

By default, the script uses [fsdp1.yaml](../accelerate_config/fsdp1.yaml) and [fsdp2.yaml](../accelerate_config/fsdp2.yaml) for FSDP and FSDP2 training respectively.

**Use Custom FSDP Arguments**:

Pass additional FSDP parameters using the `FSDP_ARGS` environment variable:

```bash
FSDP_ARGS="--fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer" \
./launch_llamafactory.sh llama_config.yaml
```

> **_NOTE:_** The default `fsdp*.yml` files use `LlamaDecoderLayer` as the transformer layer class. If your model uses a different layer class, you can either pass `--fsdp_transformer_layer_cls_to_wrap <your_layer_class>` to the `launch_llamafactory.sh` script or provide a custom FSDP configuration file.

**Custom Config File**:

Specify your own FSDP configuration:

```bash
./launch_llamafactory.sh llama_config.yaml \
  --accelerate_config /path/to/custom_fsdp.yaml
```

### Training using CLI

For QAT/QAD training using llamafactory_cli, run

```sh
./launch_llamafactory.sh train llama_config.yaml
```

## Configuration Guide

This section explains how to configure quantization parameters in your training setup.

### YAML Configuration Structure

Your configuration file should follow same structure as sft example from [llama3_full_sft.yaml](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_full/llama3_full_sft.yaml) with addition of modelopt configuration:

```yaml
# Model Configuration
model:
  model_name_or_path: /path/to/your/model
  trust_remote_code: true

### method
stage: sft                    # Supervised Fine-Tuning
do_train: true
finetuning_type: full         # Full parameter fine-tuning

### dataset
dataset: your_dataset
cutoff_len: 4096             # Maximum sequence length
max_samples: 100000          # Maximum training samples

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 1
bf16: true                   # Use bfloat16 precision

val_size: 0.1               # Validation split ratio

### ModelOpt Configuration
modelopt:
  quant_cfg: NVFP4_DEFAULT_CFG            # Quantization format
  calib_size: 1024                        # Calibration dataset size
  compress: false                         # Enable weight compression
  distill: false                          # Modify distill to true for QAD
  teacher_model: /path/to/teacher/model   # For QAD (optional)
```

> **_NOTE:_** `compress: true` enables weight compression and will by default use [ddp.yaml](../accelerate_config/ddp.yaml).
> **_NOTE:_** When training without [cli](#training-using-cli), avoid using deepspeed option in the YAML configuration file.

## Deployment

The final QAT/QAD model after training is similar in architecture to that of PTQ model. It simply has updated weights as compared to the PTQ model. It can be deployed to TensorRT-LLM (TRTLLM) or to TensorRT just like a regular **ModelOpt** PTQ model if the quantization format is supported for deployment.

To run QAT/QAD model with TRTLLM, run:

```sh
cd ../../llm_ptq

./scripts/huggingface_example.sh --model <path-to-QAT/QAD-model> --quant nvfp4
```

See more details on deployment of quantized model [here](../../llm_ptq/README.md).

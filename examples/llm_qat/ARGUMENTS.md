# Argument Reference

_Auto-generated — do not edit by hand._

## DistillArguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--distill` | `bool` | `False` | Enable training with knowledge distillation. |
| `--teacher_model` | `str` | `None` | The name or path of the teacher model. |
| `--criterion` | `str` | `"logits_loss"` | Distillation loss criterion. Currently only 'logits_loss' is supported. |
| `--temperature` | `float` | `1.0` | Softmax temperature for softening logits in KD loss. Used by both standard and Liger KD loss. |
| `--liger_jsd_beta` | `float` | `0.0` | JSD beta coefficient in [0, 1]. 0=forward KL, 1=reverse KL. Only used when --use_liger_kernel is enabled. |

## DataArguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_config` | `str` | `"configs/dataset/blend.yaml"` | Path to a dataset blend YAML config file. |
| `--train_samples` | `int` | `20000` | Number of training samples to use. |
| `--eval_samples` | `int` | `2000` | Number of evaluation samples to use. |
| `--dataset_seed` | `int` | `42` | Random seed for dataset shuffling. |
| `--dataset_cache_dir` | `str` | `".dataset_cache/tokenized"` | Directory for caching tokenized datasets. |
| `--shuffle` | `bool` | `True` | Whether to shuffle dataset sources (reservoir sampling). |
| `--shuffle_buffer` | `int` | `10000` | Buffer size for streaming shuffle. |
| `--num_proc` | `int` | `16` | Number of CPU workers for tokenization. |

## ModelArguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name_or_path` | `str` | `"meta-llama/Llama-2-7b-hf"` | HuggingFace model ID or local path to a pretrained model. |
| `--model_max_length` | `int` | `8192` | Maximum sequence length. Sequences will be right-padded (and possibly truncated). |
| `--attn_implementation` | `str` | `None` | Attention implementation: 'flash_attention_2', 'flash_attention_3', 'sdpa', or 'eager'. |

## QuantizeArguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--recipe` | `str` | `None` | Path to a quantization recipe YAML file (built-in or custom). Built-in recipes can be specified by relative path, e.g. 'general/ptq/nvfp4_default-fp8_kv'. |
| `--calib_size` | `int` | `512` | Specify the calibration size for quantization. The calibration dataset is used to setup the quantization scale parameters. |
| `--calib_batch_size` | `int` | `1` | Batch size for calibration data during quantization. |
| `--compress` | `bool` | `False` | Whether to compress the model weights after quantization for QLoRA. This is useful for reducing the model size. |
| `--quantize_output_dir` | `str` | `"quantized_model"` | Directory to save the quantized model checkpoint. |

## TrainingArguments

Extends [HuggingFace TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). Only additional arguments are shown below.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--trainable_params` | `list[str]` | `None` | Glob patterns (fnmatch) for parameters that should be trainable. All other parameters will be frozen. Mutually exclusive with frozen_params. |
| `--frozen_params` | `list[str]` | `None` | Glob patterns (fnmatch) for parameters that should be frozen. Mutually exclusive with trainable_params. |
| `--lr_config` | `str` | `None` | Path to a YAML file mapping fnmatch patterns to optimizer kwargs (e.g. lr, weight_decay). First matching pattern wins per parameter. See examples/llm_qat/configs/train/lr_config_example.yaml. |
| `--save_dtype` | `str` | `"bfloat16"` | Dtype string to write into the saved model's config.json (e.g. 'bfloat16', 'float16'). Set to None to preserve the original dtype. |
| `--manual_gc` | `bool` | `False` | Run `gc.collect()` before each training/prediction step to work around GPU memory leaks during QAT/distillation. |
| `--liger_ce_label_smoothing` | `float` | `0.0` | Label smoothing for Liger fused CE loss. Only used when --use_liger_kernel is enabled. |
| `--lora` | `bool` | `False` | Whether to add LoRA (Low-Rank Adaptation) adapter before training. When using real quantization, the LoRA adapter must be set, as quantized weights will be frozen during training. |

# Post-training quantization (PTQ)

Quantization is an effective model optimization technique that compresses your models. Quantization with Model Optimizer can compress model size by 2x-4x, speeding up inference while preserving model quality.

Model Optimizer enables highly performant quantization formats including NVFP4, FP8, INT8, INT4 and supports advanced algorithms such as SmoothQuant, AWQ, SVDQuant, and Double Quantization with easy-to-use Python APIs.

This section focuses on Post-training quantization, a technique that reduces model precision after training to improve inference efficiency without requiring retraining.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | \[[Link](#pre-requisites)\] | |
| Getting Started | Learn how to optimize your models using PTQ to reduce precision and improve inference efficiency | \[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
| Support Matrix | View the support matrix to see quantization compatibility and feature availability across different models | \[[Link](#support-matrix)\] | |
| AutoQuantize | Automatically chooses layers/precisions for mixed precision quantization to enhanced inference performance and accuracy tradeoffs | \[[Link](#autoquantize)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#optimal-partial-quantization-using-auto-quantize)\] |
| Real Quant | Real Quant compresses model weights in a low-precision format to reduce memory requirements of quantization. | \[[Link](https://nvidia.github.io/Model-Optimizer/guides/_compress_quantized_models.html)\] | |
| Framework Scripts | Example scripts demonstrating quantization techniques for optimizing Hugging Face / NeMo / Megatron-LM models | \[[Link](#framework-scripts)\] | |
| Evaluate Accuracy | Evaluate your model's accuracy! | \[[Link](#evaluate-accuracy)\] | |
| Exporting Checkpoints | Export to Hugging Face Unified Checkpoint and deploy on TRT-LLM/vLLM/SGLang | \[[Link](#exporting-checkpoints)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html)\] |
| Pre-Quantized Checkpoints | Ready to deploy Hugging Face pre-quantized checkpoints | \[[Link](#pre-quantized-checkpoints)\] | |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

### Docker

For Hugging Face models, please use the TensorRT-LLM docker image (e.g., `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4`).
For NeMo models, use the NeMo container (e.g., `nvcr.io/nvidia/nemo:25.09`).
Visit our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) for more information.

Also follow the installation steps below to upgrade to the latest version of Model Optimizer and install example-specific dependencies.

### Local Installation

For Hugging Face models, install Model Optimizer with `hf` dependencies using `pip` from [PyPI](https://pypi.org/project/nvidia-modelopt/) and install the requirements for the example:

```bash
pip install -U nvidia-modelopt[hf]
pip install -r requirements.txt
```

For TensorRT-LLM deployment, please use the TensorRT-LLM docker image or follow their [installation docs](https://nvidia.github.io/TensorRT-LLM/installation/index.html).
Similarly, for vLLM or SGLang deployment, please use their installation docs.

## Getting Started

### 1. Quantize (Post Training Quantization)

With the simple API below, you can very easily use Model Optimizer to quantize your model. Model Optimizer achieves this by converting the precision of your model to the desired precision, and then using a small dataset (typically 128-512 samples) to [calibrate](https://nvidia.github.io/Model-Optimizer/guides/_basic_quantization.html) the quantization scaling factors. The accuracy of PTQ is typically robust across different choices of calibration data, by default Model Optimizer uses a mix of [`cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail) and [`nemotron-post-training-dataset-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2). Users can try other datasets by easily modifying the `calib_set`.

```python
import modelopt.torch.quantization as mtq

# Setup the model
model = AutoModelForCausalLM.from_pretrained("...")

# Simplified example set up a calibration data loader with the desired calib_size
calib_set = get_dataloader(num_samples=calib_size)

# Prepare the calibration set and define a forward loop
def forward_loop(model):
    for batch in calib_set:
        model(batch)

# PTQ with in-place replacement to quantized modules
model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)
```

### 2. Export Quantized Model

Once your model is quantized, you can now export that model to a checkpoint for easy deployment. \
We provide two APIs to export the quantized model:

- Unified Hugging Face checkpoints, which can be deployed on TensorRT-LLM (Pytorch and C++ backends), [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).
- (Legacy) TensorRT-LLM checkpoints, a format that works with TensorRT-LLM C++ backend only.

#### Unified Hugging Face Checkpoints

```python
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir,  # The directory where the exported files will be stored.
    )
```

Please reference our [framework scripts](#framework-scripts) and our [docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html) for more details.

## Support Matrix

### Hugging Face Supported Models

| Model | fp8 | int8_sq | int4_awq | w4a8_awq<sup>1</sup> | nvfp4<sup>5</sup> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| LLAMA 3.x | ✅ | ❌ | ✅ | ✅<sup>3</sup> | ✅ |
| LLAMA 4 <sup>6</sup> | ✅ | ❌ | ❌ | ❌ | ✅ |
| Mixtral | ✅ | ❌ | ✅<sup>2</sup> | ❌ | ✅ |
| Phi-3,4 | ✅ | ✅ | ✅ | ✅<sup>3</sup> | - |
| Phi-3.5 MOE | ✅ | ❌ | ❌ | ❌ | - |
| Llama-Nemotron Super | ✅ | ❌ | ❌ | ❌ | ✅ |
| Llama-Nemotron Ultra | ✅ | ❌ | ❌ | ❌ | ❌ |
| Gemma 3 | ✅<sup>2</sup> | - | ✅ | - | - |
| QWen 2, 2.5 <sup>4</sup> | ✅ | ✅ | ✅ | ✅ | ✅ |
| QWen3 MOE, Next <sup>6</sup> | ✅ | - | - | - | ✅ |
| QwQ | ✅ | - | - | - | ✅ |
| DeepSeek V3, R1, V3.1, V3.2<sup>7</sup> | - | - | - | - | ✅ |
| GLM-4.7<sup>8</sup> | ✅ | - | - | - | ✅ |
| Kimi K2 | - | - | - | - | ✅ |
| MiniMax M2.1 | - | - | - | - | ✅ |
| T5 | ✅ | ✅ | ✅ | ✅ | - |
| Whisper | ✅ | ❌ | ❌ | ❌ | - |

> *This is a subset of the models supported. For the full list please check the [TensorRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/reference/precision.html#support-matrix)*

> *<sup>1.</sup>The w4a8_awq is an experimental quantization scheme that may result in a higher accuracy penalty.* \
> *<sup>2.</sup>For some models, there is only support for exporting quantized checkpoints.* \
> *<sup>3.</sup>W4A8_AWQ is only available on some models but not all* \
> *<sup>4.</sup>For some models, KV cache quantization may result in a higher accuracy penalty.* \
> *<sup>5.</sup>A selective set of the popular models are internally tested. The actual model support list may be longer. NVFP4 inference requires Blackwell GPUs and TensorRT-LLM v0.17 or later* \
> *<sup>6.</sup>Some models currently support export to HF format only.* \
> *<sup>7.</sup>[PTQ for DeepSeek](../deepseek/README.md)* \
> *<sup>8.</sup>GLM-4.7 has MTP (Multi-Token Prediction) layers that are automatically loaded and excluded from quantization.*

> *The accuracy loss after PTQ may vary depending on the actual model and the quantization method. Different models may have different accuracy loss and usually the accuracy loss is more significant when the base model is small. If the accuracy after PTQ is not meeting the requirement, please try either modifying [hf_ptq.py](./hf_ptq.py) and disabling the KV cache quantization or using the [QAT](./../llm_qat/README.md) instead.*

> You can also create your own custom config using [this](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#custom-calibration-algorithm) guide.

### NeMo Supported Models

Please refer to the [NeMo 2.0 PTQ documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/quantization/quantization.html#support-matrix) for supported models.

## Framework Scripts

### Hugging Face Example [Script](./scripts/huggingface_example.sh)

For LLM models like [Llama-3](https://huggingface.co/meta-llama):

```bash
# Install model specific pip dependencies if needed

export HF_PATH=<the downloaded LLaMA checkpoint from the Hugging Face hub, or simply the model card>
scripts/huggingface_example.sh --model $HF_PATH --quant [fp8|nvfp4|int8_sq|int4_awq|w4a8_awq] --tp [1|2|4|8]
```

> *By default `trust_remote_code` is set to false. Please turn it on if model calibration and eval requires it using `--trust_remote_code`.*

> *If the Huggingface model calibration fails on a multi-GPU system due to mismatched tensor placement, please try setting CUDA_VISIBLE_DEVICES to a smaller number.*

> *FP8 calibration over a large model with limited GPU memory is not recommended but possible with the [accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling) package. Please tune the device_map setting in [`example_utils.py`](./example_utils.py) if needed for model loading and the calibration process can be slow.*

> *Huggingface models trained with `modelopt.torch.speculative` can be used as regular Huggingface models in PTQ. Note: there is a known issue with Huggingface models loaded across multiple GPUs for inference (i.e., "Expected all tensors to be on the same device, but found at least two devices..."). When encountered this error in PTQ of speculative decoding models, try reducing the number of GPUs used.*

> *Calibration by default uses left padding_side for the Huggingface tokenizer as it usually leads to lower accuracy loss. The exported tokenizer files restores the default padding_side.*

> *If a GPU OOM error occurs during model quantization despite sufficient memory, setting the --use_seq_device_map flag can help. This enforces sequential device mapping, distributing the model across GPUs and utilizing up to 80% of each GPU's memory.*

> *You can add `--low_memory_mode` to the command to lower the memory requirements of the PTQ process. With this mode, the script will compress model weights to low precision before calibration. This mode is only supported for FP8 and NVFP4 with max calibration.*

#### Deepseek R1

[PTQ for DeepSeek](../deepseek/README.md) shows how to quantize the DeepSeek model with FP4 and export to TensorRT-LLM.

#### VLM calibration with image-text pairs (e.g., Nemotron VL)

For vision-language models, calibration quality can likely improve by using image-text pairs instead of text-only data, especially on visual understanding tasks:

```bash
python hf_ptq.py \
  --pyt_ckpt_path <huggingface_model_card> \
  --qformat nvfp4 \
  --export_path <quantized_ckpt_path> \
  --trust_remote_code \
  --calib_with_images \
  --calib_size 512
```

> Note: when `--calib_with_images` is set, `--calib_size` must be a single value, and the calibration dataset is nvidia/nemotron_vlm_dataset_v2.
This functionality is currently in beta and has been tested on `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`.

### NeMo Example Script

NeMo 2.0 framework PTQ and TensorRT-LLM deployment examples are maintained in the NeMo GitHub repo. Please refer to the [NeMo PTQ documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/quantization/quantization.html) for more details.

### Megatron-LM Example Script

Megatron-LM framework PTQ and TensorRT-LLM deployment examples are maintained in the Megatron-LM GitHub repo. Please refer to the examples [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt).

## AutoQuantize

[AutoQuantize (`mtq.auto_quantize`)](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize) is a PTQ algorithm which quantizes a model by searching for the best quantization format per-layer while meeting performance constraints specified by the user. `AutoQuantize` streamlines the trade-off of model accuracy and performance.

Currently `AutoQuantize` supports only `auto_quantize_bits` as the performance constraint (for both weight-only
quantization and weight & activation quantization). `auto_quantize_bits` constraint specifies the effective number of bits for the quantized model.

You may specify an `auto_quantize_bits` constraint such as 4.8 for mixed precision quantization using `NVFP4_DEFAULT_CFG` & `FP8_DEFAULT_CFG`.
`AutoQuantize` will automatically quantize highly sensitive layers in `FP8_DEFAULT_CFG` while keeping less sensitive layers in `NVFP4_DEFAULT_CFG` (and even skip quantization for any extremely sensitive layers) so that
the the final mixed precision quantized model has an effective quantized bits of 4.8. This model would give a better accuracy than the model quantized with vanilla `NVFP4_DEFAULT_CFG` configuration since the more aggressive `NVFP4_DEFAULT_CFG` quantization was not applied for the highly sensitive layers.

Here is an example usage for `AutoQuantize` algorithm (Please see [auto_quantize](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize) API for more details):

```python

    import modelopt.torch.quantization as mtq

    # Define the model & calibration dataloader
    model = ...
    calib_dataloader = ...

    # Define forward_step function.
    # forward_step should take the model and data as input and return the output
    def forward_step(model, data):
        output =  model(data)
        return output

    # Define loss function which takes the model output and data as input and returns the loss
    def loss_func(output, data):
        loss = ...
        return loss


    # Perform AutoQuantize
    model, search_state_dict = mtq.auto_quantize(
        model,
        constraints = {"auto_quantize_bits": 4.8},
        # supported quantization formats are listed in `modelopt.torch.quantization.config.choices`
        quantization_formats = ["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"]
        data_loader = calib_dataloader,
        forward_step=forward_step,
        loss_func=loss_func,
        ...
        )
```

### AutoQuantize for Hugging Face models

`AutoQuantize` can be performed for Huggingface LLM models like [Llama-3](https://huggingface.co/meta-llama) as shown below:

[Script](./scripts/huggingface_example.sh)

```bash
export HF_PATH=<the downloaded LLaMA checkpoint from the Hugging Face hub, or simply the model card>
# --auto_quantize_bits specifies the constraint for `AutoQuantize`
# --quant specifies the formats to be searched for `AutoQuantize`
# NOTE: auto_quantize_bits cannot be lower than the number of bits for the smallest quantization format in --quant
scripts/huggingface_example.sh --model $HF_PATH --quant w4a8_awq,fp8 --auto_quantize_bits 4.8 --calib_batch_size 4
```

The above example perform `AutoQuantize` where the less quantization accuracy sensitive layers are quantized with `w4a8_awq` (specified by `--quant w4a8_awq`) and the more sensitive layers
are kept un-quantized such that the effective bits is 4.8 (specified by `--auto_quantize_bits 4.8`).

The example scripts above also have an additional flag `--tasks`, where the actual tasks run in the script can be customized. The allowed tasks are `quant,mmlu,lm_eval,livecodebench` specified in the script [parser](./scripts/parser.sh). The tasks combo can be specified with a comma-separated task list. Some tasks like mmlu can take a long time to run. To run lm_eval tasks, please also specify the `--lm_eval_tasks` flag with comma separated lm_eval tasks [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

> *If GPU out-of-memory error is reported running the scripts, please try editing the scripts and reducing the max batch size to save GPU memory.*

> *NOTE: AutoQuantize requires backpropagation of the model. Models without backpropagation support (e.g., Llama-4) will not work with AutoQuantize.*

## Real Quant

When working with large language models, memory constraints can be a significant challenge. ModelOpt provides a workflow for initializing HF models with compressed weights across multiple GPUs to dramatically reduce memory usage. Check `--low_memory_mode` option in hf_ptq.py for more details.

```python
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins import init_quantized_weights
from transformers import AutoModelForCausalLM, AutoConfig

# Step 1: Initialize the model with compressed weights
with init_quantized_weights(mtq.NVFP4_DEFAULT_CFG):
    model = AutoModelForCausalLM.from_pretrained(ckpt_path)

# Step 2: Calibrate the model
mtq.calibrate(model, algorithm="max", forward_loop=calibrate_loop)
```

## Multi-Node Post-Training Quantization with FSDP2

ModelOpt enables quantization of LLMs across multiple GPU nodes using various quantization formats. It leverages HuggingFace's Accelerate library and FSDP2 for distributed model sharding and calibration.

### Usage

For distributed execution across multiple nodes, use the `accelerate` library. A template configuration file (`fsdp2.yaml`) is provided and can be customized for user specific requirements.

On each node run the following command:

```bash
accelerate launch --config_file fsdp2.yaml \
    --num_machines=<num_nodes> \
    --machine_rank=<current_node_rank> \
    --main_process_ip=<node0_ip_addr> \
    --main_process_port=<port> \
    --fsdp_transformer_layer_cls_to_wrap=<decoder_layer_name>
     multinode_ptq.py \
    --pyt_ckpt_path <path_to_model> \
    --qformat <fp8/nvfp4/nvfp4_awq/int8> \
    --kv_cache_qformat <fp8/nvfp4/nvfp4_affine/none> \
    --batch_size <calib_batch_size> \
    --calib_size <num_calib_samples> \
    --dataset <dataset> \
    --export_path <export_path> \
    --trust_remote_code
```

The exported checkpoint can be deployed using TensorRT-LLM/ vLLM/ SGLang. For more details refer to the [deployment section](#deployment) of this document.

> *Performance Note: FSDP2 is designed for training workloads and may result in longer calibration and export times. For faster calibration, maximize the batch size based on available GPU memory and choose the right number of GPUs to avoid unnecessary communication.*

## Evaluate Accuracy

### TensorRT-LLM Validation

A list of accuracy validation benchmarks are provided in the [llm_eval](../llm_eval/README.md) directory. Right now MMLU, and MTbench are supported in this example by specifying the `--tasks` flag running the scripts mentioned above. For MTBench, the task only runs the answer generation stage. Please follow [fastchat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) to get the evaluation judge score.

The `benchmark_suite.py` script is used as a fast performance benchmark. For details, please refer to the [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/)

This example also covers the [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness), MMLU and the human eval accuracy benchmarks, whose details can be found [here](../llm_eval/README.md). The supported lm_eval evaluation tasks are listed [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)

## Exporting Checkpoints

Model Optimizer supports provide two paths to export the quantized model:

- Unified Hugging Face checkpoints, which can be deployed on TensorRT-LLM (Pytorch and C++ backends), [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).
- (Legacy) TensorRT-LLM checkpoints, a format that works with TensorRT-LLM C++ backend only.

The unified checkpoint<sup>1</sup> format design reflects two key characteristics: 1. The layer structures and tensor names remain aligned with the original Hugging Face checkpoint, and 2. The same checkpoint can be deployed across multiple inference frameworks without modification. A unified checkpoint can be exported using the following commands:

> *<sup>1.</sup>Unified checkpoint export currently does not support sparsity. Speculative decoding is only supported in unified checkpoint export. For legacy deployment, exported unified checkpoint then needs a TensorRT-LLM checkpoint converter (e.g., [this](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/eagle/convert_checkpoint.py)) to convert and build the TensorRT engine(s) for deployment. Alternatively, call TensorRT-LLM LLM-API to deploy the unified checkpoints e.g., check examples [here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/README.md).*

### API

```python
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir,  # The directory where the exported files will be stored.
    )
```

### Quantize and Export

```bash
python hf_ptq.py --pyt_ckpt_path <huggingface_model_card> --qformat fp8 --export_path <quantized_ckpt_path> --trust_remote_code
```

### Hugging Face framework [Script](./scripts/huggingface_example.sh)

Alternatively, the framework script `huggingface_example.sh` also supports quantize and export:

```bash
scripts/huggingface_example.sh --model <huggingface_model_card> --quant fp8
```

### Deployment

______________________________________________________________________

#### TRT-LLM

```python
from tensorrt_llm import LLM

llm_fp8 = LLM(model="<the exported model path>")
print(llm_fp8.generate(["What's the age of the earth? "]))
```

#### vLLM

```python
from vllm import LLM

llm_fp8 = LLM(model="<the exported model path>", quantization="modelopt")
print(llm_fp8.generate(["What's the age of the earth? "]))
```

#### SGLang

```python
import sglang as sgl

llm_fp8 = sgl.Engine(model_path="<the exported model path>", quantization="modelopt")
print(llm_fp8.generate(["What's the age of the earth? "]))
```

### Unified HF Checkpoint Deployment Model Support Matrix

| Model | Quant format | TRT-LLM | vLLM | SGLang |
| :---: | :---: | :---: | :---: | :---: |
| LLAMA 3.x | FP8 | ✅ | ✅ | ✅ |
| LLAMA 3.x | FP4 | ✅ | ✅ | ✅ |
| LLAMA 4 | FP8 | ✅ | - | ✅ |
| LLAMA 4 | FP4 | ✅ | - | - |
| DS-R1 | FP8 | ✅ | ✅ | ✅ |
| DS-R1 | FP4 | ✅ | ✅ | ✅ |
| DS-V3 | FP8 | ✅ | ✅ | ✅ |
| DS-V3 | FP4 | ✅ | ✅ | ✅ |
| QWen3 | FP8 | ✅ | ✅ | ✅ |
| QWen3 | FP4 | ✅ | ✅ | - |
| QWen3 MoE | FP8 | ✅ | ✅ | ✅ |
| QWen3 MoE | FP4 | ✅ | - | - |
| QWen2.5 | FP8 | ✅ | ✅ | ✅ |
| QWen2.5 | FP4 | ✅ | ✅ | - |
| QwQ-32B | FP8 | ✅ | ✅ | ✅ |
| QwQ-32B | FP4 | ✅ | ✅ | - |
| Mixtral 8x7B | FP8 | ✅ | ✅ | ✅ |
| Mixtral 8x7B | FP4 | ✅ | - | - |

### (Legacy) TensorRT-LLM Checkpoints

The user can specify the inference time TP and PP size and the export API will organize the weights to fit the target GPUs.

```python
from modelopt.torch.export import export_tensorrt_llm_checkpoint

with torch.inference_mode():
    export_tensorrt_llm_checkpoint(
        model,  # The quantized model.
        decoder_type,  # The type of the model, e.g gpt, gptj, or llama.
        dtype,  # The exported weights data type.
        export_dir,  # The directory where the exported files will be stored.
        inference_tensor_parallel,  # The number of GPUs used in the inference time tensor parallel.
        inference_pipeline_parallel,  # The number of GPUs used in the inference time pipeline parallel.
        use_nfs_workspace,  # If exporting in a multi-node setup, please specify a shared directory like NFS for cross-node communication.
    )
```

### Build the TensorRT-LLM engines

After the TensorRT-LLM checkpoint export, you can use the `trtllm-build` build command to build the engines from the exported checkpoints. Please check the [TensorRT-LLM Build API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/architecture/workflow.md#build-apis) documentation for reference.

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

### Technical Resources

There are many quantization schemes supported in the example scripts:

1. The [FP8 format](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) is available on the Hopper and Ada GPUs with [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) greater than or equal to 8.9.

1. The [INT8 SmoothQuant](https://arxiv.org/abs/2211.10438), developed by MIT HAN Lab and NVIDIA, is designed to reduce both the GPU memory footprint and inference latency of LLM inference.

1. The [INT4 AWQ](https://arxiv.org/abs/2306.00978) is an INT4 weight only quantization and calibration method. INT4 AWQ is particularly effective for low batch inference where inference latency is dominated by weight loading time rather than the computation time itself. For low batch inference, INT4 AWQ could give lower latency than FP8/INT8 and lower accuracy degradation than INT8.

1. The W4A8 AWQ is an extension of the INT4 AWQ quantization that it also uses FP8 for activation for more speed up and acceleration.

1. The [NVFP4](https://blogs.nvidia.com/blog/generative-ai-studio-ces-geforce-rtx-50-series/) is one of the new FP4 formats supported by NVIDIA Blackwell GPU and demonstrates good accuracy compared with other 4-bit alternatives. NVFP4 can be applied to both model weights as well as activations, providing the potential for both a significant increase in math throughput and reductions in memory footprint and memory bandwidth usage compared to the FP8 data format on Blackwell.

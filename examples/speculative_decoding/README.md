# Speculative Decoding

[![Documentation](https://img.shields.io/badge/Docs-NVIDIA--Model--Optimizer-blue?logo=readthedocs&style=flat-square)](https://nvidia.github.io/Model-Optimizer/guides/5_speculative_decoding.html)

Speculative decoding accelerates auto-regressive generation in large language models (LLMs) by leveraging a lightweight draft model to predict the next γ tokens. The main LLM then verifies these candidate tokens in a single forward pass. If the draft model correctly predicts α tokens, the LLM can accept and generate α+1 tokens per verification step, significantly improving generation speed.

This folder contains an end-to-end runnable speculative decoding fine‑tuning pipeline in which Llama‑3.2‑1B (Hugging Face) is trained on the Daring‑Anteater dataset.

This example focuses on training with Hugging Face. To train with Megatron‑LM, see the [Megatron‑LM example](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt).

## Contents

<div align="center">

| **Section** | **Description** | **Jump To** |
| :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional dependencies | \[[Link](#pre-requisites)\] |
| Simplified Workflow | Train, evaluate, and export EAGLE model with one-line command | \[[Link](#getting-started-simplified-workflow)\] |
| Online Training | Train draft model alongside base model in GPU memory | \[[Link](#training-draft-model-with-online-base-model)\] |
| Offline Training | Train draft model using pre-computed hidden states | \[[Link](#training-draft-model-with-offline-base-model)\] |
| After Training | Evaluation, export and deployment | \[[Link](#model-validation)\] |
| Custom Datasets | Other dataset options and custom data format | \[[Link](#custom-datasets)\] |
| Support Matrix | Supported models for speculative decoding training | \[[Link](#support-matrix)\] |
| Speculation Module Checkpoints | View pre-trained speculation modules ready to deploy! | \[[Link](#speculation-module-checkpoints)\] |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] |

</div>

## Pre-Requisites

### Docker

Please use the PyTorch docker image (e.g., `nvcr.io/nvidia/pytorch:25.08-py3`) or visit our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) for more information.

Also follow the installation steps below to upgrade to the latest version of Model Optimizer and install dataset and example-specific dependencies.

### Local Installation

Install Modelopt with `hf` dependencies and other requirements for this example:

```bash
pip install -U nvidia-modelopt[hf]
pip install -r requirements.txt
```

### Data Preparation

We use [Daring-Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater) dataset in this example. Prepare data by:

```bash
python prepare_input_conversations/add_daring_anteater.py
```

See the [Custom Datasets](#custom-datasets) section for other dataset options and instructions for user-provided data.

For higher acceptance rates, consider training on **model-generated conversations** — see [Best Practices: Data Synthesis](https://nvidia.github.io/Model-Optimizer/guides/5_speculative_decoding.html#data-synthesis) for step-by-step instructions.

## Getting Started: Simplified Workflow

```bash
bash train_eagle3_and_export.sh --base_model meta-llama/Llama-3.2-1B-Instruct
```

This one-line command runs a minimal example workflow of training and exporting an EAGLE draft model in Modelopt. Specifically, it

- Initializes the draft model with [default settings](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/speculative/eagle/default_config.py#L18)
- Fine-tunes the model on the [Daring-Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater) dataset
- Evaluates the acceptance rate on [MT-Bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts)
- Exports a checkpoint ready for deployment

## Training Draft Model with Online Base Model

For small base models that fit in GPU memory, we can collocate them with draft models and train with the following command:

```bash
./launch_train.sh --model $BASE_MODEL \
            --output_dir $OUTPUT_DIR \
            --data input_conversations/daring-anteater.jsonl  \
            --num_epochs $NUM_EPOCH \
            --eagle_config eagle_config.json
```

FSDP2 is used by default. To enable context parallelism for long-context training, specify `--cp_size n`.
The saved modelopt checkpoint is similar in architecture to HF models. It can be further optimized through **ModelOpt**, e.g., PTQ and QAT.

To customize the draft model architecture (number of layers, MLP size, etc.) or enable draft vocabulary compression, see [Best Practices: Configuring the Draft Model](https://nvidia.github.io/Model-Optimizer/guides/5_speculative_decoding.html#configuring-the-draft-model) and [Draft Vocabulary Compression](https://nvidia.github.io/Model-Optimizer/guides/5_speculative_decoding.html#draft-vocabulary-compression).

## Training Draft Model with Offline Base Model

For large models, you can export intermediate hidden states to disk and train only the draft model. This significantly reduces GPU memory requirements, but requires several to tens of terabytes of disk storage depending on dataset size.

### Dumpping Hidden States to Disk

We support two backends for generating base model hidden states. For better effciency, it is recommended to use TRT-LLM:

```bash
python collect_hidden_states/compute_hidden_states_trtllm.py \
            --model $BASE_MODEL \
            --input-file input_conversations/daring-anteater.jsonl \
            --output-dir $HIDDEN_STATES_DIR
```

**NOTE**: TRT-LLM installation needed for the above command.

Alternatively, you can generate the same hidden states with HF:

```bash
python collect_hidden_states/compute_hidden_states_hf.py \
            --model $BASE_MODEL \
            --input-file input_conversations/daring-anteater.jsonl  \
            --output-dir $HIDDEN_STATES_DIR
```

**NOTE**: See [`run_hf_compute_hiddens_dp.sh`](./collect_hidden_states/run_hf_compute_hiddens_dp.sh) and [`run_trtllm_compute_hiddens_dp.sh`](./collect_hidden_states/run_trtllm_compute_hiddens_dp.sh) for a simple example using data parallelism (DP) to accelerate hidden state generation.

### Train Draft Model with Dumped Hidden States

Once we finish dumping hidden states, launch offline training with an extra `--offline-data` argument:

```bash
./launch_train.sh --model $BASE_MODEL \
            --output_dir $OUTPUT_DIR \
            --data $DATA \
            --num_epochs $NUM_EPOCH \
            --eagle_config eagle_config.json \
            --offline-data $HIDDEN_STATES_DIR
```

## Model Validation

For online training checkpoints, we can run in-framework evaluation on MT-bench:

```bash
python scripts/ar_validate.py --model_path $ONLINE_CKPT
```

**Note**: In-framework evaluation is supported only for online training. For offline training checkpoints, please export the model and evaluate it using serving frameworks.

## Export

```bash
python scripts/export_hf_checkpoint.py --model_path $OUTPUT_DIR --export_path $EXPORT_PATH
```

This exports the model from a ModelOpt checkpoint to a deployment-compatible format.

## Deployment

The exported checkpoint can be deployed on TRT-LLM or SGLang.

### TRT-LLM

To serve the checkpoint with TRT-LLM, run trtllm-serve with:

```bash
trtllm-serve <base_model_checkpoint> --host 0.0.0.0 --port 8000 --backend pytorch --max_batch_size 32 --max_num_tokens 8192 --max_seq_len 8192 --extra_llm_api_options extra-llm-api-config.yml
```

, with `extra-llm-api-config.yml` being

```yaml
enable_attention_dp: false
disable_overlap_scheduler: true
enable_autotuner: false

cuda_graph_config:
    max_batch_size: 1

speculative_config:
    decoding_type: Eagle
    max_draft_len: 3
    speculative_model_dir: <draft_model_checkpoint>

kv_cache_config:
    enable_block_reuse: false
```

Please refer to [TRT-LLM Doc: Speculative Decoding](https://nvidia.github.io/TensorRT-LLM/examples/llm_speculative_decoding.html) for detailed usage.

### vLLM

Please refer to [VLLM Doc: Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) for detailed usage.

Optionally, you can convert the exported checkpoint to contain target model information, which is accepted by vLLM to simplify depployment:

```bash
python scripts/convert_to_vllm_ckpt.py --input <exported_ckpt> --verifier <target_model> --output <output_dir>
```

### SGLang

Please refer to [SGLang Doc: Speculative Decoding](https://docs.sglang.ai/advanced_features/speculative_decoding.html#EAGLE-3-Decoding) for detailed usage.

### SpecDec Bench

One can also use [examples/specdec_bench](../specdec_bench) to validate the trained Eagle3 checkpoints in a variety of frameworks (vLLM, SGLang, TRT-LLM) on a set of datasets.

### Deploying Quantized model

See more details on deployment of quantized model to TRTLLM [here](../llm_ptq/README.md).

## Custom Datasets

In addition to `daring-anteater`, we provide scripts for adding several other commonly used datasets in `prepare_input_conversations`:

```text
prepare_input_conversations/
    ├── add_daring_anteater.py
    ├── add_mtbench.py
    ├── add_sharegpt.py
    ├── add_ultrachat.py
    └── example_make_prompt_dataset.sh
```

To use your own datasets, please preprocess your data into a `.jsonl` file with each line in the format:

```json
{
    "conversation_id": <unique id>,
    "conversations": [{"role":<user or assistant>, "content":<content>}]
}
```

## Support Matrix

| Model | Medusa | EAGLE1/2 | EAGLE3 |
| :---: | :---: | :---: | :---: |
| LLAMA 2 | ✅ | ✅ | ✅ |
| LLAMA 3, 3.1 | ✅ | ✅ | ✅ |
| Mistral | ✅ | ✅ | ✅ |
| Phi 3 | ✅ | ✅ | ✅ |
| QWen 1.5,2,2.5,3 | ✅ | ✅ | ✅ |

## Speculation Module Checkpoints

Ready-to-deploy speculation module checkpoints \[[🤗 Hugging Face - NVIDIA Speculative Decoding Modules Collection](https://huggingface.co/collections/nvidia/speculative-decoding-modules)\]
Deployable on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [SGLang](https://github.com/sgl-project/sglang)!\
More models coming soon!

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

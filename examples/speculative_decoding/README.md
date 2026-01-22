# Speculative Decoding

[![Documentation](https://img.shields.io/badge/Docs-NVIDIA--Model--Optimizer-blue?logo=readthedocs&style=flat-square)](https://nvidia.github.io/Model-Optimizer/guides/5_speculative_decoding.html)

Speculative decoding accelerates auto-regressive generation in large language models (LLMs) by leveraging a lightweight draft model to predict the next γ tokens. The main LLM then verifies these candidate tokens in a single forward pass. If the draft model correctly predicts α tokens, the LLM can accept and generate α+1 tokens per verification step, significantly improving generation speed.

This folder contains an end-to-end runnable speculative decoding fine‑tuning pipeline in which Llama‑3.2‑1B (Hugging Face) is trained on the [UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset.

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
| Advanced Usage | Data synthesis, vocab compression, and configuration | \[[Link](#advanced-usage)\] |
| Support Matrix | Supported models for speculative decoding training | \[[Link](#support-matrix)\] |
| Speculation Module Checkpoints | View pre-trained speculation modules ready to deploy! | \[[Link](#speculation-module-checkpoints)\] |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] |

</div>

## Pre-Requisites

### Docker

Please use the PyTorch docker image (e.g., `nvcr.io/nvidia/pytorch:25.06-py3`) or visit our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) for more information.

Also follow the installation steps below to upgrade to the latest version of Model Optimizer and install dataset and example-specific dependencies.

### Local Installation

Install Modelopt with `hf` dependencies and other requirements for this example:

```bash
pip install -U nvidia-modelopt[hf]
pip install -r requirements.txt
```

### Data Preparation

We support a range of input datasets. In this example, we will use the [UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset.

```bash
python prepare_input_conversations/make_dataset.py -f prepare_input_conversations/example_data_config.yaml --full-conversations
```

See [other-datasets](#other-datasets) section for other dataset options and instruction for user-provided data.

Omit `--full-conversations` if you plan to run synthetic data generation (see [data-synthesis](#data-synthesis)).

## Getting Started: Simplified Workflow

```bash
bash train_eagle3_and_export.sh --base_model meta-llama/Llama-3.2-1B-Instruct --num_gpu 4
```

This one-line command runs a minimal example workflow of training and exporting an EAGLE draft model in Modelopt. Specifically, it

- Initializes the draft model with [default settings](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/speculative/eagle/default_config.py#L18)
- Fine-tunes the model on the dataset
- Evaluates the acceptance rate on [MT-Bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts)
- Exports a checkpoint ready for deployment

## Training Draft Model with Online Base Model

For small base models that fit in GPU memory, we can collocate them with draft models and train with the following command:

```bash
./launch_train.sh --model $BASE_MODEL \
            --output_dir $OUTPUT_DIR \
            --data input_conversations/train.jsonl  \
            --num_gpu $NUM_GPU \
            --num_epochs $NUM_EPOCH \
            --eagle_config eagle_config.json
```

This command will launch `main.py` with `accelerate`. See [section: interact with modelopt.torch.speculative](#interact-with-modelopttorchspeculative) for more details.
The saved modelopt checkpoint is similar in architecture to HF models. It can be further optimized through **ModelOpt**, e.g., PTQ and QAT.

## Training Draft Model with Offline Base Model

For large models, you can export intermediate hidden states to disk and train only the draft model. This significantly reduces GPU memory requirements, but requires several to tens of terabytes of disk storage depending on dataset size.

### Dumpping Hidden States to Disk

We support two backends for generating base model hidden states. For better effciency, it is recommended to use TRT-LLM:

```bash
python collect_hidden_states/compute_hidden_states_trtllm.py \
            --model $BASE_MODEL \
            --input-file input_conversations/train.jsonl \
            --output-dir $HIDDEN_STATES_DIR
```

**NOTE**: TRT-LLM installation needed for the above command.

Alternatively, you can generate the same hidden states with HF:

```bash
python collect_hidden_states/compute_hidden_states_hf.py \
            --model $BASE_MODEL \
            --input-file input_conversations/train.jsonl  \
            --output-dir $HIDDEN_STATES_DIR
```

**NOTE**: See [`run_hf_compute_hiddens_dp.sh`](./collect_hidden_states/run_hf_compute_hiddens_dp.sh) and [`run_trtllm_compute_hiddens_dp.sh`](./collect_hidden_states/run_trtllm_compute_hiddens_dp.sh) for a simple example using data parallelism (DP) to accelerate hidden state generation.

### Train Draft Model with Dumped Hidden States

Once we finish dumping hidden states, launch offline training with an extra `--offline-data` argument:

```bash
./launch_train.sh --model $BASE_MODEL \
            --output_dir $OUTPUT_DIR \
            --data $DATA \
            --num_gpu $NUM_GPU \
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

## Advanced Usage

### Other Datasets

In addition to the default dataset, we support adding several other commonly used datasets in `prepare_input_conversations/make_dataset.py`:

- MTBench (for debugging)
- ShareGPT
- UltraChat
- Daring-Anteater
- Magpie (Full 1M, and 500k and 300k filtered)
- Nemotron Post-Training Dataset V2

To use your own datasets, please preprocess your data into a `.jsonl` file with each line in the format:

```json
{
    "conversation_id": <unique id>,
    "conversations": [{"role":<user or assistant>, "content":<content>}]
}
```

### Data Synthesis

To achieve higher acceptance rates during speculative decoding, it is beneficial to use conversations generated by the base model as training data. This ensures that the draft model's output distribution closely aligns with that of the base model.

To prepare such data, we launch an inference server with the base model:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.2-1B-Instruct --api-key token-abc123 --port 8000  --tensor-parallel-size 1
```

Note: Add `--quantization=modelopt` flag for quantized models.

Then, we generate conversations with the base model using the prepared prompts:

```bash
python scripts/server_generate.py --data_path input_conversations/train.jsonl --output_path synthetic/train.jsonl
```

To add a system prompt, use the `--system_prompt <system_prompt_text>` argument.

For large scale data generation, please see [SLURM prepare data](SLURM_prepare_data.md) for SLURM support.

### Draft Vocabulary Compression

We can optionally use smaller vocab size for the draft model for faster training and inference. E.g. Llama3.2-1B has a vocab size of 128256. In this example, we construct a draft vocab mapping of size 32k by finding the most commonly appeared vocabs in our training set:

```bash
python scripts/calibrate_draft_vocab.py --model meta-llama/Llama-3.2-1B-Instruct --data input_conversations/train.jsonl --draft_vocab_size 32000 --save_dir draft_vocab_cache
```

This will produce a `d2t.pt` file in `save_dir`, which is the mapping from draft token to target token. During inference, draft tokens can be mapped back to target tokens by `target_token = draft_token + d2t[draft_token]`.

### Configuring Draft Model

For EAGLE‑1 and EAGLE‑3 we provide a [default model architecture config](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/speculative/config.py#L37) in ModelOpt. You can override default settings by providing an additional JSON dict. In this example, we override `draft_vocab_size` in `eagle_config.json`:

```json
{
    "draft_vocab_size": 32000
}
```

### Interact with `modelopt.torch.speculative`

`main.py` provides an example for converting a HF base model for speculative decoding and training it. It consists of a few simple steps:
First, load the base model and tokenizer from Hugging Face:

```python
model = transformers.AutoModelForCausalLM.from_pretrained(
    "<path to your pretrained model>"
)
```

Then, load default eagle config and make necessary overwrites:

```python
# Load default config
config = {
    "eagle1": EAGLE1_DEFAULT_CFG,
    "eagle3": EAGLE3_DEFAULT_CFG,
}[training_args.mode]["config"]

# overwrite config with custom config
config["eagle_architecture_config"].update({"<overwrite_keys>": "<overwrite_values>"})

# Mandatory: hidden size, vocab size and max position embeddings must match base model
config["eagle_architecture_config"].update(
    {
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
    }
)
```

Then, we convert model to a speculative decoding model:

```python
mtsp.convert(model, [("eagle", config)])
```

This will modify the model in-place with eagle training forward, making it compatible with HF trainer:

```python
# Create a trainer
trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
trainer._move_model_to_device(model, trainer.args.device)

# Enable HF checkpointing so that the saved model will contain the speculative decoding module
mto.enable_huggingface_checkpointing()

trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_state()
trainer.save_model("<path to the output directory>")
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

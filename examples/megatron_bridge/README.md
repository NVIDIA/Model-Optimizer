# Megatron Bridge

This directory contains examples of using Model Optimizer with [NeMo Megatron-Bridge](https://github.com/NVIDIA-Nemo/Megatron-Bridge) framework for pruning, distillation, quantization, etc.

<div align="center">

| **Section** | **Description** | **Link** |
| :------------: | :------------: | :------------: |
| Pre-Requisites | Development environment setup | \[[Link](#pre-requisites)\] |
| Pruning | Examples of pruning a model using Minitron algorithm | \[[Link](#pruning)\] |
| Distillation | Examples of distillation a pruned or quantized model | \[[Link](#distillation)\] |
| Quantization | Examples of quantizing a model | \[[Link](#quantization)\] |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] |

</div>

## Pre-Requisites

Running these examples requires many additional dependencies to be installed (e.g., Megatron-Bridge, Megatron-core, etc.), hence we strongly recommend directly using the NeMo container (e.g., `nvcr.io/nvidia/nemo:26.02`) which has all the dependencies installed.

To get the latest ModelOpt features and examples scripts, mount your Model-Optimizer repo to the container.

```bash
export MODELOPT_DIR=${PWD}/Model-Optimizer # or set to your local Model-Optimizer repository path if you have cloned it
if [ ! -d "${MODELOPT_DIR}" ]; then
  git clone https://github.com/NVIDIA/Model-Optimizer.git ${MODELOPT_DIR}
fi

export DOCKER_IMAGE=nvcr.io/nvidia/nemo:26.02
docker run \
  --gpus all \
  --shm-size=16GB \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v ${MODELOPT_DIR}:/opt/Model-Optimizer \
  -v ${MODELOPT_DIR}/modelopt:/opt/venv/lib/python3.12/site-packages/modelopt \
  -w /opt/Model-Optimizer/examples/megatron_bridge \
  ${DOCKER_IMAGE} bash
```

Once inside the container, you need to login with your HuggingFace token to download gated datasets / models.
Note that the default dataset for pruning and quantization is [`nemotron-post-training-dataset-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2), which is gated.

```bash
hf auth login --token <your token>
```

## Pruning

This section shows how to prune a HuggingFace model using Minitron algorithm in Megatron-Bridge framework. Checkout other available pruning algorithms, supported frameworks and models, and general pruning getting-started in the [pruning README](../pruning/README.md).

Example usage to prune Qwen3-8B to 6B on 2-GPUs (Pipeline Parallelism = 2) while skipping pruning of `num_attention_heads` using following defaults:
    1024 samples from [`nemotron-post-training-dataset-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) for calibration,
    at-most 20% depth (`num_layers`) and 40% width is pruned per prunable hparam (`hidden_size`, `ffn_hidden_size`, ...),
    top-10 candidates are evaluated for MMLU score (5% sampled data) to select the best model.

```bash
torchrun --nproc_per_node 2 prune_minitron.py \
    --pp_size 2 \
    --hf_model_name_or_path Qwen/Qwen3-8B \
    --prune_target_params 6e9 \
    --hparams_to_skip num_attention_heads \
    --output_hf_path /tmp/Qwen3-8B-Pruned-6B
```

Example usage for manually pruning to a specific architecture using following defaults:
    1024 samples from [`nemotron-post-training-dataset-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) for calibration.

```bash
torchrun --nproc_per_node 2 prune_minitron.py \
    --pp_size 2 \
    --hf_model_name_or_path Qwen/Qwen3-8B \
    --prune_export_config '{"hidden_size": 3584, "ffn_hidden_size": 9216}' \
    --output_hf_path /tmp/Qwen3-8B-Pruned-6B-manual
```

To see the full usage for advanced configurations, run:

```bash
torchrun --nproc_per_node 1 prune_minitron.py --help
```

> [!TIP]
> If number of layers in the model is not divisible by number of GPUs i.e. pipeline parallel (PP) size, you can configure
> uneven PP by setting `--num_layers_in_first_pipeline_stage` and `--num_layers_in_last_pipeline_stage`.
> E.g. for Qwen3-8B with 36 layers and 8 GPUs, you can set both to 3 to get 3-5-5-5-5-5-5-3 layers per GPU.

## Distillation

This section shows how to distill a student model from a teacher model in the Megatron-Bridge framework.

This can be used stand-alone or after pruning (see [Pruning](#pruning)) / quantization (see [Quantization](#quantization)) to recover accuracy of the model by distilling from the original model (teacher).

The [distill.py](distill.py) script loads student and teacher models from HuggingFace checkpoints and saves the distilled model to `<output_dir>/checkpoints` in Megatron distributed checkpoint format.

### Data Preparation

The distillation script expects pre-tokenized data in Megatron's binary format (`.bin` / `.idx` files).

You can tokenize your JSONL datasets using the following command:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths /path/to/data1.jsonl /path/to/data2.jsonl ... \
    --json_keys text \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir /path/to/tokenized/data/qwen3 \
    --workers 32 \
    --max_sequence_length 256_000
```

Instead of `--jsonl_paths`, you can also pass a directory path to the `--input_dir` argument to tokenize all JSONL files in the directory.
We are setting a maximum sequence length of 256k to avoid rare OOM errors in tokenization if text is too long.

If you want to download and tokenize a dataset from Hugging Face Hub directly, you can use the following command:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-Pretraining-SFT-v1 \
    --hf_name Nemotron-SFT-General \
    --hf_split train \
    --hf_max_samples_per_split 10_000_000 \
    --json_keys text \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir /path/to/tokenized/data/qwen3 \
    --workers 32 \
    --max_sequence_length 256_000
```

The [Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1) dataset is huge, so it will take a while to download and tokenize. You can also split the large `.jsonl` into multiple files (e.g. 10M samples per file using `split -l 10000000 -d --additional-suffix=.jsonl <file>.jsonl <file>_part`) and tokenize them parallelly.
To quickly test the script, you can try the [nvidia/Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample) dataset.

If you skip `--hf_name`, it will download and tokenize all subsets for the dataset.
If you skip `--hf_split`, it will download and tokenize all splits for the subset.
If you skip `--hf_max_samples_per_split`, it will download and tokenize all samples for the split.

### Distillation with Real Data

Example usage to distill a 4B student (HF) from an 8B teacher (HF) on 8 GPUs (TP=8, PP=1):

```bash
torchrun --nnodes 1 --nproc_per_node 8 distill.py \
    --tp_size 8 \
    --teacher_hf_path Qwen/Qwen3-8B \
    --student_hf_path Qwen/Qwen3-4B \
    --data_paths 1.0 /path/to/tokenized/data/qwen3 \
    --data_path_to_cache /path/to/cache/dataset_indices_qwen3 \
    --seq_length 8192 \
    --mbs 1 \
    --gbs 768 \
    --train_iters 15000 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_warmup_iters 50 \
    --eval_interval 100 \
    --eval_iters 32 \
    --log_interval 10 \
    --output_dir /output/qwen3_8b_to_4b_distill
```

Tensorboard logging is enabled by default and logs are saved to `<output_dir>/tensorboard` directory.
To use Weights & Biases for logging, set the `WANDB_API_KEY` environment variable and pass the `--wandb_project` argument.
Optionally, you can also pass `--wandb_entity` and `--wandb_exp_name` arguments to group runs under a project and experiment name.

To see all available arguments:

```bash
torchrun --nproc_per_node 1 distill.py --help
```

### Quick Test with Mock Data

Example usage with mock data for quick testing (no pre-tokenized data needed):

```bash
torchrun --nproc_per_node 8 distill.py \
    --tp_size 8 \
    --teacher_hf_path Qwen/Qwen3-0.6B \
    --student_hf_path Qwen/Qwen3-0.6B \
    --use_mock_data \
    --seq_length 512 \
    --mbs 1 \
    --gbs 8 \
    --train_iters 100 \
    --eval_interval 10 \
    --eval_iters 4 \
    --output_dir /tmp/test_distill
```

### Slurm Usage

To run the distillation script on a Slurm cluster for multi-node training, you just need use `python` instead of `torchrun` and set the number of nodes using `#SBATCH --nodes=<num_nodes>` clause in your Slurm script.

### Convert Megatron checkpoint to Hugging Face format

To convert the Megatron checkpoint from last iteration (or any intermediate iteration) to Hugging Face format, you need the pruned model config (`--output_hf_path` from `prune_minitron.py` script) and the distilled megatron checkpoint dir (`<distill_output_dir>/checkpoints/iter_<iter_number>`) to run the following command:

```bash
uv run python /opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py export \
    --hf-model <path_to_pruned_hf_ckpt> \
    --megatron-path <distill_output_dir>/checkpoints/iter_<iter_number> \
    --hf-path <path_to_save_distilled_hf_ckpt>
```

For more details, you can refer to the checkpoint conversion scripts in the [Megatron-Bridge README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/conversion).

## Quantization

TODO

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

# Megatron Bridge

This directory contains examples of using Model Optimizer with [NeMo Megatron-Bridge](https://github.com/NVIDIA-Nemo/Megatron-Bridge) framework for pruning, distillation, quantization, etc.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Development environment setup | \[[Link](#pre-requisites)\] | |
| Pruning | Examples of pruning a model using Minitron algorithm | \[[Link](#pruning)\] | |
| Distillation | Examples of distillation a pruned or quantized model | \[[Link](#distillation)\] | |
| Quantization | Examples of quantizing a model | \[[Link](#quantization)\] | |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

Running these examples requires many additional dependencies to be installed (e.g., Megatron-Bridge, Megatron-core, etc.), hence we strongly recommend directly using the NeMo container (e.g., `nvcr.io/nvidia/nemo:26.02`) which has all the dependencies installed.

To get the latest ModelOpt features and examples, you can mount your latest ModelOpt cloned repository to the container at `/opt/Megatron-Bridge/3rdparty/Model-Optimizer` or pull the latest changes once inside the docker container (`cd /opt/Megatron-Bridge/3rdparty/Model-Optimizer && git checkout main && git pull`).

## Pruning

This section shows how to prune a HuggingFace model using Minitron algorithm in Megatron-Bridge framework. Checkout other available pruning algorithms, supported frameworks and models, and general pruning getting-started in the [pruning README](../pruning/README.md).

Example usage to prune Qwen3-8B to 6B on 2-GPUs (Pipeline Parallelism = 2) while skipping pruning of `num_attention_heads` using following defaults:
    1024 samples from [`nemotron-post-training-dataset-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) for calibration,
    at-most 20% depth (`num_layers`) and 40% width is pruned per prunable hparam (`hidden_size`, `ffn_hidden_size`, ...),
    top-10 candidates are evaluated for MMLU score (5% sampled data) to select the best model.

```bash
torchrun --nproc_per_node 2 /opt/Megatron-Bridge/3rdparty/Model-Optimizer/examples/megatron_bridge/prune_minitron.py \
    --hf_model_name_or_path Qwen/Qwen3-8B \
    --prune_target_params 6e9 \
    --hparams_to_skip num_attention_heads \
    --output_hf_path /tmp/Qwen3-8B-Pruned-6B
```

Example usage for manually pruning to a specific architecture using following defaults:
    1024 samples from [`nemotron-post-training-dataset-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) for calibration.

```bash
torchrun --nproc_per_node 2 /opt/Megatron-Bridge/3rdparty/Model-Optimizer/examples/megatron_bridge/prune_minitron.py \
    --hf_model_name_or_path Qwen/Qwen3-8B \
    --prune_export_config '{"hidden_size": 3584, "ffn_hidden_size": 9216}' \
    --output_hf_path /tmp/Qwen3-8B-Pruned-6B-manual
```

To see the full usage for advanced configurations, run:

```bash
python /opt/Megatron-Bridge/3rdparty/Model-Optimizer/examples/megatron_bridge/prune_minitron.py --help
```

> [!TIP]
> If number of layers in the model is not divisible by number of GPUs i.e. pipeline parallel (PP) size, you can configure
> uneven PP by setting `--num_layers_in_first_pipeline_stage` and `--num_layers_in_last_pipeline_stage`.
> E.g. for Qwen3-8B with 36 layers and 8 GPUs, you can set both to 3 to get 3-5-5-5-5-5-5-3 layers per GPU.

## Distillation

TODO

## Quantization

TODO

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

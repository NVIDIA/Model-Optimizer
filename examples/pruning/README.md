# Pruning

Model pruning is a technique that removes redundant or less important parameters/connections from a neural network to reduce complexity and improve efficiency while maintaining performance.

Pruning can involve removal (prune) of Linear and Conv layers; and Transformer attention, MLP, MoE, Mamba, and depth of the model.

This section focuses on applying Model Optimizer's state-of-the-art complementary pruning modes to enable you to search for the best subnet architecture from your provided base model:

1. [Minitron](https://arxiv.org/pdf/2408.11796): A pruning method developed by NVIDIA Research for pruning GPT (and later extended to Mamba, MoE, and Hybrid Transformer Mamba) models in NVIDIA Megatron-LM (M-LM) or Megatron-Bridge (M-Bridge) framework. It uses the activation magnitudes to prune the embedding hidden size; mlp ffn hidden size; transformer attention heads; mamba heads and head dimension; MoE number of experts, ffn hidden size, and shared expert intermediate size; and number of layers of the model.
1. FastNAS: A pruning method recommended for Computer Vision models. Given a pretrained model, FastNAS finds the subnet which maximizes the score function while meeting the given constraints.
1. GradNAS: A light-weight pruning method recommended for language models like Hugging Face BERT, GPT-J. It uses the gradient information to prune the model's linear layers and attention heads to meet the given constraints.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | \[[Link](#pre-requisites)\] | |
| Getting Started | Learn how to use the pruning API | \[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/3_pruning.html)\] |
| Support Matrix | View the support matrix to see available pruning algorithms and their compatibility with different models and frameworks | \[[Link](#support-matrix)\] | |
| Examples | Examples of different pruning methods | \[[Link](#examples)\] | |
| Pruning Guidelines | Guidelines for choosing how and how much to prune for best results | \[[Link](#pruning-guidelines)\] | |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

For Minitron pruning for Megatron-LM / Megatron-Bridge models, use the NeMo container (e.g., `nvcr.io/nvidia/nemo:26.02`) which has all the dependencies installed.

For FastNAS pruning for PyTorch Computer Vision models, no additional dependencies are required.

For GradNAS pruning for Hugging Face BERT / GPT-J, no additional dependencies are required.

## Getting Started

As part of the pruning process, you will need to set up the training and/or validation data loaders, and optionally define a validation score function (Minitron, FastNAS) or loss function (GradNAS) and specify the desired pruning constraints (See [Support Matrix](#support-matrix) for available pruning constraints).

To prune your model, you can simply call the `mtp.prune` API and save the pruned model. If the model is pruned using Minitron, you can use your standard saving and loading functions since it is a homogeneous pruning; while for FastNAS or GradNAS, you need to use `mto.save` and `mto.restore` to save and restore the heterogeneous pruned model.

### Minitron

Minitron pruning supports two types:

1. **Manual Pruning**: Manually specify the target dimensions for each pruning axis (e.g., `constraints = {"export_config": {"hidden_size": 3072, "ffn_hidden_size": 9216}}`)
2. **NAS-based Auto Pruning (New)**: Specify a target parameter count (e.g., `constraints = {"params": 6e9}`) and let the algorithm automatically search for the best architecture that maximizes a user-defined score function (e.g. MMLU, negative validation loss, etc.)

Please see example snippets of both modes for Minitron pruning on Megatron-Bridge Qwen3-8B model below. For end-to-end examples script (M-LM / M-Bridge framework), please refer to the examples below.

#### Common Setup

```python
import torch
import modelopt.torch.prune as mtp
from modelopt.torch.utils.plugins.mbridge import (
    get_hf_mbridge_calibration_loop,
    load_mbridge_model_from_hf,
)

# Import the Megatron-Bridge Qwen3-8B model from Hugging Face checkpoint
bridge, provider, model, unwrapped_model, tokenizer = load_mbridge_model_from_hf(
    hf_model_name_or_path="Qwen/Qwen3-8B",
    provider_overrides={
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
        "seq_length": 4096,
    },
)

# Set up the forward loop to run on 1024 train samples
forward_loop = get_hf_mbridge_calibration_loop(
    model=model,
    provider=provider,
    tokenizer=tokenizer,
    hf_model_name_or_path="Qwen/Qwen3-8B",
    dataset_name="nemotron-post-training-dataset-v2",
    num_samples=1024,
)

# Run pruning on the unwrapped model
mtp.prune(  # in-place pruning
    unwrapped_model,
    mode="mcore_minitron",
    constraints=constraints,  # Shown below for both types
    dummy_input=None,  # Not used
    config=config,  # Shown below for both types
)
```

> [!Note]
> Fine-tuning / distillation is required after pruning to recover the accuracy. Please refer to [end-to-end pruning and distillation tutorial](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/qwen/pruning-distillation) for more details.

#### 1. Manual Pruning

This mode can be useful when you know the exact dimensions you want to prune to (e.g. fitting a specific latency / memory budget). Alternatively, you can also use this mode to export top-K architectures (searched using NAS-based auto pruning) and perform short Knowledge Distillation on them before selecting the best architecture.

```python
# Specify the pruning constraints (Check Support Matrix for available pruning dimensions)
# Save minitron scores at checkpoint so we can re-run pruning with different constraints without running the forward loop again
constraints = {"export_config": {"num_layers": 32, "hidden_size": 3584, "ffn_hidden_size": 10240}}
config = {"forward_loop": forward_loop, "checkpoint": "/path/to/cache/pruning/scores.pth"}

mtp.prune(...)
```

**Under the Hood:**

1. **Importance Scoring**: Runs forward passes on calibration data (512-1024 samples) to compute activation magnitudes for each neuron/head/layer (takes ~5 minutes for an 8B model)
2. **Ranking**: Ranks all parameters within each pruning dimension (e.g., all hidden dimensions, all attention heads) by their importance scores
3. **Pruning**: Removes the least important parameters to meet the specified target dimensions in `export_config`
4. **Weight Slicing**: Slices the model weights according to the pruned architecture (homogeneous pruning - all layers pruned uniformly)

> [!TIP]
> Checkout the [Pruning Guidelines](#pruning-guidelines) section for more details on how to choose the best pruning strategy and distillation hyperparameters.

#### 2. NAS-based Auto Pruning

This mode can be useful when you don't know the exact dimensions you want to prune to and want the algorithm to search for the best architecture that maximizes a user-defined score function at the cost of longer runtime.

```python
# Define the score function to maximize (e.g., MMLU, negative validation loss, etc.)
# The algorithm will search for the best architecture that maximizes this score
from modelopt.torch.utils.plugins.megatron_mmlu import megatron_mmlu

def score_func(m):
    return megatron_mmlu(m, tokenizer, percentage=0.05)  # 5% sampled data for faster eval

# Specify target parameter count and configure the auto pruning algorithm
# Save minitron scores at checkpoint so we can resume pruning without running the forward loop again
constraints = {"params": 6e9}  # Prune to 6B parameters
config = {
    "forward_loop": forward_loop,
    "checkpoint": "/path/to/cache/pruning/scores.pth",
    "score_func": score_func,
    # Optional: Configure search space constraints (showing defaults)
    "max_width_pruning": 0.4,  # Maximum 40% per width pruning hparams (hidden_size, ffn_hidden_size, etc.)
    "max_depth_pruning": 0.2,  # Maximum 20% per depth pruning hparam (num_layers)
    "hparams_to_skip": [],  # Disable pruning specific hparams, e.g., ["num_attention_heads"]
    "top_k": 10,  # Number of top architectures to evaluate (using 20 may result in better pruned model at the cost of 2x time)
}

mtp.prune(...)
```

**Under the Hood:**

1. **Importance Scoring**: Same as manual pruning - computes activation magnitudes for all parameters (takes ~5 minutes for an 8B model)
2. **Search Space Construction**: Generates a search space of possible architectures based search space config and other configs (`max_width_pruning`, `max_depth_pruning`, `hparams_to_skip`)
3. **Architecture Search**: Find candidate architectures that meet the parameter constraint and evaluate `top_k` (based on number of parameters) of them using `score_func` e.g. MMLU, negative validation loss, etc. (takes ~10 mins per candidate for an 8B model pruning)
4. **Best Architecture Selection**: Returns the architecture (best `export_config`) with the highest actual score from the top-K evaluated architectures
5. **Weight Slicing**: Slices the model weights according to the best pruned architecture found

> [!Note]
> As per the [original paper](https://arxiv.org/pdf/2407.14679), ideally we need to perform a short Knowledge Distillation on ~2B tokens for all top-K candidate architectures before evaluating the score function, which will take a lot longer to prune, require splitting the pruning process into multiple stages and a lot more compute for pruning but can lead to better pruned model. If you are interested to do this, you can take the top-K candidate's `export_config` from the pruning logs and then export all models separately and perform Knowledge Distillation on each of them before evaluating the score function.

#### Advanced Configuration

For finer control over the search space (e.g., granularity of pruning choices), you can configure the divisors:

```python
# Configure search space granularity (showing defaults)
ss_config = mtp.mcore_minitron.get_mcore_minitron_config(
    hidden_size_divisor=256,
    ffn_hidden_size_divisor=512,
    mamba_head_dim_divisor=8,
    num_moe_experts_divisor=8,
    num_layers_divisor=2,
)

# Use the custom search space config
mtp.prune(unwrapped_model, mode=[("mcore_minitron", ss_config)], ...)
```

If your model parameters are already sorted and you just want to prune the weights, you can skip the sorting step by setting `"skip_sorting": True` in `config` instead of passing `forward_loop`.

## Support Matrix

| **Algorithm** | **Model** | **Pruning Constraints** |
| :---: | :---: | :---: |
| Minitron | Megatron-core (M-LM, M-Bridge) based GPT / Mamba / MoE / Hybrid LLM Models<sup>1</sup> | **Manual:** `export_config` with width (`hidden_size`, `ffn_hidden_size`, `num_attention_heads`, `mamba_num_heads`, `mamba_head_dim`, `num_moe_experts`, `moe_ffn_hidden_size`, `moe_shared_expert_intermediate_size`) and/or depth (`num_layers`) pruned values<br>**Auto:** `params` (requires `score_func` in config) |
| FastNAS | Computer Vision models | `flops`, `params` |
| GradNAS | HuggingFace BERT, GPT-J | `flops`, `params` |

> *<sup>1.</sup>Only models in Pipeline Parallelism (PP) are supported. Hugging Face models can be imported into M-Bridge/M-LM format as long as they are [supported](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html#supported-models) by the framework.*

## Examples

### Minitron Pruning for Megatron-Bridge/ Megatron-LM Framework LLMs (e.g. Qwen 3, Nemotron Nano)

Checkout the Minitron pruning example for [Megatron-Bridge Framework](../megatron_bridge/README.md#pruning) or [Megatron-LM Framework](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt#-pruning) which showcases the usage of the powerful Minitron pruning algorithm developed by NVIDIA Research for pruning LLMs like Llama-3.1-8B, Qwen3-8B, Nemotron-Nano-9B-v2, Nemotron-3-Nano-30B-A3B, etc.
Both frameworks support importing from a Hugging Face pretrained checkpoint.

\[Deprecated\] You can also look at the NeMo tutorial notebooks [here](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/qwen/pruning-distillation) which showcase the usage of Minitron pruning followed by distillation for Qwen3-8B step-by-step in NeMo framework. Hugging Face models can also be converted to NeMo format and used subsequently as shown in the tutorial.

Some of the models pruned using Minitron method followed by distillation and post-training are:

- [Minitron Collection on Hugging Face](https://huggingface.co/collections/nvidia/minitron)
- [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)

### FastNAS Pruning for PyTorch Computer Vision Models

Check out the FastNAS pruning example usage in the [documentation](https://nvidia.github.io/Model-Optimizer/guides/3_pruning.html#pruning-and-subnet-search).

You can also take a look at FastNAS pruning interactive notebook [cifar_resnet](./cifar_resnet.ipynb) in this directory
which showcases the usage of FastNAS for pruning a ResNet 20 model for the CIFAR-10 dataset. The notebook
also shows how to profile the model to understand the search space of possible pruning options and demonstrates
how to save and restore pruned models.

### GradNAS Pruning for HuggingFace Language Models (e.g. BERT)

Checkout the BERT pruning example in [chained_optimizations](../chained_optimizations/README.md) directory
which showcases the usage of GradNAS for pruning BERT model for Question Answering followed by fine-tuning
with distillation and quantization. The example also demonstrates how to save and restore pruned models.

## Pruning Guidelines

### Minitron

This section provides recommendations for choosing pruning strategies and distillation hyperparameters for Minitron pruning to help achieve the best latency-accuracy trade-offs.

#### Depth Pruning

Depth pruning reduces the number of layers (`num_layers`) in the model.

**Advantages:**

- Simpler to configure - only 1 parameter to tune
- Faster inference than width-pruned models at a fixed number of parameters

**Recommendations:**

- Up to **1/3rd parameter reduction** can generally result in a model above the Pareto frontier with good latency-accuracy trade-off (when using a good quality dataset for distillation with ~80-100B tokens)
- For pruning **>50%**, use iterative pruning: compress by 30%, perform distillation, then compress again

**Examples:**

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (`num_layers=36`) → 6B (`num_layers=24`)
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) (`num_layers=32`) → 4.5B (`num_layers=16`)

#### Width Pruning

Width pruning reduces model dimensions per layer such as `hidden_size`, `ffn_hidden_size`, `num_attention_heads`, `mamba_num_heads`, `mamba_head_dim`, `num_moe_experts`, `moe_ffn_hidden_size`, and `moe_shared_expert_intermediate_size`.

**Advantages:**

- Better accuracy than depth-pruned models at a fixed number of parameters

**Recommendations:**

- Start with pruning `hidden_size` and `ffn_hidden_size` as the simplest configuration
- Up to **1/3rd parameter reduction** can generally result in a model above the Pareto frontier with good latency-accuracy trade-off (when using a good quality dataset for distillation with ~80-100B tokens)
- **Axis sensitivity:** MLP dimensions (`ffn_hidden_size`) can typically be pruned more aggressively than embedding dimensions (`hidden_size`) and attention/Mamba dimensions (`num_attention_heads`, `mamba_num_heads`, `mamba_head_dim`)
- For pruning **>50%**, use iterative pruning: compress by 30%, perform distillation, then compress again

**Examples:**

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (`ffn_hidden_size=12288`, `hidden_size=4096`) → 6B (`ffn_hidden_size=9216`, `hidden_size=3584`)
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) (`ffn_hidden_size=14336`, `hidden_size=4096`) → 4.5B (`ffn_hidden_size=9216`, `hidden_size=3072`)
- [Nemotron-H-8B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K) (`ffn_hidden_size=21504`, `hidden_size=4096`, `mamba_num_heads=128`) → [Nemotron-H-4B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-4B-Base-8K) (`ffn_hidden_size=12288`, `hidden_size=3072`, `mamba_num_heads=112`) - See [paper](https://arxiv.org/pdf/2504.11409)

#### Depth and Width Pruning

For optimal results, combine depth and width pruning. This will require more tuning to find the best architecture.

**Examples:**

- [NVIDIA-Nemotron-Nano-12B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2) (`ffn_hidden_size=20480`, `hidden_size=5120`, `num_layers=62`) → [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) (`ffn_hidden_size=15680`, `hidden_size=4480`, `num_layers=56`) - See [paper](https://arxiv.org/pdf/2508.14444)

#### General Pruning Guidelines

- **Pruning ratio:** Anything **>50% pruning is hard to recover**. For such aggressive pruning, iterative pruning (compress → distill → compress again) is recommended.
- **Latency-accuracy trade-off:** The more pruning you do, the faster your model will be at the cost of lower accuracy. Choose based on your requirements.
- **Dataset quality:** Use a high-quality dataset for distillation. If you don't have a specific dataset, [Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1) is recommended.
- **Post-training:** Further post-training (e.g., instruction tuning, preference alignment) is needed after pruning and distillation on pre-training datasets to improve reasoning capabilities. A good dataset for post-training is [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2).

#### Distillation Hyperparameters

After pruning, distillation is required to recover model accuracy. Below are recommended starting hyperparameters for distillation:

| **Hyperparameter** | **Recommendation** |
| :---: | :---: |
| **Sequence Length** | 8192 (or 4096 if dataset has smaller sequences) |
| **Global Batch Size (GBS)** | 768 |
| **Micro Batch Size (MBS)** | As large as your GPU memory can accommodate |
| **Learning Rate (LR)** | 1e-4 → 1e-5 (linear decay) for 30-50% pruning<br>• More compression → higher LR<br>• Less compression → lower LR<br>• As model gets larger → reduce LR to avoid divergence |
| **Warmup Steps** | 100 |
| **Training Max Steps** | Num training tokens / (Seq len × GBS)<br>• Recommended: 80-100B tokens |
| **Data Composition** | • Standard models: 100% pre-training data<br>• Reasoning models: 70% reasoning data + 30% pre-training data |

> [!TIP]
> If you know the maximum learning rate used during the original training, a good rule of thumb for knowledge distillation is to use **1/5th of that maximum LR** when compressing by ~50%.

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

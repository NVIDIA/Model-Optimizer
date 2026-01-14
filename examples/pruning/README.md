# Pruning

Model pruning is a technique that removes redundant or less important parameters/connections from a neural network to reduce complexity and improve efficiency while maintaining performance.

Pruning can involve removal (prune) of Linear and Conv layers; and Transformer attention, MLP, MoE, Mamba, and depth of the model.

This section focuses on applying Model Optimizer's state-of-the-art complementary pruning modes to enable you to search for the best subnet architecture from your provided base model:

1. [Minitron](https://arxiv.org/pdf/2408.11796): A pruning method developed by NVIDIA Research for pruning GPT (and later extended to Mamba, MoE, and Hybrid Transformer Mamba) models in NVIDIA Megatron-LM or NeMo framework. It uses the activation magnitudes to prune the embedding hidden size; mlp ffn hidden size; transformer attention heads; mamba heads and head dimension; MoE number of experts, ffn hidden size, and shared expert intermediate size; and number of layers of the model.
1. [Puzzletron](../puzzletron/README.md): An advanced pruning method by NVIDIA using Mixed Integer Programming (MIP) based NAS search algorithm.
1. FastNAS: A pruning method recommended for Computer Vision models. Given a pretrained model, FastNAS finds the subnet which maximizes the score function while meeting the given constraints.
1. GradNAS: A light-weight pruning method recommended for language models like Hugging Face BERT, GPT-J. It uses the gradient information to prune the model's linear layers and attention heads to meet the given constraints.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | \[[Link](#pre-requisites)\] | |
| Getting Started | Learn how to use the pruning API | \[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/3_pruning.html)\] |
| Support Matrix | View the support matrix to see available pruning algorithms and their compatibility with different models and frameworks | \[[Link](#support-matrix)\] | |
| Pruning Guidelines | Guidelines for choosing how and how much to prune for best results | \[[Link](#pruning-guidelines)\] | |
| Examples | Examples of different pruning methods | \[[Link](#examples)\] | |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

For Minitron pruning for Megatron-LM / NeMo models, use the NeMo container (e.g., `nvcr.io/nvidia/nemo:25.09`) which has all the dependencies installed.

For FastNAS pruning for PyTorch Computer Vision models, no additional dependencies are required.

For GradNAS pruning for Hugging Face BERT / GPT-J, no additional dependencies are requisred.

## Getting Started

As part of the pruning process, you will need to set up the training and/or validation data loaders, and optionally define a validation score function (FastNAS) or loss function (GradNAS) and specify the desired pruning constraints (See [Support Matrix](#support-matrix) for available pruning constraints).

To prune your model, you can simply call the `mtp.prune` API and save the pruned model. If the model is pruned using FastNAS or GradNAS, you need to use `mto.save` and `mto.restore` to save and restore the pruned model; while for Minitron pruning, you can use your standard saving and loading functions since it is a homogeneous pruning.

Please see an example snippet of Minitron pruning for Megatron-Core GPT model below (for other algorithms, please refer to the examples below).

```python
import modelopt.torch.prune as mtp
from megatron.core.models.gpt import GPTModel
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.transformer.transformer_config import TransformerConfig

# Load the Megatron-Core GPTModel with ModelOpt transformer layer spec
config = TransformerConfig(...)
model = GPTModel(
    config=config,
    transformer_layer_spec=get_gpt_modelopt_spec(config, remap_te_layernorm=True),
    ...
)

# Set up the forward loop to run on 512-1024 train samples
# For Megatron-LM framework, you can use the following utility function
from megatron.training.training import evaluate_and_print_results

def forward_loop(_):
    evaluate_and_print_results(prefix, forward_step, train_iterator, model, ...)


# Specify the pruning constraints (Check Support Matrix for available pruning dimensions)
export_config = {
    "hidden_size": 3072,
    "ffn_hidden_size": 9216,
}


# Run the pruning process (if model is a list then pass model[0] to the prune API)
# Save minitron scores at scores_path so we can re-run pruning with different export configs without running the forward loop again
# NOTE: Skip scores_path on re-running if you want to change the dataset and re-calibrate
model, pruning_scores = mtp.prune(
    model,
    mode="mcore_minitron",
    constraints={"export_config": export_config},
    dummy_input=None,  # Not used
    config={"forward_loop": forward_loop, "scores_path": "modelopt_minitron_scores.pth"},
)
```

If your model parameters are already sorted, you can skip the sorting step by setting `"skip_sorting": True` in `config` instead of passing `forward_loop`.

> [!Note]
> Fine-tuning / distillation is required after pruning to recover the accuracy. Please refer to [end-to-end pruning and distillation tutorial](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/qwen/pruning-distillation) for more details.

## Support Matrix

| **Algorithm** | **Model** | **Pruning Constraints** |
| :---: | :---: | :---: |
| Minitron | Megatron-core / NeMo based GPT / Mamba / MoE / Hybrid LLM Models<sup>1</sup> | Export config with width (`hidden_size`, `ffn_hidden_size`, `num_attention_heads`, `mamba_num_heads`, `mamba_head_dim`, `num_moe_experts`, `moe_ffn_hidden_size`, `moe_shared_expert_intermediate_size`) and/or depth (`num_layers`) values |
| FastNAS | Computer Vision models | flops, parameters |
| GradNAS | HuggingFace BERT, GPT-J | flops, parameters |

> *<sup>1.</sup>Only Pipeline Parallel models are supported. Hugging Face models can be converted to Megatron-LM/NeMo format and used subsequently.*

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

## Examples

### Minitron Pruning for Megatron-LM / NeMo Framework LLMs (e.g. Qwen 3, Nemotron Nano)

Checkout the Minitron pruning example for the [Megatron-LM Framework](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt#-pruning) or [NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/pruning/pruning.html) which showcases the usage of the powerful Minitron pruning algorithm developed by NVIDIA Research for pruning LLMs like Llama 3.1 8B, Qwen 3 8B, Nemotron Nano 12B v2, etc.
Both frameworks support importing from a Hugging Face pretrained checkpoint.

You can also look at the NeMo tutorial notebooks [here](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/qwen/pruning-distillation) which showcase the usage of Minitron pruning followed by distillation for Qwen 3 8B step-by-step in NeMo framework. Hugging Face models can also be converted to NeMo format and used subsequently as shown in the tutorial.

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

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

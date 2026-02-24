# DMS Architecture and Advanced Options

This document describes DMS internals, configuration options, and how to extend the codebase.

## Code Details

### Eviction Decisions

DMS supports two ways to compute the eviction decision:

- **Extracted from a single neuron of a key or query vector**: see Section 3.1 of [Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference](https://arxiv.org/pdf/2403.09636). Enable with `dms_separate_alpha=False`.
- **Produced by a learned linear projection (adapter) from the hidden state**: see Section 3.2 of [Inference-Time Hyper-Scaling with KV Cache Compression](https://arxiv.org/pdf/2506.05345). Enable with `dms_separate_alpha=True`.

You can also choose the granularity of eviction decisions:

- `dms_alpha_per: "head"`: decisions are made independently per attention head (KV cache lengths may differ across heads).
- `dms_alpha_per: "layer"`: decisions are shared across heads within a layer (all heads in the layer keep the same number of tokens).

During training, decision logits are augmented with Gumbel noise to enable differentiable gating (`dms.core.get_gating_with_noise`). During inference, a hard threshold is used.

### Attention

The DMS attention implementation (given decision logits) can be found in `dms/attention.py` (see `dms_attn_train_mode`).

### Loss Function

Training uses knowledge distillation with forward KL divergence between student and teacher logits, computed in `dms/training/engine.py` (`distillation_loss`). This is combined with a DMS compression loss that encourages the model to match the target eviction fraction.

### DMS Schedule

The compression ratio increases linearly from `initial_cr` (typically 1.0) to `final_cr` (e.g., 16.0) over `final_step` training steps. See `dms_schedule()` in `dms/training/engine.py`.

## Advanced Options

### Chunked Prefill

Chunked prefill reduces peak memory usage during the prefill phase by processing the input sequence in fixed-size chunks. Set the chunk size (in tokens) via:

```python
Qwen3ForCausalLMDMS.from_pretrained(..., dms_chunked_prefill=4096)
```

### Cache Preallocation

The paged KV cache uses a dynamically resizable per-attention-layer block table (similar to `std::vector` in C++), growing as needed during generation. If you know your maximum context length ahead of time, you can preallocate to avoid runtime reallocations:

```python
Qwen3ForCausalLMDMS.from_pretrained(..., dms_preallocate_for_tokens=2048)
```

## Retrofitting a New Model Family

To add DMS support for a new model family, create a new directory under `models/`:

```bash
models/new_model/
├── configuration_new_model_dms.py  # Config extending the base model config
├── extract.py                      # Checkpoint extraction
├── modeling_new_model_dms.py       # Model with DMS attention
└── train.py                        # Training entry point
```

The model-specific code should:

1. Extend the model's config class with DMS parameters (see `models/qwen3/configuration_qwen3_dms.py`).
2. Override the attention forward pass and call:
   - `dms.core.prepare_attention_input`
   - `dms.attention.dms_attention`
3. Add `dms_proj_alpha` and `dms_proj_alpha_norm` layers to the attention layer.
4. Add a YAML config under `configs/`.

Core DMS operations (`prepare_attention_input`, `dms_attention`, `post_process_attention_output`) are model-agnostic; model-specific code provides its Q/K/V projections and any required norms as inputs.

## Adding a New Dataset

To add a new training dataset, edit `dms/training/data.py`:

1. Define `filter_fn` and `extract_fn` for your dataset.
2. Create a `DatasetInfo` instance.

Example:

```python
def my_dataset_filter_fn(ds_elem):
    return ds_elem["quality_score"] > 0.8

def my_dataset_extract_fn(ds_elem):
    return {
        "conversation": [
            {"role": "user", "content": ds_elem["prompt"]},
            {"role": "assistant", "content": ds_elem["response"]},
        ]
    }

MyNewDataset = DatasetInfo(
    args=("org/my-dataset",),
    kwargs={"split": "train"},
    filter_fn=my_dataset_filter_fn,
    extract_fn=my_dataset_extract_fn,
)
```

Then reference it in your YAML config:

```yaml
data:
  blend: "MyNewDataset:0.5,OpenR1Math220k:0.5"
```

## Checkpoint Resume

To resume training from the latest checkpoint, set the following in your YAML config:

```yaml
hf_trainer:
  resume_from_checkpoint: "auto"
```

This auto-detects the latest `checkpoint-N` directory under the output directory. You can also specify an explicit path:

```yaml
hf_trainer:
  resume_from_checkpoint: outputs/qwen3_8b/checkpoint-300
```

Resume works because:

- The Hugging Face Trainer restores optimizer state, LR scheduler state, the training step counter, and RNG states.
- The DMS schedule is deterministic given the current training step.
- Gumbel noise is seeded from `step + process_index + grad_acc_step`.

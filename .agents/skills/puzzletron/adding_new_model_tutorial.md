# Adding a New Model to Puzzletron with Claude

This tutorial shows the full conversation between a user and Claude when adding
Qwen3.5-0.8B support to Puzzletron. Use it as a guide for how to interact with Claude
when you want to compress a model that Puzzletron doesn't yet support.

---

## The journey

### Step 1: Download the model and try to run it

The user starts simply — download the model and run the pipeline.

> **User:** can you download <https://huggingface.co/Qwen/Qwen3.5-0.8B> to /workspace/hf_models/...

Claude downloads the model using `huggingface-cli`:

```text
Successfully downloaded 13 files to /workspace/hf_models/Qwen3.5-0.8B
```

> **User:** run puzzletron all for this model

Claude checks whether the model is supported — first by trying to load it with Transformers, then by looking for an existing Puzzletron descriptor and compression config. It finds two blockers:

1. Transformers doesn't recognise model type `qwen3_5`:

```text
ValueError: The checkpoint you are trying to load has model type `qwen3_5`
but Transformers does not recognize this architecture.
```

2. There is no Puzzletron descriptor for `qwen3_5` in `modelopt/torch/puzzletron/anymodel/models/` and no compression example config under `examples/puzzletron/configs/`

---

### Step 2: Fix the Transformers version

> **User:** so please install it using python -m pip

Claude upgrades:

```bash
python3 -m pip install --upgrade transformers
# → Successfully installed transformers-5.12.1
```

Claude confirms the model now loads:

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("/workspace/hf_models/Qwen3.5-0.8B")
print(config.model_type)  # qwen3_5  ✓
```

---

### Step 3: Implement the descriptor, converter, and configs

> **User:** ok, implement the descriptor and compression example config for Qwen3.5-0.8B

Claude implements everything in one go:

**Model descriptor** (`modelopt/torch/puzzletron/anymodel/models/qwen3_5/qwen3_5_model_descriptor.py`)

**Converter**

**Registration** (`anymodel/models/__init__.py`): gated behind `transformers >= 4.57.0`

**Compression example config** (`examples/puzzletron/configs/qwen3_5-0.8b_pruneffn_memory/`):

`intermediate_size_list` chosen by scaling the Llama-3.1-8B ratios (~21 %, 42 %, 60 %, 83 %
of teacher) to the 0.8B's `intermediate_size = 3584`:

```text
[1280, 2560, 3584, 5120]
```

MIP constraints set proportionally to Llama 3.1-8B:

```yaml
target_memory: 20_000  # 20 GiB
num_params: 1_500_000_000
```

Claude then verifies that both the descriptor and converter register correctly:

```python
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
print('qwen3_5' in ModelDescriptorFactory.CLASS_MAPPING)  # True  ✓
```

---

### Step 4: Run the pipeline

> **User:** run puzzletron all for Qwen3.5-0.8B on 4 GPUs

Claude constructs the `torchrun` command directly with the Qwen3.5-0.8B config path and runs the full pipeline. The user monitors progress with:

```text
/puzzletron all progress
```

Example output mid-run:

```text
Overall: Puzzletron full pipeline (steps 1–8)
────────────────────────────────────────────────────────────────────
  Status      Step  Description                          Elapsed
────────────────────────────────────────────────────────────────────
  [DONE]            1/8: starting puzzletron pipeline      0m 1s
  [DONE]            2/8: converting model to Puzzletron heterogeneous format (single-gpu)     0m 3s
  [DONE]            3/8: scoring pruning activations (multi-gpu)     0m 56s
  [DONE]            4/8: pruning the model and saving pruned checkpoints (single-gpu)     0m 10s
  [DONE]            5/8: building replacement library and subblock statistics (single-gpu)     0m 10s
  [RUNNING]         6/8: calculating one block scores (multi-gpu) (127/264 solutions)    20m 6s
  [ ]               7/8: running MIP and realizing models (multi-gpu)
  [ ]               8/8: puzzletron pipeline completed (multi-gpu)
────────────────────────────────────────────────────────────────────
  Started:   10:35:54
  Elapsed:   21m 26s  |  Remaining: ~28m estimated
```

Step 6 (one-block scoring) is the longest step — it scores all candidate solutions using a proxy metric (cosine embedding loss on hidden states). The number of solutions depends on the model size and `intermediate_size_list`; for Qwen3.5-0.8B with 4 sizes across 28 layers it is 264. Use `eval_samples` in the base YAML to trade off speed vs. score quality (default 128; 8 is useful for quick iteration).

---

### Step 5: Check the compressed model accuracy

Once the pipeline completes, check the accuracy of the MIP-selected compressed model against the teacher:

```text
/puzzletron mip losses
```

Example output for Qwen3.5-0.8B:

| Metric | Teacher | Compressed (solution_0) |
|---|---|---|
| `lm_loss` | 1.1067 | 3.8808 |
| `token_accuracy_top_1` | 0.7365 | 0.2915 |
| `token_accuracy_top_5` | 0.9079 | 0.5500 |
| `token_accuracy_top_10` | 0.9399 | 0.6451 |

The results are read from `<puzzle_dir>/mip/puzzle_solutions/<target>/solutions--validation/solution_0.json` (and `teacher.json` in the same directory).

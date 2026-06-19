# Adding a New Model to Puzzletron with Claude

This tutorial shows the full conversation between a user and Claude when adding
Qwen3.5-2B support to Puzzletron. Use it as a guide for how to interact with Claude
when you want to compress a model that Puzzletron doesn't yet support.

---

## The journey

### Step 1: Download the model and try to run it

The user starts simply — download the model and run the pipeline.

> **User:** can you download <https://huggingface.co/Qwen/Qwen3.5-2B> to /workspace/hf_models/...

Claude downloads the model using `huggingface-cli`:

```text
Successfully downloaded 13 files to /workspace/hf_models/Qwen3.5-2B
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
config = AutoConfig.from_pretrained("/workspace/hf_models/Qwen3.5-2B")
print(config.model_type)  # qwen3_5  ✓
```

---

### Step 3: Implement the descriptor, converter, and configs

> **User:** ok, implement the descriptor and compression example config for Qwen3.5-2B

Claude implements everything in one go:

**Model descriptor** (`modelopt/torch/puzzletron/anymodel/models/qwen3_5/qwen3_5_model_descriptor.py`)

**Converter**

**Registration** (`anymodel/models/__init__.py`): gated behind `transformers >= 4.57.0`

**Compression example config** (`examples/puzzletron/configs/qwen3_5-2B_pruneffn_memory/`):

`intermediate_size_list` chosen by scaling the Qwen3-8B ratios (~21 %, 42 %, 60 %, 81 %
of teacher) to the 2B's `intermediate_size = 6144`:

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

> **User:** run puzzletron all for Qwen3.5-2B on 4 GPUs

Claude constructs the `torchrun` command directly with the Qwen3.5-2B config path and runs the full pipeline. The user monitors progress with:

```text
/puzzletron all progress
```

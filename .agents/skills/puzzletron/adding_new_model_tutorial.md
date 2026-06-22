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

> **User:** show mip losses for Qwen3.5-0.8B

Claude runs `/puzzletron mip losses` and presents the results as a comparison table:

Example output for Qwen3.5-0.8B:

```text
Metric                             Teacher   Compressed (solution_0)
--------------------------------------------------------------------
target_memory                   20,389 MiB                10,000 MiB
--------------------------------------------------------------------
lm_loss                             1.1067                    3.8808
token_accuracy_top_1                0.7365                    0.2915
token_accuracy_top_5                0.9079                      0.55
token_accuracy_top_10               0.9399                    0.6451

Results from: /workspace/puzzle_dir_qwen3_5-0.8b/mip/puzzle_solutions/target_memory_10000MiB-num_params_1_5G/solutions--validation
Sweep results: use /puzzletron mip sweep losses
```

The results are read from `<puzzle_dir>/mip/puzzle_solutions/<target>/solutions--validation/solution_0.json` (and `teacher.json` in the same directory). The teacher memory is taken from the sweep CSV if a sweep was also run.

---

### Step 6: Run the MIP sweep and check sweep losses

If the sweep is enabled in the config YAML (`mip.sweep.enabled: true`), run it after the full pipeline:

> **User:** run sweep for Qwen3.5-0.8B

Claude runs the MIP step with the Qwen3.5-0.8B config on the requested number of GPUs. Monitor progress with:

```text
/puzzletron mip progress
```

Example output mid-run:

```text
Overall: Puzzletron step 7/8 — MIP sweep (6 compression rates)
──────────────────────────────────────────────────────────────
  Status      Phase                              Elapsed
──────────────────────────────────────────────────────────────
  [DONE]      Prep (teacher memory + rate list)       <1s
  [DONE]      compression_rate=0.5                0m 44s
  [DONE]      compression_rate=0.6                0m 36s
  [DONE]      compression_rate=0.7                0m 37s
  [DONE]      compression_rate=0.8                0m 37s
  [RUNNING]   compression_rate=0.9 — validating (8/8 batches)    0m 28s
  [ ]         compression_rate=1.0               pending
──────────────────────────────────────────────────────────────
  Started:   00:03:28
  Finished:  00:06:30 (in progress)
  Elapsed:   3m 2s
  Completed: 4/6 compression rates
  Remaining: 1m 17s estimated
```

Once complete, view accuracy across all compression rates:

> **User:** show mip sweep losses

Claude runs `/puzzletron mip sweep losses` and presents the results:

```text
  rate    target_mem    actual_mem    num_params   lm_loss    top_1    top_5    top_10
--------------------------------------------------------------------------------------
0.5000    10194.3640    10143.2768   888,813,280    3.2367   0.3663   0.6384    0.7251
0.6000    12233.2368    11719.5001   909,901,856    2.6377   0.4434   0.7198    0.7981
0.7000    14272.1096    14083.8350   941,534,720    1.8532   0.5855   0.8176    0.8735
0.8000    16310.9824    15660.0582   962,623,296    1.5385   0.6448   0.8576    0.9046
0.9000    18349.8552    18024.3931   994,256,160    1.2447   0.7064   0.8914    0.9278
1.0000    20388.7280    20388.7280  1,025,889,024    1.1067   0.7365   0.9079    0.9399

Results from: /workspace/puzzle_dir_qwen3_5-0.8b/mip_sweep_results.csv
```

Use this table to pick the compression rate that best meets your accuracy/memory budget.

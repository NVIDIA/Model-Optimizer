---
name: puzzletron
description: "End-to-end workflow for model pruning and MIP-based optimization. Commands: mip, all, add-model, eval (list/mmlu/mmlu-pro), distill (run/list/summary/progress). Usage: /puzzletron COMMAND [ARGS]"
license: Apache-2.0
---

# Puzzletron

## Routing

**STEP 1 — Check args before doing anything else. This is MANDATORY.**

- If args are **empty**, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**
- If the first word of args does **not exactly match** `mip`, `all`, `add-model`, `eval`, or `distill`, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**

---

**Puzzletron** — end-to-end workflow for model pruning and MIP-based optimization.

Available commands:
- `mip <nproc_per_node>` — Run the MIP step (nproc_per_node: number of GPUs per node)
- `mip progress` / `mip sweep progress` — Show live MIP progress with timing summary (works for both single and sweep runs)
- `mip losses` — Show teacher vs. compressed model accuracy for the single constrained MIP solution
- `mip sweep run <nproc_per_node> [--config <path>]` — Enable sweep in config YAML and run MIP across all compression rates
- `mip sweep losses` — Show accuracy across all compression rates from a completed sweep
- `all <nproc_per_node>` — Run the full Puzzletron pipeline (nproc_per_node: number of GPUs per node)
- `all progress` — Show live full pipeline progress with timing summary
- `add-model <hf_model_path>` — Implement descriptor, converter, and configs for an unsupported model
- `eval list [puzzle_dir]` — List all available checkpoints (teacher + sweep solutions) with their index numbers; auto-discovers puzzle_dir if omitted
- `eval progress [puzzle_dir]` — Show per-checkpoint MMLU status, full-vs-limited accuracy, and phase-aware timing; auto-discovers puzzle_dir if omitted
- `eval mmlu <index|hf_model_path> [--puzzle_dir <dir>] [--limit <N>] [--batch_size <B>]` — Evaluate a checkpoint on MMLU (5-shot); pass index from `eval list` or a direct path; default `--limit 10` (smoke test)
- `eval mmlu-pro <index|hf_model_path> [--puzzle_dir <dir>] [--limit <N>] [--batch_size <B>] [--gpus <ids>...]` — Evaluate MMLU-Pro subjects concurrently; results use a limit-specific directory
- `distill run [--puzzle_dir <dir>] [--ratio <r>] [--nproc_per_node <n>] [--train_iters <n>] [--output_dir <dir>] [--use_mock_data] [--data_paths <p>...]` — Run distillation for a MIP solution
- `distill list [puzzle_dir]` — List all distillation runs with status
- `distill summary [puzzle_dir]` — Compare datasets, recipes, checkpoints, MMLU, and disk usage across runs
- `distill progress [--puzzle_dir <dir>] [--ratio <r>]` — Show training progress for distillation runs
- `distill tokenize --hf_dataset <name> --output_dir <dir> --tokenizer <path> [--hf_name <s>] [--hf_split <s>] [--hf_max_samples <n>] [--json_keys text|messages]` — Tokenize an HF dataset to Megatron binary format for distillation

Usage: `/puzzletron <command> [args]`

---

**STEP 2 — Only if the first word of args exactly matches a command name, execute it. Never reach this step if args were empty.**

## Permission handling

- Run read-only monitoring commands normally, without requesting escalation. This includes
  `ps`, `grep`, `rg`, `find`, `tail`, `sed`, `stat`, log inspection, result inspection, and
  Puzzletron progress scripts under `.agents/skills/puzzletron/`.
- Do not proactively mark a read-only monitoring command as requiring elevated permissions.
  Request escalation only after the command actually fails because of sandbox restrictions and
  the result is necessary to continue.
- Treat GPU access and job launch separately from monitoring. A CUDA command may require
  escalation even when its logs, result files, and process status can be inspected without it.
- Reuse an existing approved command rule when it exactly covers the required operation; do not
  escalate adjacent read-only checks merely because the launched job required escalation.

## Command: all

Parse `nproc_per_node` from args using either positional or flag syntax:
- Positional: second word is a number, e.g. `all 2`
- Flag: `--nproc_per_node <value>` anywhere in args, e.g. `all --nproc_per_node 2`

- If the second word is exactly `progress`, execute the **all progress** sub-command below.
- If no `nproc_per_node` value can be found, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- If the value does not match `^[0-9]+$`, ask the user: "nproc_per_node must be a positive integer." and **STOP**.
- Otherwise use the parsed value and run the full pipeline.

### all \<nproc_per_node\>

First extract `puzzle_dir` from the config YAML:

```bash
PUZZLE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml'))['puzzle_dir'])")
```

Then run the pipeline, writing the log into the puzzle dir:

```bash
mkdir -p $PUZZLE_DIR && set -o pipefail && export PYTHONPATH=$PYTHONPATH:. && \
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml \
  2>&1 | tee $PUZZLE_DIR/log.txt | grep "Puzzletron Progress"
```

Stream output to the user as it arrives. When the command finishes, report the exit code.

### all progress

First extract `puzzle_dir` from the config YAML:

```bash
PUZZLE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml'))['puzzle_dir'])")
```

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/all_progress.py $PUZZLE_DIR
```

**How remaining time is estimated (`all_progress.py`):**

The script uses per-step baseline durations and a live scale factor derived from completed steps:

| Step | Description | Baseline | Source |
|------|-------------|----------|--------|
| 1 | starting pipeline | 1s | measured Qwen3-8B/8GPU |
| 2 | model conversion (single-gpu) | 29s | measured Qwen3-8B/8GPU |
| 3 | activation scoring (multi-gpu) | 2m8s | measured Qwen3-8B/8GPU |
| 4 | pruning & saving (single-gpu) | 58s | measured Qwen3-8B/8GPU |
| 5 | replacement library (single-gpu) | 15s | measured Qwen3-8B/8GPU |
| 6 | one block scores (multi-gpu) | 53m7s | measured Qwen3-8B/8GPU |
| 7 | MIP solve | 5m15s | measured Qwen3-8B/8GPU |
| 8 | completion | ~0s | measured Qwen3-8B/8GPU |

When adding support for a new model or GPU count, update `_BASELINE_S` in `all_progress.py` with actual measured step durations from a completed run.

`scale_factor` = mean(actual / baseline) across all **completed** steps except step 1 (excluded because it is always trivially fast and would skew the factor). Starts at 1.0 until at least one non-trivial step finishes.

`remaining = max(0, step_est(current) − elapsed_in_current) + Σ step_est(future)`

where `step_est(s) = baseline[s] × scale_factor`.

**At the very start of step 1** (nothing completed yet): `scale_factor = 1.0`, so the estimate equals the sum of all baselines for steps 1–8 ≈ **6 minutes**. This is a rough prior based on measured/estimated values; it converges as steps complete.

For the current running step, if sub-step progress is available (batch count or solution count), remaining time within that step is computed from the observed throughput rate instead.

## Command: mip

Parse `nproc_per_node` from args using either positional or flag syntax:
- Positional: second word is a number, e.g. `mip 2`
- Flag: `--nproc_per_node <value>` anywhere in args, e.g. `mip --nproc_per_node 2`

- If the second word is exactly `progress`, execute the **mip progress** sub-command below.
- If the second word is exactly `losses`, execute the **mip losses** sub-command below.
- If the second word is exactly `sweep` and the third word is exactly `progress`, execute the **mip progress** sub-command below (sweep and single use the same progress view).
- If the second word is exactly `sweep` and the third word is exactly `run`, execute the **mip sweep run** sub-command below.
- If the second and third words are exactly `sweep losses`, execute the **mip sweep losses** sub-command below.
- If no `nproc_per_node` value can be found, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- If the value does not match `^[0-9]+$`, ask the user: "nproc_per_node must be a positive integer." and **STOP**.
- Otherwise use the parsed value and run the MIP step.

### mip \<nproc_per_node\>

First extract `puzzle_dir` from the config YAML:

```bash
PUZZLE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml'))['puzzle_dir'])")
```

Then run the MIP step, writing the log into the puzzle dir:

```bash
mkdir -p $PUZZLE_DIR && set -o pipefail && export PYTHONPATH=$PYTHONPATH:. && \
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml \
  --mip-only 2>&1 | tee $PUZZLE_DIR/log.txt | grep "Puzzletron Progress"
```

Stream output to the user as it arrives. When the command finishes, report the exit code.

### mip progress

First extract `puzzle_dir` from the config YAML:

```bash
PUZZLE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml'))['puzzle_dir'])")
```

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/mip_progress.py $PUZZLE_DIR
```

### mip losses

First extract `puzzle_dir` from the config YAML:

```bash
PUZZLE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml'))['puzzle_dir'])")
```

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/mip_losses.py $PUZZLE_DIR
```

### mip sweep run

Parse args:
- `nproc_per_node` — fourth word or `--nproc_per_node <value>`. If missing, ask: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- `--config <path>` — optional path to the main YAML config file. If omitted, ask the user which model config to use (show available configs under `examples/puzzletron/configs/`) and **STOP** if they don't provide one.

**Step 1 — Resolve config and puzzle_dir:**

```bash
CONFIG=<config_path>
PUZZLE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['puzzle_dir'])")
```

**Step 2 — Check if sweep is already enabled; enable it if not:**

```bash
python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
sweep = (cfg.get('mip') or {}).get('sweep') or {}
print('enabled' if sweep.get('enabled') else 'disabled')
"
```

If the output is `disabled`, edit the YAML to add the sweep block under `mip:`:

```yaml
  sweep:
    enabled: true
    memory_compression_rates: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    output_csv: ${puzzle_dir}/mip_sweep_results.csv
```

Tell the user what was changed. If already `enabled`, tell the user sweep is already configured and proceed.

**Step 3 — Run the MIP sweep in background:**

```bash
mkdir -p $PUZZLE_DIR && set -o pipefail && export PYTHONPATH=$PYTHONPATH:. && \
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config <config_path> \
  --mip-only 2>&1 | tee $PUZZLE_DIR/log.txt | grep "Puzzletron Progress"
```

Run in background (`run_in_background=true`). Tell the user the log path (`$PUZZLE_DIR/log.txt`) immediately.

**Step 4 — Show initial progress** after ~15 seconds:

```bash
python3 .agents/skills/puzzletron/mip_progress.py $PUZZLE_DIR
```

Present output in a fenced code block. The sweep runs 6 compression rates sequentially; each takes ~5 minutes, so the full sweep takes ~30 minutes.

### mip sweep losses

First extract `puzzle_dir` from the config YAML:

```bash
PUZZLE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml'))['puzzle_dir'])")
```

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/mip_sweep.py $PUZZLE_DIR
```

## Command: eval

- If the second word is not exactly `list`, `progress`, `mmlu`, or `mmlu-pro`, tell the user: "Unknown eval sub-command. Available: `list`, `progress`, `mmlu`, `mmlu-pro`." and **STOP**.

### eval list

Parse `puzzle_dir` from args (third word or `--puzzle_dir <value>`). It is optional.

Run the following Bash command, including `<puzzle_dir>` as an argument when provided, or omitting it to trigger auto-discovery:

```bash
python3 .agents/skills/puzzletron/eval_list.py [<puzzle_dir>]
```

Present the output to the user wrapped in a fenced code block (``` ... ```).

### eval progress

Parse `puzzle_dir` from args (third word or `--puzzle_dir <value>`). It is optional.

Run the following Bash command, including `<puzzle_dir>` as an argument when provided, or omitting it to trigger auto-discovery:

```bash
python3 .agents/skills/puzzletron/eval_progress.py [<puzzle_dir>]
```

Present the output to the user wrapped in a fenced code block (``` ... ```).

For running evals, detect active `lm_eval_hf.py` processes via `ps` and show the current named phase on a second line:

```text
  [RUNNING]   teacher               ...  /workspace/.../Qwen3-8B
                          phase: Loading weights  phase progress 62% (157/254)  total elapsed 0m38s  phase remaining ~0m15s

  [RUNNING]   distill:0.9x          ...  /workspace/.../hf
                          phase: Running loglikelihood requests  phase progress 89% (49989/56168)  total elapsed 4m19s  phase remaining ~0m19s  overall remaining ~0m24s
```

Interpret the timing fields as follows:

- `phase progress` is progress only within the named phase. It is not an overall evaluation percentage.
- `total elapsed` covers the entire evaluation process, including all completed phases.
- `phase remaining` is the ETA for only the named phase.
- `overall remaining` is the ETA until the evaluation finishes. Show it only when it can be estimated reliably.

Do not derive phase remaining from total process time. Read tqdm's own ETA for loading, tokenization, and loglikelihood. Aggregate the repeatedly-reset per-subtask bars into one `Building contexts` phase and estimate that phase from completed task intervals. During the final loglikelihood phase, show `overall remaining` as the phase ETA plus a five-second result-saving allowance. Do not show an overall ETA during earlier phases without calibrated timings for their future phases.

When saved JSONs include both limited and full evaluations, prefer the newest full evaluation. If only a limited result exists, append `(limit=N)` to its accuracy so it cannot be mistaken for a full score.

Status values:
- `[RUNNING]` — active `lm_eval_hf.py` process found; shows total elapsed and phase timing when available
- `[DONE]`    — results JSON written with MMLU accuracy
- `[ ]`       — no active process and no saved result; an empty result directory is also pending

To inspect the parser against a saved log while developing it, run:

```bash
python3 .agents/skills/puzzletron/eval_progress.py --log-file <eval_log>
```

### eval mmlu

Parse args:
- `index_or_path` — third word. If missing, ask: "Please provide a checkpoint index (from `eval list`) or a direct HF model path." and **STOP**.
- `--puzzle_dir <dir>` — optional; used when resolving an index.
- If `index_or_path` matches `^[0-9]+$`, resolve it to a path by running `python3 .agents/skills/puzzletron/eval_list.py [<puzzle_dir>]` and picking the Nth entry (0-based) from the output lines. If the index is out of range, tell the user and **STOP**.
- Otherwise treat `index_or_path` as a literal `hf_model_path`.
- `--limit <N>` — optional integer; default `10` if not provided.
- `--batch_size <B>` — optional integer; default `4` if not provided.

Derive `output_path` as `<hf_model_path>/eval_results/mmlu` (always; not user-configurable).

Run the following Bash command, substituting the parsed values:

```bash
env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
  PYTHONPATH=.:$PYTHONPATH python examples/llm_eval/lm_eval_hf.py \
  --model hf \
  --model_args pretrained=<hf_model_path>,dtype=bfloat16,parallelize=True \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size <batch_size> \
  --output_path <hf_model_path>/eval_results/mmlu \
  [--limit <N>]
```

(Replace `[--limit <N>]` with the actual `--limit <N>` flag; always include it since the default is 10.)

**Note:** Unset the complete distributed environment for a single-process evaluation. Setting only
`WORLD_SIZE=1` is insufficient when the interactive node exports `RANK`, `LOCAL_RANK`,
`MASTER_ADDR`, and `MASTER_PORT`; Accelerate may then initialize a TCP rendezvous and collide with
another evaluation using the inherited port.

**Note on output file location:** lm_eval does not write results directly into `--output_path`. It creates a subdirectory named after the full model path with `/` replaced by `__`, then writes `results_<timestamp>.json` inside it. For example, for a model at `/workspace/foo/bar`, results land at:

```text
<output_path>/__workspace__foo__bar/results_<timestamp>.json
```

`eval_progress.py` handles this automatically via a recursive glob.

Stream output to the user as it arrives. When the command finishes:
- Report the exit code.
- Show a results summary table with: model path, total questions evaluated, loglikelihood requests, and the mmlu/category accuracy scores parsed from the output.

### eval mmlu-pro

Resolve `index_or_path` and optional `--puzzle_dir` as for `eval mmlu`. Parse `--limit <N>`
(default `10`), `--batch_size <B>` (default `4`), and `--gpus <ids>...` (default `0`).
Run in the background because the evaluation can take several minutes:

```bash
python3 .agents/skills/puzzletron/eval_mmlu_pro.py <hf_model_path> \
  --limit <N> --batch-size <B> --gpus <ids>...
```

The limit applies to each of MMLU-Pro's 14 subjects. Results are stored under
`<hf_model_path>/eval_results/mmlu_pro_limit_<N>/`, with a manifest recording the limit,
model, tasks, batch size, and GPU allocation. Never store a limited run in the full-run directory.

## Command: add-model

Parse `hf_model_path` from args (the second word). If missing, ask: "Please provide the HuggingFace model path (local or hub)." and **STOP**.

Then follow the steps below to implement full Puzzletron support for the model.

### Step 1 — Check if already supported

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('<hf_model_path>', trust_remote_code=True)
supported = cfg.model_type in ModelDescriptorFactory.CLASS_MAPPING
print(f'model_type: {cfg.model_type}')
print(f'already supported: {supported}')
"
```

If already supported, tell the user and **STOP**.

If `AutoConfig` raises an error about an unrecognised model type, the installed Transformers version is too old. Check the version, upgrade with `python3 -m pip install --upgrade transformers`, then re-run.

### Step 2 — Inspect the architecture

Run the following to understand what you are implementing:

```bash
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('<hf_model_path>', trust_remote_code=True)
print(cfg)
# If it has a nested text_config, print that too
if hasattr(cfg, 'text_config'):
    print('--- text_config ---')
    print(cfg.text_config)
"
```

Key things to note:
- **`model_type`** — this becomes the registration key for both descriptor and converter.
- **Nested `text_config`** — VLMs (e.g. Qwen3.5) wrap language model params inside `config.text_config`. Use `config.text_config` wherever you need `num_hidden_layers`, `intermediate_size`, `num_key_value_heads`. The converter must save `text_config` (not the full VLM config) so downstream code can access these fields directly.
- **Hybrid attention** — check `cfg.text_config.layer_type_list` (or similar). If some layers are linear/recurrent and others are full attention, `attn_no_op_post_init` must branch on `decoder_layer.layer_type`.
- **MoE** — if `num_experts` > 1 the model uses a MoE FFN and is not currently supported by the FFN pruning path; skip FFN pruning for such models.
- **Weight name prefixes** — inspect the checkpoint index to understand the layout:

```bash
python3 -c "
import json, collections
idx = json.load(open('<hf_model_path>/model.safetensors.index.json'))
prefixes = collections.Counter()
for n in idx['weight_map']:
    prefixes['.'.join(n.split('.')[:3])] += 1
for p, c in sorted(prefixes.items()):
    print(f'{c:4d}  {p}')
"
```

If weight names use a prefix like `model.language_model.*` rather than `model.*`, the converter must implement `convert_weight_name` to remap them, and `get_weight_groups` must handle both the original checkpoint names (used during conversion) and the remapped names (used when saving pruned checkpoints). See the Qwen3_5 descriptor/converter for the reference implementation of this pattern.

### Step 3 — Create the files

Create the following files (use an existing descriptor as a reference — `qwen3_5` for VLMs with nested config and weight remapping, `llama` or `qwen2` for standard text-only models):

**`modelopt/torch/puzzletron/anymodel/models/<model_type>/__init__.py`**

```python
from .<model_type>_converter import *
from .<model_type>_model_descriptor import *
```

**`modelopt/torch/puzzletron/anymodel/models/<model_type>/<model_type>_model_descriptor.py`**

Must implement (inheriting from `ModelDescriptor`):
- `decoder_layer_cls()` → the HF decoder layer class
- `input_embedding_name()` → e.g. `"model.embed_tokens"`
- `output_embedding_name()` → e.g. `"lm_head"`
- `final_norm_name()` → e.g. `"model.norm"`
- `layer_block_name(index)` → e.g. `f"model.layers.{index}"`
- `block_config_to_layer_overrides(block_config)` → dict with `intermediate_size` and `num_key_value_heads`
- `attn_no_op_post_init(decoder_layer)` → replace attention + input norm with no-ops
- `mlp_no_op_post_init(decoder_layer)` → replace MLP + post-attention norm with no-ops
- `layer_name_predicates(num_layers)` → regex dict grouping weights into `embeddings`, `lm_head`, `block_N_ffn`, `block_N_attention`
- `init_rotary_embedding(model, runtime)` → re-initialise rotary embedding after subblock load

**Critical:** `layer_name_predicates` patterns must match the **converted** `model.*` names (not the original VLM checkpoint names). If the checkpoint uses a different prefix, override `get_weight_groups` to normalise names before matching and restore originals in the returned groups (so `param_to_file` lookups in `convert_model_weights` still work). See `Qwen3_5ModelDescriptor.get_weight_groups` for the reference pattern.

**`modelopt/torch/puzzletron/anymodel/models/<model_type>/<model_type>_converter.py`**

Must implement (inheriting from `Converter`):
- `create_block_configs_from_main_config(config)` → list of `BlockConfig`, one per layer
- `convert_configs_in_dirs(input_dir, output_dir)` → if the model has a nested `text_config`, save that instead of the full VLM config so `num_hidden_layers` is accessible at the top level
- `convert_weight_name(name)` → remap checkpoint weight names to converted model names (identity if no remapping needed)

**Register in `modelopt/torch/puzzletron/anymodel/models/__init__.py`** — gate behind the minimum Transformers version that introduced the model:

```python
if _Version(_transformers_version) >= _Version("X.Y.Z"):
    from .<model_type> import *
```

**Compression config** at `examples/puzzletron/configs/<model_type>-<size>_pruneffn_memory/`:
- Base YAML (`<model_type>.yaml`): `descriptor: <model_type>`, MIP constraints
- Main YAML (override): `input_hf_model_path`, `dataset_path`, `puzzle_dir` (use a **model-specific path** to avoid collisions with other models), `pruning.intermediate_size_list`
- Pruning YAML: points `layer_descriptor._target_` at the new `FFNIntermediateLayerDescriptor` subclass

Choose `intermediate_size_list` by scaling the Llama-3.1-8B ratios (~21%, 42%, 60%, 83% of teacher) to the new model's `intermediate_size`.

### Step 4 — Verify registration

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
from modelopt.torch.puzzletron.anymodel.converter import ConverterFactory
print('descriptor:', '<model_type>' in ModelDescriptorFactory.CLASS_MAPPING)
print('converter: ', '<model_type>' in ConverterFactory.CLASS_MAPPING)
"
```

Both must print `True`. If not, check the `__init__.py` import chain and the `@register_decorator` keys.

### Step 5 — Tell the user what was created

List the files created, confirm registration, and suggest running the pipeline:

```text
run puzzletron all for <model_name> on <N> GPUs
```

## Command: distill

- If the second word is not exactly `run`, `list`, `summary`, `progress`, or `tokenize`, tell the user: "Unknown distill sub-command. Available: `run`, `list`, `summary`, `progress`, `tokenize`." and **STOP**.

### distill run

Parse args:
- `--puzzle_dir <dir>` — optional; auto-discovered if omitted
- `--ratio <r>` — optional float (e.g. `0.9`); required if multiple sweep ratios exist
- `--nproc_per_node <n>` — number of GPUs per node; required. If missing, ask: "Please provide the number of GPUs per node (--nproc_per_node)." and **STOP**.
- `--train_iters <n>` — number of training iterations; optional. If omitted and `--data_paths` is provided, auto-compute as one epoch (see Step 1b). If omitted and `--use_mock_data` is provided, ask: "Please provide the number of training iterations (--train_iters)." and **STOP**.
- `--use_mock_data` — flag; use mock/dummy data instead of a real dataset
- `--data_paths <p>...` — one or more tokenized data paths (weight1 path1 weight2 path2 ...); mutually exclusive with `--use_mock_data`
- `--mbs <n>` — micro-batch size; optional, default `1`
- `--gbs <n>` — global batch size; optional, default equals `nproc_per_node` (one sample per GPU)
- `--output_dir <dir>` — optional override for the output directory (default: resolved from puzzle_dir + ratio). Use when running multiple datasets for the same ratio (e.g. `--output_dir /workspace/puzzle_dir_llama3_2-3b/distillation/0.9x-nemotron`).

If neither `--use_mock_data` nor `--data_paths` is provided, ask: "Please provide either --use_mock_data or --data_paths <paths>." and **STOP**.

**Step 1 — Resolve paths** by running:

**Step 1b — Auto-compute `train_iters` if omitted and `--data_paths` is provided:**

```bash
python3 -c "
import os, sys
args = sys.argv[1:]
# args: weight1 path1 weight2 path2 ... gbs seq_len
gbs, seq_len = int(args[-2]), int(args[-1])
data_paths = args[:-2][1::2]  # extract paths from weight/path pairs
total_tokens = sum(os.path.getsize(p + '.bin') // 4 for p in data_paths if os.path.isfile(p + '.bin'))
total_seqs = total_tokens // seq_len
print(max(1, total_seqs // gbs))
" <data_paths_space_separated> <gbs> 4096
```

Report to the user: `"Auto-computed train_iters=<N> (1 epoch, <total_seqs> sequences, gbs=<gbs>)"`. Use this value for `<train_iters>` in Step 2.

```bash
python3 .agents/skills/puzzletron/distill_resolve.py [<puzzle_dir>] [--ratio <ratio>]
```

(Include `<puzzle_dir>` positionally if provided; include `--ratio <ratio>` if provided.)

Capture stdout and `eval` it to obtain shell variables: `STUDENT_PATH`, `TEACHER_PATH`, `OUTPUT_DIR`, `HF_EXPORT_PATH`, `RATIO_LABEL`. If the script exits non-zero, show its stderr to the user and **STOP**.

If `--output_dir` was provided, override `OUTPUT_DIR` and set `HF_EXPORT_PATH=<output_dir>/hf`.

**Step 2 — Run distillation:**

Compute `gbs` = `nproc_per_node` if not explicitly provided (ensures GBS is divisible by data-parallel size). Build the command:

```bash
set -o pipefail && export PYTHONPATH=$PYTHONPATH:. && export HF_HOME=/workspace/hf_cache && \
torchrun --nproc_per_node <nproc_per_node> examples/megatron_bridge/distill.py \
  --student_hf_path <STUDENT_PATH> \
  --teacher_hf_path <TEACHER_PATH> \
  --output_dir <OUTPUT_DIR> \
  --hf_export_path <HF_EXPORT_PATH> \
  --student_hf_model <TEACHER_PATH> \
  --train_iters <train_iters> \
  --mbs <mbs> \
  --gbs <gbs> \
  --lr_warmup_iters 50 \
  --eval_interval 100 \
  --log_interval 10 \
  [--use_mock_data | --data_paths <data_paths>] \
  2>&1 | tee <OUTPUT_DIR>/log.txt
```

Replace bracketed placeholders with resolved values. Use `--use_mock_data` or `--data_paths <data_paths>` (space-separated) depending on which was provided.

**Run in background** — distillation takes hours. Launch the torchrun command with `run_in_background=true` so the agent stays responsive. Tell the user the log file path (`<OUTPUT_DIR>/log.txt`) immediately.

**Show progress immediately after launch** — once the background job is started, wait ~30 seconds for initialization, then run:

```bash
python3 .agents/skills/puzzletron/distill_progress.py [<puzzle_dir>]
```

Show the progress output to the user. If the run hasn't written its first iteration yet, tell the user it's still initializing and to check again shortly with `/puzzletron distill progress`.

**Key gotchas:**
- Always set `HF_HOME=/workspace/hf_cache` — the default `/tmp` cache fills up quickly on this machine.
- Always pass `--student_hf_model <TEACHER_PATH>` (the local teacher path), NOT a HuggingFace hub model ID — hub downloads fail due to `/tmp` space limits.
- `gbs` must be divisible by `nproc_per_node`; defaulting to `nproc_per_node` is safe for smoke tests.

### distill list

Parse `puzzle_dir` from args (third word or `--puzzle_dir <value>`). It is optional.

Run the following Bash command, including `<puzzle_dir>` as an argument when provided, or omitting it to trigger auto-discovery:

```bash
python3 .agents/skills/puzzletron/distill_list.py [<puzzle_dir>]
```

Present the output to the user wrapped in a fenced code block (``` ... ```).

### distill summary

Parse `puzzle_dir` from args (third word or `--puzzle_dir <value>`). It is optional.

Run the following Bash command, including `<puzzle_dir>` as an argument when provided, or
omitting it to trigger auto-discovery:

```bash
python3 .agents/skills/puzzletron/distill_summary.py [<puzzle_dir>]
```

Present the output to the user wrapped in a fenced code block (``` ... ```). The command
uses the argument dump in each `log.txt`, saved checkpoint directories, HF MMLU result
JSONs, and allocated file sizes. It does not infer recipes from directory names. Full MMLU
results take precedence over limited smoke tests. Use the reported matched-recipe groups
when making dataset comparisons; runs outside the same group differ in at least one model,
optimizer, schedule, validation, logging, or parallelism field.

### distill progress

Parse args:
- `puzzle_dir` — third word or `--puzzle_dir <value>`; optional
- `--ratio <r>` — optional; filter to a specific ratio

Run the following Bash command, including arguments as applicable:

```bash
python3 .agents/skills/puzzletron/distill_progress.py [<puzzle_dir>] [--ratio <ratio>]
```

Present the output to the user wrapped in a fenced code block (``` ... ```).

The output shows for each run:
- Dataset name(s) and token count processed so far / total (from `run_config.yaml` in the latest checkpoint)
- Status (RUNNING / STOPPED / DONE), current iteration out of total, and ETA with elapsed/remaining time (running only)
- ASCII sparklines for training/validation objective loss and validation student CE (higher bar = higher loss, so a descending curve means improvement)
- Convergence verdict: `CONVERGING` (>2% improvement over last 3 checkpoints), `DIMINISHING RETURNS` (0.5–2%), `PLATEAU` (<0.5%), or `DIVERGING`

Example output for a running job:

```text
  Ratio:      0.9x-nemotron-full_correct_dataset
  Output dir: /workspace/puzzle_dir_llama3_2-3b/distillation/0.9x-nemotron-full_correct_dataset
  Dataset:    nvidia/Nemotron-Post-Training-Dataset-v2_default_math, nvidia/Nemotron-Post-Training-Dataset-v2_default_stem
  Tokens:     49.8M / 124.1M  (GBS=8, seq=4096)
  Status:     RUNNING  (iter 1520/3787)
  Started:    11:59:35
  Elapsed:    9m 36s
  Iter time:  0.3s/iter (avg last 5)
  Remaining:  ~10m 58s (2267 iters left)
  HF export:  not yet
  Log file:   /workspace/puzzle_dir_llama3_2-3b/distillation/0.9x-nemotron-full_correct_dataset/log.txt

  Train loss: █▇▆▅▅▄▄▃▃▃▂▂▂▁▁▁  (0.335 → 0.235)
  Val loss:   ▇█▅▆▄▃▂▃▄▃▁▁▁▁▁  (0.336 → 0.241)
  Convergence: CONVERGING  (-4.2% over last 3 checkpoints)
  Student CE: █▇▆▅▅▄▄▃▃▃▂▂▂▁▁  (2.840 → 2.410)
```

Example output for a completed job (Dataset and Tokens always shown when `run_config.yaml` is present):

```text
  Ratio:      0.9x-nemotron-full_correct_dataset
  Output dir: /workspace/puzzle_dir_llama3_2-3b/distillation/0.9x-nemotron-full_correct_dataset
  Dataset:    nvidia/Nemotron-Post-Training-Dataset-v2_default_math, nvidia/Nemotron-Post-Training-Dataset-v2_default_stem
  Tokens:     124.1M / 124.1M  (GBS=8, seq=4096)
  Status:     DONE (HF exported)
  Checkpoints: iters 3400, 3500, 3600, 3700, 3787
  HF export:  /workspace/puzzle_dir_llama3_2-3b/distillation/0.9x-nemotron-full_correct_dataset/hf

  Train loss: █▇▇▆▆▆▅▅▅▅▄▄▄▄▄▃▃▃▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁  (0.335 → 0.157)
  Val loss:   ▇█▆▇▆▅▅▅▅▅▄▄▄▄▄▄▃▃▂▃▃▂▂▂▁▂▂▂▁▁▁▁▁▁▁▁▁▁  (0.336 → 0.158)
  Convergence: DIVERGING  (+0.0% over last 3 checkpoints — consider stopping)
  Student CE: █▇▆▅▅▄▄▃▃▃▂▂▂▁▁  (2.840 → 2.410)
```

**Loss data sources:**
- **Running job**: parsed in real-time from `<output_dir>/log.txt` — objective loss matches `iteration <N>/<total> ... total loss: <val>` and `validation loss at iteration <N> ... total loss value: <val>`; student CE matches `lm loss value: <val>` on validation lines
- **Stopped/done jobs**: read from `<output_dir>/tb_logs/` TensorBoard event files via `EventAccumulator`

### distill tokenize

Tokenize a HuggingFace dataset into Megatron binary format (`.bin` / `.idx`) for use with `distill run --data_paths`.

Parse args:
- `--hf_dataset <name>` — HuggingFace dataset repo ID (e.g. `nvidia/Nemotron-Post-Training-Dataset-v2`); required.
- `--hf_name <subset>` — dataset config/subset name (e.g. `default`, `General`); optional.
- `--hf_split <split>` — dataset split (e.g. `train`, `stem`, `math`); optional, defaults to all splits.
- `--hf_max_samples <n>` — cap samples per split; optional.
- `--output_dir <dir>` — output directory for `.bin`/`.idx` files; required.
- `--tokenizer <path>` — local HF tokenizer path; required. If missing, ask the user.
- `--json_keys <key>` — `text` for pretraining data, `messages` for chat/instruction data; default `text`.
- `--workers <n>` — tokenization workers; optional, default `32`.

Run the following Bash command, substituting parsed values. Include `--append_eod` only when `--json_keys text`:

```bash
HF_TOKEN=<token_if_needed> HF_HOME=/workspace/hf_cache \
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
  --hf_dataset <hf_dataset> \
  [--hf_name <hf_name>] \
  [--hf_split <hf_split>] \
  [--hf_max_samples_per_split <hf_max_samples>] \
  --hf_streaming \
  --json_keys <json_keys> \
  --tokenizer <tokenizer> \
  --output_dir <output_dir> \
  --workers <workers> \
  --max_sequence_length 256_000 \
  [--append_eod]
```

When the command finishes, report the output prefixes (printed by the script) and the `.bin` file size in MB + token count (file size ÷ 4).

**Key gotchas:**
- `HF_TOKEN` must be set explicitly in the command prefix — exporting it in the shell is not enough for background jobs.
- `gated: manual` datasets (e.g. `Nemotron-Pretraining-SFT-v1`) require NVIDIA to manually approve access before downloading.
- `gated: auto` datasets (e.g. `Nemotron-Post-Training-Dataset-v2`) grant access immediately after accepting terms on the Hub page.
- Token count = `.bin` file size ÷ 4 bytes. Aim for ≥2.5B tokens for a meaningful run at GBS=768, seq_len=8192 (400 iters).

# End-to-End Walkthrough: Qwen3-8B DSpark (Data Synthesis → Streaming Training)

A complete worked example of training a speculative-decoding drafter, from raw
prompts to an evaluated checkpoint. It uses **Qwen3-8B** as the target and
**DSpark** as the draft algorithm, driven end-to-end by the
[launcher](../../tools/launcher/).

Qwen3-8B is small enough to run the whole flow on a handful of GPUs, so this
doubles as the recommended first run before scaling to a large MoE target — the
pipeline shape is identical, only node counts and a few model-specific fields
change.

## Why these two steps

**Data synthesis.** A drafter is trained to predict what the *target model*
would say. Off-the-shelf SFT corpora contain some other model's answers, so
training on them teaches the drafter the wrong distribution and acceptance
length suffers. Step 1 therefore keeps the prompts but regenerates every
assistant turn with the target model itself.

**Streaming training.** DSpark trains against the target's hidden states. The
classic route dumps them to disk first, which is slow and enormous. Streaming
mode instead runs a live `vllm serve` and ships hidden states to the trainer
over NIXL RDMA — no dump, no round-trip.

## What DSpark is

DSpark = the DFlash backbone + a lightweight sequential (**Markov**) head + a
**confidence** head. The Markov head adds a prefix-dependent transition bias to
the backbone's base logits, which induces a causal block distribution and lets
the draft generate a block semi-autoregressively. The confidence head predicts
per-position acceptance. It trains with a three-term loss:

```text
dflash_ce_loss_alpha * CE  +  dflash_l1_loss_alpha * TVD  +  dflash_confidence_head_alpha * BCE
```

Head architecture and loss weights live in
[`dspark.yaml`](../../modelopt_recipes/general/speculative_decoding/dspark.yaml);
the launcher YAMLs below only override data and schedule fields.

## Prerequisites

- A Slurm cluster with the launcher configured (see
  [launcher docs](../../tools/launcher/docs/configuration.md)).
- `Qwen/Qwen3-8B` present under the launcher's `/hf-local/` mount.
- Two container images: a vLLM image (serve + training) and a TensorRT-LLM image
  (dataset build). Both are pinned in the YAMLs.

Set your cluster environment once:

```bash
export SLURM_HOST=localhost          # run on the login node
export SLURM_ACCOUNT=<your_account>
export SLURM_PARTITION=<your_partition>
export SLURM_HF_LOCAL=<hf_models_dir>
export SLURM_JOB_DIR=<experiments_dir>
export NEMORUN_HOME=$PWD
```

---

## Step 1 — Data synthesis

Regenerate assistant turns with Qwen3-8B over a prompt corpus.

```bash
cd tools/launcher
uv run launch.py --yaml examples/Qwen/Qwen3-8B/hf_synth.yaml --yes
```

[`hf_synth.yaml`](../../tools/launcher/examples/Qwen/Qwen3-8B/hf_synth.yaml)
runs a Slurm **array job**. Each task starts its own `vllm serve` on one node and
calls [`common/query.py`](../../tools/launcher/common/query.py) on its own shard
of the dataset. `query.py` replays each sample's user turns, generates a fresh
assistant reply, **discards the original assistant turns**, and writes
`train-{shard}-{total}.jsonl`.

The knobs worth knowing:

| Field | Meaning |
|---|---|
| `--data` | Input prompt corpus (HF id or local path). Ships pointing at `nvidia/Speculative-Decoding-Multilingual-Prompt-v2`. |
| `--save` | Output dir. Set `output_dir` in `global_vars`. |
| `--num-shards` / `array` | Shard count and the array range. Keep them consistent. |
| `--num-proc` | Per-worker request concurrency. |
| `--tensor-parallel-size` | Must match `gpus_per_node`. |

Practical notes, mostly learned the hard way:

- **Resumable by construction.** A shard whose output file already exists is
  skipped, so a requeued or re-dispatched job picks up where it stopped. Write
  `--save` to a persistent mount, not to scratch.
- **Concurrency vs. yield.** Pushing `--num-proc` very high can *lower* total
  yield: requests start timing out under the stampede and those samples come out
  user-only. If you see a yield dip, lower it before raising it.
- **Reasoning models need context headroom.** With `--max-model-len` too tight,
  any prompt longer than the generation cap gets a 400 back and is dropped. This
  loss is *systematic* — it removes the longest, hardest prompts and quietly
  biases the corpus short. Keep `max-model-len >= max_prompt + max_output`.
- **Check yield before training.** Count records that actually have an assistant
  turn. A corpus that is silently 60% prompt-only will train, and the result will
  just be mysteriously bad.

Prefer to skip synthesis on a first pass? `task_0` of the training YAML builds a
conversation set from public SFT data, and the pipeline runs standalone. Expect a
lower acceptance length — that gap is exactly what this step buys.

---

## Step 2 — Streaming DSpark training

```bash
cd tools/launcher
uv run launch.py --yaml examples/Qwen/Qwen3-8B/hf_streaming_dspark.yaml --yes
```

[`hf_streaming_dspark.yaml`](../../tools/launcher/examples/Qwen/Qwen3-8B/hf_streaming_dspark.yaml)
is a three-task pipeline:

| Task | Nodes | What it does |
|---|---|---|
| `task_0` | 1 | Build input conversations → `/scratchspace/data/train.jsonl` |
| `task_1` | 2 | Node 0 `vllm serve`, node 1 trainer; exports to `/scratchspace/export` |
| `task_2` | 1 | vLLM smoke test — acceptance length |

**To train on your synthesized corpus** (the point of Step 1), edit the YAML: set
`skip: true` on `task_0`, and point `data.data_path` in `task_1` at the synthesis
`output_dir`:

```yaml
  task_0:
    skip: true
    ...

  task_1:
    args:
      ...
      - data.data_path=/hf-local/modelopt/qwen3-8b-synth-v1
```

CLI overrides use dotted paths for scalar fields, e.g.:

```bash
uv run launch.py --yaml examples/Qwen/Qwen3-8B/hf_streaming_dspark.yaml \
  pipeline.task_1.slurm_config.nodes=4 --yes
```

Always preview the resolved config before submitting:

```bash
uv run launch.py --yaml examples/Qwen/Qwen3-8B/hf_streaming_dspark.yaml --dryrun --yes -v
```

### The fields that actually matter

**Capture ids.** `EAGLE_CAPTURE_IDS` selects which target layers the serve
captures. The draft consumes 5 evenly-spaced layers plus the final hidden state.
For Qwen3-8B (36 layers), `build_target_layer_ids(36,5) = [1,9,17,25,33]`; vLLM's
ids are those **+1**, plus the final layer:

```text
EAGLE_CAPTURE_IDS: "[2,10,18,26,34,36]"
```

Getting the final id wrong is the classic silent failure: capturing the
second-to-last layer instead of the true final one trains fine, shows a healthy
loss curve, and caps acceptance length. If loss looks good but acceptance
plateaus, check this first.

No spaces in the value — nemo_run emits `export FOO=value` unquoted, so a space
splits the variable.

**Draft dims are not inherited.** The draft is an independent Qwen3 model; it
does *not* pick up the base model's GQA/FFN dims. `dspark.yaml` sets them
explicitly for Qwen3-8B. Retargeting to a different base means updating
`num_attention_heads`, `num_key_value_heads`, `head_dim` and `intermediate_size`
to match — otherwise you silently train a wrong-shaped draft.

**`answer_only_loss=false`.** The streaming corpus is prompt-only (the serve
generates the response and we capture *its* hidden states), so there is no
assistant span to mask. Train over the full sequence.

**`dflash_block_size`** is the semi-AR generation block and must divide
`training_seq_len`.

**`report_to=none`.** `dspark.yaml` defaults to tensorboard, which hard-fails if
tensorboard isn't installed in the serve container.

**Batch size and LR stay at the recipe defaults.** `dspark.yaml` ships
`per_device_train_batch_size=1`, `learning_rate=6e-4`, `warmup_ratio=0.04` —
tuned for a from-scratch draft on a single GPU. The Kimi-K2.6 and MiniMax-M3
examples override these to a larger batch and a gentle `1e-4` because they run 8
GPUs per node and warm-start a large backbone. Don't copy those numbers to
Qwen3-8B: at `training_seq_len=4096` on one GPU, a batch of 4 will OOM.

### Scaling up to a large target

The same pipeline drives large MoE targets — compare this YAML against
[`MiniMaxAI/MiniMax-M3`](../../tools/launcher/examples/MiniMaxAI/MiniMax-M3/hf_streaming_dspark_multi_node.yaml)
and
[`moonshotai/Kimi-K2.6`](../../tools/launcher/examples/moonshotai/Kimi-K2.6/hf_streaming_dspark_multi_node.yaml).
What changes:

| Concern | Qwen3-8B | Large MoE target |
|---|---|---|
| Topology | 2 nodes × 1 GPU | `SERVE_NODES` serve replicas + DDP trainers, 8 GPU/node |
| Draft dims | recipe defaults already match | must be set explicitly to match the base |
| `rope_theta` | default `1e6` matches base | pin the base's value onto the draft |
| Mask token | `151669` | a free id in that model's vocab |
| `trust_remote_code` | not needed | needed at serve *and* export |
| Chat template | not needed (`answer_only_loss=false`) | a `{% generation %}`-tagged copy if masking |

`SERVE_NODES` splits the allocation: nodes `0..SERVE_NODES-1` each run an
independent whole-node serve replica, the rest are DDP trainers, and each trainer
rank fetches its shard round-robin across replicas. See the header of
[`train_eagle_streaming.sh`](../../tools/launcher/common/eagle3/train_eagle_streaming.sh)
for the full knob list.

On clusters using EFA rather than InfiniBand, NIXL needs
`NIXL_BACKENDS=LIBFABRIC`. Note that UCX segfaults at agent init on EFA nodes, so
LIBFABRIC is required there even for single-node runs (see the commented block in
the YAML).

---

## Step 3 — Evaluate

`task_2` serves the exported drafter and reports acceptance length.

**Do not use the training-time AR eval.** `dspark.yaml` sets
`estimate_ar: false` deliberately: that eval runs the DFlash backbone *only*,
without applying the Markov head, so its number describes the backbone rather
than the model you trained. Acceptance length comes from the vLLM smoke test or
the offline [specdec_bench](../specdec_bench/) harness.

`task_2` needs a vLLM build with DSpark support. If your image predates it,
startup fails on the `--speculative-config` method name; pin a newer nightly, or
set `skip: true` and use specdec_bench on `/scratchspace/export`.

Reading the numbers: training accuracy, acceptance length, and accepted-fraction
are three different things and are easy to confuse. Acceptance length is the
deployment-facing one.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Training hangs before step 1, no error | Corpus schema issue — a prompt-only `messages` field can stall the streaming dataset silently. Inspect one record. |
| Loss curve healthy, acceptance length plateaus | Wrong final capture id (see above). |
| Trainer OOM on the serve node | Lower `SERVE_GPU_MEM_UTIL`, `SERVE_MAX_MODEL_LEN`, `SERVE_MAX_NUM_SEQS`. |
| Serve never becomes ready | Raise `SERVE_READY_TIMEOUT`; large models load slowly on cold cache. |
| `export FOO=value` splitting | A space in an env value. Remove it. |
| Synthesis yield well under 100% | Lower `--num-proc`, or raise `--max-model-len`. |

## Reference

- [`dspark.yaml`](../../modelopt_recipes/general/speculative_decoding/dspark.yaml) — head arch and loss weights
- [`hf_synth.yaml`](../../tools/launcher/examples/Qwen/Qwen3-8B/hf_synth.yaml) — synthesis
- [`hf_streaming_dspark.yaml`](../../tools/launcher/examples/Qwen/Qwen3-8B/hf_streaming_dspark.yaml) — streaming training
- [`hf_streaming_dflash.yaml`](../../tools/launcher/examples/Qwen/Qwen3-8B/hf_streaming_dflash.yaml) — DFlash, same transport
- [README](README.md) — speculative decoding overview, other algorithms
- [SLURM_prepare_data.md](SLURM_prepare_data.md) — the alternative `server_generate.py` synthesis path

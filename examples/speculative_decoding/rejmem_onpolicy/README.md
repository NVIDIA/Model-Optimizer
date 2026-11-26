# On-policy rejection-memory training of DFlash drafters

Trains a DFlash drafter on its own deployment: vLLM serves the target with the current drafter,
which reads the rows the previous verify rejected (rejection memory), and every drafting round of
those rollouts becomes one training block, with that round's memory. The loop is synchronous, so
the data is strictly on-policy:

1. each trainer rank takes its next prompts and generates them on its node's servers, with the
   deployment's sampling and speculative settings;
2. the servers' capture hook writes one record per request (committed tokens and target features,
   every round's anchor, drafts, accepted count and memory rows, and the drafter version);
3. the trainer checks that every record came from the drafter after exactly the pushes before this
   step, builds the batch (`modelopt.torch.speculative.plugins.rejmem.batch`) and takes one
   optimizer step (DDP across nodes);
4. the new drafter weights are pushed to the servers through CUDA IPC
   (`modelopt.torch.speculative.plugins.rejmem.weight_push`), which confirm the new version; only
   then does the next rollout start.

The trainer also pushes its drafter before the first rollout, so the servers' launch export does
not matter. Each prompt is used once. A record from any drafter version other than the last one
pushed aborts the run on every rank.

## Layout

| Path | Role |
|---|---|
| `train_onpolicy.py` | the loop: one torchrun rank per node |
| `rollout.py` | config schema, prompt source, HTTP client, record loading, `serve` launcher |
| `configs/qwen3.5-9b.yaml` | Qwen3.5-9B target, DFlash drafter warm-started from an SFT export |
| `launch/node.sh` | one node: 3 rollout servers (GPUs 0-2) + 1 trainer rank (GPU 3) |
| `launch/onpolicy.sbatch` | Slurm + pyxis: `node.sh` on every node |

## Configuration

The experiment YAML is a ModelOpt DFlash recipe (`model`, `training`, `dflash`; the drafter keys
must describe the init export in `dflash.dflash_init_checkpoint`) plus an `onpolicy` section, the
single source of the deployment regime for both the trainer and `python rollout.py serve`:

- `sampling`: the target's sampling, sent with every `/v1/completions` request (default T=1,
  top-p 1, top-k off). Greedy deployment: `onpolicy.sampling.temperature=0
  onpolicy.draft_sample_method=greedy`.
- `draft_sample_method`, `num_speculative_tokens`: the servers' `--speculative-config`
  (`num_speculative_tokens` must be `dflash_block_size - 1`). `server_args` adds other `vllm serve`
  flags to the ones the loop needs (rejmem worker extension, prefix caching off, IPC weight
  transfer, listening on 127.0.0.1 only), which `rollout.py serve` always sets.
- `memory`: rejection memory on or off, in the servers (`MODELOPT_REJMEM_CONSUMER`) and in training.
  Off gives the on-policy baseline without memory; records carry the server's setting and a
  mismatch with the trainer is an error.
- `data_dir`, `max_prompt_len`, `prompts_per_step`: single-turn prompts (system + first user turn)
  of a jsonl corpus of `{"conversations": [...]}` rows, rendered with the target's chat template,
  thinking on when the reference reply has a `</think>`. Each response may use the rest of
  `training_seq_len`. `prompts_per_step` is per node; the global batch is that times the nodes.
  Each node shuffles the rows of 64 shards at a time, so every step mixes the corpus's subsets.

Any field can be overridden on the command line as `key.path=value`.

## Running

All paths are host paths; the sbatch script mounts them at the container paths of the config.
`SERVE_PYTHONPATH` and `TRAIN_PYTHONPATH` (mounted with `MOUNTS`) add Python packages after this
repository, e.g. the Transformers version the trainer needs if the image has another one.

```bash
IMG=<vLLM 0.33.0 image .sqsh> CODE=$PWD TARGET=<Qwen3.5-9B dir> DRAFT0=<drafter export dir> \
DATA=<prompt corpus dir> OUT=<output dir> \
[MOUNTS=<host>:<ctr>,...] [SERVE_PYTHONPATH=<ctr paths>] [TRAIN_PYTHONPATH=<ctr paths>] \
[EXTRA="key.path=value ..."] \
  sbatch -N 2 -A <account> -p <partition> -t 4:00:00 \
  examples/speculative_decoding/rejmem_onpolicy/launch/onpolicy.sbatch
```

`OUT` receives `metrics.jsonl` (per step: loss, accuracy, rollout acceptance length, prompts with
thinking, prompts skipped as too long, responses cut at `max_tokens`, generated tokens, record bytes,
and the rollout / record / train / push times), the server logs and `checkpoint-<step>/`. A
checkpoint holds the drafter export (`export/`, loadable by vLLM), the optimizer and scheduler, and
each rank's prompt cursor and RNG. Submitting the same command again resumes from the latest
checkpoint, and `metrics.jsonl` keeps the steps up to it.

# VLM checkpoint evaluation

Use this evaluator to test a local Qwen 3.5 checkpoint on image and video
benchmarks. A profile chooses the benchmarks, examples, model-loading path,
and evaluation settings. Use the same profile for every model being compared.

The common choices are:

- `short-native-v2` for the maintained three-benchmark campaign screen;
- `short-vllm-v2` and `smoke-vllm-v1` for materialized heterogeneous campaign checkpoints;
- `short-all-native-v2` for a broader eight-benchmark regression screen;
- `core3-full-native-v1` and `core3-full-vllm-v1` for paired full-dataset
  teacher references on RealWorldQA, MMMU validation, and MVBench;
- `short-v1`, `short-native-v1`, `short-all-native-v1`, and `full-v1` with an
  environment that matches each profile's pinned evaluator revision.

For text-only IFEval and GSM8K evaluation, use the separate
[text checkpoint evaluator](checkpoint_evaluation.md).

## Prepare the worker environment

Run evaluation in the default Puzzletron worker image described in the
[worker environment guide](environment_setup.md). The image pins the
`lmms-eval` source revision and dependencies recorded in
`examples/puzzletron/ci_environment.json`, includes the native Qwen 3.5 image
and video backend, and preserves each task's output-token budget for vLLM. No
evaluator overlay or separate VLM requirements install is needed. Do not modify
the evaluator checkout inside the image. Preflight rejects evaluator revisions
that do not match the selected profile.

## Understand the two execution paths

| Path | Profiles | What it does |
| --- | --- | --- |
| Qwen-specific Transformers | `short-native-v1`, `short-native-v2`, `short-all-native-v1`, `short-all-native-v2`, `core3-full-native-v1` | Loads the checkpoint directly with the Qwen 3.5 model loader. Video prompts include timestamps for sampled frames. |
| General vLLM | `short-v1`, `short-vllm-v2`, `smoke-vllm-v1`, `full-v1`, `core3-full-vllm-v1` | Runs the checkpoint through vLLM and converts inputs to general image and video messages. Video prompts do not include frame timestamps. Materialized heterogeneous checkpoints require this path. |

The native and vLLM paths produce different prompts, so their scores represent
the complete paths and do not isolate the inference engine.
The heterogeneous-checkpoint `short-vllm-v2` and `smoke-vllm-v1` profiles pin
FlashAttention 2 because the runtime's FlashAttention 3 scheduler does
not support their per-layer attention geometry.

## Choose an evaluation

Use a versioned profile when scores will be compared across checkpoints:

| Profile | Coverage | Examples evaluated |
| --- | --- | --- |
| `short-v1` | RealWorldQA, MMMU, and MVBench | 344 predefined examples; MMMU uses four per subject and MVBench uses eight per category |
| `short-native-v1` | Same examples as `short-v1`, using the Qwen-specific Transformers adapter | The same 344 predefined examples with timestamps added to video prompts |
| `short-native-v2` | Fixed coverage of RealWorldQA, MMMU, and MVBench | 64 RealWorldQA rows, four rows from each of 30 MMMU subjects, and eight rows from each of 20 MVBench tasks |
| `short-all-native-v1` | All eight judge-free benchmarks, using the Qwen-specific Transformers adapter | 690 predefined examples: the same 344 plus 346 from five additional benchmarks |
| `short-all-native-v2` | Fixed coverage of all eight judge-free benchmarks | 690 predefined examples distributed across the recorded source strata |
| `core3-full-native-v1` | Native backend on RealWorldQA, MMMU validation, and MVBench | All 765, 900, and 4,000 rows respectively |
| `core3-full-vllm-v1` | vLLM backend on the same three datasets | All 765, 900, and 4,000 rows respectively |
| `full-v1` | Eight judge-free image and video benchmarks | Every available example in each pinned dataset version |

Use `short-native-v2` for campaign comparisons. Use
`short-all-native-v2` when broader image and video regression coverage is more
important than matching the campaign screen. The v1 profiles require their
pinned evaluator revisions. No short profile replaces a full-data profile for
complete benchmark reporting.

The paired `core3-full-*` profiles provide teacher references. They pin the
Qwen 3.5 0.8B Hub snapshot and
reject other checkpoints or runtime-setting overrides. Keep the two backend
results separate: their prompt construction and MVBench frame annotations
differ by design, so their score delta is not a pure inference-engine effect.

Profiles keep the evaluator, datasets, selected examples, video sampling,
answer generation, and batch size fixed. They disable Qwen thinking so the
model returns the short answer expected by these benchmarks. Puzzletron stops
before evaluation if the installed evaluator or cached data do not match the
selected profile.

The preflight report and normalized result include an `output_budget_contract`
for every selected task. It records the requested and effective output-token
budget and the adapter-specific resolution rule. In the maintained three-task
profiles, RealWorldQA and MVBench use 16 tokens and MMMU uses 128. The native
adapter applies each task budget directly. The pinned generic vLLM adapter
treats its model-level `max_new_tokens` value as a lower bound, so the report
also records that limitation; backend score differences therefore include
generation-policy and prompt-path differences, not just engine behavior.

`short-native-v2` selects
64 positions across all 765 RealWorldQA test rows, four positions within each
30-row MMMU subject, and eight positions within each 200-row MVBench task. Its
profile records the generator version, population and stratum counts, selected
index quantiles, recorded source-row identities, and a digest of those identities.
These fixed rows provide a regression screen, not a representative
full-benchmark estimate. Compare a teacher and every candidate with the same
profile.

`short-all-native-v2` adds midpoint samples from
five video benchmarks: 24 rows from each of three VideoMMMU tasks, four rows
from each of 18 Video-MME duration-and-domain strata, 68 rows across the
LongVideoBench validation split, and 10 rows from each of seven MLVU task
types. Its 64 PerceptionTest rows are balanced across the 13 observed
area-and-reasoning strata: the first 12 in lexical order receive five rows and
the last receives four. Before selecting any rows, the generated adapters
verify the exact source population size and, where the pinned metadata exposes
the selected stratification fields, the complete source stratum counts. They
also verify recorded upstream IDs for Video-MME, MLVU, and PerceptionTest.
LongVideoBench is stratified only by its validation split. This profile is a
fixed regression screen, not a full-benchmark quality estimate.

`full-v1` excludes MMVU because complete MMVU evaluation requires an external
judge. Run MMVU separately through `mmvu-smoke` or the judge-enabled `full`
suite.

These additional options are intended for setup checks and targeted diagnosis:

| Option | Coverage | Examples evaluated |
| --- | --- | --- |
| `short` | RealWorldQA and MMMU | Eight samples per task, repeated twice |
| `realworldqa-mmmu-prefix100-repeat2` | RealWorldQA and MMMU | First 100 rows of each task, repeated twice |
| `quick` | RealWorldQA, MMMU, and MVBench | 344 predefined examples from a required profile file |
| `adapter-smoke` | Video-MME and PerceptionTest | Eight samples per task |
| `video-mmmu-smoke` | VideoMMMU | Eight samples |
| `mmvu-smoke` | MMVU | Eight judge-free rows |
| `longvideobench-smoke` | LongVideoBench | Eight samples |
| `mlvu-smoke` | MLVU | Eight samples |
| `full` | All nine pinned image and video benchmarks | Complete datasets; MMVU judge calls required |

All suites pin their dataset and upstream task revisions. Generation is
deterministic. Video suites sample at 2 frames per second with at most 32
frames.

## Cache benchmark data

Evaluation reads every selected dataset revision from an explicit Hugging Face
cache root. The campaign `prepare_dataset` stage fills and validates this cache
from the configured evaluation tasks. The repository stores only profile
metadata and exact-row selectors; it does not store benchmark records or media.
To prepare the common image and video cache independently, run:

```bash
export HF_HOME=/path/to/huggingface-cache

python -m examples.puzzletron.evaluation.vlm.preparation.benchmark_data \
  --hf-home "$HF_HOME" \
  --tasks realworldqa,mmmu_val,mvbench,video_mmmu,videomme,longvideobench_val_v,mlvu_dev,perceptiontest_val_mc
```

The command downloads each listed exact pinned snapshot and safely extracts
media only for tasks that declare a preparation directory. For `short-native-v2`,
prepare `realworldqa`, `mmmu_val`, and `mvbench`. Use `--download-only` and
`--extract-only` to split transfer and extraction across jobs, or
`--range-resume` for a resumable single-writer download. Run the command with
`--help` to list all supported dataset task names. Preparation records exact
snapshot and media inventories. Ordinary resume uses file metadata as a fast
path and checks recorded hashes after metadata changes; incomplete owned media
is rebuilt from the pinned snapshot only when the host supports atomic directory
exchange. Otherwise the existing root is preserved and preparation fails.

Video suites also require an installed `decord`-compatible reader. The
Puzzletron requirements select the supported reader for the current platform
where one is available. Preflight stops before evaluation with an installation
error if the environment has no compatible reader.

## Check the setup before using a GPU

Add `--preflight-only` to any evaluation command to check the model, evaluator,
cached datasets, video files, and credentials without loading model weights or
running inference.

## Run a versioned profile

For comparison profiles, the checkpoint path is an invocation input. Use the
same command with a teacher or any materialized Qwen 3.5 student checkpoint:

```bash
python -m examples.puzzletron.evaluation.vlm.run \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/short-all-native-v2 \
  --hf-home "$HF_HOME" \
  --profile short-all-native-v2
```

The `core3-full-*` profiles instead require the exact pinned local Qwen 3.5
0.8B Hub snapshot. For example:

```bash
python -m examples.puzzletron.evaluation.vlm.run \
  --checkpoint "$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17" \
  --output-dir /path/to/results/core3-full-native-v1/realworldqa \
  --hf-home "$HF_HOME" \
  --profile core3-full-native-v1 \
  --profile-task realworldqa \
  --preflight-only
```

Use `full-v1` only with its pinned evaluator revision. Always run
`--preflight-only` before consuming GPU time.

To run tasks in parallel, run one profile task per job with
`--profile PROFILE --profile-task TASK`. This works for `full-v1`, the paired
`core3-full-*` profiles, and `short-all-native-v2`. A complete result needs one
successful job for every task in the selected profile.

Grouped tasks can use multiple batch-1 workers without changing inference
batching. Add `--profile-task-shard INDEX/COUNT` to `mvbench` or `video_mmmu`,
where `INDEX` is zero-based. For example, split MVBench as `0/8` through `7/8`
on an eight-GPU node and combine all leaf-task metrics.

## Run the default smoke evaluation

```bash
python -m examples.puzzletron.evaluation.vlm.run \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/vlm-smoke \
  --hf-home "$HF_HOME" \
  --suite short
```

Run with `--preflight-only` first. It reports missing or incompatible model
files, evaluator packages, datasets, video files, and credentials without
starting model evaluation.

The `quick` suite additionally requires `--quick-manifest` with its pinned
344-row selection. The `full` suite requires explicit
`--allow-judge-calls`, `--mmvu-judge-api-type`, and `--mmvu-judge-model`
options plus the corresponding OpenAI or Azure credentials. Other suites do
not accept judge options and use a loopback-only disabled judge configuration.

## Interpret results and limitations

Each execution creates an `attempt_<id>/` directory containing the command,
logs, raw evaluator output, and normalized metrics in `summary.json`. Repeating
a run creates another attempt directory instead of overwriting earlier output.

The execution-path table above identifies each profile's adapter. The native
and generic vLLM paths construct different model inputs, so keep the profile
name with every score and do not combine scores as if only the inference engine
changed. The preflight report records backend identity, frame policy,
generation policy, and exact evaluator revision.

MMMU results also include `mmmu_parser_audit`. Its per-sample status counts use
`parsed`, `parsed_open`, `invalid_open`, and `fallback_random`, making invalid
open-ended parses and random-fallback scoring visible in each result.

If preflight fails, address the reported checkpoint, revision, cache, decoder,
or credential mismatch before retrying. Inspect `stderr.txt` in the attempt
directory when the evaluator subprocess starts but does not complete.

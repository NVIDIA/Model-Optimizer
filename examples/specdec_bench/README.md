# Speculative Decoding (SpecDec) Bench

## Installation

This benchmark is meant to be a lightweight layer on top of an existing vLLM/SGLang/TRTLLM installation. For example, no install
is required if one is running in the following dockers: `vllm/vllm-openai:v0.11.0` (vLLM), `lmsysorg/sglang:v0.5.4.post2` (SGLang), or
`nvcr.io/nvidia/tensorrt-llm/release:1.2.0` (TRT-LLM).

Next

```bash
cd examples/specdec_bench
```

## Purpose

Collect relevant metrics on acceptance rate, timing, and outputs for Speculative Decoding methods.
Acceptance rate refers to the number of tokens generated on every iteration.  For a standard Autoregressive LLM, this number
is just 1.

## Getting Started

A basic example run script is provided which benchmarks MTBench (a standard 160 prompts spanning 8 categories).
MTBench is available [here](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts)

### Running MTBench on GPT OSS + Eagle3

Download `nvidia/gpt-oss-120b-Eagle3` to a local directory `/path/to/eagle`.

```bash
python3 run.py \
    --model_dir openai/gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --draft_model_dir /path/to/eagle \
    --mtbench question.jsonl \
    --tp_size 1 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --num_requests 80 \
    --engine TRTLLM \
    --concurrency 1 \
    --postprocess gptoss
```

### Running Random ids on GPT OSS + Eagle3

Download `nvidia/gpt-oss-120b-Eagle3` to a local directory `/path/to/eagle`.

```bash
python3 run.py \
    --model_dir openai/gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --draft_model_dir /path/to/eagle \
    --random_isl 1024 \
    --tp_size 1 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --num_requests 40 \
    --engine TRTLLM \
    --concurrency 1
```

### Running [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) on Llama 3.3 70B + Eagle 3

The `prepare_data.py` script now supports SPEED-Bench v1.1. SPEED-Bench v1.0 is no longer supported.

1. Install the requirements file using `pip install -r requirements.txt`

2. Prepare the data using the provided script:

```bash
python3 prepare_data.py --dataset speed --config all
```

The data will be saved to `data/` directory, each config type (qualitative, throughput_1k, ...) to each own directory.

#### License

GOVERNING TERMS: This dataset is governed by the NVIDIA Evaluation Dataset License Agreement.

ADDITIONAL INFORMATION: MIT for bigcode/humanevalpack, RUCAIBox/MMATH, RUCAIBox/BAMBOO and EQ-Bench. Apache 2.0 for Writing Bench and Spec-Bench. CC BY 4.0 for FBK-MT/MCIF. MIT and Apache 2.0 for tianyang/repobench_python_v1.1, JetBrains-Research/lca-project-level-code-completion and tianyang/repobench_java_v1.1.

NOTICE: For each dataset a user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose. The `prepare_data.py` script automatically fetches data from all the source datasets.

Additional details are in [HuggingFace dataset repository](https://huggingface.co/datasets/nvidia/SPEED-Bench).

#### Qualitative split

```bash
python3 run.py \
    --model_dir meta-llama/Llama-3.3-70B-Instruct \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --draft_model_dir yuhuili/EAGLE3-LLaMA3.3-Instruct-70B \
    --dataset speed \
    --dataset_path data/speed/qualitative \
    --tp_size 8 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --engine TRTLLM \
    --concurrency 32 \
    --show_progress
```

#### Throughput split

```bash
python3 run.py \
    --model_dir meta-llama/Llama-3.3-70B-Instruct \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --draft_model_dir yuhuili/EAGLE3-LLaMA3.3-Instruct-70B \
    --dataset speed \
    --dataset_path data/speed/throughput_1k \
    --tp_size 8 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --engine TRTLLM \
    --concurrency 32 \
    --show_progress
```

For longer context (>8192 tokens), please use the following configuration when using TRTLLM:

```yaml
engine_args:
  max_seq_len: 131072   # Model max context length (for Llama 3.3 70B)
  enable_chunked_prefill: true
```

```bash
python3 run.py \
    --model_dir meta-llama/Llama-3.3-70B-Instruct \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --draft_model_dir yuhuili/EAGLE3-LLaMA3.3-Instruct-70B \
    --dataset speed \
    --dataset_path data/speed/throughput_16k \
    --tp_size 8 \
    --ep_size 1 \
    --draft_length 3 \
    --output_length 4096 \
    --engine TRTLLM \
    --concurrency 32 \
    --show_progress \
    --runtime_params runtime_args_long_context.yaml
```

## Measuring acceptance and throughput with AIPerf

`run.py` drives an in-process engine and derives acceptance from its own token
bookkeeping. That is accurate, but it makes a number comparable only to other
`specdec_bench` numbers, and it cannot measure the serving stack under load.
`aiperf/run_aiperf.sh` measures both through a served vLLM endpoint and
[AIPerf](https://github.com/ai-dynamo/aiperf), the standard LLM benchmarking
client, in a layout the Speed visualizer can publish directly.

Both paths are supported; neither replaces the other. Use `run.py` when you want
a single self-contained process and no server; use the AIPerf path when the
number has to line up with other AIPerf measurements, when you need throughput
under concurrency, or when you want to publish to the Speed view.

Requires a vLLM build exposing `--per-request-spec-decode-metrics` and `aiperf`
built from main (the `spec_al_*` and `speed_bench_*` public datasets are not in
a released tag yet).

```bash
MODEL=/path/to/target \
DRAFT_MODEL=/path/to/draft \
MODEL_REVISION=<sha> DRAFT_REVISION=<sha> \
SPEC_METHOD=dflash NUM_SPEC_TOKENS=3 \
TP_SIZE=4 GPU_COUNT=4 \
DATASETS="spec_al_mtbench" \
CONCURRENCIES="1" \
OUTPUT_LENGTH=4096 TEMPERATURE=0 \
ARTIFACT_ROOT=./aiperf_results \
    bash aiperf/run_aiperf.sh
```

Checkpoints that ship custom modelling code (MiniMax, many MoE variants) also
need `TRUST_REMOTE_CODE=1`. It is off by default because it lets the
checkpoint's own Python execute, which is not a safe unconditional default when
`MODEL` may name a remote repo.

The script probes `metrics.speculative_decoding` on one request before the
sweep, and requires that block to be non-empty *and* non-zero. A build without
the flag produces records with no acceptance field; a build that accepts the
speculative config and then ignores it produces records reporting AL ~1.0,
which reads as a real measurement of a useless draft. Both fail before any
sweep time is spent.

### Datasets

Any AIPerf public dataset, named in `DATASETS`:

- acceptance: `spec_al_mtbench`, `spec_al_gsm8k`, `spec_al_humaneval`,
  `spec_al_math500`, `spec_al_mbpp`
- SPEED-Bench categories: `speed_bench_qualitative`, `speed_bench_coding`,
  `speed_bench_math`, `speed_bench_reasoning`, `speed_bench_qa`,
  `speed_bench_rag`, `speed_bench_stem`, `speed_bench_writing`,
  `speed_bench_roleplay`, `speed_bench_humanities`,
  `speed_bench_summarization`, `speed_bench_multilingual`
- SPEED-Bench throughput shapes: `speed_bench_throughput_{1k,2k,8k,16k,32k}`,
  each also in `_low_entropy` / `_high_entropy` / `_mixed` variants

### Acceptance versus throughput

Acceptance is a per-step property, so measure it at `CONCURRENCIES="1"`, where
no queueing distorts the per-step statistics. Throughput is the opposite: it
only means something under load, so sweep it. `CONCURRENCIES` defaults to
`1 4 16 32 64 128 256 512`, the Speed view's `batch` axis.

The server starts once and stays up for every dataset and concurrency.
Restarting per point would waste the allocation and risk a different serving
configuration between cells, which the publisher rejects.

Set `SPEC_METHOD=none` for a matched baseline arm: same hardware, image,
precision, parallelism, workload and budgets, with speculation off.

### Publishing

The script writes one directory per dataset (a *sweep*) containing one
subdirectory per concurrency (a *cell*):

```text
${ARTIFACT_ROOT}/<dataset>/
  c1/
    metadata.json                     client type, serving command, hardware
    recipe.json                       model/draft revisions, GPU count
    aiperf0_aiperf_results/
      profile_export_aiperf.json      AIPerf's own export, untouched
  c4/ ...
```

That is what [specdec-visualizer](https://modelopt.nvidia.com/speed)'s
publishers consume:

```bash
# Throughput / Pareto -> MLflow experiment specdec_bench/btk_results
python3 btk_results.py ${ARTIFACT_ROOT}/<dataset> --name <sweep-name> --publish

# Acceptance -> MLflow experiment specdec_bench/results
python3 acceptance_results.py ${ARTIFACT_ROOT}/<dataset>/c1/aiperf0_aiperf_results \
    --manifest <manifest>.json --publish
```

Pass `--al-run-id` to `btk_results.py` to tie a throughput sweep to the
acceptance run it should be read with. Datasets are the outer loop because the
publisher treats the dataset as part of a sweep's workload identity and rejects
a directory mixing them; concurrency is the dimension allowed to vary.

Uploading the BTK archive to the IB platform is a separate step. `btk_results.py`
records the linkage from a `.btk_uploaded` receipt and does not upload itself.

### Cross-validating against the in-process harness

`aiperf/compare_harnesses.py` reports both AL conventions from the two harnesses side by side:

```bash
python3 aiperf/compare_harnesses.py \
    --specdec-dir /path/to/run_py_save_dir \
    --aiperf-dir  /path/to/aiperf_artifact_dir
```

Running both on one node (MiniMax-M2.7 + DFlash K=3, MT-Bench, greedy, 4096-token generations,
identical prompts, 4xH100):

```text
harness                   per-request AL   pooled AL       steps
----------------------------------------------------------------
specdec_bench                     1.9798      1.7506      193329
aiperf                            2.0413      1.7865      174514

emitted tokens per step (share of steps):
  length                        1        2        3        4
  specdec_bench            56.3%    22.3%    11.5%     9.9%
  aiperf                   54.4%    23.1%    12.1%    10.5%

pooled AL delta: +0.0359 (2.0% of specdec_bench)
```

Two means are reported because they answer different questions and are easy to confuse.
**Per-request AL** averages over requests, counting each equally; **pooled AL** divides total
emitted tokens by total verification steps, so long requests dominate. They diverge whenever
acceptance correlates with response length, which it usually does — quoting one against the
other is a common way to manufacture a discrepancy that isn't there.

The harnesses also bin differently: `specdec_bench` histograms tokens *emitted* per step
(1..K+1, bonus token included), AIPerf histograms drafts *accepted* per step (0..K). The script
normalises AIPerf's to the emitted-length convention so the rows line up.

## Uploading results to S3

Each `run.py` invocation writes a result directory containing `configuration.json`,
`timing.json`, `acceptance_rate.json`, and (when applicable) `mtbench.json` / `specbench.json`.
`upload_to_s3.py` is a single-file, drop-in tool that uploads one run — or an entire sweep —
to any S3-compatible bucket:

```bash
python upload_to_s3.py /path/to/run_or_sweep_dir s3://your-bucket/some/prefix \
    --endpoint https://your-s3-endpoint \
    --key-id YOUR_KEY_ID \
    --secret YOUR_SECRET
```

`--endpoint`, `--key-id`, and `--secret` default to the `S3_ENDPOINT`, `S3_KEY_ID`, and
`S3_SECRET` environment variables. Omit `--endpoint` (or set `S3_ENDPOINT=""`) to use AWS S3's
default endpoint. Use `--dry-run` to preview the upload plan, and `--skip-existing` to skip
runs already present at the destination instead of failing.

The tool handles two directory layouts and mirrors them into S3:

- **Flat** — `LOCAL_DIR/run_name/{configuration,timing,...}.json`
- **Sweep** — `LOCAL_DIR/sweep_name/run_name/{configuration,timing,...}.json`

`LOCAL_DIR`'s basename is preserved in the destination prefix, so re-uploads from the same
source land in the same place.

## Notes

The goal of this benchmark is to provide an easy way to configure, run, and compare speculative implementations across frameworks in an apples-to-apples method.
This benchmark sends request in a single-threaded fashion, so running large concurrency (>256) may result in python async scheduling delays and skew metrics.
If larger concurrency is needed, it is recommended to fully deploy the model using `vllm serve`, `python -m sglang.launch_server`, or `trtllm-serve` (for vLLM, SGlang, or TRTLLM respectively) and
use a more robust benchmarking client like NVIDIA AI Perf.

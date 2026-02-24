# Speculative Decoding (SpecDec) Bench

## Installation

This benchmark is meant to be a lightweight layer ontop of an existing vLLM/SGLang/TRTLLM installation. For example, no install
is required if one is running in the following dockers: `vllm/vllm-openai:v0.11.0` (vLLM), `lmsysorg/sglang:v0.5.4.post2` (SGLang), or
`nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4` (TRT-LLM).

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

1. Install the requirements file using `pip install -r requirements_speed.txt`

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

## Notes

The goal of this benchmark is to provide an easy way to configure, run, and compare speculative implementations across frameworks in an apples-to-apples method.
This benchmark sends request in a single-threaded fashion, so running large concurrency (>256) may result in python async scheduling delays and skew metrics.
If larger concurrency is needed, it is recommended to fully deploy the model using `vllm serve`, `python -m sglang.launch_server`, or `trtllm-serve` (for vLLM, SGlang, or TRTLLM respectively) and
use a more robust benchmarking client like NVIDIA AI Perf.

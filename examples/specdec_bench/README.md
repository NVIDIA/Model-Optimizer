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
python3 run.py --model_dir openai/gpt-oss-120b --tokenizer openai/gpt-oss-120b --draft_model_dir /path/to/eagle --mtbench question.jsonl --tp_size 1 --ep_size 1 --draft_length 3 --output_length 4096 --num_requests 80 --engine TRTLLM --concurrency 1 --postprocess gptoss

```

### Running Random ids on GPT OSS + Eagle3

Download `nvidia/gpt-oss-120b-Eagle3` to a local directory `/path/to/eagle`.

```bash
python3 run.py --model_dir openai/gpt-oss-120b --tokenizer openai/gpt-oss-120b --draft_model_dir /path/to/eagle --random_isl 1024 --tp_size 1 --ep_size 1 --draft_length 3 --output_length 4096 --num_requests 40 --engine TRTLLM --concurrency 1

```

## Notes

The goal of this benchmark is to provide an easy way to configure, run, and compare speculative implementations across frameworks in an apples-to-apples method.
This benchmark sends request in a single-threaded fashion, so running large concurrency (>256) may result in python async scheduling delays and skew metrics.
If larger concurrency is needed, it is recommended to fully deploy the model using `vllm serve`, `python -m sglang.launch_server`, or `trtllm-serve` (for vLLM, SGlang, or TRTLLM respectively) and
use a more robust benchmarking client like NVIDIA AI Perf.

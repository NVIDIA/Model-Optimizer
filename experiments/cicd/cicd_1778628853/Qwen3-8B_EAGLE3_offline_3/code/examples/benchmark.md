# Model Optimizer Benchmark Reference

This document summarizes performance and accuracy measurements of [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) for a few popular models.
The benchmark in the following tables is provided as reference points and **should not be considered as the peak
performance** that can be delivered by Model Optimizer. All performance numbers are tested with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) or [TensorRT](https://developer.nvidia.com/tensorrt-getting-started).

## 1. Post-training quantization (PTQ) for LLMs

### 1.1 Performance

Config: H200, nvidia-modelopt v0.21.1, TensorRT-LLM v0.15, latency measured with [trtllm-bench](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-overview.md#for-non-gh200-systems-1).
Inference speedup are compared to the BF16 baseline. **Speedup is normalized to the GPU count**.

> Benchmark scenario: Input tokens 2048, output tokens 128. Real performance may vary based on the target usecases and flags used to build the TensorRT-LLM engine.

> Memory saving is not reported here as TensorRT-LLM occupies all the remaining available GPU memory for KV caching.

> If the GPU memory is the limitation, lower bit quantization may have better GPU-count-normalized throughput gain with fewer TP.

| | | BF16 (8B:TP1, 70B:TP2) | | FP8 (TP1) | | |INT4 AWQ (TP1)| | |W4A8 AWQ (TP1)| |
|:------------:|:----------:|:----------------------:|:-:|:------------:|:-------:|:-:|:------------:|:-------:|:-:|:------------:|:-------:|
| Model | Batch Size | Tokens/sec | | Tokens/sec | Speedup | | Tokens/sec | Speedup | | Tokens/sec | Speedup |
| Llama3.1-8B | 1 | 173.80 | | 245.03 | 1.41x | | 231.75 | 1.33x | | 239.70 | 1.38x |
| | 8 | 803.11 | | 1,051.17 | 1.31x | | 599.72 | 0.75x | | 801.72 | 1.00x |
| | 64 | 1,679.74 | | 2,190.93 | 1.30x | | 1,392.78 | 0.83x | | 1,930.86 | 1.15x |
| Llama3.1-70B | 1 | 45.81 | | 43.46 | 1.90x | | 44.10 | 1.93x | | 46.31 | 2.02x |
| | 8 | 182.61 | | 182.07 | 1.99x | | 93.98 | 1.03x | | 140.02 | 1.53x |
| | 64 | 401.50 | | 420.64 | 2.10x | | 176.68 | 0.88x | | 345.43 | 1.72x |

### 1.2 Accuracy

The table below shows the MMLU loss in percentage compared to BF16 baseline.
Config: H100, nvidia-modelopt v0.21.1, TenorR-LLM v0.15.
Note that typically FP8 is the go-to choices for H100. 4-bit AWQ methods is recommended when GPU memory is a constraint.
More benchmark with earlier version of Model Optimizer can be found in this [TensorRT-LLM README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md#benchmark).

| Model | MMLU loss FP8 |MMLU loss INT4 AWQ|MMLU loss W4A8 AWQ|
|:-----------------------:|:-------------:|:----------------:|:----------------:|
| Llama3.1-8B (instruct) | 1.50% | 5.66% | 6.00% |
| Llama3.1-70B (instruct) | 0.38% | 1.07% | 1.20% |

## 2. PTQ for Stable Diffusion

The following table shows inference speedup for INT8 and FP8 on a Stable Diffusion XL 1.0 base model compared to the FP16 baseline.
Config: Image resolution=1024×1024, 30 steps. TensorRT v9.3. num-warmup-runs=1. Batch size=1.

| GPU | INT8 Latency (ms) | FP8 Latency (ms) | Speedup (INT8 v.s. FP16) | Speedup (FP8 v.s. FP16) |
|:--------------:|:-----------------:|:----------------:|:------------------------:|:-----------------------:|
| RTX 6000 Ada | 2,479.19 | 2,441.16 | 1.43x | 1.45x |
| RTX 4090 | 2,058.11 | 2,161.38 | 1.20x | 1.14x |
| L40S | 2,338.88 | 2,167.82 | 1.25x | 1.35x |

## 3. Quantization-aware training

The below table demonstrates the validation loss of Quantization-aware training (QAT) compared to PTQ of a Llama 2 7B model using nvidia-modelopt v0.11.0.
The baseline is fine-tuned on the target dataset. Note that we use INT4 to showcase that QAT can better preserve model accuracy at low precision. This implies that QAT can be applied with a low training cost, enabling generative AI applications that are sensitive to accuracy drop to preserve accuracy even at ultra-low precisions where both weight and activations are 4-bit for [NVIDIA Blackwell platform](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/).

| Method | Dataset | Val loss - BF16 Baseline | Val loss - PTQ | Val loss - QAT (lower is better) |
|:----------------------------:|:--------------------:|:------------------------:|:--------------:|:--------------:|
| INT4 Weight, FP16 Activation | samsum | 1.036 | 1.059 | **1.044** |
| INT4 Weight, INT8 Activation | samsum | 1.036 | 3.321 | **1.294** |
| INT4 Weight, FP16 Activation | databricks-dolly-15k | 1.151 | 1.305 | **1.172** |
| INT4 Weight, INT8 Activation | databricks-dolly-15k | 1.151 | 2.313 | **1.640** |

## 4. Sparsity

### 4.1 Performance

The table shows the inference speedup of a sparsified Llama 2 70B model compared to the baseline dense model in different batch sizes.
The benchmark with batch_size=896 is part of [MLPerf Inference v4.0](https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/).
Config: NVIDIA H100 80GB GPU. FP8, TP=1, PP=1 for all sparsified models. The dense model needs TP=2 due to larger weight sizes.

| Batch Size | Inference speedup (compared to the FP8 dense model) |
|:----------:|:---------------------------------------------------:|
| 32 | 1.62x |
| 64 | 1.52x |
| 128 | 1.35x |
| 896 | 1.30x |

### 4.2 Accuracy

We recommend using sparsity with fine-tuning to avoid accuracy degradation.
The following table shows the comparison of validation loss of a Llama 2 70B using sparsity with and without fine-tuning. Finetuning and validation are done on the Open-Orca dataset.

| Method | Validation loss (lower is better) |
|:--------------------------------:|:---------------------------------:|
| FP8 (baseline) | 0.721 |
| FP8 + SparseGPT, no fine-tuning | 2.724 |
| FP8 + Sparsity, with fine-tuning | **1.01** |

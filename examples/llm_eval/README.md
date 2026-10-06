# Evaluation scripts for LLM tasks

This folder includes popular 3rd-party LLM benchmarks for LLM accuracy evaluation.

The following instructions show how to evaluate the Model Optimizer quantized LLM with the benchmarks, including the TensorRT-LLM deployment.

## PyTorch KL evaluation prototype

`kl_eval.py` compares an unquantized BF16 base model with a separately calibrated fake-quant copy. Supply a model and a fixed PTQ recipe; calibration reuses the `hf_ptq.py` workflow and stops before checkpoint export. Install ModelOpt and the [hf_ptq requirements](../hf_ptq/requirements.txt) as described in the [hf_ptq setup](../hf_ptq/README.md).

Run from the repository root:

```sh
python examples/llm_eval/kl_eval.py \
    --model Qwen/Qwen3.8-27B \
    --recipe general/ptq/nvfp4_default-kv_fp8_cast \
    --output kl_results.json
```

This model/recipe combination has been validated with the default evaluation settings. The recipe includes NVFP4 weights and activations with FP8 KV-cache cast quantization. Other built-in PTQ recipe names or YAML paths can be supplied through `--recipe`.

- Evaluation uses 100 non-overlapping, 128-token windows from the tokenized WikiText-2-raw-v1 test split, selected with seed 0. Text is joined with blank lines and encoded without a chat template or added special tokens.
- The BF16 model greedily generates up to 512 tokens per prompt, stopping at EOS. Both models then receive the identical prompt and continuation. Only the generated-token predictions, including EOS, contribute to the score.
- Full-vocabulary KL is `KL(BF16 || fake-quant)` over all vocabulary tokens. Conditional top-k KL selects the BF16 model's top 128 token IDs at each position and separately normalizes both models over those same IDs. It excludes tail mass and is not an approximation with an `OTHER` bucket.
- Log-softmax and KL use FP32, in chunks of 32 positions. Each reported mean first averages positions within an example, then averages the example scores equally. Units are nats. Non-finite scores raise an error rather than being dropped.
- The JSON contains only the overall `full_vocab_kl` and `conditional_topk_kl` means by default. Add `--detailed_results` to save a report with `summary`, per-example scores and token counts, prompt/continuation token IDs, and resolved recipe and run settings. Logits are held for one example at a time and are not saved.

Override evaluation settings with `--num_examples`, `--prompt_tokens`, `--max_new_tokens`, `--top_k`, and `--seed`. Calibration is separate from WikiText evaluation and inherits `hf_ptq.py` defaults: the CNN/DailyMail + Nemotron mixture, 1,024 samples, maximum length 512, and automatic batch sizing. Override those with `--dataset`, `--calib_size`, `--calib_seq`, and `--batch_size`; these flags affect calibration only. Evaluation processes one prompt at a time.

The default [Nemotron calibration dataset](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) is gated. Authenticate with a Hugging Face account that has access before running; if using an isolated `HF_HOME`, make the token available through `HF_TOKEN_PATH`.

Run in one process with sufficient GPU memory for **two model copies**, calibration workspace, and one example's logits. The existing hf_ptq loader can place models across visible GPUs; `--gpu_max_mem_percentage` controls its budget. The initial prototype requires an unquantized BF16 checkpoint with text-only causal generation and calibration. AutoQuantize, layerwise export recipes, and distributed evaluation are excluded. `--trust_remote_code` is opt-in, and `--attn_implementation` is forwarded to the hf_ptq loader.

## NeMo Evaluator

[NeMo Evaluator](https://docs.nvidia.com/nemo/evaluator/latest/get-started/quickstart/index.html#self-hosted-options) is the recommended way to evaluate a large choice of benchmarks on quantized checkpoints generated from [hf_ptq](../hf_ptq). Quantized checkpoints can be served with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), or [SGLang](https://github.com/sgl-project/sglang) and then evaluated using NeMo Evaluator.

## LM-Eval-Harness

[LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness) provides a unified framework to test generative language models on a large number of different evaluation tasks.

The supported eval tasks are [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

For guidance on shortening research iteration cycles while preserving meaningful model comparisons, see
[ModelOpt for Researchers: Fast Experimentation Workflows](../researcher_guide/README.md#efficient-evaluation-with-lm-eval-harness).

### Baseline

Both standard HuggingFace models and heterogeneous pruned checkpoints produced by Puzzletron are supported.

- For models which fit on a single GPU:

```sh
python lm_eval_hf.py --model hf --model_args pretrained=<HF model folder or model card> --tasks <comma separated tasks> --batch_size 4
```

For a quick smoke test, add `--limit 10` to any of the above commands to evaluate on only 10 samples per task.

- To fit one model across multiple GPUs (model sharding) and enable larger batches that may speed up evaluation:

```sh
python lm_eval_hf.py --model hf --model_args pretrained=<HF model folder or model card>,parallelize=True --tasks <comma separated tasks> --batch_size 4
```

> **Note (Slurm interactive nodes):** On Slurm interactive nodes, `WORLD_SIZE` is set to the number of available GPUs in the shell environment. Running `python` directly causes `lm_eval` to hang waiting for peer ranks that were never spawned. Prepend `WORLD_SIZE=1` to the `python` commands above to fix this. This does not limit GPU usage — `parallelize=True` independently enables model parallelism across all available GPUs within the single process. The `accelerate launch` command manages `WORLD_SIZE` itself and does not require this workaround.

- For data-parallel evaluation with model-sharding:

`--num_processes` controls how many model copies evaluate samples concurrently. More
copies usually make evaluation faster but leave fewer GPUs for each copy. With `N`
GPUs, each copy uses approximately `N / num_processes` GPUs. For example, on 8 GPUs,
8 processes run eight single-GPU copies. Choose the largest number of processes for
which each model copy fits.

```sh
accelerate launch --multi_gpu --num_processes <num_copies_of_your_model> \
    lm_eval_hf.py --model hf \
    --tasks <comma separated tasks> \
    --model_args pretrained=<HF model folder or model card>,parallelize=True \
    --batch_size 4
```

### Quantized (simulated)

- For simulated quantization with any of the default quantization formats:

Multi-GPU evaluation without data-parallelism:

```sh
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG|MXFP8_DEFAULT_CFG]
python lm_eval_hf.py --model hf \
    --tasks <comma separated tasks> \
    --model_args pretrained=<HF model folder or model card>,parallelize=True \
    --quant_cfg <MODELOPT_QUANT_CFG> \
    --batch_size 4
```

> **_NOTE:_** `MXFP8_DEFAULT_CFG` is one the [OCP Microscaling Formats (MX Formats)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) family which defines a set of block-wise dynamic quantization formats. The specifications can be found in the [official documentation](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). Currently we support all MX formats for simulated quantization, including `MXFP8 (E5M2, E4M3), MXFP6 (E3M2, E2M3), MXFP4, MXINT8`. However, only `MXFP8 (E4M3)` is in our example configurations, users can create their own configurations for other MX formats by simply modifying the `num_bits` field in the `MXFP8_DEFAULT_CFG`.

> **_NOTE:_** ModelOpt's triton kernels give faster NVFP4 simulated quantization. For details, please see the [installation guide](https://nvidia.github.io/Model-Optimizer/getting_started/_installation_for_Linux.html#accelerated-quantization-with-triton-kernels).

For data-parallel evaluation, launch with `accelerate launch --multi_gpu --num_processes <num_copies_of_your_model>` (as shown earlier).

- For simulated optimal per-layer quantization with `auto_quantize`:

Multi-GPU evaluation without data-parallelism:

```sh
# MODELOPT_QUANT_CFG_TO_SEARCH: Comma-separated list of the formats auto_quantize searches over.
# Pick each one from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG|MXFP8_DEFAULT_CFG|NONE],
# where NONE lets auto_quantize leave a layer unquantized.
# EFFECTIVE_BITS: Effective bits constraint for auto_quantize

# Example settings for an optimally quantized model with W4A8 & FP8 with effective bits of 4.8:
# MODELOPT_QUANT_CFG_TO_SEARCH=W4A8_AWQ_BETA_CFG,FP8_DEFAULT_CFG,NONE
# EFFECTIVE_BITS=4.8

python lm_eval_hf.py --model hf \
    --tasks <comma separated tasks> \
    --model_args pretrained=<HF model folder or model card>,parallelize=True \
    --quant_cfg <MODELOPT_QUANT_CFG_TO_SEARCH> \
    --auto_quantize_bits <EFFECTIVE_BITS> \
    --batch_size 4
```

For data-parallel evaluation, launch with `accelerate launch --multi_gpu --num_processes <num_copies_of_your_model>` (as shown earlier).

- If evaluating encoder-decoder models such as T5, keep `--model hf`: lm-eval detects the
  encoder-decoder architecture from `config.json`. There is no `hf-seq2seq` backend in the
  supported lm-eval versions (>= 0.4.12); add `backend=seq2seq` to `--model_args` only for
  checkpoints lm-eval cannot classify on its own.

```sh
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG|MXFP8_DEFAULT_CFG]
python lm_eval_hf.py --model hf --model_args pretrained=t5-small --quant_cfg <MODELOPT_QUANT_CFG> --tasks <comma separated tasks> --batch_size 4
```

If `trust_remote_code` needs to be true, please append the command with the `--trust_remote_code` flag.

### TensorRT-LLM

Uses the `trtllm` backend built into lm-eval (>= 0.4.12), which loads the quantized
checkpoint directly with the TensorRT-LLM LLM API.

```sh
python lm_eval_trtllm.py --model trtllm \
    --model_args model=<Quantized checkpoint dir>,tokenizer=<HF model folder>,tensor_parallel_size=<tp>,max_batch_size=<max batch size>,max_input_len=4096,max_output_len=512,kv_cache_free_gpu_memory_fraction=0.8 \
    --tasks <comma separated tasks> \
    --batch_size <max batch size>
```

> **_NOTE:_** Loglikelihood tasks (mmlu, hellaswag, arc, ...) need **TensorRT-LLM >=
> 1.3.0rc11**, which is when the engine started returning the requested token in every
> `prompt_logprobs` entry. Earlier releases return only the top-1 token per position, so a
> continuation token's logprob cannot be recovered and the run aborts with a clear error.
> Generative tasks (gsm8k, ifeval) are unaffected.

> **_NOTE:_** Set `max_input_len` and `max_output_len` explicitly. They default to 2048 and
> 512, and prompts longer than `max_input_len` are silently truncated — 5-shot MMLU or
> gsm8k prompts exceed 2048 tokens. `max_seq_len` of the engine is their sum.

> **_NOTE:_** `tensor_parallel_size` defaults to 1; set it to the number of GPUs the
> checkpoint needs. `pipeline_parallel_size` is also supported.

> **_NOTE:_** Use `lm_eval_trtllm.py` rather than the plain `lm_eval` CLI. lm-eval 0.4.12's
> `trtllm` backend misaligns TensorRT-LLM's `prompt_logprobs` by one position, so every
> loglikelihood task (hellaswag, mmlu, arc, ...) fails with a `KeyError`;
> `lm_eval_trtllm.py` overrides the alignment. It goes away once the fix lands upstream.

> **_NOTE:_** `kv_cache_free_gpu_memory_fraction` is the share of the GPU memory left after
> loading the weights that the KV cache may take. TensorRT-LLM's own default of 0.9 can leave
> too little room for the `prompt_logprobs` buffers and OOM on a large-memory GPU, so
> `lm_eval_trtllm.py` defaults it to 0.8; lm-eval's backend drops the key, which is why this
> entry point forwards it. `huggingface_example.sh` passes its
> `--kv_cache_free_gpu_memory_fraction` (default 0.8) through.

> **_NOTE:_** Other than the KV cache fraction, the backend forwards only a fixed set of
> arguments to TensorRT-LLM, so the remaining tuning the old `lm_eval_tensorrt_llm.py`
> applied is not reachable: expert parallelism is left at the TensorRT-LLM default (MoE
> checkpoints can fail in DeepEP kernels on some GPUs, e.g. SM 12.0). Lower
> `tensor_parallel_size` if you hit that.

`lm_eval_tensorrt_llm.py` (`--model trt-llm`) has been removed; use the command above.

## MMLU

[Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300). A score (0-1, higher is better) will be printed at the end of the benchmark.

### Setup

Download data

```bash
mkdir -p data
wget --connect-timeout=20 --read-timeout=60 --tries=3 -c \
    https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
```

Run the commands below from `examples/llm_eval`; `mmlu.py` resolves its default `--data_dir data/mmlu` relative to the current directory.

### Baseline

```bash
python mmlu.py --model_name causal --model_path <HF model folder or model card>
```

### Quantized (simulated)

```bash
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG|MXFP8_DEFAULT_CFG]
python mmlu.py --model_name causal --model_path <HF model folder or model card> --quant_cfg <MODELOPT_QUANT_CFG>
```

### auto_quantize (simulated)

```bash
# MODELOPT_QUANT_CFG_TO_SEARCH: Comma-separated list of the formats auto_quantize searches over.
# Pick each one from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG|MXFP8_DEFAULT_CFG|NONE],
# where NONE lets auto_quantize leave a layer unquantized.
# EFFECTIVE_BITS: Effective bits constraint for auto_quantize

# Example settings for an optimally quantized model with W4A8 & FP8 with effective bits of 4.8:
# MODELOPT_QUANT_CFG_TO_SEARCH=W4A8_AWQ_BETA_CFG,FP8_DEFAULT_CFG,NONE
# EFFECTIVE_BITS=4.8

python mmlu.py --model_name causal --model_path <HF model folder or model card> --quant_cfg $MODELOPT_QUANT_CFG_TO_SEARCH --auto_quantize_bits $EFFECTIVE_BITS --batch_size 4
```

### Evaluate with TensorRT-LLM

```bash
python mmlu.py --model_name causal --model_path <HF model folder or model card> --checkpoint_dir <Quantized checkpoint dir>
```

## LiveCodeBench

[LiveCodeBench](https://livecodebench.github.io/) is a holistic and contamination-free evaluation benchmark of LLMs for code that continuously collects new problems over time.

We support running LiveCodeBench against a local running OpenAI API compatible server. For example, quantized TensorRT-LLM checkpoint or engine can be loaded using [trtllm-serve](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html) command. Once the local server is up, the following command can be used to run the LiveCodeBench:

```bash
bash run_livecodebench.sh <custom defined model name> <prompt batch size in parallel> <max output tokens> <local model server port>
```

## Simple Evals

[Simple Evals](https://github.com/openai/simple-evals) is a lightweight library for evaluating language models published from OpenAI. This eval includes "simpleqa", "mmlu", "math", "gpqa", "mgsm", "drop" and "humaneval" benchmarks.

Similarly, we support running simple evals against a local running OpenAI API compatible server. Once the local server is up, the following command can be used to run the Simple Evals:

```bash
bash run_simple_eval.sh <custom defined model name> <comma separated eval names> <max output tokens> <local model server port> [num examples per eval]
```

The optional fifth argument caps the number of examples per eval (`--examples`); omit it to run the full eval.

## Customize quantization method for evaluation

An example of customized quantization config is shown in `quantization_utils.py`. It allows users to test accuracy of a custom method without the need of modifying the whole deployment framework, e.g., TensorRT-LLM, vLLM, SGLang, etc. Users can disable quantization of specific layers to debug the cause of accuracy drop, or explore a promising new quantization method.

```bash
python lm_eval_hf.py --model hf \
    --tasks <comma separated tasks> \
    --model_args pretrained=<HF model folder or model card>,parallelize=True \
    --quant_cfg MY_QUANT_CONFIG \
    --batch_size 4
```

## Evaluating with LM-Eval-Harness via vLLM

The `run_lm_eval_vllm.sh` script provides a convenient way to run evaluations using the `lm-evaluation-harness` library against a model served with vLLM's OpenAI-compatible API endpoint.

This is useful for evaluating quantized models deployed with vLLM or any model served via its OpenAI API interface. More importantly, for new models that are neither supported **natively** by vLLM or Transformers, but Transformers compatible, they can still be evaluated with vLLM's endpoint! By Transformers compatible, it needs to satisfy the following:

- The model directory must have the correct structure (e.g. `config.json` is present)
- `config.json` must contain `auto_map.AutoModel`.
- Customization should be done in the base model (e.g. in `MyModel`, not `MyModelForCausalLM`).

### Prerequisites

1. **Install vLLM:** Follow the installation instructions at [https://docs.vllm.ai/en/latest/getting_started/installation.html](https://docs.vllm.ai/en/latest/getting_started/installation.html).

### Usage

1. **Start the vLLM OpenAI-compatible Server:**
   In a separate terminal, launch the vLLM server with your desired model. For example:

   ```bash
   # Example using vLLM's built-in server
   vllm serve <your_model_name_or_path> \
       --port 8000 \
       --tensor-parallel-size <tp_size> # Adjust as needed
   ```

   Replace `<your_model_name_or_path>` with the actual model identifier (e.g., `Qwen/Qwen3-30B-A3B`) and adjust the `--port` and `--tensor-parallel-size` if necessary. You may also need to disable vllm v1 by `export VLLM_USE_V1=0` if you encounter issues.

   To serve a modelopt quantized model, add `--quantization modelopt`, for example:

   ```bash
   # Example using vLLM's built-in server
   vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 \
       --quantization modelopt \
       --port 8000 \
       --tensor-parallel-size <tp_size> # Adjust as needed
   ```

   To generate the quantized model such as `nvidia/Llama-3.1-8B-Instruct-FP8`, please refer to instructions [here](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq#deploy-fp8-quantized-model-using-vllm-and-sglang). Note currently modelopt quantized model support in vLLM is limited, we are working on expanding the model and quant formats support.

1. **Make the script executable (if not already):**

   ```bash
   chmod +x run_lm_eval_vllm.sh
   ```

1. **Run the evaluation script from the `examples/llm_eval` directory:**

   ```bash
   ./run_lm_eval_vllm.sh <model_name> [port] [task]
   ```

   - `<model_name>`: The name of the model being served (this is passed to `lm_eval`, e.g., `Qwen/Qwen3-30B-A3B`).
   - `[port]`: (Optional) The port the vLLM server is listening on. Defaults to `8000`. Note, it must match the number when launch the server.
   - `[task]`: (Optional) The `lm_eval` task(s) to run. Defaults to `mmlu`. Can be a single task or a comma-separated list (e.g., `"mmlu,hellaswag"`).

### Examples

- **Evaluate Qwen3-30B-A3B on MMLU (default task and port):**

  ```bash
  # Assumes vLLM server running with Qwen/Qwen3-30B-A3B on port 8000
  ./run_lm_eval_vllm.sh Qwen/Qwen3-30B-A3B
  ```

- **Evaluate a model on Hellaswag using port 8001:**

  ```bash
  # Assumes vLLM server running with <model_name> on port 8001
  ./run_lm_eval_vllm.sh <model_name> 8001 hellaswag
  ```

- **Evaluate on multiple tasks:**

  ```bash
  # Assumes vLLM server running with <model_name> on port 8000
  ./run_lm_eval_vllm.sh <model_name> 8000 "arc_easy,winogrande"
  ```

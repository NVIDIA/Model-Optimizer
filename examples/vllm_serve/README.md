# Serve fakequant models with vLLM

This is a simple example to demonstrate calibrating and serving ModelOpt fakequant models in vLLM.

Compared with realquant, fakequant is 2-5x slower, but doesn't require dedicated kernel support and facilitates research.

The general fakequant example is tested with vLLM 0.9.0 and 0.19.1. The compact
NVFP4 attention worker documented below requires vLLM 0.15.0 or newer.

## Prepare environment

Follow the following instruction to build a docker environment, or install vllm with pip.

```bash
docker build -f examples/vllm_serve/Dockerfile -t vllm-modelopt .
```

## Calibrate and serve fake quant model in vLLM

Step 1: Configure quantization settings.  
You can either edit the `quant_config` dictionary in `vllm_serve_fakequant.py`, or set the following environment variables to control quantization behavior:

| Variable        | Description                                      | Default             |
|-----------------|--------------------------------------------------|---------------------|
| QUANT_DATASET   | Dataset name for calibration                     | cnn_dailymail       |
| QUANT_CALIB_SIZE| Number of samples used for calibration           | 512                 |
| QUANT_CFG       | Quantization config                              | None                |
| KV_QUANT_CFG    | KV-cache quantization config                     | None                |
| QUANT_FILE_PATH | Optional path to exported quantizer state dict `quantizer_state.pth` | None |
| MODELOPT_STATE_PATH | Optional path to exported `vllm_fq_modelopt_state.pth` (restores quantizer state and parameters) | None |
| CALIB_BATCH_SIZE | Calibration batch size                           | 1                  |
| RECIPE_PATH      | Optional path to a ModelOpt PTQ recipe YAML  | None |

Set these variables in your shell or Docker environment as needed to customize calibration.

Step 2: Run the following command, with all supported flag as `vllm serve`:

```bash
python vllm_serve_fakequant.py <model_path> -tp 8 --host 0.0.0.0 --port 8000
```

For vLLM versions that expose `--moe-backend`, this launcher defaults to `--moe-backend triton`.
ModelOpt expert fakequant needs a decomposed MoE backend so both expert GEMMs are visible during
calibration.

Step 3: test the API server with curl:

```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions"     -H "Content-Type: application/json"     -d '{
          "model": "<model_path>",
          "messages": [
              {"role": "user", "content": "Hi, what is your name"}
          ],
          "max_tokens": 8
        }'

```

Step 4 (Optional): using lm_eval to run evaluation

```bash
lm_eval --model local-completions --tasks gsm8k --model_args model=<model_name>,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=128,tokenizer_backend=None
```

## Load QAT/PTQ model and serve in vLLM (WIP)

Step 1: export the model with bf16 weights and quantizer state. To export the model:

- For **HF** models, use `examples/hf_ptq/hf_ptq.py` with `--vllm_fakequant_export`:

```bash
python ../hf_ptq/hf_ptq.py \
  --pyt_ckpt_path <MODEL_PATH> \
  --recipe <PATH_TO_RECIPE> \
  --calib_size 512 \
  --export_path <EXPORT_DIR> \
  --vllm_fakequant_export \
  --trust_remote_code
```

  This creates `<EXPORT_DIR>/vllm_fq_modelopt_state.pth` (ModelOpt quantizer state for vLLM fake-quant reload) and saves the HF-exported model under `<EXPORT_DIR>` (config/tokenizer/weights).

  Note: `--pyt_ckpt_path` can point to either an HF checkpoint or a ModelOpt-saved checkpoint (e.g., a QAT/QAD checkpoint produced by `examples/llm_qat/train.py`). If the input checkpoint is already quantized, the script will **skip re-quantization** and only export artifacts for vLLM fakequant reload.

- For **MCore** models, export the model with flag `--export-vllm-fq` as described in [Megatron-LM README](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt#-nvfp4-quantization-qauntization-aware-training-and-model-export). This generates `quantizer_state.pth`, which contains quantizer tensors for vLLM reload via `QUANT_FILE_PATH`.

Step 2: use the exported artifacts when serving:

- **HF export**: pass the exported `vllm_fq_modelopt_state.pth` via `MODELOPT_STATE_PATH`

```bash
# HF
MODELOPT_STATE_PATH=<vllm_fq_modelopt_state.pth> python vllm_serve_fakequant.py <model_path> -tp 8 --host 0.0.0.0 --port 8000
```

- **MCore export**: pass the exported `quantizer_state.pth` via `QUANT_FILE_PATH` and set `QUANT_CFG` to match the MCore quantization recipe

```bash
# MCore
QUANT_CFG=<quant_cfg> QUANT_FILE_PATH=<quantizer_state.pth> python vllm_serve_fakequant.py <model_path> -tp 8 --host 0.0.0.0 --port 8000
```

## Serve a model with sparse attention in vLLM

Apply ModelOpt sparse attention at serve time. Right after model load, the launcher replaces each native attention implementation with its matching ModelOpt adapter: `ModelOptSparseAttentionImpl` for FlashAttention or `ModelOptSparseFlashInferImpl` for FlashInfer. Both adapters use the same Triton kernel with paged KV cache support.

The configuration is read from the checkpoint's `config.json` `sparse_attention_config` block, written by ModelOpt's HF export. The launcher restores calibrated skip-softmax metadata and N:M sparse-softmax metadata (`sparsity_n`, `sparsity_m`, `dense_sink_tokens`, `dense_recent_tokens`). Checkpoints exported with both metadata entries use ModelOpt Triton for sparse prefill launches; launches without active sparse work delegate back to the native backend selected by vLLM.

Workflow:

1. Calibrate and export the model with `examples/llm_sparsity/attention_sparsity/hf_sa.py`. This writes `sparse_attention_config` into the exported checkpoint's `config.json`.
2. Serve the exported checkpoint with `--enforce-eager` (CUDA graph capture is not yet validated with the sparse attention kernel — see Known Problems):

   ```bash
   python vllm_serve_sparse_attn.py <EXPORT_DIR> --enforce-eager -tp 8 --host 0.0.0.0 --port 8000
   ```

If the checkpoint has no `sparse_attention_config`, the sparse-only installer passes through and vLLM runs unchanged. Whole-model fakequant flows remain handled by `vllm_serve_fakequant.py`; the compact attention-only path is below.

### Calibrate skip-softmax thresholds through vLLM

Instead of the HF path in step 1, thresholds can be calibrated directly through vLLM — over the paged KV cache, for both prefill and decode, with tensor parallelism:

```bash
# One-time: fetch the RULER essay haystack
bash ../llm_sparsity/attention_sparsity/download_ruler_data.sh

python calibrate_sparse_attn.py <CKPT> \
  --calib_data_dir ../llm_sparsity/attention_sparsity/data \
  --target_sparse_ratio 0.5 \
  --decode_tokens 32 --tensor_parallel_size 8 --update_checkpoint_config
```

Calibration prompts default to the **RULER dataset** via the same `RulerDatasetBuilder` the HF calibration path uses (`--calib_samples` / `--calib_max_seqlen` mirror the HF defaults of 24 / 32768), so vLLM- and PyTorch-calibrated thresholds are fit on identical data. `--prompts_file` (one prompt per line) substitutes custom calibration data.

`install_vllm_skip_softmax_calibration` (called by `sparse_attn_worker.SkipSoftmaxCalibWorker` at model load) swaps calibration adapters onto every attention layer after validating all of them — eager execution is required, model and KV-cache dtypes must be fp16/bf16, and no attention Q/K/P/V fakequant may be active. During `llm.generate`, the paged Triton calibration kernel computes full dense attention — no sparsification is applied to generation, though the dense kernel's numerics differ slightly from the native backend's — while counting, per candidate threshold, how many KV tiles the skip criterion would drop. The driver then collects **raw tile counts from every TP rank** (each rank only measures its head shard), merges them, fits `scale_factor = a * exp(b * sparsity)` once per phase, and writes the same canonical `sparse_attention_config` block the HF export produces — preserving any exported N:M sparse-softmax groups — so the serving workflow above picks it up unchanged.

Calibration and serving measure skipping at the same fixed 128x128 tile geometry (active skip-softmax launches bypass the autotuner), so the sparsity realized at serve time matches the calibrated `(a, b)` model.

The reusable serving policies live in `modelopt/torch/sparsity/attention_sparsity/plugins/vllm_runtime.py`. `install_vllm_sparse_attention_from_checkpoint` installs checkpoint-driven sparse-only attention, while `install_vllm_nvfp4_attention` installs fixed NVFP4 Q/K/P/V with optional checkpoint sparsity. Both validate every selected layer before publishing any replacement implementation and return a `VllmAttentionInstallReport` with the installed layer names and backend counts.

`sparse_attn_worker.py` only invokes these APIs after vLLM loads the model. It retains `SparseAttnWorker` as the launcher's default and provides `QuantSparseAttnWorker` for the compact NVFP4 policy. Other vLLM integrations can invoke the same library APIs directly:

```python
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm_runtime import (
    install_vllm_nvfp4_attention,
)

report = install_vllm_nvfp4_attention(model_runner, sparse_cfg="checkpoint")
```

Limitations:

- vLLM V1 chunked prefill and prefix-cache suffix attention are supported by offsetting query positions into the longer KV span. This applies to sparse-only serving; quantized attention installs and skip-softmax calibration reject `enable_prefix_caching` (quantize-on-write and per-request measurement both require uncached prefills).
- `SparseAttnWorker` CUDA graph capture is not validated yet — use `--enforce-eager`.

### Compact NVFP4 attention worker

vLLM 0.15.0 or newer is required when either worker activates a ModelOpt attention transform. Importing `SparseAttnWorker`, or using it with no checkpoint sparse metadata, does not resolve quant-only APIs.

Use the same launcher with the compact worker. By default, vLLM selects the backend for the model and platform; NemotronH on Blackwell selects FlashInfer:

```bash
python vllm_serve_sparse_attn.py <MODEL_PATH> -tp 8 \
  --no-enable-prefix-caching \
  --worker-cls sparse_attn_worker.QuantSparseAttnWorker
```

The installer supports both FlashInfer and FlashAttention, and the worker prints the installed adapter counts. Pass `--attention-backend FLASHINFER` or `--attention-backend FLASH_ATTN` only when an explicit override is needed.

This attention-only path applies a fixed dynamic block-16 NVFP4 fakequant format to Q/K/P/V. Q is dynamic; missing K/V scales default to global scale 1.0, and P defaults to amax 1.0. Existing scalar attention amax values are preserved, but this path does not calibrate or restore them itself. It does not re-quantize realquant Linear or MoE weights. An optional checkpoint `sparse_attention_config` is still honored for N:M sparse softmax; calibrated skip-softmax groups are rejected in combination with attention quantization, because quantized Q/K/P change the score distribution the skip thresholds were calibrated on.

Decode uses a fixed 32-split, 128-key-tile schedule. P QDQ consumes split-local,
unnormalized online-softmax probabilities, so changing that schedule can change
quantized results; split count is part of the numerical contract.

K is QDQ before its cache write, while V is written pristine. Complete 16-token V groups are finalized once in cache; an incomplete tail remains pristine and is QDQ on read. P@V therefore sees uniform fakequant values without re-quantizing the tail.

Supported configurations are regular decoder self-attention with FlashInfer or FlashAttention, fp16/bf16 model and KV cache, equal Q/K/V head dimensions that are multiples of 16, and DCP 1. The FlashInfer adapter preserves both NHD and HND cache strides and separates mixed decode/prefill launches so each phase keeps its own kernel contract. The default `FULL_AND_PIECEWISE` mode remains enabled for fixed N:M and attention-only NVFP4; checkpoints with calibrated decode `threshold_scale_factor` must use a non-`FULL` decode graph mode such as `--enforce-eager` because the live sequence length is not replayed as a Python scalar.

Unsupported features are sliding window, ALiBi, softcap, sinks, FP8 KV cache, cross/encoder/MLA attention, KV sharing or transfer, prefix caching, speculative decoding, DBO/ubatching, and `FULL` mixed/prefill CUDA graphs.

## Known Problems

1. **MCore reload does not use `MODELOPT_STATE_PATH`**; use `QUANT_FILE_PATH` and make sure `QUANT_CFG` matches the quantization recipe used for the original MCore model (otherwise quantizer keys/config won’t align).
2. KV cache quantization export and reload is not supported in MCore yet.
3. **`NVFP4_KV_CFG` and `NVFP4_AFFINE_KV_CFG` require `--enforce-eager`**; these configs use a dynamic-block Triton kernel for KV-cache quantization that is incompatible with CUDA graph capture (the kernel grid is computed from Python-level tensor shapes, which get baked in at capture time). Without `--enforce-eager`, the captured grid will be wrong for different batch sizes, producing incorrect outputs.

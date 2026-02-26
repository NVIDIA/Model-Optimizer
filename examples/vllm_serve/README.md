# Serve fakequant models with vLLM

This is a simple example to demonstrate calibrating and serving ModelOpt fakequant models in vLLM.

Compared with realquant, fakequant is 2-5x slower, but doesn't require dedicated kernel support and facilitates research.

This example is tested with vllm 0.9.0 and 0.11.2

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
| QUANT_CFG       | Quantization format                              | NVFP4_DEFAULT_CFG   |
| KV_QUANT_CFG    | Quantization format for KV Cache                 | None                |
| AMAX_FILE_PATH  | Optional path to amax file (for loading amax)    | None                |
| KV_ATTN_MATH    | Attention math precision for KV fake-quant       | None                |

### KV_ATTN_MATH=fp8 — NVFP4 KV emulation with FP8 attention math

Setting `KV_ATTN_MATH=fp8` together with an NVFP4 KV config enables a more
accurate emulation of NVFP4 KV cache on hardware where the attention BMM runs
in FP8 precision (e.g., Blackwell with FlashInfer or TRT-LLM kernels).

**How it works:**

1. `KV_QUANT_CFG` (e.g. `NVFP4_KV_CFG`) applies NVFP4 fake-quantization to K
   and V, introducing NVFP4-level quantization noise into the BF16 tensors.
2. vLLM writes those BF16 K/V values into an FP8 KV cache
   (`--kv-cache-dtype fp8`) with **scale=1.0** — a direct dtype cast.
3. FlashInfer (or TRT-LLM on SM100) reads the FP8 KV cache and executes the
   attention BMM in FP8 precision.

**Required launch flags:**

```bash
KV_QUANT_CFG=NVFP4_KV_CFG \
KV_ATTN_MATH=fp8 \
python vllm_serve_fakequant.py <model_path> \
    --kv-cache-dtype fp8 \
    -tp 8 --host 0.0.0.0 --port 8000
```

`--kv-cache-dtype fp8` is **mandatory** when `KV_ATTN_MATH=fp8` is set.
Without it, vLLM stores K/V in BF16 and FlashInfer runs BF16 attention,
defeating the purpose of this mode.

The `KV_ATTN_MATH=fp8` flag locks the FP8 KV cache scales to 1.0 and disables
vLLM's dynamic scale computation, ensuring the direct-cast assumption holds.

**Combining with weight/activation quantization:**

```bash
QUANT_CFG=NVFP4_DEFAULT_CFG \
KV_QUANT_CFG=NVFP4_KV_CFG \
KV_ATTN_MATH=fp8 \
python vllm_serve_fakequant.py <model_path> \
    --kv-cache-dtype fp8 \
    -tp 8 --host 0.0.0.0 --port 8000
```

Set these variables in your shell or Docker environment as needed to customize calibration.

Step 2: Run the following command, with all supported flag as `vllm serve`:

```bash
python vllm_serve_fakequant.py <model_path> -tp 8 --host 0.0.0.0 --port 8000
```

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

Overwrite the calibrated amax value with prepared values from either QAT/PTQ.

Step 1: export the model with bf16 weights and amax values. To export the model:

- For HF model use `modelopt.torch.export.export_hf_vllm_fq_checkpoint` function.
- For MCore model use `modelopt.torch.export.export_mcore_gpt_to_hf_vllm_fq` function.

Step 2: configure <quant_amax.pth> from exported model using AMAX_FILE_PATH environment variable in step 1. For example:

```bash
AMAX_FILE_PATH=<vllm_amax.pth> QUANT_CFG=<quant_config> python vllm_serve_fakequant.py <model_path> -tp 8 --host 0.0.0.0 --port 8000
```

## Known Problems

1. AWQ is not yet supported in vLLM.
2. QAT checkpoint export doesn't have KV Cache quantization enabled. KV Cache fake quantization works for PTQ.
3. Mixed precision checkpoint doesn't work currently.

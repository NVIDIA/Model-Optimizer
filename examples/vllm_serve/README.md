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
| AMAX_FILE_PATH  | Optional path to amax file (for loading amax)    | None                |

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

## Important Notes

**Amax Synchronization across Tensor Parallel (TP):**

- **For non-per-tensor quantization**: It is **recommended** to use an amax file (via `AMAX_FILE_PATH`) because amax synchronization across TP/EP is not automatically handled. Without an amax file, the amax values can be different across different TP ranks, leading to inconsistent results compared to real-quantization.

- **For per-tensor quantization**: If you are not using an amax file, you need to enable amax synchronization across TP ranks. An example implementation is provided in `fakequant_worker.py` (lines 190-198):

```python
for name, buffer in model.named_buffers():
    if name.endswith("_amax"):
        torch.distributed.all_reduce(
            buffer, op=torch.distributed.ReduceOp.MAX, group=get_tp_group().device_group
        )
torch.distributed.barrier()
```

## Known Problems

1. AWQ is not yet supported in vLLM.
2. PTQ/QAT checkpoint doesn't work with KV Cache quantization enabled.
3. Mixed precision checkpoint doesn't work currently.

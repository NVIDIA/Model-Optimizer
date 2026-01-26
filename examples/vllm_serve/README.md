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
| QUANT_CFG       | Quantization config                              | None                |
| KV_QUANT_CFG    | KV-cache quantization config                     | None                |
| QUANT_FILE_PATH | Optional path to exported quantizer state dict `quantizer_state.pth` | None |
| MODELOPT_STATE_PATH | Optional path to exported `vllm_fq_modelopt_state.pth` (restores quantizer state and parameters) | None |
| CALIB_BATCH_SIZE | Calibration batch size                           | 1                  |

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

Step 1: export the model with bf16 weights and quantizer state. To export the model:

- For **HF** models, use `hf_ptq_export.py`:

```bash
python  hf_ptq_export.py\
  --pyt_ckpt_path <MODEL_PATH> \
  --quant_cfg NVFP4_DEFAULT_CFG \
  --export_path <EXPORT_DIR> \
  --trust_remote_code
```

  This creates `<EXPORT_DIR>/vllm_fq_modelopt_state.pth` (ModelOpt quantizer state for vLLM fake-quant reload) and saves the HF-exported model under `<EXPORT_DIR>` (config/tokenizer/weights).
  Note: `--pyt_ckpt_path` can point to either an HF checkpoint or a ModelOpt-saved checkpoint (e.g., a QAT/QAD checkpoint produced by `examples/llm_qat/main.py`). If the input checkpoint is already quantized, the script will **skip re-quantization** and only export artifacts for vLLM fakequant reload.

- For **MCore** models, use `modelopt.torch.export.export_mcore_gpt_to_hf_vllm_fq`:

  ```python
  from modelopt.torch.export import export_mcore_gpt_to_hf_vllm_fq
  export_mcore_gpt_to_hf_vllm_fq(
          unwrapped_model,  # Quantized MCore model
          args.pretrained_model_name,  # HF model id/path (for config/tokenizer)
          export_dir=args.export_dir,  # Directory where exported files will be stored
      )

  ```

  This generates `quantizer_state.pth`, which contains quantizer tensors for vLLM reload via `QUANT_FILE_PATH`.

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

## Known Problems

1. **MCore reload does not use `MODELOPT_STATE_PATH`**; use `QUANT_FILE_PATH` and make sure `QUANT_CFG` matches the quantization recipe used for the original MCore model (otherwise quantizer keys/config won’t align).
2. AWQ reload is not supported yet
3. KV cache quantization export and reload is not supported in MCore yet.

# Error Prevention & Recovery Guide

## Environment Setup Checklist

Run these checks before starting any quantization job:

```bash
nvidia-smi                           # GPU available and healthy
python --version                     # Needs 3.10+
python -c "import torch; print(torch.cuda.is_available())"  # CUDA works
pip show nvidia-modelopt 2>/dev/null || echo "Check PYTHONPATH for dev install"
pip show vllm                        # vLLM installed
nel --version                        # nel installed
```

## Error Patterns by Category

### Environment Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'modelopt'` | modelopt not installed or not on PYTHONPATH | `pip install nvidia-modelopt` or set `PYTHONPATH=$REPO_ROOT` |
| `ModuleNotFoundError: No module named 'example_utils'` | PYTHONPATH missing examples dir | `PYTHONPATH=examples/llm_ptq:$REPO_ROOT` |
| `Multiple distributions found for package modelopt` | Both pip install and dev install exist | Set `PYTHONPATH=$REPO_ROOT` to prefer dev install |
| `RuntimeError: CUDA not available` | No GPU or driver issue | Check `nvidia-smi`, reinstall CUDA toolkit |
| `ImportError: transformers version` | transformers too old | `pip install --upgrade transformers` |

### Model Loading Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: '<model_type>'` | transformers doesn't know this model | `pip install --upgrade transformers` |
| `ValueError: does not recognize this architecture` | Same as above | `pip install --upgrade transformers` |
| `OSError: <model> does not appear to have a file named config.json` | Wrong model path | Check path exists and contains config.json |
| `trust_remote_code` required | Model has custom code | Add `--trust-remote-code` or `trust_remote_code=True` |
| `torch.cuda.OutOfMemoryError` on load | Model too large for GPU | Use `device_map="auto"` or more GPUs |

### Quantization Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: module 'modelopt.torch.quantization' has no attribute '*_CFG'` | Installed version missing config | Use sed patching: `getattr(mtq, "CFG_NAME", {})` |
| `RuntimeError: NaN detected in quantized output` | Bad calibration data or scales | Increase `--calib_size`, try different algorithm |
| `Calibration hangs / very slow` | GPU underutilized | Check batch size, reduce calib_size |
| `CUDA out of memory during calibration` | Model + calibration data too large | Reduce `--calib_size` (try 256, then 128) |
| `AssertionError in quantizer` | Incompatible tensor shapes | Check model architecture, may need custom handling |

### Export Errors

| Error | Cause | Fix |
|-------|-------|-----|
| No safetensors files in export dir | Export failed silently | Check stderr, verify model was quantized |
| Missing hf_quant_config.json | `enable_huggingface_checkpointing()` not called | Ensure it's called before export (hf_ptq.py does this) |
| Missing tokenizer files | Original model path lost | Copy tokenizer files from original model dir |
| `PermissionError` on export | No write access to export dir | Check directory permissions |

### Deployment Errors (vLLM)

| Error | Cause | Fix |
|-------|-------|-----|
| `Address already in use` | Port conflict | Change port or `kill $(lsof -t -i:8199)` |
| `torch.cuda.OutOfMemoryError` on serve | Model too large for GPU memory | `--gpu-memory-utilization 0.8` or reduce `--max-model-len` |
| `ValueError: quantization not supported` | vLLM version too old | `pip install --upgrade vllm` |
| Server starts but returns errors | Checkpoint incomplete or corrupt | Re-export, verify checkpoint files |
| Health check returns 503 | Server still loading | Wait longer (large models take minutes) |

### Evaluation Errors (nel)

| Error | Cause | Fix |
|-------|-------|-----|
| `nel: command not found` | nel not installed | `pip install nemo-evaluator-launcher` |
| `Connection refused` | vLLM server not running | Start vLLM server first |
| `Timeout` on requests | Model too slow or parallelism too high | Increase `request_timeout` or reduce `parallelism` |
| `Task not found` | Wrong task name | Run `nel ls tasks` for valid names |
| nel hangs indefinitely | vLLM server crashed | Check vLLM health, restart if needed |

## Common Pitfalls

1. **Forgetting PYTHONPATH** — Always set `PYTHONPATH=examples/llm_ptq:$REPO_ROOT` before running hf_ptq.py
2. **Wrong model path** — Use absolute paths, not relative
3. **Not checking vLLM health** — Always poll `/health` before running benchmarks
4. **Reusing export dir** — Old files may conflict; use a fresh directory per format
5. **Killing vLLM too early** — Wait for graceful shutdown before starting a new server

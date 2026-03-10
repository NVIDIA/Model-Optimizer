# Export & Checkpoint Reference

## Export Function

```python
import modelopt.torch.opt as mto
from modelopt.torch.export import export_hf_checkpoint

mto.enable_huggingface_checkpointing()  # must be called before export
export_hf_checkpoint(model, export_dir="/path/to/output")
```

**Note:** `hf_ptq.py` calls `mto.enable_huggingface_checkpointing()` automatically at line 106.

## Output Files

A quantized HuggingFace checkpoint contains:

| File | Description |
|------|-------------|
| `model-00001-of-NNNNN.safetensors` | Sharded weight files with quantized weights and scales |
| `model.safetensors.index.json` | Shard index mapping tensor names to files |
| `config.json` | Model architecture config (copied from original) |
| `hf_quant_config.json` | Quantization metadata (format, scales, block sizes) |
| `tokenizer.json` | Tokenizer (copied from original) |
| `tokenizer_config.json` | Tokenizer config (copied from original) |
| `special_tokens_map.json` | Special tokens (copied from original) |
| `generation_config.json` | Generation defaults (if present in original) |
| `vocab.json`, `merges.txt` | BPE vocab files (if applicable) |

## `hf_quant_config.json`

This file records the quantization configuration so deployment frameworks (vLLM, TRT-LLM) can auto-detect the format:

```json
{
  "producer": {"name": "modelopt", "version": "0.x.x"},
  "quantization": {
    "quant_algo": "FP8",
    "kv_cache_quant_algo": null
  }
}
```

Deployment frameworks read `quant_algo` to determine how to load and dequantize weights.

## Checkpoint Size Expectations

| Format | Approx Size vs Original |
|--------|------------------------|
| fp8 | ~100% (same number of params, lower precision) |
| int8 | ~50-60% |
| int4_awq | ~30-40% |
| nvfp4 | ~30-40% |

Actual size depends on model architecture and how many layers are quantized.

## Verification Checklist

After export, verify the checkpoint is valid:

```bash
# 1. Check safetensors files exist
ls $EXPORT_PATH/*.safetensors

# 2. Check config.json exists and is valid JSON
python -c "import json; json.load(open('$EXPORT_PATH/config.json')); print('config.json OK')"

# 3. Check hf_quant_config.json exists
python -c "import json; c=json.load(open('$EXPORT_PATH/hf_quant_config.json')); print('quant_algo:', c.get('quantization',{}).get('quant_algo'))"

# 4. Check tokenizer files exist
ls $EXPORT_PATH/tokenizer* $EXPORT_PATH/special_tokens_map.json
```

If any file is missing, the export may have failed silently. Check stderr output from the export command.

## Common Export Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Missing tokenizer files | Original model dir didn't have tokenizer | Copy tokenizer files manually |
| Empty safetensors | Model wasn't quantized properly | Re-run quantization, check mtq.print_quant_summary() |
| Missing hf_quant_config.json | enable_huggingface_checkpointing() not called | Ensure it's called before export |
| Very large checkpoint | Scales stored at high precision | Expected; deployment framework handles this |

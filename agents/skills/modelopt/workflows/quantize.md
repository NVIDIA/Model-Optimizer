# Workflow: Quantize Model with hf_ptq.py

## Inputs (provided by caller)

- `MODEL_PATH`: Absolute path to the model directory
- `QFORMAT`: Quantization format (e.g., `nvfp4`, `fp8`, `int4_awq`, `int8_sq`, `w4a8_awq`)
- `EXPORT_PATH`: Where to save the quantized checkpoint
- `GPU_IDS`: Comma-separated GPU indices (e.g., `0` or `0,1`)
- `CALIB_SIZE`: Calibration dataset size (default: 512)
- `CALIB_ALGO`: (optional) Calibration algorithm: `max` (default), `mse`, `percentile`, `histogram`. Use `mse` if `max` gives poor accuracy.
- `BATCH_SIZE`: (optional) Calibration batch size. Use `1` for very deep models (100+ layers) to avoid OOM.
- `REPO_ROOT`: Root of the TensorRT-Model-Optimizer repo

## Procedure

### 1. Create patched script

The installed modelopt package may not have all `*_CFG` config constants referenced in `hf_ptq.py`. Create a patched copy that uses safe `getattr` fallbacks:

```bash
PATCH_DIR="$(mktemp -d)/modelopt_patch"
mkdir -p "$PATCH_DIR"
sed 's/mtq\.\([A-Z][A-Z0-9_]*_CFG\)\b/getattr(mtq, "\1", {})/g' \
  "$REPO_ROOT/examples/llm_ptq/hf_ptq.py" > "$PATCH_DIR/hf_ptq_patched.py"
```

Verify the patch worked:

```bash
# File should be non-empty and contain getattr substitutions
test -s "$PATCH_DIR/hf_ptq_patched.py" && grep -c 'getattr(mtq' "$PATCH_DIR/hf_ptq_patched.py" && echo "Patch OK" || echo "Patch FAILED"
```

### 2. Run quantization

```bash
PYTHONPATH="$REPO_ROOT/examples/llm_ptq:$REPO_ROOT" \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python "$PATCH_DIR/hf_ptq_patched.py" \
  --pyt_ckpt_path "$MODEL_PATH" \
  --qformat "$QFORMAT" \
  --export_path "$EXPORT_PATH" \
  --calib_size $CALIB_SIZE
```

**Optional flags** (append to the command above if needed):

- `--calib_algo $CALIB_ALGO` -- calibration algorithm (default: depends on format config). Use `mse` if accuracy is poor with defaults.
- `--batch_size $BATCH_SIZE` -- calibration batch size. Use `1` for very deep models (100+ layers) to avoid OOM.

This may take 5-60 minutes depending on model size and calibration size.

### 3. Verify success

Check the export directory has model files:

```bash
ls "$EXPORT_PATH"/*.safetensors "$EXPORT_PATH"/config.json 2>/dev/null && echo "SUCCESS" || echo "FAILED"
```

### 4. Error recovery

If quantization fails, check the error and apply the appropriate fix:

| Error Pattern | Action |
|--------------|--------|
| `KeyError: '<model_type>'` or `does not recognize this architecture` | **Read and follow** `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/tools/patch-model.md`, then retry from step 2 |
| `AttributeError: module 'modelopt.torch.quantization' has no attribute` | The sed patching missed something. Check the specific attribute name and manually add `getattr(mtq, "ATTR_NAME", {})` in the patched script |
| `ModuleNotFoundError: No module named 'example_utils'` | Verify `PYTHONPATH` includes `$REPO_ROOT/examples/llm_ptq` |
| `torch.cuda.OutOfMemoryError` | Retry with `--calib_size 256`. If still OOM, try `--calib_size 128` |
| Any other error | Read the full traceback. Fix the issue in the patched script and retry. Max 3 attempts. |

## Supported Formats (common)

| Format | Description | Typical Accuracy Drop |
|--------|-------------|----------------------|
| `fp8` | 8-bit float (E4M3) | < 1% |
| `int8_sq` | 8-bit int + SmoothQuant | 1-2% |
| `w4a8_awq` | 4-bit weights, 8-bit activations | 2-3% |
| `int4_awq` | 4-bit int + AWQ | 2-4% |
| `nvfp4` | NVIDIA 4-bit float (recommended for Blackwell) | 2-5% |

For the full list of 14+ formats (including `nvfp4_awq`, `nvfp4_mse`, `mxfp8`, `fp8_pb_wo`, etc.), **read** `/home/scratch.kaix_coreai/workspace/trt_model_optimizer_dev/modelopt_agent/agents/skills/references/02_quantization_formats.md`.

## Output

- Quantized checkpoint at `EXPORT_PATH` (HuggingFace safetensors format)
- Report success/failure to the caller

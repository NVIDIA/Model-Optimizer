# Tool: Patch Model for Compatibility

## Inputs (provided by caller)

- `MODEL_PATH`: Absolute path to the model directory

## Procedure

### 1. Read model config

```bash
python -c "
import json
c = json.load(open('$MODEL_PATH/config.json'))
print('model_type:', c.get('model_type', 'MISSING'))
print('architectures:', c.get('architectures', []))
print('has_vision:', 'vision_config' in c)
print('has_auto_map:', 'auto_map' in c)
print('trust_remote_code_files:', [f for f in __import__('os').listdir('$MODEL_PATH') if f.startswith('modeling_')])
"
```

Note the `model_type` and whether it has `vision_config`.

### 2. Test if transformers recognizes the model

```bash
python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True); print('OK:', c.model_type)"
```

If this succeeds, the model is compatible. **Return to caller -- no patching needed.**

### 3. If model_type is not recognized

The error will look like: `KeyError: '<model_type>'` or `ValueError: ... does not recognize this architecture`

**Fix: Upgrade transformers**

```bash
pip install --upgrade transformers
```

Re-run the test from step 2. If it passes, **return to caller.**

### 4. If upgrade doesn't help

Check if the model has custom code:

```bash
ls "$MODEL_PATH"/modeling_*.py 2>/dev/null
python -c "import json; c=json.load(open('$MODEL_PATH/config.json')); print('auto_map:', c.get('auto_map', 'NONE'))"
```

If custom modeling files exist or `auto_map` is present:

- The model requires `trust_remote_code=True`
- Verify the quantization script passes this flag
- If `hf_ptq.py` doesn't pass `trust_remote_code`, add it to the patched script's model loading call

### 5. VLM detection

If `vision_config` is present in `config.json`:

- This is a Vision-Language Model (VLM)
- Standard text-only quantization applies to the text backbone only
- Inform the user: "This model has a vision component. Quantization will apply to the text backbone."

## Common Model Type Issues

| Model Type | Min Transformers Version | Notes |
|------------|-------------------------|-------|
| `qwen3_5` | >= 4.53 | Qwen3.5 VLM, very new |
| `qwen3` | >= 4.51 | Qwen3 text models |
| `deepseek_v3` | >= 4.50 | DeepSeek V3 |
| Custom models | any | Need `trust_remote_code=True` |

## Output

- Report whether patching was needed and what was done
- Return to caller to retry quantization

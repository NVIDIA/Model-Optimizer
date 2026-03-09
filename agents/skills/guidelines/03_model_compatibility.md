# Model Compatibility Guide

## Transformers Version Matrix

Models require minimum transformers versions. If quantization fails with `KeyError` on `model_type`, upgrade transformers.

| Model Family | model_type | Min transformers | Notes |
|-------------|------------|-----------------|-------|
| Llama 3.x | `llama` | >= 4.40 | Llama 3, 3.1, 3.2 |
| Qwen3 | `qwen3` | >= 4.51 | Text-only models |
| Qwen3.5 | `qwen3_5` | >= 4.53 | Vision-language model |
| DeepSeek V3 | `deepseek_v3` | >= 4.50 | MoE architecture |
| Mistral/Mixtral | `mistral` | >= 4.40 | Including MoE variants |
| Gemma 2 | `gemma2` | >= 4.42 | |
| Phi-3 | `phi3` | >= 4.41 | |
| Phi-4 | `phi4` | >= 4.48 | |
| Command R+ | `cohere` | >= 4.40 | |

**Rule:** When in doubt, run `pip install --upgrade transformers` before quantization.

## Checking Model Compatibility

```bash
# 1. Read model_type from config.json
python -c "import json; print(json.load(open('MODEL_PATH/config.json')).get('model_type'))"

# 2. Test if transformers recognizes it
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('MODEL_PATH', trust_remote_code=True)"
```

If step 2 fails, follow the `patch-model.md` sub-skill.

## VLM (Vision-Language Model) Detection

Check for `vision_config` in `config.json`:

```python
import json
config = json.load(open("MODEL_PATH/config.json"))
is_vlm = "vision_config" in config
```

**VLM handling:**

- hf_ptq.py auto-detects VLMs via `get_language_model_from_vl()`
- Only the **text backbone** is quantized; the vision encoder is left in original precision
- The vision encoder gets `enable: False` in its quantization config
- VLMs may have a different `architectures` entry (e.g., `Qwen3_5ForConditionalGeneration` vs `Qwen3ForCausalLM`)

## trust_remote_code Models

Some models include custom modeling code that isn't part of the transformers library.

**Detection:**

```python
config = json.load(open("MODEL_PATH/config.json"))
has_auto_map = "auto_map" in config
has_custom_code = any(f.startswith("modeling_") for f in os.listdir("MODEL_PATH") if f.endswith(".py"))
```

If either is true:

- Pass `trust_remote_code=True` to `AutoConfig.from_pretrained()` and `AutoModelForCausalLM.from_pretrained()`
- hf_ptq.py already does this by default

## MoE (Mixture of Experts) Models

MoE models (e.g., Mixtral, DeepSeek V3, Qwen MoE) have special considerations:

- **Expert weights are quantized independently** — each expert gets its own scales
- **Routing is not quantized** — the gating network stays in original precision
- **More calibration data helps** — MoE models benefit from larger `--calib_size` (1024+) to cover more expert paths
- **ModelOpt's HF plugin handles MoE routing** automatically via `QuantizedMoEModule`

## Known Problematic Architectures

| Architecture | Issue | Workaround |
|-------------|-------|------------|
| Very new models (< 1 week) | transformers may not support | Wait for transformers release or use `trust_remote_code` |
| Custom attention (sliding window, etc.) | Non-standard attention may confuse quantizer | Usually works; if not, try `nvfp4_mlp_only` to skip attention |
| Shared embeddings (`tie_word_embeddings=True`) | Export must handle shared weights correctly | ModelOpt handles this; verify with `mtq.print_quant_summary()` |
| Multi-modal encoders | Vision/audio encoders shouldn't be quantized | hf_ptq.py auto-handles via `extract_and_prepare_language_model_from_vl()` |
| Very deep models (100+ layers) | Calibration may be slow | Reduce `--calib_size`, use `--batch_size 1` |

## Checking vLLM Compatibility

Not all quantized formats work with all vLLM versions:

```bash
# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# Test loading
python -c "from vllm import LLM; llm = LLM('MODEL_PATH', trust_remote_code=True); print('OK')"
```

| vLLM Version | FP8 | INT4_AWQ | NVFP4 | W4A8 |
|-------------|-----|----------|-------|------|
| >= 0.6.0 | Yes | Yes | No | Partial |
| >= 0.8.0 | Yes | Yes | Yes | Yes |
| >= 0.9.0 | Yes | Yes | Yes | Yes |

If vLLM can't load the checkpoint, try `pip install --upgrade vllm`.

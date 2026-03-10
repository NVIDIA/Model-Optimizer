# Quantization API Reference

## Module

```python
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto  # for enable_huggingface_checkpointing()
```

## Core Functions

### `mtq.quantize(model, config, forward_loop)`

Main entry point. Quantizes and calibrates the model **in-place**.

```python
import modelopt.torch.quantization as mtq

def calibrate(model):
    for batch in calib_dataloader:
        model(batch)

model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=calibrate)
```

**Parameters:**

- `model` (`nn.Module`): The PyTorch model to quantize
- `config` (`dict`): Quantization config dict with `quant_cfg` and `algorithm` keys. Use predefined configs (e.g., `mtq.FP8_DEFAULT_CFG`) or custom dicts.
- `forward_loop` (`callable | None`): Function that runs calibration data through the model. Pass `None` for weight-only quantization.

**Returns:** The quantized model (same object, modified in-place).

### `mtq.print_quant_summary(model)`

Prints all quantizer modules and their states. Useful for debugging.

```python
mtq.print_quant_summary(model)
# Output: lists each layer's quantizer type, bit width, and calibration status
```

### `export_hf_checkpoint(model, export_dir)`

Exports the quantized model as a HuggingFace checkpoint.

```python
from modelopt.torch.export import export_hf_checkpoint

export_hf_checkpoint(model, export_dir="/path/to/output")
```

**Prerequisite:** Call `mto.enable_huggingface_checkpointing()` before export (done automatically in hf_ptq.py).

**Output files:** See `references/04_export_checkpoint.md`.

## hf_ptq.py Flow

The reference quantization script at `examples/llm_ptq/hf_ptq.py` follows this pipeline:

```
1. load_model(args)           → Load HF model + tokenizer
2. quantize_main(model, args) → Insert Q/DQ nodes, run calibration, compute scales
3. export_hf_checkpoint()     → Save quantized checkpoint as safetensors
```

## Config System

Configs are dicts with this structure:

```python
{
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "default": {"num_bits": 8, "axis": None},
    },
    "algorithm": "max",  # or "mse", "smoothquant", "awq"
}
```

Predefined configs: `mtq.FP8_DEFAULT_CFG`, `mtq.INT8_SMOOTHQUANT_CFG`, `mtq.INT4_AWQ_CFG`, `mtq.NVFP4_DEFAULT_CFG`, etc.

The `QUANT_CFG_CHOICES` dict in hf_ptq.py maps CLI format names to these configs.

# ModelOpt Agents

Claude Code skills for interactive model optimization with NVIDIA ModelOpt.

## Skills

### `modelopt-optimizer`

Interactive model optimization workflow: quantize, benchmark speed, evaluate accuracy, and iterate.

```
modelopt-optimizer/
├── SKILL.md                    # Top-level orchestrator
└── sub-skills/
    ├── quantize.md             # Quantize with hf_ptq.py + error recovery
    ├── patch-model.md          # Fix model architecture incompatibilities
    ├── vllm-benchmark.md       # Deploy on vLLM + speed benchmark
    └── accuracy-eval.md        # Accuracy evaluation via nel (NeMo Evaluator)
```

### Workflow

```
Step 1: Gather info (model path, qformat, GPUs)
Step 2: Quantize (quantize.md + patch-model.md if needed)
Step 3: Deploy + speed benchmark (vllm-benchmark.md)
Step 4: Accuracy evaluation (accuracy-eval.md)
Step 5: Present combined results (speed + accuracy)
Step 6: User satisfied? yes -> done / no -> lighter recipe, loop
```

### Usage

In Claude Code, say:

```
quantize /path/to/model with nvfp4
```

Or invoke directly:

```
/modelopt-optimizer
```

### Supported Quantization Formats

| Format | Description | Typical Accuracy Drop |
|--------|-------------|----------------------|
| `fp8` | 8-bit float (E4M3) | < 1% |
| `int8_sq` | 8-bit int + SmoothQuant | 1-2% |
| `w4a8_awq` | 4-bit weights, 8-bit activations | 2-3% |
| `int4_awq` | 4-bit int + AWQ | 2-4% |
| `nvfp4` | NVIDIA 4-bit float | 2-5% |

### Dependencies

- NVIDIA ModelOpt (`nvidia-modelopt` or dev install)
- vLLM (`pip install vllm`)
- NeMo Evaluator Launcher (`pip install nemo-evaluator-launcher`)
- HuggingFace Transformers (latest recommended)

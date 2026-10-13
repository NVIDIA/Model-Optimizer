# Evaluation

## Results

Qwen3-8B was evaluated using NEL with vLLM on one H100 per evaluation. QAD starts from the PTQ checkpoint and uses the BF16 model as teacher.

| Checkpoint | Deployed precision | MMLU accuracy ↑ | WikiText-2 perplexity ↓ |
|---|---|---:|---:|
| Teacher (BF16) | BF16 | 74.94% | 13.485 |
| PTQ (NVFP4) | NVFP4 weights / BF16 activations | 73.59% | 13.883 |
| PTQ → QAD (NVFP4) | NVFP4 weights / BF16 activations | 73.76% | 13.684 |

QAD improves both metrics over PTQ but remains below the BF16 teacher. The quantized checkpoints use Marlin's W4A16 fallback on H100; these results do not measure native NVFP4 W4A4 execution on Blackwell. KV cache is BF16 for all checkpoints.

## File structure

NeMo Evaluator Launcher (NEL) configurations and evaluation results for the [Hugging Face QAT/QAD tutorial](../README.md).

- `nel_configs/`: shared benchmark recipes and vLLM deployment settings.
- `results/<model-and-hardware>/<benchmark-and-scope>/`: each evaluation's config `nel_config.yaml`.

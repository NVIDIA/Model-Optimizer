# PDD for Qwen-Image

Parallel Decoding Distillation (PDD) trains one Qwen-Image student call to predict several
consecutive rectified-flow updates. The student keeps the original transformer backbone and widens
only its output projection to 128 velocity heads. During training it samples different aligned
block lengths up to 64, so the same checkpoint can use different supported block schedules at
inference.

The provided schedules use the 128-interval grid as follows:

| Schedule | Block sizes | Transformer calls |
|---|---|---:|
| `pdd-2` | `[64, 64]` | 2 |
| `pdd-4` | `[32, 32, 32, 32]` | 4 |
| `pdd-8` | `[16, 16, 16, 16, 16, 16, 16, 16]` | 8 |

## Training

Install the shared requirements from the repository root, then launch with released AutoModel
APIs. No AutoModel, Diffusers, or Qwen source changes are required.

```bash
pip install -r examples/diffusers/fastgen/requirements.txt
export MODELOPT_FASTGEN_DATASET_CACHE_DIR=/absolute/path/to/qwen_image_cache
torchrun --standalone --nproc-per-node=8 \
  examples/diffusers/fastgen/pdd/finetune.py \
  --config examples/diffusers/fastgen/pdd/configs/qwen_image.yaml
```

The cache must contain `metadata.json`, its declared shards, cached tensors, and
`negative_prompt_embedding.pt`. The environment variable overrides the configured cache root. All
metadata, tensor, and negative-embedding paths must still resolve inside that effective root.

Training deterministically derives disjoint train and validation membership from metadata ordinals;
it does not rewrite the cache or require separate split manifests. The default recipe uses 2,000
validation samples, learning rate `2e-5`, 128 heads, and sampled block lengths from 4 through 64.
Checkpoints include the student, optimizer, scheduler, RNG, trainer, and exact replayable sampler
state needed to resume the next committed batch.

Start with a one-node smoke and scale only after it passes; project training runs are capped at 16
nodes.

## Export and inference

```bash
torchrun --standalone --nproc-per-node=8 \
  examples/diffusers/fastgen/pdd/export_qwen_image.py \
  --config examples/diffusers/fastgen/pdd/configs/qwen_image.yaml \
  --checkpoint LATEST --output-dir /path/to/pdd-export

python examples/diffusers/fastgen/pdd/inference_qwen_image.py \
  --export-dir /path/to/pdd-export --schedule pdd-4 \
  --prompt-id red-cube-0001 --prompt "a small red cube on a white table" \
  --seed 42 --height 1024 --width 1024 \
  --output /path/to/pdd4.png --result-json /path/to/pdd4.json
```

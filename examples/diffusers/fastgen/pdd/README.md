# PDD for Qwen-Image

Parallel Decoding Distillation (PDD) trains one Qwen-Image student call to predict several
consecutive rectified-flow updates. The student keeps the original transformer backbone and widens
only its output projection to 128 velocity heads. During training it samples aligned block starts
and target spans from 1 through 64 intervals, so the same checkpoint can use different supported
block schedules at inference.

The provided schedules use the 128-interval grid as follows:

| Schedule | Block sizes | Transformer calls |
|---|---|---:|
| `pdd-2` | `[64, 64]` | 2 |
| `pdd-4` | `[32, 32, 32, 32]` | 4 |
| `pdd-8` | `[16, 16, 16, 16, 16, 16, 16, 16]` | 8 |

## Training

Install the shared requirements from the repository root, then launch with released AutoModel
APIs. No AutoModel, Diffusers, or Qwen source changes are required.

The example uses the ordinary Diffusers Qwen transformer without replacing or monkeypatching its
forward. Diffusers owns Qwen timestep conversion and converts each text mask into the joint
text/image attention mask, so padded text tokens do not participate as attention keys. ModelOpt
owns only PDD projection conversion, latent packing, condition validation, and packed per-token
classifier-free guidance using the Qwen pipeline formula. Guidance-embedded models and PEFT remain
outside this first example.

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
PDD's committed sampler makes the next batch, sample IDs, RNG, optimizer, and scheduler state
exactly replayable without requiring per-payload hashes. Hashless caches must therefore remain
immutable for the duration of a run: metadata and the negative embedding are bound into the
checkpoint identity, but an in-place tensor payload change cannot be detected.

Set `data.dataloader.verify_payload_hashes: true` to require the `cache_sha256` field written by
the shared preprocessor and authenticate every tensor's bytes before loading it. This stricter mode
has additional read and SHA-256 cost and fails immediately when a hash is missing or mismatched.

Training deterministically derives disjoint train and validation membership from metadata ordinals;
it does not rewrite the cache or require separate split manifests. The default recipe uses 2,000
validation samples, learning rate `2e-5`, per-rank batch size 4, 128 heads, start indices aligned by
4, and target spans from 1 through 64 intervals. The learning rate is the cached-Qwen project
treatment; the MR210 reference arm uses `5e-5` with 1,000 warmup steps. On 16 four-GPU nodes the
default per-rank batch gives global batch size 256; other GPU topologies must set per-rank batch to
`256 / world_size` because this recipe does not use gradient accumulation.
Checkpoints include the student, optimizer, scheduler, RNG, trainer, and exact replayable sampler
state needed to resume the next committed batch. FP32 master parameters and Adam state are sharded
while forward/backward uses the configured model dtype and gradient reduction remains FP32. The
adapter casts packed image/text inputs to that compute dtype; the ordinary Diffusers Qwen forward
owns timestep conversion.

Start with a one-node smoke and scale only after it passes; project training runs are capped at 16
nodes. Checkpointed training, export, and inference require the remote model ID and exact lowercase
40-character Hugging Face commit in the provided config; local model directories are limited to
low-level hermetic setup tests because the frozen teacher is rebuilt rather than checkpointed.

## Export and inference

The v1 export restores the sharded checkpoint with the same total process count that created it;
cross-world-size DCP resharding is not yet part of this example. The command below therefore
applies to a checkpoint trained with eight ranks. For a 64-rank checkpoint, use the cluster
launcher with 64 export ranks before running single-process inference.

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

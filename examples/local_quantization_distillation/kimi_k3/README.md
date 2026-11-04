# Kimi-K3: one pretrained MoE, IQ1_S PTQ → Bypass

This study compares four IQ1_S initializers and repairs **each** one. It measures
held-out MSE at the complete first routed MoE output (`model.layers.1.mlp`),
including its native router, latent projections, normalization inside the MoE,
shared experts and SiTU. Only routed-expert gate/up/down weights are quantized
and their native scales optimized. External decoder normalization and attention
are outside this repair objective.

The teacher is the **released checkpoint**, whose routed experts are already
MXFP4, decoded by AutoModel's native Kimi adapter to BF16. It is not an unavailable
original BF16 teacher. This is an isolated-layer fidelity study, with no whole-model
PPL, KLD or downstream-quality claim.

## Run

Install this ModelOpt checkout, a matching NeMo AutoModel environment with native
Kimi/KDA support, and the optional tokenizer dependency. The tested environment
uses NeMo AutoModel revision `bf6cae1d32d3a34a88aa6352f62f0cf1a9bb156a`,
PyTorch 2.14 / CUDA 13.4, and four GB300 GPUs (284 GiB each) with EP4.
These commands assume a compatible environment is already installed:

```bash
python -m pip install -r examples/local_quantization_distillation/kimi_k3/requirements.txt
export STUDY=/persistent/path/kimi-first-moe
python -m examples.local_quantization_distillation.kimi_k3.prepare --output "$STUDY"

# Run on one allocated four-GPU node. These commands do not allocate nodes.
torchrun --standalone --nproc_per_node=4 \
  -m examples.local_quantization_distillation.kimi_k3.capture --root "$STUDY"
torchrun --standalone --nproc_per_node=4 \
  -m examples.local_quantization_distillation.kimi_k3.study --root "$STUDY"
```

Preparation pins `moonshotai/Kimi-K3` revision
`f831ab66814297da540d832a5235f8e904f29d06` and Nemotron Post-Training Dataset v2
revision `5c89e01dd720ae0f4058445ed49c5fb68a03c76e`. Only three required source
shards are downloaded. Only selected tensors are loaded from them; the full 2.8T
model, vision tower and language-model head are never materialized. The pinned
released TikToken tokenizer and chat encoder are loaded with `trust_remote_code`;
model execution uses native AutoModel implementations.

Capture runs embeddings, the preceding dense block and the target block's
attention/normalization, then stops immediately before its MoE. The target MoE
is unallocated during capture. All subsequent PTQ/repair work loads only that
one MoE and uses native EP4. Kimi's attention-residual state, KDA document resets
and native packed metadata remain intact in the short prefix. The prefix is
unloaded before PTQ/repair starts. TP is unsupported by native Kimi; CP is
unnecessary for token-independent MoE replay.

## Exact comparison

| Setting | Value |
|---|---|
| Initializers | Unweighted IQ1_S; importance-only (`imatrix`); GPTQ; importance-weighted GPTQ |
| GPTQ | Block 256, damping 0.01, unfused; same calibration inputs |
| Capture context | 8,192 packed tokens, original chat encoding and document boundaries |
| Calibration | 64 packs = 524,288 tokens |
| Repair training | 1,024 captured packs, split into 2K samples = 8,388,608 unique tokens |
| Budget | 512 updates × eight samples globally; two samples per EP rank, no accumulation |
| Optimizer | Adagrad, scale LR 0.01, frozen native codes, FP32 log-scale masters |
| Schedule | 5% warmup then cosine decay; gradient clip 1; final state |
| MSE validation | 32 separate packs = 262,144 tokens |
| Final held-out MSE | 64 separate packs = 524,288 tokens |

All four partitions are conversation-disjoint and deduplicated. Every method sees
identical inputs in identical order. The loss includes every selected token,
including prompts; no language-model labels or logit distillation are used.

The study stores only exact BF16 **inputs**, not teacher outputs, using approximately
129.5 GiB of temporary activation storage. It retains one teacher and one current
candidate, runs methods sequentially, and saves small metrics/provenance only.
No per-method model or optimizer checkpoints are kept. Completed methods are
skipped on rerun; an interrupted method starts again from PTQ. Complete activation
captures can be reused with matching source, data and topology. They are removed
after all four methods succeed, unless `--keep-activations` is specified.

For a smoke check, add `--smoke` to both capture and study and use
`--methods unweighted` for study. This uses two capture rows per rank per split
and two repair updates. Smoke results live in `smoke/` and are not final results.

## Results

`RESULTS.md` and `results.json` report each initializer's held-out MSE before and
after repair, normalized MSE, reduction percentage, valid token counts, timings,
training curves and expert exposure. Source/data revisions and the repair budget
are recorded with the results. Compare the same final held-out split across every
row. Read-only held-out measurements use ModelOpt's scoped decoded-weight
cache; it is closed and disabled before repair starts. Cache hit/miss and peak
retained-byte counters are included in each diagnostic. Local timers are component wall times; use scheduler accounting for allocated
GPU-hours. The main tutorial's [full-model results](../README.md) use KLD instead.

## Recorded comparison

Measured with the source, data, hardware and 512-update recipe above. Each row
uses the same 524,288 held-out tokens, separate from calibration, training and
MSE validation. Lower MSE is better.

| IQ1_S initializer | PTQ held-out MSE | After Bypass | Reduction |
|---|---:|---:|---:|
| Unweighted | 1.5861824e-05 | 1.1926605e-05 | 24.81% |
| Importance-only | 1.2602503e-05 | 9.9510914e-06 | 21.04% |
| GPTQ | 6.9347793e-06 | 6.4845498e-06 | 6.49% |
| Importance-weighted GPTQ | 5.8258669e-06 | 5.5370421e-06 | 4.96% |

Importance-weighted GPTQ plus Bypass has the lowest final MSE. Each initializer
benefits from repair; the improvement is smaller for the stronger GPTQ starting
points. This measures fidelity of this one MoE to the released teacher, without
a whole-model quality claim. Each repair call, including its MSE validation and finalization, takes
approximately 4–4.6 minutes on the tested four-GPU node. This excludes activation
capture, PTQ and final held-out diagnostics. The successful four-method job used
27:09 of wall time (1.81 allocated GPU-hours), including a two-update parity
smoke and process startup; earlier prefix capture and debugging are excluded.

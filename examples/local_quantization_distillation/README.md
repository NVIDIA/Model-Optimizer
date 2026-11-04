# Local quantization distillation with AutoModel

Bypass repairs the output of each selected quantized MLP/MoE independently. A
teacher forward supplies its original input and target output; the student unit
replays that input and immediately backpropagates its own loss. All selected units
train during the same pass. Hidden states are retained only for the current
microbatch, rather than written as a dataset or kept for the entire model.

For a unit `u`, the default objective is
`sum_valid ||student_u(teacher_input_u) - teacher_output_u||² / valid_elements`.
It includes all valid tokens, including prompts. There is no language-model loss
or logit distillation in repair. The unit includes exactly the module selected by
its path: a complete MoE includes its router, routed/shared experts and combine;
normalization outside that module is outside the objective. Choose a decoder
block explicitly to include its normalization, attention and residual path.

## Defaults

| Setting | GLM tutorial |
|---|---|
| Source | Original GLM-5.3-Flash teacher; specify its path |
| PTQ | Routed experts, GPTQ IQ2_S; other modules unchanged |
| GPTQ | Damping 0.01, update block 256, importance weighting on, unfused |
| Data | Nemotron Post-Training Dataset v2: stem/chat/math/code |
| Context | 32,768 packed tokens; calibration uses a 4,096-token prefix |
| Update batch | 8 sequences globally; microbatch 1, accumulation as needed |
| Budget | 512 updates, 4,096 **unique** packs, 134,217,728 training tokens |
| Optimizer | Adagrad, no weight decay, FP32 positive log-scale masters |
| Learning rates | Scales 0.01; optional shadow weights 0.00001 |
| Schedule | 5% linear warmup, cosine decay to zero; gradient clip 1 |
| Validation | 128 independent packs, baseline and final MSE |
| Selection | Final state; independent best-validation state is optional |
| Saves | Every 128 updates; publish every rank before retiring the prior save |

These are starting values, not a claim that one recipe is optimal for every
format/model. The generic API defaults to context 4,096; the GLM YAML explicitly
selects 32K. Data size, context, batch, steps and topology are explicit settings.

## GPTQ → repair → held-out diagnostics

Install this checkout and a compatible NeMo AutoModel environment. Run from the
repository root so the example modules are importable. Work in a persistent data
directory and use your cluster's allocation/launcher; these commands do not
allocate nodes. Never place caches or model files in a home directory on MARS.

```bash
export TEACHER=/path/to/original/glm53-flash
export REPAIR_DATA=/path/to/unique-packed-corpus
export EXPERIMENT=/path/to/local-repair
python -m examples.local_quantization_distillation.prepare_data \
  --tokenizer "$TEACHER" --output "$REPAIR_DATA"
```

Preparation pins dataset/tokenizer revisions, renders the original chat template,
deduplicates rendered conversations, and assigns each conversation to one split
by a stable hash. A conversation may span multiple packs, but its token intervals
never repeat or cross splits. Native AutoModel neat packing supplies document
IDs and reset positions. Exhausting unique data is an error; there is no epoch
wrap or distributed sampler padding. The manifest records file hashes, source
revisions, provenance and actual token budgets. A longer budget shares the same
training prefix when the source and seed are unchanged.

Copy [the GLM YAML](configs/glm53_flash_iq2_s.yaml), set `model.source` to
`$TEACHER`, and choose the realized topology in `parallelism`. The default uses
FSDP2 without EP/CP/TP. Increase EP/CP according to the model and allocation. EP
and CP have different ownership: CP shards tokens; EP distributes expert weights
and does not create extra independent samples. The global batch stays eight.
The example uses native eager experts during GPTQ so calibration sees exactly
the routed inputs, then restores grouped GEMMs for repair. It prints the complete GPTQ recipe before quantization.

```bash
# Replace N with the number of allocated local GPUs; use your multinode
# torchrun rendezvous options when the model needs more than one node.
torchrun --standalone --nproc_per_node=N \
  -m examples.local_quantization_distillation.repair \
  --config /path/to/glm-repair.yaml --data "$REPAIR_DATA" \
  --ptq-only --output "$EXPERIMENT/ptq"

torchrun --standalone --nproc_per_node=N \
  -m examples.local_quantization_distillation.repair \
  --config /path/to/glm-repair.yaml --data "$REPAIR_DATA" \
  --checkpoint "$EXPERIMENT/ptq" --output "$EXPERIMENT/repaired" \
  --checkpoint-dir "$EXPERIMENT/resume"

# Same held-out packs and full-vocabulary KL(teacher || candidate).
torchrun --standalone --nproc_per_node=N \
  -m examples.local_quantization_distillation.evaluate \
  --config /path/to/glm-repair.yaml --data "$REPAIR_DATA" \
  --checkpoint "$EXPERIMENT/ptq" --output "$EXPERIMENT/ptq-diagnostics.json"
torchrun --standalone --nproc_per_node=N \
  -m examples.local_quantization_distillation.evaluate \
  --config /path/to/glm-repair.yaml --data "$REPAIR_DATA" \
  --checkpoint "$EXPERIMENT/repaired" --output "$EXPERIMENT/repair-diagnostics.json"
```

Use the same mesh and model skeleton for native checkpoint restore. Add
`--resume-from "$EXPERIMENT/resume"` to the repair command after interruption.
Resume restores native fields, optimizer/scheduler position, RNG and the exact
consumed data prefix. Keep the starting PTQ source and packed data unchanged.

PPL scores valid within-document next-token targets. KL scores all valid input
positions, over the full vocabulary with bounded FP32 token/vocabulary chunks.
Prompts contribute to both when they have valid targets. All candidates use the
same diagnostic split, disjoint from training, MSE validation and calibration.
Reports include actual target/token counts, teacher PPL, timing and local peak
CUDA allocation. Scheduler accounting is the source of **allocated GPU-hours**;
do not infer it from a single rank's compute timer.

## Choose initializers by teacher KLD

This tutorial measures fidelity to the **released teacher**, not language-model
quality on its unknown original training distribution. We do not have that
training corpus. Nemotron v2 is a disjoint surrogate corpus, so a candidate with
lower CE/PPL here need not reproduce the teacher more faithfully. Select and
retain initializers by **full-vocabulary KL(teacher || candidate)** only. Keep
PPL as a secondary diagnostic, never as a repair-selection criterion.

Bypass training still minimizes each unit's hidden-state MSE over all valid
tokens; KLD is the held-out, whole-model criterion for selecting its PTQ starting
point and assessing its result. A KLD improvement is evidence of closer teacher
predictions on this corpus, not a substitute for downstream benchmark testing.

## Additional models and optional format providers

Use a native AutoModel source and configure that model's weight/activation
quantizers in YAML. Family-owned descriptors select decoder MLP/MoE boundaries
while excluding routers and auxiliary towers. The same repair entry point accepts
mixed formats; it does not depend on a model-specific study controller.

Optional backend providers register their own repair and native checkpoint
handlers. Import a provider explicitly before PTQ or native restore; see
[advanced configuration](ADVANCED.md#optional-external-format-plugins).

## Measured full-model examples

All candidates below use the same conversation-disjoint Nemotron diagnostic
split for their model: 4,194,304 valid KL positions. Lower KL means closer
predictions to the released teacher; it does not establish downstream quality.

| Model | Routed/MLP format | Lowest-KL PTQ | PTQ KLD | After Bypass KLD | Reduction |
|---|---|---|---:|---:|---:|
| MiMo-V2.6-Flash-RL | Routed experts IQ1_M | Importance-weighted GPTQ | 0.602981 | 0.508429 | 15.68% |
| Qwen3.8-27B | MLP NVFP4 W4A4 | GPTQ | 0.164113 | 0.160786 | 2.03% |

These runs use 512 updates, eight unique packed 32K sequences per update and
the optimizer/schedule above. MiMo's reference is its released mixed MXFP4/FP8
checkpoint decoded to BF16; Qwen's reference is its released BF16 checkpoint.
Source revisions are `f657ba2f7d24c9adc5730675baaa2d34de878a49` (MiMo) and
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Qwen). Measurements predate the
new native repair optimizations and do not measure their training speedup.

[Full PTQ comparison and measurement conditions](RESULTS.md).

## One-layer Kimi-K3 study

[The Kimi-K3 study](kimi_k3/README.md) compares four IQ1_S initializers and repairs
each, using real inputs to the first routed MoE and a separate held-out MSE split.
It loads only the short activation-capture prefix and the single target MoE,
never the full model. This local study reports MSE rather than whole-model KLD.

## Configuration and limits

[Advanced configuration](ADVANCED.md) covers mixed formats, activation
quantization, exact repair boundaries, EP/CP/TP/FSDP2/PP, native persistence,
family adapters and optional external providers.

[Native model coverage](MODEL_COVERAGE.md) distinguishes tiny compatibility
checks from pretrained studies. [Validation](VALIDATION.md) records correctness
and component-performance evidence. [Review map](REVIEW_MAP.md) explains source
ownership for contributors.

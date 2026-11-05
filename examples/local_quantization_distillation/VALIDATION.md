# Development validation

These measurements use tiny native modules on NVIDIA GB300 GPUs and validate
execution/serialization, not GLM-5.3-Flash quality or full-model performance.

| Check | Result |
|---|---|
| Native GLM, two sparse decoder blocks, IQ2_S GPTQ → two whole-MoE scale updates → native reload → packed PPL/full-vocab KL | All four ranks passed; test body 8.61 seconds (EP2 + CP2 + FSDP2) |
| Diagnostic accounting in that GLM probe | 128 KL positions, 112 within-document next-token targets |
| Native GLM diagnostic peak CUDA allocation after repair/reload | 86,900,224 bytes per rank |
| Slurm allocation for that probe | 158 seconds × 4 GPUs = 0.17556 allocated GPU-hours, including container/import/teardown |
| Earlier integrated native suite | All 14 cases passed on all four ranks in 56.61–56.62 seconds after independent-review fixes; allocation 218 seconds × 4 GPUs = 0.24222 allocated GPU-hours |
| Native expert calibration | Remote-only EP inputs captured after gather; grouped GEMMs restored after GPTQ; ordinary observation hooks remain compatible |
| Native Mamba W4A4 | All three projections calibrated and participate in gradients; packed reset, two updates and exact native reload passed |
| Native format forward/gradient/reload | INT8, IQ2_S, NVFP4 W4A4, MXFP4, MXFP6 E3M2/E2M3, MXFP8 E5M2, MXINT8, MX E1M2 and E4M3-scaled FP6 E3M2/FP8 E5M2 passed |
| Warm format repair/reload, two updates on tiny MLPs | 0.12–0.77 seconds per format; first-build time excluded |
| Mixed NVFP4/IQ2_S TP2 + DP2/FSDP2 | Exact repaired global weights and native DCP reload; strict FP32 output max error 7.15e-7; replicated scale gradients agree |
| Uneven data lanes | Both Gloo and GPU ranks reject uneven validation exhaustion; mixed TP/FSDP also rejects differently partitioned training windows |
| Native EP and FSDP2/TP/CP/physical PP | Short reference-update, checkpoint/resume and native restore checks passed; PP verifies student replay follows teacher schedule completion |

The GLM probe uses randomly initialized weights, short synthetic packed tokens,
and the tutorial's actual layerwise GPTQ helper. It is intentionally too small
to establish a quality claim, learning-rate recommendation or training speedup.
The published YAML's 512-update/32K budget has not been run as part of validation.

Timer fields distinguish synchronized repair-call wall time and process
GPU-hours from scheduler allocation hours. Memory fields report CUDA allocations,
not total device reservation, external runtime memory or a guarantee that the
full model fits. Native DCP round trips use a matching topology; deployment
export and untested format/topology combinations retain their own restrictions.

The independent review's three findings were addressed: collective agreement on
microbatch availability, one master for shared activation thresholds, and
teacher-then-student PP execution. The curated focused CPU suite passed 161 cases in 54.82 seconds after removing
11 redundant new cases. A subsequent local-distillation and attention recheck
passed 43 cases in 6.80 seconds. Required repository pre-commit checks passed.

## Model-family review

The [revision-pinned compatibility matrix](MODEL_COVERAGE.md) passed all 31
cases: 27 native checkpoint/config variants run IQ2_S GPTQ → two Bypass updates
→ exact-logit native reload; four Qwen GDN variants run real W4A4 calibration,
all-weight-scale gradient checks, unchanged activation ranges and exact reload.
This covers 10 natively supported checkpoints among the top 15 distinct open-weight
Artificial Analysis entries and all 18 identified official generative Qwen
3.5/3.6/3.8 checkpoints. The five unregistered architectures are listed explicitly.

Job 1527875 passed the checked-in matrix in four independent GPU lanes (8, 8,
8 and 7 cases; slowest lane 113.29 seconds). Its allocation was 260 seconds ×
four GPUs = 0.28889 allocated GPU-hours. A development probe measured peak CUDA
allocation of 74.6–367.6 MB across 23 passing language-family variants; these are
tiny-prefix measurements and exclude total reserved/external device memory.

The native/distributed suite now has 15 cases. Thirteen passed in job 1527996;
the existing self-attention CP reference/resume check passed on the focused
rerun 1528184 with its original compiler stance and reusable native kernel cache.
The force-eager/cold-cache harness had hit its 240-second backward timeout.
The new Flash-Next full-decoder case passed on all four ranks in job 1528257
(29.23-second body), using the model's native EP2 + CP2 + FSDP2 setup. It verifies
packed boundaries, padding exclusion, actual scale gradients and replay of
a native CP context containing a real process group.

The review fixed packed GLM DSA token masks, DeepSeek V4.1 FFN/packing aliases,
Qwen GDN fused-convolution quantizer participation, complete attention-class
registration before save/reload, and Flash-Next packed CP/context snapshots.
A follow-up independent source review found no remaining substantive issues
in those changes. These checks do not establish pretrained quality, a full-model
training budget, or every format/topology combination.

## Additional format verification

The published-format matrix covers native IQ2_S, IQ1_M, IQ2_XS and NVFP4 W4A4
across 29 tiny native model/config variants, plus four targeted GDN cases.
These tiny probes establish compatibility, not pretrained quality or full-model
throughput.

## Full-main review cleanup (2026-11-04)

The full working tree, including untracked source, was reviewed against main at
`7d442467504cdae155812baf15242628d96971b5`. See `REVIEW_MAP.md` for ownership and
review order. AutoModel integration is now one optional plugin package; generic
training code does not import that runtime. Obsolete manual FSDP DTensor
writeback was removed in favor of native parameter ownership.

The review reproduced and corrected stale native-payload policy keys, compressed
NVFP4 tiled-scale layout, backend derived-scale cache invalidation, supported
NVFP4 AWQ/MXFP4 pinned export gates, partial-block INT4/NF4/MXFP8 handling and
secondary PPL overflow. The targeted CPU regression set passed **37 tests in
10.74 seconds**. Following the expert-observation addition, all nine generic
local-distillation cases passed again in 3.93 seconds.

Native prefix GPTQ/repair/reload checks for MiMo IQ1_M and Qwen NVFP4 W4A4
passed (job 1538132). CP replay/resume, mixed TP/FSDP native DCP,
and embedding-only physical PP checks passed on all four ranks (job 1538356).
The remote-input EP reference update, now including a globally unobserved expert,
passed on all four ranks (job 1538930, 6.66-second body). It verifies exposure
counts without changing the numerical reference update.

A small public HF export probe passed for repaired NVFP4 AWQ and weight-only
MXFP4. Existing AWQ resmoothing legitimately re-encoded one up projection;
MXFP4 preserved all three payloads. This supports the documented distinction
between exact native persistence and deployment transforms.

**Coverage limit:** the validation image does not contain TensorRT-LLM. The
real NVFP4 GEMM regression was extended in the existing TRT suite but could not
run here. CPU tiled-field tests and native GPU fake-quant paths do not replace
that serving-backend test. No full format/topology Cartesian-product claim is made.

The completed full MiMo study uses its frozen runtime snapshot, predating these
cleanup changes and new expert counters. Its final quality measurements are
recorded in [RESULTS.md](RESULTS.md), separately from the tiny compatibility checks.

Final native GPU checks passed in job 1538987: both MiniMax multimodal text-family
variants with quantized vision weights resolve only decoder repair defaults
(6.58-second body); compressed NF4 FP32 and BF16 prepare/update/finalize/native
reload preserve exact outputs and packed codes (0.75-second body). NF4 follows
the native CUDA BF16 scale/output rounding. Restore creates native field buffers
on CPU, so the test follows restore with an explicit device move before forward.
All repository hooks passed on all 80 changed/new files, including mypy, Ruff,
licenses, CUDA formatting and documentation checks.

## Native MoE replay optimization (2026-11-04)

The shared runner batches compatible expert scale projections and decodes IQ1_M,
IQ2_S and IQ2_XS integer levels in one GPU launch. The native model still owns
routing, activation functions, shared experts and EP dispatch. A primitive check
compares all three decoders bit-for-bit against their original implementations,
with exact FP32/BF16 forward outputs and matching scale gradients (job 1541362).

Job 1541412 measures a full-width native MoE forward/backward component with
EP4 and 8,192 tokens per rank. Each configuration uses its model-owned native
configuration and legal synthetic packed weights. Three synchronized samples
per path report median maximum-rank time and peak allocated CUDA memory:

| Native family | IQ format | Reference → optimized time | Speedup | Peak allocation, reference → optimized |
|---|---|---:|---:|---:|
| GLM-5.3-Flash | IQ2_S | 2.343 → 0.101 seconds | 23.16× | 28.18 → 23.59 GiB |
| MiMo-V2.6-Flash | IQ1_M | 2.696 → 0.093 seconds | 28.97× | 26.43 → 22.35 GiB |
| DeepSeek-V4-Flash | IQ2_S | 2.090 → 0.064 seconds | 32.47× | 17.21 → 13.13 GiB |

Outputs match exactly; maximum absolute scale-gradient differences are below
4e-9. These are synthetic component measurements against the previous reference
repair path, **not whole-model training speedups**. Attention, teacher execution,
optimizer updates and checkpoint I/O are outside the timed region.

The same job passed five focused distributed scenarios on every rank: two
physical PP layouts (including an embedding-only stage), EP reference replay,
packed CP MLP replay and mixed TP/FSDP ownership. PP verifies that supported
teacher heads receive one token while hidden-state repair retains all valid
tokens. Full-vocabulary diagnostic logits remain unchanged.

The expanded public matrix passed **120/120 cases** in job 1541412 (four GPU
lanes, 106.21–114.48 seconds per lane). It now includes the exact
DeepSeek-V4-Flash configuration and Kimi-K3 text prefix. The hash-routing check
first failed before the side-input replay fix, then passed with identical
teacher/student expert IDs. Generic repair checks passed 10 cases in 1.92 seconds.

## Individual IQ replay and adapter finalization

Job 1542457 passed the extended individual IQ1_M/IQ2_S/IQ2_XS native replay
checks for FP32 and BF16 on all four GPUs. The same check first failed in job
1542411 because individual states still invoked the reference unpacker. It now
proves exact outputs, matching projected scale gradients and unchanged finalized
outputs. CPU and unsupported-codec fallbacks remain available.

Native writeback now belongs to each format state; the policy resolves the
current weight binding and delegates, including within existing FSDP windows.
The focused public CPU suite passed 47 cases in 4.68 seconds, covering native
payloads, transform folding, activation handling, compressed formats and resume.
A fresh independent reviewer also passed ten generic CPU cases. Required
repository hooks passed.

Job 1542510 passed eight focused native/distributed cases on every rank in
20.18 seconds: the native format matrix, NF4 FP32/BF16, EP, CP, mixed TP/FSDP
and two PP layouts. Its allocation completed successfully and was released.
A synthetic 11,008×4,096 individual-quantizer forward/backward probe measured:

| Format | Compute dtype | Reference → native seconds | Speedup | Peak allocation saved |
|---|---|---:|---:|---:|
| IQ1_M | BF16 | 0.020009 → 0.000795 | 25.16× | 0.452 GiB |
| IQ2_S | BF16 | 0.016409 → 0.000827 | 19.83× | 0.451 GiB |
| IQ2_XS | BF16 | 0.017111 → 0.000821 | 20.84× | 0.451 GiB |
| IQ2_S | FP32 | 0.016227 → 0.000838 | 19.36× | 0.283 GiB |

These are medians of five synchronized samples with legal synthetic packed
payloads. They measure quantizer forward/backward, excluding matrix multiplication,
teacher execution, optimizer and I/O; they are not whole-model speedups.

## Optimizer/reporting and additional native families (November 5, 2026)

The compatible-field optimizer uses bounded 64 MiB groups while preserving one
optimizer per repair unit, canonical collective order and the legacy per-field
optimizer checkpoint representation. The independent CPU reference checks cover
Adagrad and AdamW updates plus exact scale-only/joint resume. All 73 focused CPU
repair/calibration cases passed, including prefix-replay PTQ resume.

A full-size representative IQ1_S scale bank (896 FP32 fields, 260,112,384
parameters) was benchmarked with matched updates. Grouping reduced 896 optimizer
groups to 16: mean optimizer-step time was 65.49 ms before and 8.52 ms after,
with maximum parameter difference zero. Peak allocated CUDA memory increased
from 6,336,020,480 to 6,463,225,856 bytes (about 121 MiB). This is a component
benchmark, not a full-repair throughput claim; the 64 MiB bound avoids an
unbounded foreach allocation across the complete MoE.

Job 1569709 passed 132 native GPU cases on AutoModel revision
`752a71ebf731f84b19ca9af5e7c242be003082bd`: 32 representative model prefixes
across IQ2_S, IQ1_M, IQ2_XS and NVFP4 W4A4, plus four GDN checks. This includes
Muse Spark, Mistral Large 4 and Nemotron 3.5 Super VL. The new cases include packed
padding; Muse/Nemotron also execute their real native image encoders. The same
job passed 12 distributed scenarios on every one of four ranks: native/shared-runner
EP updates and expert coverage, GLM PTQ/diagnostics, packed CP MLP/attention/decoder
repair, TP/FSDP2 including mixed formats, and physical PP with resume. The earlier
AutoModel revision `bf6cae1d32d3a34a88aa6352f62f0cf1a9bb156a` also passed the
native GLM IQ2_S GPTQ/repair/reload control.

Job 1569798 passed the 12 new-family cases again plus the existing selective
Kimi first-MoE test, completing all 133 family-matrix items. The Mistral IQ2_S case
also rewinds a layerwise PTQ checkpoint at a shared-K/V producer, restores a fresh
model, and matches uninterrupted PTQ logits exactly. Prefix replay recomputes its
native layout instead of serializing arbitrary Python objects.

Coverage remains a compatibility result on small random models. It establishes
neither pretrained quality nor every family/format/distributed-topology combination.
See [model coverage](MODEL_COVERAGE.md) for native model and modality limits.

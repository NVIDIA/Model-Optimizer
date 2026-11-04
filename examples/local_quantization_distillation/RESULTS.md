# Full-model PTQ and Bypass results

These are paired full-vocabulary KL(teacher || candidate) measurements on
conversation-disjoint Nemotron Post-Training Dataset v2 diagnostic data.
Each model uses 4,194,304 valid token positions; lower is better. PPL is
secondary on this surrogate corpus and does not select initializers.

## MiMo-V2.6-Flash-RL / routed IQ1_M

| PTQ method | PTQ KLD | Bypass KLD |
|---|---:|---:|
| gptq | 0.622360 | — |
| gptq_importance | 0.602981 | 0.508429 |
| imatrix | 0.917330 | — |
| unweighted | 1.178278 | — |

Only the minimum-KLD initializer receives repair; dashes mean repair was not run.

## Qwen3.8-27B / MLP NVFP4 W4A4

| PTQ method | PTQ KLD | Bypass KLD |
|---|---:|---:|
| awq_clip | 0.197120 | — |
| awq_full | 0.184751 | — |
| awq_lite | 0.177476 | — |
| gptq | 0.164113 | 0.160786 |
| local_hessian | 0.170606 | — |
| max | 0.178047 | — |
| mse | 0.173246 | — |

Only the minimum-KLD initializer receives repair; dashes mean repair was not run.

## Recipe and reference

Both studies use 512 updates, eight unique packed 32K sequences per update,
Adagrad scale LR 0.01, 5% warmup/cosine decay, frozen codes and final-state
selection. Calibration, training, validation and diagnostics are separate splits.
MiMo uses its released mixed MXFP4/FP8 weights decoded to BF16; Qwen uses its
released BF16 checkpoint. Model revisions are recorded in the main tutorial.
These results predate the new native repair optimizations. They establish
fidelity on the diagnostic corpus, not downstream benchmark performance.

The [Kimi single-MoE study](kimi_k3/README.md) instead compares all four
PTQ→repair pairs by local held-out MSE. It does not measure whole-model KLD.

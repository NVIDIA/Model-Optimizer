# Plain Neumann candidate: qualification result

**Decision: reject this candidate for the pinned KDA checkpoint and retain exact
solve.** The software implements an explicit differentiable polynomial, but none
of the predeclared degrees passed the model-quality screening margin. These
configs are research examples, not recommended quantization recipes.

## Matched validation screen

Implementation revision: `fc6ba9dcd06383e430cb5e29e27830d86c3b4369`.
The [fixed plan](../examples/llm_qat/linear_attention/solve_study_plan.json) used
`arcee-ai/AFM-4.5B-Base-KDA-Only` revision
`01ad2e06ee4f1214193c17b69e09105a9b257e80` and WikiText-2 raw revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. All trials used identical token
hashes, 32 validation blocks of 128 predicted tokens (4,096 total), seed 2026,
and no training. Projection weights were unchanged. Controls and candidates ran
on H100 with FP32 working arithmetic, BF16 outer autocast, and TF32 disabled.

The acceptance rule required the upper end of the descriptive 95% paired-block
bootstrap interval for the NLL increase to be at most 0.02. This interval does
not establish broad downstream-task quality.

| Policy | Validation perplexity | Mean NLL increase vs exact | 95% interval |
| --- | ---: | ---: | --- |
| Exact materialized | 13.6647 | 0 | — |
| FP8 operands, exact solve | 13.6684 | 0.000272 | — |
| Neumann degree 3 | 644,766.6 | 10.7618 | [10.5895, 10.9342] |
| Neumann degree 7 | 611,215.2 | 10.7084 | [10.5448, 10.8766] |
| Neumann degree 15 | 362,093.8 | 10.1848 | [10.0034, 10.3771] |
| Neumann degree 31 | 98,961.5 | 8.8877 | [8.7282, 9.0551] |

No degree was selected. The plan's held-out test/training comparison was therefore
not launched, and test data remained untouched. A separate one-step degree-31
integration smoke test had finite gradients for all trainable attention parameters
and updated query weights; its loss was 11.8963 with pre-clip gradient norm 616.6.
That verifies training plumbing, not quality recovery.

## Why the polynomial fails

A diagnostic observed the exact model path without changing it and retained the
largest-Frobenius-norm chunk/head matrix from each of 36 layers on the first
validation block. Maximum absolute interaction was only 0.9922, while maximum
matrix Frobenius norm was 37.67. Entry magnitudes below one and triangular
nilpotence do not prevent intermediate matrix powers from growing.

At degree 3, the maximum kernel-versus-independent-FP64-polynomial relative
error was `1.34e-7`, and the corresponding gradient error was `8.79e-8`.
Yet the polynomial-versus-exact-inverse relative error reached 1,242 and its
relative residual reached 24,468. Approximation error dominates this case.

At degree 31, the polynomial-versus-exact error reached `3.92e15`; FP32 kernel
and gradient errors relative to the FP64 polynomial also reached about 1.4%.
A matrix-only degree-63 diagnostic showed catastrophic cancellation: even the
FP64 power sum differed from the exact inverse by up to 1.71 relative error, and
the FP32 candidate was much worse. **Degree 63 is not a safe numerical fallback.**
It is exact only in exact arithmetic. The implementation never silently falls
back or changes degree.

## Cost and correctness evidence

On H100, 64 synthetic FP32 chunk matrices with strict-lower Gaussian standard
deviation 0.04, three warmups and 20 interleaved samples gave these median total
forward/backward wall times:

| Solve | Wall time, ms |
| --- | ---: |
| Exact triangular solve | 0.392 |
| Triton degree 3 plus recomputed backward | 0.778 |
| Triton degree 7 plus recomputed backward | 0.928 |
| Triton degree 15 plus recomputed backward | 1.071 |
| Triton degree 31 plus recomputed backward | 1.253 |

Complete KDA prefill at `[1,1024,4,64]` took 22.116 ms for exact materialized
arithmetic, 22.515 ms for degree 3, and 22.695 ms for degree 7. Fused FLA took
2.462 ms. These measurements establish no speed advantage.

The implementation passed 84 H100 tests, 167 CPU numerical/configuration tests,
and six comparison-audit tests. Tests include actual-polynomial gradients,
residual identity, composed QDQ, packed tails, saved policy, and KDA/Megatron
training integration. Those tests establish implementation behavior in their
specified input regimes; they do not imply uniform numerical conditioning or
acceptable model quality. Exact solve remains the supported baseline while a
better-conditioned approximation is investigated separately.

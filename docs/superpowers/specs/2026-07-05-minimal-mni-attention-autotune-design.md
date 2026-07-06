# Minimal MNI-Aligned Attention Autotune

## Status

Approved in conversation on 2026-07-05.

## Context

The compact attention branch currently builds a union of dense and P-QDQ Triton
configurations, uses a long semantic cache key, and prunes configurations according
to P-QDQ and sparse-mask flags. This makes the launch policy difficult to review.

The authoritative MNI reference uses one prefill schedule for dense and every
Q/K/P/V QDQ combination. Its KV tile is fixed at 32, independent of quantization
flags. B200 measurements of the ModelOpt kernel with that fixed KV tile selected
query tiles 16, 64, and 128 at sequence lengths 512, 2048, and 8192 respectively;
all selected two stages and four warps.

## Goals

- Use one small, identical eligible configuration set for all normal forward modes.
- Fix the KV tile to 32 to match the MNI prefill numerical schedule.
- Retain query-tile autotuning for short and long prefill efficiency.
- Remove the union configuration construction and mode-aware pruning function.
- Keep sparsity-counter measurement as a direct single launch because autotune
  trials would mutate its atomic counters.

## Non-Goals

- Reproduce MNI decode tiling or replace the compact split-K decode kernel.
- Align Q/K/P/V carrier dtypes, dot precision, global-scale lifecycle, or every
  other MNI arithmetic detail.
- Redesign skip-softmax or N:M dense-region semantics.
- Claim that KV tile 32 is the unrestricted throughput winner. It is the MNI
  compatibility contract; unrestricted B200 tuning selected a different tile.

## Design

Normal forward launches use one Triton autotuner with three configurations:

```text
BLOCK_M = {16, 64, 128}
BLOCK_N = 32
num_stages = 2
num_warps = 4
```

The autotune key contains only the shape regime and QDQ properties that can change
the best query tile:

```text
N_CTX, HEAD_DIM, Q_IS_FP32, P_QDQ, V_QDQ
```

K QDQ does not need a kernel key because it is materialized before this kernel and
does not change the kernel's dtype or temporary storage. Under pytest, the eligible
set is reduced to the single `BLOCK_M=16, BLOCK_N=32` configuration so tests do not
run benchmark warmups.

The following implementation elements are removed:

- `_FWD_BLOCK_M_CHOICES`
- `_FWD_BLOCK_N_CHOICES`
- `_P_QDQ_BLOCK_M_CHOICES`
- `_FWD_AUTOTUNE_KEYS`
- `_prune_fwd_configs`
- Tests coupled to Triton's early-pruning internals

The direct measurement launch remains outside the autotuner. Its existing stable
measurement geometry remains unchanged in this refactor; changing skip-softmax
calibration geometry is separate work.

## Behavioral Consequences

- Dense, Q-only, K-only, P-only, V-only, and combined Q/K/P/V paths see the same
  eligible physical tile set.
- P-QDQ output no longer depends on a performance-selected KV tile because every
  eligible configuration uses KV tile 32.
- The selected query tile may differ by sequence-length bucket and QDQ combination,
  but query tiling does not change P-QDQ's per-row block-16 quantization groups.
- Normal P-QDQ plus 2:4 or skip-softmax uses the same small configuration set instead
  of a special fixed autotune profile. The BM16 candidate remains available as the
  low-resource fallback.

## Testing

- Assert the production configuration set has exactly the three declared tile
  shapes and fixed launch parameters.
- Assert the autotune key is the reduced five-field key.
- Assert all non-measurement QDQ combinations route through `_attn_fwd`.
- Retain the direct measurement-launch tests.
- Update the P-QDQ oracle to use KV tile 32.
- Run focused CPU routing tests, P-QDQ GPU tests, and a real dense Triton launch.
- Run repository pre-commit hooks on every touched file.

## Known Risk

MNI's `BLOCK_M` fuses GQA heads and query tokens, while ModelOpt's `BLOCK_M` counts
query tokens for one head. Matching the numeric value does not make the physical
grids identical. This design aligns the P-QDQ-sensitive KV tile and simplifies the
policy without claiming bitwise MNI kernel equivalence.

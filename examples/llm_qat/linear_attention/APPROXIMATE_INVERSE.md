# Guarded Neumann inverse: retained experimental result

The GDN prefill study retains an experimental polynomial replacement for the
exact unit-lower triangular solve. It is not enabled by a built-in recipe and is
not a qualified optimization.

For an interaction matrix `L`, degree `d` evaluates
`I - L + L^2 - ... + (-L)^d`. Degrees 3, 7, 15, and 31 use a binary composition
order. A sufficient exact-arithmetic truncation bound exists when
`r = ||L||_inf < 1`: the omitted tail is no larger than
`r^(d+1) / (1-r)` in the matrix infinity norm. The experimental policy runs the
polynomial only when that bound is below the configured threshold. Other
matrices fall back individually to the exact solve by default; an explicit
`unsafe_action: error` turns fallback into a diagnostic failure.

```yaml
linear_attention:
  - module_name: "*"
    cfg:
      backend: gdn_prefill
      precision: gdn_prefill
      solve:
        method: guarded_neumann
        degree: 15
        max_tail_bound: 0.001
        unsafe_action: exact
```

The bound covers truncation in exact arithmetic, not FP32 roundoff, model
quality, or speed. Qualification still needs output, state, gradient, and model
quality comparisons plus a native-kernel benchmark. Report the safe-matrix and
fallback rates; a result that mostly falls back is not an optimization.

## Why the guard is mandatory

The earlier unguarded candidate failed its pinned KDA validation screen. Against
an exact validation perplexity of 13.6647, degrees 3, 7, 15, and 31 produced
perplexities of 644766.6, 611215.2, 362093.8, and 98961.5. The isolated H100
forward/backward implementation was also slower than the exact triangular solve.
Those results reject the plain Neumann method; they are not a training-recovery
result.

Strict triangularity does not make degree 63 numerically safe. Although the
series terminates in exact arithmetic, correlated lower-triangular interactions
create large intermediate powers and catastrophic cancellation. The test suite
retains a correlation-0.9 matrix where the FP32 degree-63 relative error exceeds
one million and verifies that the guarded execution refuses it.

This tail change is kept to preserve the negative result and provide a safe
instrument for future GDN measurements. It should remain the final,
non-release-qualified item in the linear-attention stack until a real workload
shows a high safe-matrix rate, near-lossless accuracy, and a native speedup.

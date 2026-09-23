# Approximate triangular solve for linear-attention QAT

GDN and KDA materialized prefill form a strictly lower-triangular interaction
matrix `L` and apply `(I + L)^-1` to the WY key/value right-hand sides. Exact
triangular solve remains the default. An explicitly selected Neumann policy
materializes `A_p = sum((-L)^j, j=0..p)` instead. The existing `wy_key` and
`wy_value` operand quantizers therefore still see an actual inverse operand.

```python
policy = {
    "backend": "matmul",
    "solve": {"method": "neumann", "degree": 7, "implementation": "triton"},
}
# Supply policy as cfg in a linear_attention module rule to mtq.quantize.
```

The degree is a required integer in `[0,63]` for chunk size 64. It is saved in
ModelOpt metadata and activates the numerical path even if all quantizers are
disabled. There is no automatic degree choice, residual-triggered fallback, or
exact-inverse surrogate gradient. The all-disabled exact policy still uses the
original framework path.

**Qualification:** the plain Neumann candidate failed the pinned KDA quality
screen and showed no measured speed benefit. See the
[study result](linear_attention_solve_study.md). Keep exact solve for that model.

## Numerical and gradient contracts

- The independent reference adds powers in ascending order. The implementation
  composes geometric-series blocks in a fixed binary order, reducing the number
  of matrix multiplications. Different association can change floating-point
  rounding; both evaluate the same polynomial.
- `implementation="torch"` uses differentiable PyTorch matmuls in the prefill
  working dtype (FP32, or FP64 for double inputs). An outer BF16 autocast context
  does not lower the solve arithmetic.
- `implementation="triton"` requires CUDA FP32 `[... ,64,64]` matrices. One
  program evaluates one matrix using IEEE FP32 dot products. Backward reconstructs
  the same binary polynomial using PyTorch autograd and returns its first-order
  derivative. It does not differentiate an exact inverse or detach recurrent
  state. Higher-order gradients are unsupported by this candidate.
- Use `torch.set_float32_matmul_precision("highest")` for qualification, including
  recomputed backward and the surrounding prefill sites. Operand QDQ remains a
  separate policy; the polynomial's internal matmuls are FP32.
- Tail chunks are zero padded before the solve; the causal valid submatrix and
  gradients agree with an unpadded polynomial reference.

The diagnostic identity is `I - (I + L) A_p = (-L)^(p+1)`. Strict triangularity
implies degree 63 is exact in exact arithmetic. It does not guarantee a useful
low-degree approximation: intermediate powers can amplify error. Inspect solve
residual, output/state drift, gradients, and model quality independently.

## Qualification and measurement

Tests compare independent polynomial values and derivatives, the residual
identity, exact degree-63 behavior, and full GDN/KDA prefill outputs and states.
They include packed tails, initial-state gradients, BF16 outer autocast, FP8 WY
operand composition, saved policies, and actual model-layer optimizer updates.
A deliberately poor degree-3 case verifies the absence of a hidden fallback.

```bash
PYTHONPATH=. python examples/llm_qat/linear_attention/benchmark_solve.py \
  --degrees 3 7 15 31 --matrices 64 --repeats 20 --output solve.json

PYTHONPATH=. python examples/llm_qat/linear_attention/benchmark_prefill.py \
  --attention kda --solve-degrees 3 7 --repeats 20 --output prefill.json
```

The first command reports isolated forward/backward cost and numerical error on
explicitly described synthetic matrices. The second measures complete prefill.
Neither establishes end-to-end model speed or native low-precision execution.
The candidate's PyTorch backward and materialized prefill remain research paths.

Use the [pinned model study](../examples/llm_qat/linear_attention/README.md) for
quality comparisons. Degree-specific configs are provided for 3, 7, 15, and 31,
with and without FP8 prefill operands. Choose the degree on validation data and
compare a trained candidate against an equally trained exact control. Report a
failed quality threshold as a failed candidate, rather than silently increasing
the degree or switching back to exact solve during execution.

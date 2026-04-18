# Quantization-Aware Benchmark Recommendations

When evaluating a quantized checkpoint, prioritize benchmarks that are sensitive to precision loss.

## Sensitivity ranking

| Priority | Benchmarks | Why |
|----------|-----------|-----|
| **Always include** | MMLU | General knowledge — typically shows measurable accuracy loss from quantization |
| **Recommended** | GSM8K, ARC-Challenge | Math reasoning and general reasoning — sensitive to precision loss |
| **Good to add** | HumanEval, Winogrande | Code generation and commonsense — catches subtle degradation |
| **Less useful for quant comparison** | IFEval | Instruction following — typically less affected, but worth including for aggressive quantization like FP4 |

## Recommended sets by use case

| Use case | Benchmarks |
|----------|-----------|
| Quick sanity check | MMLU |
| Standard quant validation | MMLU, GSM8K, ARC-Challenge |
| Thorough evaluation | MMLU, GSM8K, ARC-Challenge, HumanEval, Winogrande |
| Code-focused model | HumanEval, MBPP, MMLU |
| Reasoning model | GSM8K, MATH-500, GPQA, MMLU |

## How to use

Present these recommendations to the user and ask which to include. If the user already specified benchmarks, keep their choice but mention any accuracy-sensitive benchmarks they may have missed.

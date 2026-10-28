# Experimental GDN prefill operand QAT

ModelOpt can expose the actual operands of the eight matrix multiplications in
the chunk-64 Gated DeltaNet (GDN) prefill algorithm. This is a training-time
fake-QDQ path for studying FP8 or NVFP4 sensitivity and adaptation. It does not
compress a checkpoint and does not provide a native low-precision prefill
kernel or a serving speedup.

The supported sites are the key interaction, WY value and key solves, state
read, output-state read, output scores, output values, and state update. Every
site owns independent left- and right-operand quantizers. The default solve
remains exact FP32. A separate
[guarded Neumann research policy](APPROXIMATE_INVERSE.md) retains the rejected
approximation experiment without promoting it as a recipe.

Two complete recipes are available:

- `general/ptq/gdn_prefill_fp8_dynamic`: dynamic per-row E4M3 QDQ.
- `general/ptq/gdn_prefill_nvfp4_dynamic`: dynamic block16 NVFP4 QDQ. ModelOpt
  invokes the NVFP4 quantizer separately for every chunk and head, so the
  second-level global scale cannot couple unrelated requests or heads.

Both recipes use dynamic scales and need no calibration pass. They are intended
for QAT or QAD: load the recipe before training and retain the resulting
ModelOpt metadata in the training checkpoint. They apply only to full GDN
prefill with BF16 Q/K/V, prepared scalar log-decay gates, `chunk_size=64`, and
no recurrent-state or ReplaySSM QDQ in the same layer.

The bundled linear-attention training example keeps the prompt label mask but
passes each sequence's full valid length to this profile, so all model tokens
use prefill arithmetic and right padding does not alter the recurrent state.
This differs intentionally from state QAT, where the prompt length switches the
student to serving-aligned recurrent decode arithmetic.

KDA is deliberately unsupported. Its pairwise operand
`k[j] * exp(G[i] - G[j])` is not produced by a GEMM kernel. Materializing and
quantizing that tensor would train against a numerical operation that a native
implementation does not execute and would greatly increase memory use.

Before promotion beyond numerical emulation, a deployment must supply a native
kernel with exactly the same operand formats and scale boundaries, followed by
model-quality and end-to-end performance qualification. Until then, results
from these recipes establish only numerical sensitivity and training recovery.

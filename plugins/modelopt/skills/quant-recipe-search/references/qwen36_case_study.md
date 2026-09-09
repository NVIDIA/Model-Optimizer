# Qwen3.6 Case Study

This is an optional example, not the generic rule.

## Main Lesson

For Qwen3.6-35B-A3B, AutoQuant was useful for exploration but did not find the best benchmark frontier by itself. Manual module-family ablations found the best recipe frontier.

The strongest recipe pattern was:

- NVFP4: routed MoE experts and `lm_head`.
- FP8: self-attention and large linear-attention projections.
- BF16/no quant: linear-attention A/B, conv-style pieces, routers/gates, shared-expert gate, and VLM/MTP siblings.
- Final evaluation artifact: native ModelOpt W4A4+W8A8 served with vLLM.
- Compressed-tensors W4A16/W8A8 was an explored/exported variant, not the final case-study evaluation target.

## Sensitivity Findings

- FP8/W8A8 was effectively lossless.
- Routed MoE tolerated NVFP4 when attention was protected.
- `lm_head` NVFP4 was worthwhile because it was a large active decode cost.
- Self-attention NVFP4 hurt LCB for modest active-byte savings.
- Linear attention was sensitive and needed targeted ablation.
- Shared experts were low active-byte value and created runtime/coherence risk in early mixed checkpoints.

## Operational Findings

- LCB was the best early-warning benchmark; GPQA could look recovered while LCB remained low.
- Active GiB/token was a better optimization objective than checkpoint BPE.
- Runtime settings such as parser, FP8-KV, backend, and token caps affected both accuracy and verbosity, so they had to be tracked separately from recipe changes.

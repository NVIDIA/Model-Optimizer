# Vendor Reference and Sampling

Find the model card (Hugging Face, build.nvidia.com) and read the full text. Extract every setting the
evaluation depends on, and say where each came from. The card is the source of truth; the tables here are
references, not constraints.

## What to take from the model card

Reading the card for the settings a run needs (sampling, the reasoning switch, parsers, context length,
parallelism, the serving image) is the evaluation tool's job, so use its skill for it (the tool's reference names
it). Do not re-derive those settings here.
What this guidance adds is specific to quantized checkpoints:

- **Cross-check sampling against the NVFP4 table.** Required for any NVFP4 checkpoint or a same-family sibling:
  [nvfp4-modelcard-sampling.md](nvfp4-modelcard-sampling.md) is a dated snapshot of published settings, and for
  anything newer the card is the source of truth. Trust a card sentence only when it ties the values to the
  benchmarks ("Benchmarked with...", "...were evaluated with...") or a "Recommended Sampling" row; never read
  evaluation settings out of a quickstart snippet (`SamplingParams(temperature=0.8, top_p=0.95)` is boilerplate
  copied across unrelated models).
- **A quantized checkpoint is served like its baseline.** Change only what the quantization requires, and list
  every difference (the serving record in `modelopt-deployment`, step 5c, is where it is written). The tool's model
  skill may state the same rule; follow it there.
- **Hardware images.** The default serving image supports common GPU architectures only. For ARM64 or GPUs with
  a non-standard compute capability (for example sm_121), use an NGC vLLM image, and ask the user for the
  architecture if the card does not say. A tool's model skill may not cover this.
- **Ask the user to confirm** the settings you propose, and ask directly if no model card exists.

## Output budget

1. **The tool's benchmark policy comes first.** The evaluation tool owns the benchmark's output budget and
   repeat count; follow it and do not override it. Keep sampling, thinking, context and timeout settings
   separate from per-response budgets.
2. Fetch the model card before recording a value; consult its linked evaluation recipes.
3. Look for a budget used for the exact task, model and harness. A published recipe linked from the card
   may set a shared default; use it only for the evaluations it covers. A budget for another harness does not
   apply even when the benchmark name matches. Do not take the highest number mentioned, quickstart or
   OpenCode examples, generic output-length recommendations, or `generation_config.json` defaults.
4. Consult the NVFP4 sampling table as a reference, with the same applicability rule; do not infer output
   budgets from same-family rows or recommended sampling.
5. **Record, do not override.** If the card names a budget that differs from the tool's, report both side
   by side in the handoff. Where the tool has no budget of its own, its reference says whether it has a numeric
   fallback; never invent one.
6. Never write a generic default with a "card not yet checked" comment: fetch the card and apply the
   evaluation-specific guidance, or confirm that none is disclosed.

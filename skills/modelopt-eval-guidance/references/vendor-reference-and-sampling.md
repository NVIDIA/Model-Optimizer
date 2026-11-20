# Vendor Reference and Sampling

Find the model card (Hugging Face, build.nvidia.com) and read the full text. Extract every setting the
evaluation depends on, and say where each came from. The card is the source of truth; the tables here are
references, not constraints.

## What to extract

- **Sampling (`temperature`, `top_p`).** Trust only a sentence that ties the values to the benchmarks
  ("Benchmarked with...", "...were evaluated with...", "We evaluate the model using...") or a "Recommended
  Sampling" row. The `SamplingParams(temperature=0.8, top_p=0.95)` in a quickstart snippet is boilerplate
  copied across unrelated models; never read evaluation settings out of it. Then cross-check
  [nvfp4-modelcard-sampling.md](nvfp4-modelcard-sampling.md), required for any NVFP4 checkpoint or a
  same-family sibling: it is a dated snapshot of published settings, so for anything newer trust the card.
- **Context length.** The maximum model length the card or recipe requires.
- **Output length.** Record any output budget the card ties to the applicable evaluation, with its source.
  Apply the rules in the next section.
- **Reasoning switch.** Whether reasoning is on or off by default and how it is toggled (a system prompt such as
  `/think`, or a chat-template keyword). The keyword drifts across model generations: read the card or
  `chat_template.jinja`, do not extrapolate, and set only the one key the model uses. Known examples:
  `enable_thinking` (Qwen3.5 and 3.6, GLM 5.1), `thinking` (Kimi K2.6, DeepSeek V3.2 and V4; the DeepSeek
  encoder is Python, so an unused keyword can error instead of being ignored). Reasoning effort, if
  configurable (for example DeepSeek V4 `reasoning_effort`): use the highest effort the card documents unless the
  user signals a cost or latency preference, and honor any tied requirement (V4 Think Max needs a maximum model
  length of at least 393216).
- **Serving settings.** Tensor and data parallelism, the serving framework and its minimum version (a newer
  vLLM dropped `rope-scaling`, which some Qwen models use), the container for ARM64 or non-standard GPU
  architectures (for example sm_121), and any preparation the card requires before serving (downloading a
  reasoning parser or plugin; use `curl` rather than `wget`, and `pip install --no-cache-dir` inside
  containers to avoid cross-device link errors). The serving record in `modelopt-deployment` (step 5c) is where
  these are written down for the evaluation tool.
- **Anything else model-specific.** Present the findings, explain each setting, and ask the user to confirm or
  adjust. If there is no model card, ask the user directly.

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
   by side in the handoff. Where the tool has no budget of its own, NEL's numeric fallback (65536 reasoning,
   16384 non-reasoning, only when the task policy permits it) is in
   `modelopt-evaluation/references/model-card-research.md`.
6. Never write a generic default with a "card not yet checked" comment: fetch the card and apply the
   evaluation-specific guidance, or confirm that none is disclosed.

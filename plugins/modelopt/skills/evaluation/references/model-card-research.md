# Model Card Research

Use WebSearch to find the model card (HuggingFace, build.nvidia.com). Read it carefully, the FULL text, the devil is in the details. Extract ALL relevant configurations:

- Sampling params (`temperature`, `top_p`)
  - **Only trust a sentence that ties the values to the benchmarks** ("Benchmarked
    with…", "…were evaluated with…", "We evaluate the model using…") or a
    "Recommended Sampling" row. The `SamplingParams(temperature=0.8, top_p=0.95)`
    in a card's TensorRT-LLM/vLLM quickstart snippet is boilerplate copied
    verbatim across unrelated models — **never** read eval settings out of it.
  - **Then cross-check `nvfp4-modelcard-sampling.md`** — required for any NVFP4
    checkpoint or same-family sibling; see [Sampling reference](#sampling-reference).
- Context length (`deployment.extra_args: "--max-model-len <value>"`)
- **Output length (`max_new_tokens`) — mandatory extraction.** Record any budget
  explicitly tied to the applicable evaluation, with its source. Follow the
  [token-budget rule below](#max_new_tokens--mandatory-model-card-lookup)
  for selecting the value and fallback.
- TP/DP settings (to set them appropriately, AskUserQuestion on how many GPUs the model will be deployed)
- Reasoning config (if applicable):
  - reasoning on/off: use either:
    - `adapter_config.custom_system_prompt` (like `/think`, `/no_think`) and no `adapter_config.params_to_add` (leave `params_to_add` unrelated to reasoning untouched)
    - `adapter_config.params_to_add` for payload modifier (like `"chat_template_kwargs": {"enable_thinking": true/false}`) and no `adapter_config.custom_system_prompt` and `adapter_config.use_system_prompt: false` (leave `custom_system_prompt` and `use_system_prompt` unrelated to reasoning untouched).
  - **The `chat_template_kwargs` toggle key drifts across model generations — read the card / `chat_template.jinja`, don't extrapolate, and set only the one key the model uses.** Known: `enable_thinking` (Qwen3.5/3.6, GLM 5.1 — note GLM-4.x used `thinking`+`/nothink`); `thinking` (Kimi K2.6 — renamed from K2.5's `enable_thinking`; DeepSeek V3.2/V4 — Python encoder, not Jinja, so an unused kwarg can error rather than be ignored).
  - reasoning effort/budget (if configurable, e.g. DeepSeek V4 `reasoning_effort`): **default to `max`** (the highest effort the card documents), honoring any tied requirement (e.g. V4 Think Max needs `--max-model-len >= 393216`). AskUserQuestion only if the user signals a cost/latency preference.
  - etc.
- Deployment-specific `extra_args` for vLLM/SGLang (look for the vLLM/SGLang deployment command)
- Deployment-specific vLLM/SGLang versions (by default we use latest docker images, but you can control it with `deployment.image` e.g. vLLM above `vllm/vllm-openai:v0.11.0` stopped supporting `rope-scaling` arg used by Qwen models)
- ARM64 / non-standard GPU compatibility: The default `vllm/vllm-openai` image only supports common GPU architectures. For ARM64 platforms or GPUs with non-standard compute capabilities (e.g., NVIDIA GB10 with sm_121), use NGC vLLM images instead:
  - Example: `deployment.image: nvcr.io/nvidia/vllm:26.01-py3`
  - AskUserQuestion about their GPU architecture if the model card doesn't specify deployment constraints
- Any preparation requirements (e.g., downloading reasoning parsers, custom plugins):
  - If the model card mentions downloading files (like reasoning parsers, custom plugins) before deployment, add `deployment.pre_cmd` with the download command
  - Use `curl` instead of `wget` as it's more widely available in Docker containers
  - Example: `pre_cmd: curl -L -o reasoning_parser.py https://huggingface.co/.../reasoning_parser.py`
  - When using `pip install` in `pre_cmd`, always use `--no-cache-dir` to avoid cross-device link errors in Docker containers (the pip cache and temp directories may be on different filesystems)
  - Example: `pre_cmd: pip3 install --no-cache-dir flash-attn --no-build-isolation`
- Any other model-specific requirements

Remember to check `evaluation.nemo_evaluator_config` and `evaluation.tasks.*.nemo_evaluator_config` overrides too for parameters to adjust (e.g. disabling reasoning)!

Present findings, explain each setting, ask user to confirm or adjust. If no model card found, ask user directly for the above configurations.

## Sampling reference

**Cross-check `temperature` / `top_p` against `nvfp4-modelcard-sampling.md`** — the published settings for the 2026 NVFP4 checkpoints under `huggingface.co/nvidia` that disclose them (older releases and cards that publish nothing are absent — for those, read the card; `-DSpark` / `-DFlash` spec-decode variants share their base checkpoint's row, since spec decoding does not change the target's output distribution). **The card is the source of truth; this file is a reference, not a constraint** — use it to confirm a value you read, to fill a gap when the card is silent or ambiguous, and to catch a misreading. Worth consulting whenever the model is an NVFP4 checkpoint **or shares a family with one** (Qwen3.x, GLM-4.7/5.x, Kimi K2.x/K3, MiniMax M2.x/M3, DeepSeek V3.x/V4/R1, Gemma 4, Nemotron 3/3.5, Llama-Nemotron, Mistral Medium 3.5), and especially when you are unsure. It is a dated snapshot, so for anything newer than it, trust the card. See that file's "Lookup" section. For `max_new_tokens`, follow the token-budget rule below.

## `max_new_tokens` — mandatory model-card lookup

1. **Fetch the HF model card before writing the value.** Not optional. Consult its linked evaluation recipes for applicable settings.
2. Look for `max_tokens` / `max_new_tokens` / "output length" explicitly used for the applicable evaluation. A published evaluation recipe linked from the card may set a shared default: use it for tasks covered by that recipe unless a task-specific override applies. Annotate with a citing comment. Do not select the highest number mentioned, quickstart/OpenCode examples, generic output-length recommendations, or `generation_config.json` defaults.
3. **Consult `nvfp4-modelcard-sampling.md` as a reference.** Use an output budget only when its source explicitly covers the evaluated model and applicable evaluation. Re-read the card and surface discrepancies; do not infer output budgets from same-family rows or recommended sampling.
4. If no applicable evaluation-specific output budget is disclosed after checking the card and reference, fall back to: **65536** (reasoning), **16384** (non-reasoning); surface the missing evaluation guidance to the user.
5. **Forbidden:** writing `max_new_tokens: <generic_default>` with a "card not yet checked" comment. Either fetch and apply evaluation-specific guidance, or fetch and confirm none is disclosed.

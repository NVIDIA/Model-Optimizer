# Model Card Research

Use WebSearch to find the model card (HuggingFace, build.nvidia.com). Read it carefully, the FULL text, the devil is in the details. Extract ALL relevant configurations:

- Sampling params (`temperature`, `top_p`)
- Context length (`deployment.extra_args: "--max-model-len <value>"`)
- **Output length (`max_new_tokens`) — mandatory extraction.** Scan the
  card for any `max_tokens` / `max_new_tokens` / "output length"
  recommendation. Cards often list two values (e.g., Qwen3.x: `32768`
  thinking-general + `81920` math/coding). **Pick the highest value** and
  apply at the top level (no per-task overrides). If the card is genuinely
  silent on output length, note that explicitly and fall back to the
  generic default (64K reasoning / 16K non-reasoning) — never write a
  config with "card not yet checked" + generic default. See SKILL.md
  Step 3 "`max_new_tokens` — pick a single top-level value" for the full
  rule.
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

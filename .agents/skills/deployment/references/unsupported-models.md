# Deploying Unsupported Models

When deploying a model not in the validated support matrix (`support-matrix.md`), expect failures. This guide covers the iterative debug loop for getting unsupported models running on vLLM, SGLang, or TRT-LLM.

## Step 1 — Run and collect the error

Submit the deployment job. When it fails, read the full log — focus on the **first** error traceback (not "See root cause above" wrappers). Identify the file and line number in the framework source.

## Step 2 — Diagnose the root cause

Fetch the framework source at the failing line (use `gh api` for the tagged version, or `find` inside the container). Common error categories:

| Category | Symptoms | Examples |
|----------|----------|----------|
| **Weight key mismatch** | `KeyError`, `Unexpected key`, `Missing key` during weight loading | Checkpoint uses `model.language_model.layers.*` but framework expects `model.layers.*`. See [vllm#39406](https://github.com/vllm-project/vllm/pull/39406) |
| **Quantized/unquantized layer confusion** | Wrong layer type loaded, dtype errors, shape mismatches | Framework tries to load unquantized layers with FP4 kernel due to overly broad `quantization_config.ignore` patterns or missing ignore entries. See [sglang#18937](https://github.com/sgl-project/sglang/pull/18937) |
| **Missing architecture support** | `NoneType is not iterable`, `KeyError` on model type, unknown architecture | Framework's model handler doesn't recognize the text backbone type (e.g., `ministral3` not handled in vLLM's `mistral3.py` init). Fix: extend the model type mapping |
| **Transformers version mismatch** | `ImportError`, `KeyError` on config fields | Framework ships with older transformers that doesn't know the model type. Fix: upgrade transformers after installing the framework |
| **Kernel-level issues** | CUDA errors, `triton` import failures, unsupported ops | Framework lacks kernel support for this model + quantization combo |

## Step 3 — Apply a targeted fix

Focus on **small, targeted patches** to the framework source. Do not modify `config.json` or the checkpoint — fix the framework's handling instead.

### Weight key mismatches and architecture mapping gaps

Patch the framework source in the run script using `sed` or a Python one-liner. Keep patches minimal — change only what's needed to unblock the current error.

```bash
# Example: extend model type mapping in vLLM mistral3.py
FRAMEWORK_FILE=$(find /usr/local/lib -path "*/vllm/model_executor/models/mistral3.py" 2>/dev/null | head -1)
sed -i 's/old_pattern/new_pattern/' "${FRAMEWORK_FILE}"
```

> **Tip**: when locating framework source files inside containers, use `find` instead of Python import — some frameworks print log messages to stdout during import that can corrupt captured paths.

### Speeding up debug iterations (vLLM)

When iterating on fixes, use these flags to shorten the feedback loop:

- **`--load-format dummy`** — skip loading actual model weights. Useful for testing whether the model initializes, config is parsed correctly, and weight keys match without waiting for the full checkpoint load.
- **`VLLM_USE_PRECOMPILED=1 pip install --editable .`** — when patching vLLM source directly (instead of `sed`), this rebuilds only Python code without recompiling C++/CUDA extensions.

### Quantized/unquantized layer confusion

Check `hf_quant_config.json` ignore patterns against the framework's weight loading logic. The framework may try to load layers listed in `ignore` with quantized kernels, or vice versa. Fix by adjusting the framework's layer filtering logic.

### Kernel-level issues

These require framework kernel team involvement. Do NOT attempt to patch kernels. Instead:

1. Document the exact error (model, format, framework version, GPU type)
2. Inform the user: *"This model + quantization combination requires kernel support that isn't available in {framework} v{version}. I'd suggest reaching out to the {framework} kernel team or trying a different framework."*
3. Suggest trying an alternative framework (vLLM → SGLang → TRT-LLM)

## Step 4 — Re-run and iterate

After applying a fix, resubmit the job. Each iteration may reveal a new error (e.g., fixing the init error exposes a weight loading error). Continue the loop: **run → read error → diagnose → patch → re-run**.

Typical iteration count: 1-3 for straightforward fixes, 3-5 for models requiring multiple patches.

## Step 5 — Know when to stop

**Stop patching and escalate** when:

- The error is in compiled CUDA kernels or triton ops (not Python-level)
- The fix requires changes to core framework abstractions (not just model handlers)
- You've done 5+ iterations without the server starting

In these cases, inform the user and suggest: trying a different framework, checking for a newer framework version, or filing an issue with the framework team.

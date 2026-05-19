# vLLM AOT compile-cache poisoning across multimodal-on / multimodal-off runs

Applies to **any** model whose vLLM architecture supports multimodal input —
this is modality-agnostic, covering image, video, audio, or any other
modality (`vllm/multimodal/registry.py: supports_multimodal_inputs` iterates
the model's `supported_mm_limits`, which can be `{"image": N}`,
`{"video": N}`, `{"audio": N}`, `{"image": N, "video": N}`, etc.). The hazard
appears when multiple vLLM runs against the **same checkpoint** share a
`VLLM_CACHE_ROOT` and differ in whether **all** of the model's modalities
are zeroed out via `--limit-mm-per-prompt`.

## Symptom

vLLM startup crashes during `profile_run` / `_dummy_run` / CUDA-graph capture
with:

```
AttributeError: 'NoneType' object has no attribute 'size'
```

The traceback ends inside `torch/_dynamo/utils.py call_size → x.size(i)`,
after passing through `vllm/compilation/decorators.py: aot_compiled_fn`.
**There is no model-layer frame** in the failing stack — no attention op,
no MLP, no quantized linear. The compiled function is loaded from disk and
crashes in dynamo's prologue, before any decoder layer runs. The log line
just above the traceback is the smoking gun:

```
INFO ... [decorators.py:...] Directly load AOT compilation from path
  /vllm-cache/torch_compile_cache/torch_aot_compile/<hash>/rank_*/model
```

## Mechanism

vLLM's `@support_torch_compile` decorator caches one compiled `forward` per
`(aot_compile_hash_factors(vllm_config), _model_hash_key(forward))` key
(`vllm/compilation/decorators.py`). That key includes the model config and
quantization, but **does not include** `--limit-mm-per-prompt` or the
derived `supports_mm_inputs` flag.

`vllm/v1/worker/gpu_model_runner.py: _dummy_run` branches on
`supports_mm_inputs`:

```python
if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
    input_ids, inputs_embeds = self._prepare_mm_inputs(...)   # (None, Tensor)
else:
    input_ids = self.input_ids.gpu[:num_tokens_padded]        # (Tensor, None)
    inputs_embeds = None
```

`supports_mm_inputs` (`vllm/multimodal/registry.py: supports_multimodal_inputs`)
returns `False` when **every** supported modality has
`--limit-mm-per-prompt = 0`. So:

| Run config | `supports_mm_inputs` | Pattern compiled / loaded |
| --- | --- | --- |
| `--limit-mm-per-prompt '{"image":0}'` (and `"video":0` etc.) | False | `input_ids=Tensor, inputs_embeds=None` |
| default, or any modality non-zero | True | `input_ids=None, inputs_embeds=Tensor` |

The `@support_torch_compile` docstring explicitly forbids the same argument
slot from being `None` on one invocation and a Tensor on another — Dynamo
specializes on None-vs-Tensor identity per argument, so one cached graph
cannot serve both patterns. When run A populates the cache slot and run B
shares the slot but uses the opposite pattern, the prologue calls
`.size()` on what is now `None` and dies.

This is symmetric: a multimodal-first run followed by a text-only-via-image:0
run fails the same way, just with the None/Tensor roles swapped.

## How to confirm

1. **Cache hit before the crash.** Look in the server log for
   `Directly load AOT compilation from path ...` shortly before the
   traceback. A cache *hit* immediately before a `NoneType.size()` is the
   diagnostic. (A cold compile would print `Dynamo bytecode transform
   time` and `Inductor compile took ...` instead.)
2. **Config delta on `--limit-mm-per-prompt`.** Compare the failing run's
   serving args against the most recent successful runs that share
   `$VLLM_CACHE_ROOT`. If they disagree on whether any modality is
   zero-limited (or one side omits the flag while the other passes
   `{"image":0}`), the cache slot is colliding.
3. **Positive control.** Relaunch the failing config with
   `VLLM_DISABLE_COMPILE_CACHE=1` and change nothing else. If `profile_run`
   passes, the cache was the cause.

## Fix

Two parts — stop the poisoning, then heal what's already poisoned.

### Stop poisoning

For multimodal-architecture models, do **not** zero out a modality with
`--limit-mm-per-prompt '{"image":0}'` (or `"video":0`, …) on runs intended
to share a cache root with multimodal runs. The vision tower weights are
loaded from the checkpoint regardless of this flag; zeroing only flips
`supports_mm_inputs` and creates the cache hazard. Text-only inference
still works without the flag because vLLM's `_preprocess` routes both
text and multimodal prompts through the same `inputs_embeds` path when
`supports_mm_inputs=True`:

```python
# vllm/v1/worker/gpu_model_runner.py: _preprocess
# NOTE(woosuk): To unify token ids and soft tokens (vision embeddings),
# we always use embeddings (rather than token ids) as input to the
# multimodal model, even when the input is text.
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

A text-only prompt simply has `mm_embeds=[]` / `is_multimodal=False`; the
call signature into the language model is unchanged. The small cost of
keeping multimodal inputs enabled is that vLLM allocates an encoder cache
budget at startup (e.g. a few hundred MB) and prints a vision warmup line.

### Heal existing cache

Either fully wipe and let the next run repopulate:

```bash
rm -rf "$VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile/"
```

…or sidestep by separating cache roots per multimodal-ness (set a different
`VLLM_CACHE_ROOT` for the runs that need a different pattern), or just set
`VLLM_DISABLE_COMPILE_CACHE=1` on the affected runs and accept a one-time
recompile (~20-30 s) at every startup.

## See also

- `vllm/compilation/decorators.py` — `support_torch_compile` decorator and
  its docstring on the None-vs-Tensor invariant.
- `vllm/v1/worker/gpu_model_runner.py` — the input-construction branch in
  `_dummy_run` and the unified-`inputs_embeds` comment in `_preprocess`.
- `vllm/multimodal/registry.py` — how `supports_multimodal_inputs` is
  computed from `--limit-mm-per-prompt`.

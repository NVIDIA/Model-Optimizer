---
name: debugging-playbooks
description: Diagnostic playbooks for tricky failures — failures where the traceback misdirects and the first 2-3 reasonable hypotheses turn out wrong. Use when a run fails with a framework-internal-looking error (cryptic torch.compile / dynamo / NCCL / vLLM / transformers / CUDA / pyxis / enroot / NEL / SLURM / container runtime), the top frame appears to blame the wrong layer (e.g. the user's code, ModelOpt, the quantized linear, the wrapper class) but fixing that layer doesn't help, or the symptom recurs across unrelated changes. Use this skill when you've eliminated the obvious suspects and the bug hasn't budged. Don't reach for this on the first guess; reach for it when the obvious answers don't pan out. Each playbook is keyed by a literal symptom string from logs so future agents can grep for it.
---

# Debugging playbooks

When a failure surfaces a symptom that doesn't clearly map to the code under change, check whether one of the documented playbooks below already describes it. Each playbook is keyed by the literal symptom string so future agents can match by grep.

| Symptom (literal string from logs) | Playbook |
| --- | --- |
| `AttributeError: 'NoneType' object has no attribute 'size'` during vLLM `profile_run` / `_dummy_run` / CUDA-graph capture | [vllm-aot-cache-poisoning.md](references/vllm-aot-cache-poisoning.md) |

## When to add a new playbook

Add an entry when **all three** are true:

1. The root cause was non-obvious from the traceback — the immediate frame was misleading (e.g. blames ModelOpt when the bug is in vLLM).
2. The symptom is likely to recur across runs (different models, different containers).
3. There is a concrete fix (config change, env var, cache invalidation) that future agents should reach for before deeper debugging.

Each playbook should include: the literal symptom string, the actual mechanism, how to confirm the diagnosis, and the minimal fix.

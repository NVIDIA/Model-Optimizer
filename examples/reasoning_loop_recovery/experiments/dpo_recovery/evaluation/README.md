# Evaluate without prompting or penalties

Evaluation uses a **custom Python SciCode runner**, not `lm_eval` or NeMo Evaluator Launcher (NEL).
It builds requests from SciCode task definitions and prompt templates, sends them to a vLLM
OpenAI-compatible chat endpoint, and evaluates final-answer code with SciCode's official tests.
Save requests/responses, reasoning, final content, finish reason, completion-token usage, correctness,
and repetition metrics separately. Only final-answer code is scored; draft code in reasoning is not
substituted for an empty final answer.

Compare the **round-3 iterative QAD checkpoint immediately before DPO** with the exported DPO model.
Use all 55 selected subtasks and ten seeds each, 61000–61009: 440 training-target responses and
110 held-out-target responses per model. Original prompts, no prompt 7, no presence penalty.
Hold temperature, top-p, output cap, context limit, task IDs, and seeds constant.

Each model generates its own preceding SciCode steps with the same seed. Prefill supplied reference
steps and exclude prerequisite responses from target counts. Consequently, matching target seeds
does not guarantee byte-identical target prompts across models. Fail context-overflow chains rather
than silently lowering the cap. Resume from saved attempts without discarding failed generations.

Path 4's deployment used one model per GPU, TP=1, four workers at a time across a four-GPU allocation.
Twenty worker units (ten seeds × two task shards) run in five sequential groups; this is an execution
layout, not twenty separate samples per target. [evaluation.yaml](evaluation.yaml) records the settings.

Report cap hits, missing final answers, their intersection, token lengths including reasoning,
repetition, and correctness separately for training and held-out subtasks. A cap hit does not prove
repetition and the two failure counts overlap. High repetition means at least half of the overlapping
20-word windows in the reasoning are duplicates; [evaluation.yaml](evaluation.yaml) defines the score.
For path 4, both cap hits and missing-final counts
are zero, so their intersection is necessarily zero; a new high-repetition aggregate has not been
published in this result snapshot and must not be inferred from zero cap hits.

The [saved summary](results.yaml) records the complete path-4 evaluation and its baseline.
Do not use preference-pair validation loss as SciCode accuracy or pool it into the held-out results.

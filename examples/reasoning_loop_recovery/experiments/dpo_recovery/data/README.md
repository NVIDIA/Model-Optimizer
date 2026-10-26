# Prepare preference pairs

This README explains how to prepare the preference dataset for DPO recovery training on
**Qwen3.6-35B-A3B after round-3 iterative QAD**. Each example pairs two responses to the same input:
a completed, non-looping response to prefer and a repeating reasoning prefix to reject. The goal
is to teach the model to favor completed reasoning over looping, without extra instructions at
inference time. Correctness is recorded separately; preferred responses are not necessarily correct.

At a high level:

1. Collect repeating responses from existing QAD evaluations, using only the 44 training subtasks.
2. For each selected failure, reuse a completed response to the exact same input, or generate one
   with the round-3 iterative QAD model using prompt 7.
3. Filter the preferred responses, shorten the rejected reasoning to a prefix showing repetition,
   and format each pair with the original input, without prompt 7.
4. Split the 115 accepted pairs into 104 training and 11 diagnostic validation pairs. Keep the
   11 held-out SciCode subtasks separate for evaluation after training.

The sections below describe the source responses, selection rules, text format, and splits.

## Files

| File | Purpose |
| --- | --- |
| [pair_preparation.yaml](pair_preparation.yaml) | Settings for selecting saved negatives and obtaining preferred responses |
| [dataset_manifest.yaml](dataset_manifest.yaml) | Dataset filenames, format, counts, filtering outcomes, and training-file checksum |
| [subtask_split.yaml](subtask_split.yaml) | Frozen lists of 44 training and 11 held-out SciCode subtasks |
| [prompt7.txt](prompt7.txt) | Exact instruction used only when generating missing preferred responses |

These files describe dataset preparation; they do not contain the generated responses. The dataset
itself is stored separately as JSONL, with the filenames recorded in the manifest.

## Example preference pair

This simplified, invented example illustrates the shared input and the two response branches.
It is not a SciCode sample, an accepted training example, or an exact JSONL record. The real
repetition checks and length limits described below still apply.

**Shared input, without prompt 7:**

```text
Write a Python function returning the larger of two numbers.
```

**Preferred response — completed reasoning and a final answer:**

````text
Compare the two values and return the larger one.
</think>

```python
def larger(a, b):
    return a if a >= b else b
```
````

**Rejected response — an unfinished reasoning prefix showing repetition:**

```text
I should compare the two values. Let me reconsider the comparison.
I should compare the two values. Let me reconsider the comparison.
I should compare the two values. Let me reconsider the comparison.
```

Both branches are paired with the **same original input**. The rejected prefix is an output to
discourage, not an extra instruction or reasoning prefix supplied as input. The preferred branch
receives an EOS token during tokenization; the unfinished rejected branch does not.

## Two different splits

| Split | Unit | Training | Reserved | Purpose |
| --- | --- | ---: | ---: | --- |
| SciCode split | Subtasks | 44 | 11 held-out subtasks | Measure generalization to subtasks excluded from preference-data preparation |
| Preference-pair split | Response pairs | 104 | 11 validation pairs | Monitor DPO loss during training |

Reuse the frozen SciCode split from the prompt-7 experiment, stored in
[subtask_split.yaml](subtask_split.yaml). Only its **44 training subtasks** supply preference pairs.
The **11 held-out subtasks never supply pairs**, including validation pairs. Related training and
held-out subtasks may share a parent problem.

After preparing the pairs, divide them into 104 training and 11 diagnostic validation pairs.
These two pair sets may share subtasks and input prompts; pair-validation loss is not a measure
of generalization to the held-out SciCode subtasks.

## Preparation counts

| Stage | Count |
| --- | ---: |
| Candidate negatives found in saved evaluations | 249 responses |
| Selected negatives with a confirmed repetition witness | 150 responses |
| Pairs accepted after preferred-response filtering, length checks, and deduplication | 115 pairs |
| DPO training set | 104 pairs |
| Diagnostic pair-validation set | 11 pairs |

The 150 negatives are a selected subset of the 249 candidates, using the per-subtask limits and
repetition checks below. Each accepted pair contains one preferred response and one rejected
reasoning prefix for the same input.

## Find negative examples

Use **existing saved responses only** for negatives; this experiment did not generate new negative
responses. New generation is used only for missing preferred responses, as described below.

Index saved evaluations of the original QAD model and the round-3 iterative QAD model. The inventory
contained 2,639 training-target responses and 249 candidate negatives. A negative must hit the
131,072-token cap, have an empty final answer, and have a saved reasoning-repetition score ≥ 0.5.

Select up to eight negatives per subtask, round-robin across sorted subtask IDs, stopping at 160.
Within a subtask, prioritize round-3 failures, then order by SHA-256 of the source sample path.
That historical tie-breaker depends on the original inventory paths: exact selected IDs and source
fingerprints must be preserved to reproduce the dataset, rather than rerunning under new paths.

Require an auditable repeated paragraph before retaining a negative:

1. Find a prose paragraph with 80–1,400 characters after whitespace normalization.
2. Find its third exact normalized occurrence, ending after character 3,000.
3. Keep the reasoning prefix through that occurrence, provided the cut is ≤ 100,000 characters.
4. Do not append `</think>`, a final answer, or EOS to this unfinished rejected prefix.

This retained 150 candidate negatives across 37 training subtasks. A cap hit alone is insufficient:
the paragraph witness confirms actual repetition. The prefix is the **rejected output**, not a
prefix added to the model's input.

## Obtain preferred responses for the same input

Hash the original request messages after removing any appended prompt-7 instruction. Include the
full SciCode user message, including generated prerequisite code. Matching subtask IDs alone is
insufficient because previous steps can produce different target prompts.

For 39 candidates, reuse an existing completed, non-looping response with the exact same input.
Prefer round-3 responses, then correct responses, then the deterministic source-path hash ordering.
Existing candidates must be below 26,000 completion tokens. Generate the other 111 candidates by
replaying the exact failing input to **round-3 iterative QAD + prompt 7**.

[prompt7.txt](prompt7.txt) stores the exact extra instruction used for these 111 preferred-response
generations. It is not used to collect negatives. Append it to the original user message after two
newlines, then remove it from the input saved in the DPO dataset. DPO training and evaluation use
no extra instruction. The instruction is:

> If your reasoning begins repeating, summarize the unresolved issue once, resolve it as best you can, and continue toward a final answer.

Use seed `85000 + pair_index`, temperature 0.6, top-p 0.95, output cap 131,072, context 262,144,
and no presence penalty. Save raw requests, raw responses, reasoning, final content, metrics,
correctness, and the original-input hash. Remove prompt 7 when constructing the pair's input.

Reject preferred responses that are unfinished, have empty final content, have repetition score
≥ 0.5, or repeat any case-sensitive 20-whitespace-word sequence at least eight times. Record
correctness separately; it is not required for inclusion. Only 18/115 retained preferred responses
passed SciCode tests, so these are **termination preferences**, not a correctness preference dataset.

## Format preference pairs

Convert the original chat messages into the exact text format expected by the model, including
its role markers and the start of the assistant's reasoning. This is what “serialize” means here;
the saved result is still **text, not token IDs**. Remove prompt 7 before formatting the input.
Use the model's tokenizer to apply its chat template:

```python
prompt_serialized = tokenizer.apply_chat_template(
    original_messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
```

The formatted input ends with `<think>` followed by a newline, so the response continues directly
with reasoning. Store two possible responses for this same input:

- **Preferred:** completed reasoning, then `\n</think>\n\n`, then the final answer.
- **Rejected:** the reasoning prefix containing the confirmed repetition, ending at the selected cut.

During training-data processing, convert these text strings into token IDs and append one
end-of-sequence (EOS) token to the preferred response. Do not append EOS to the unfinished rejected
response. Each input plus response must fit 32,768 tokens, including EOS where applicable.
Deduplicate on `SHA256(prompt_serialized + NUL + chosen_serialized + NUL + rejected_serialized)`.

## Create the preference-pair training and validation sets

Final selection retained **115 pairs across 35 training subtasks and 86 unique inputs**. Exclusions:
23 preferred responses with a repeated 20-word cycle, nine unfinished, two high repetition,
and one overlength. Some original inputs have multiple rejected branches.

For diagnostic pair validation, sort pairs by SHA-256 of `dpo-20261024:` + pair ID and reserve the
first `max(5, floor(N/10))` pairs: **11 validation / 104 training**. This split can share subtasks
and input prompts across training and pair validation. Its loss is only a training diagnostic;
the 11 held-out SciCode subtasks remain the generalization evaluation.

JSONL rows store `pair_id`, `context`, ranked `completions`, `prompt_serialized`,
`chosen_serialized`, `rejected_serialized`, `chosen_eos: true`, and `rejected_eos: false`.
The custom `preference_processor.py` tokenizes the serialized strings without adding special tokens,
masks the input, appends EOS only for chosen, and returns NeMo RL's chosen/rejected message logs.
Using a generic processor that adds EOS to both branches changes this experiment.

Before training, audit exact input identity, absence of held-out targets, raw response provenance,
negative repetition witnesses, branch lengths, and EOS handling. [The source manifest](../training/compatibility/source_manifest.yaml)
identifies the preparation and tokenization implementations used.

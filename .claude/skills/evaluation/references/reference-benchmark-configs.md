# Reference Benchmark Task Configs

Use these task-level YAML files when the user asks for the named reference benchmarks.
Keep the task stanza shape intact unless the user explicitly asks to change sampling or
prompt settings.

For reasoning-mode comparisons, use `num_repeats >= 3` when the benchmark supports
repeats. Single-trial noise can hide or mimic low-single-digit percentage-point
effects, so do not rely on a one-shot comparison when judging small deltas.

## GPQA Diamond AA v3

Aliases: `gpqa_diamond_aa_v3`, `GPQA Diamond`, `GPQA`.

Config file: `references/gpqa_diamond_aa_v3.yaml`

## SciCode AA v2

Aliases: `scicode_aa_v2`, `SciCode`.

Config file: `references/scicode_aa_v2.yaml`

# Qwen 3.5 0.8B VLM KD learning-curve campaign

This directory packages bounded, identity-bound evidence from the September 2026
Qwen 3.5 0.8B VLM campaign. The run evaluates six students before KD and after
64, 128, and 256 cumulative KD steps on one frozen 344-row short-v1 manifest.
The FFN-3328 control also has a separately gated 512-step milestone.

## Protocol status

The measured run used an early campaign prototype with bespoke shortlist and
diversity heuristics. Those measurements remain valid observations of the
recorded checkpoints, but that selection policy is superseded. The integrated
campaign now uses the established Puzzletron selection flow: the existing
`params-90` run, LM-loss selection with `top_k: 2`, and observational serving
metrics that do not affect candidate selection. A new execution may therefore
retain different students.

The current admitted-axis recipe searches hidden width, heterogeneous FFN width,
and depth, with exact FFN-3328 and FFN-3072 controls. It does not claim a full
Qwen 3.5 axis search. Native NeMo attention and GDN pruning remain disabled
until compact and physically sliced reload equivalence is admitted.

## Runs

- [September 2026 legacy-selection learning curve](runs/20260901_legacy_selection_v1/summary.md)

The run leaf follows the provisional
`modelopt.puzzletron-result-record/v1` contract. Runtime manifests are retained
as external source evidence and referenced only by immutable hashes.

# Qwen 3.5 0.8B MIP smoke

The focused Qwen 3.5 0.8B example pins the public checkpoint revision. Its
default model config searches only the FFN intermediate sizes `[3072, 2048]`.
The `mip_smoke.yaml` run enables the composite scenario route required by
named-profile MIP while keeping depth, attention, GDN, and embedding width at
their teacher values.

The experimental `advanced.yaml` overlay covers more axes. Its target values
were derived from the pinned 0.8B geometry rather than selected by a completed
campaign, and the overlay has not been fully runtime-validated. In particular,
the `gdn_key_head_dim` reduction from 128 to 96 does not yet have physical
runtime equivalence evidence. This does not affect the FFN-only smoke route.

## Inspect the plan

The checked-in one-GPU execution plan ends at `mip`. It does not run bypass,
vLLM serving statistics, evaluation, AIPerf, or distillation. Replace every
site placeholder in the runner, then inspect the plan without launching work:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/mip_smoke.yaml \
  --runner examples/puzzletron/configs/orchestration/qwen3p5_0p8b/runner.slurm.yaml \
  --execution examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.smoke.yaml \
  --stage full --dry-run
```

## Run the GPU acceptance test

CPU plan tests cover composition and scheduling only. The real-checkpoint test
is a manual gate and is not part of generic GPU CI. Run it from a reviewed,
worker-visible checkout on one H100 80GB GPU with model access configured:

```bash
python -m pytest -v -s --run-manual \
  tests/gpu/torch/puzzletron/test_qwen3p5_0p8b_smoke.py
```

Treat the route as runtime-validated only when the test passes and confirms a
successful MIP manifest, the `params-90` active profile, and at least one
feasible MIP scenario. Retain the source revision, environment or container,
GPU model, command, and complete pytest log with the result. The test uses
isolated temporary data and cache roots and does not submit scheduler work.

# Checkpoint evaluation

Puzzletron keeps process execution separate from task policy:

- `modelopt/torch/puzzletron/evaluation/lmms.py` runs `lmms-eval` and records durable attempt artifacts.
- `checkpoint.py` provides shared example-side loading, argument, revision, and credential helpers.
- `vlm/` owns VLM model checks, benchmark suites, media preparation, and task adapters.

Modality-specific task catalogs and command-line workflows belong in sibling packages. This keeps text and VLM sampling rules, dependencies, media handling, and judge policies independent while sharing the same evaluator process contract.

## VLM layout

- `vlm/run.py` provides the small command-line entry point and runs the selected suite.
- `vlm/preflight.py` validates local inputs, media, credentials, and judge policy.
- `vlm/suites.py` defines suite membership and pinned sample-selection policy.
- `vlm/tasks.py` generates local `lmms-eval` task configurations and validates them offline.
- `vlm/profile.py` pins benchmark datasets and upstream task configurations.
- `vlm/model.py` validates the initial Qwen 3.5 VLM checkpoint contract.
- `vlm/preparation/` prepares pinned local media layouts.

Inspect the available VLM suites and arguments with:

```bash
python -m examples.puzzletron.evaluation.vlm.run --help
```

Prepare the pinned video benchmark data with:

```bash
python -m examples.puzzletron.evaluation.vlm.preparation.benchmark_data --help
```

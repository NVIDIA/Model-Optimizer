# Run real-model manual integration tests

Maintainers can run two opt-in Qwen 3.5 0.8B integration tests. The text test
checks FFN pruning, IFEval, AIPerf, and short KD. The VLM test checks image-text
scoring, physical checkpoint evaluation on RealWorldQA, AIPerf, and short VLM
KD. Each test uses one GPU and runs the full local orchestration path.

## Prepare the worker image

Use a Linux amd64 host with one H100-class GPU, Docker with NVIDIA GPU support,
network access to the model and benchmark sources, and at least 50 GiB of fast
writable scratch space. Build the repository's pinned worker image by following
[build and run the Puzzletron worker image](worker_image.md), including its GPU
check and checkout mount. Mount the scratch space at `/scratch`, then run the
remaining commands inside that container on an exclusive GPU. The worker-image
guide owns the image tag, container invocation, and runtime prerequisites.

## Collect both tests

Collection verifies the test names, manual-test flag, imports, and pytest
configuration without downloading the model or using the GPU:

```bash
/venv/bin/python -m pytest --collect-only -q --run-manual \
  tests/gpu/torch/puzzletron/test_qwen3p5_0p8b_smoke.py::test_qwen3p5_0p8b_orchestrated_full_smoke_completes \
  tests/gpu/torch/puzzletron/test_qwen3p5_0p8b_vlm_smoke.py::test_qwen3p5_0p8b_orchestrated_vlm_full_smoke_completes
```

## Run the text smoke

No environment variable is required. The test downloads the public model and
IFEval inputs into its pytest scratch directory, so network access is required.
Allow up to two hours; the test timeout is 7,300 seconds. Run:

```bash
/venv/bin/python -m pytest -q --run-manual \
  --basetemp /scratch/qwen3p5-0p8b-text-smoke \
  tests/gpu/torch/puzzletron/test_qwen3p5_0p8b_smoke.py::test_qwen3p5_0p8b_orchestrated_full_smoke_completes
```

On success, pytest exits with status 0. The test also checks the width and
slicing summaries, the active MIP profile, and IFEval completion for the saved
pre-KD and post-KD checkpoints. Keep `/scratch/qwen3p5-0p8b-text-smoke` when
diagnosing a failure; otherwise it can be removed after the run.

## Prepare and run the VLM smoke

The VLM test requires a persistent Hugging Face cache containing the pinned
RealWorldQA snapshot. Choose a path with at least 20 GiB free and prepare it
inside the worker image:

```bash
export PUZZLETRON_VLM_BENCHMARK_HF_HOME=/scratch/qwen3p5-0p8b-vlm-hf

/venv/bin/python -m examples.puzzletron.evaluation.vlm.preparation.benchmark_data \
  --hf-home "$PUZZLETRON_VLM_BENCHMARK_HF_HOME" \
  --tasks realworldqa
```

The preparation command requires network access on its first run and validates
the pinned snapshot on later runs. For cache layout and repair behavior, see
[cache benchmark data](vlm_checkpoint_evaluation.md#cache-benchmark-data).

Run the test with that variable exported:

```bash
/venv/bin/python -m pytest -q --run-manual \
  --basetemp /scratch/qwen3p5-0p8b-vlm-smoke \
  tests/gpu/torch/puzzletron/test_qwen3p5_0p8b_vlm_smoke.py::test_qwen3p5_0p8b_orchestrated_vlm_full_smoke_completes
```

Allow up to two hours and 25 minutes; the test timeout is 8,700 seconds. It
exits with status 0 only after image tensors reach the model, replacement
scoring completes, MIP produces a feasible result, and RealWorldQA completes
for the saved pre-KD and post-KD checkpoints. The persistent Hugging Face cache
can be reused. Keep the pytest scratch directory only when its logs or artifacts
are needed for diagnosis.

## Troubleshoot a failure

Read the stdout and stderr tail printed by pytest, then inspect the retained
`--basetemp` directory; campaign state and logs are under its `results`
directory. Use the [worker-image GPU check](worker_image.md#gpu-check) for
runtime failures, [cache benchmark data](vlm_checkpoint_evaluation.md#cache-benchmark-data)
for VLM cache repair, and [progress and interruption](orchestration_operations.md#progress-and-interruption)
for orchestration recovery.

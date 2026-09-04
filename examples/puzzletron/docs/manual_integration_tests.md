# Run real-model manual integration tests

Maintainers can run two opt-in Qwen 3.5 0.8B integration tests. The text test
checks FFN pruning, IFEval, AIPerf, and short KD. The VLM test checks image-text
scoring, physical checkpoint evaluation on RealWorldQA, AIPerf, and short VLM
KD. Each test uses one GPU and runs the full local orchestration path.

## Prepare the worker image

Use a Linux amd64 host with one H100-class GPU, Docker with NVIDIA GPU support,
network access to the model and benchmark sources, and at least 50 GiB of fast
writable scratch space. Build the repository's pinned worker image by following
[build the Puzzletron worker image](worker_image.md#build), then run its
[GPU check](worker_image.md#gpu-check). The image provides `/venv`, CUDA,
NeMo AutoModel, vLLM, AIPerf, and the pinned `lmms-eval` checkout.

Start the image from the repository root. Mount the checkout and scratch space
so pytest artifacts survive the container:

```bash
export PUZZLETRON_TEST_SCRATCH=/path/with-at-least-50GiB-free
mkdir -p "$PUZZLETRON_TEST_SCRATCH"
image="modelopt-puzzletron:linux-amd64-git-$(git rev-parse --short=12 HEAD)"

docker run --gpus all --ipc=host --rm -it \
  -v "$PWD:/workspace/modelopt" \
  -v "$PUZZLETRON_TEST_SCRATCH:/scratch" \
  -w /workspace/modelopt \
  "$image" bash
```

Run the remaining commands inside that container. Use an exclusive GPU because
both tests start local model-serving processes.

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

- Confirm `/venv/bin/python` is active and `torch.cuda.is_available()` is true
  with the [worker-image GPU check](worker_image.md#gpu-check).
- Confirm the selected scratch filesystem has enough free space and supports
  local process execution.
- For VLM cache errors, rerun the preparation command above. Do not point
  `PUZZLETRON_VLM_BENCHMARK_HF_HOME` at a symlink.
- Read the stdout and stderr tail printed by pytest, then inspect the retained
  `--basetemp` directory. Campaign state and logs are under the test's `results`
  directory.
- For orchestration failures and safe resume behavior, see
  [progress and interruption](orchestration_operations.md#progress-and-interruption).

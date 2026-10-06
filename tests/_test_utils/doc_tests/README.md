# Executable Markdown tests

The QAT and Megatron Bridge READMEs use the same parser, subprocess runner and
pytest adapter. Example-specific code only prepares inputs and verifies results.
The visible commands execute without command rewriting or mocked model operations.
These are trusted repository tests with the same privileges as pytest.

## Directive contract

Directives occupy their own lines outside code fences. Metadata is TOML:

````markdown
<!-- modelopt-doc-test:begin
id = "example-smoke"
profile = "cpu"
timeout_seconds = 30
-->
<!-- modelopt-doc-test:setup python
ctx.cwd = ctx.tmp
ctx.env["MESSAGE"] = "hello"
-->
<!-- modelopt-doc-test:run -->

```bash
printf '%s' "$MESSAGE" > result.txt
```

<!-- modelopt-doc-test:verify python
assert (ctx.cwd / "result.txt").read_text() == "hello"
-->
<!-- modelopt-doc-test:end -->
````

- Required metadata: unique lowercase/hyphenated `id`, `profile` (`cpu` or `gpu`),
  and positive `timeout_seconds`. Optional: `manual = true`, `min_gpus = 2`,
  and `requires = ["megatron.bridge"]` for importable modules.
- Setup precedes all selected fences; verification follows them. Both share a
  namespace with `ctx.repo`, `ctx.tmp`, `ctx.cwd` and mutable `ctx.env`.
- Each `run` selects one immediately following `bash`, `sh`, or `python` fence;
  only blank lines may intervene. Backtick, tilde, list-indented and single-level
  blockquoted fences are supported. Nested scenarios and duplicate IDs fail.
- Shell fences share one Bash process with `set -Eeuo pipefail`; shell variables,
  directory changes and exported environment persist. Verification sees the final
  `ctx.cwd` and `ctx.env`. Setup/verification Python uses those explicit fields.
- Python fences share the setup namespace. Before execution the worker applies
  `ctx.cwd` and `ctx.env`; afterward it records their final values. Keep Python
  and shell fences in separate scenarios to avoid ambiguous shared state.
- `run background` starts the final shell fence asynchronously. Verification must
  check readiness and behavior; `ctx.background_pid` identifies its shell process.
  The serving test checks HTTP health and requests a real adapter completion.
- The deadline covers setup, commands and verification. The parent kills the
  process group on success, failure or timeout. Commands must not detach, disable
  error checking, or replace/exit the shell. Workspace artifacts remain in pytest's
  temporary directory; internal control files are removed. Failures include output
  and Markdown source locations.
- The runner prepends the checkout and test utilities to `PYTHONPATH`. The separate
  installation test deliberately removes it and uses a disposable virtualenv so
  the installed package can be tested without replacing the checkout environment.
- Collection uses `require_coverage=True`: every fence must be selected or have
  `<!-- modelopt-doc-test:skip A specific reason. -->` immediately before it.
  This is an accounting check, **not** proof that excluded/manual snippets passed.
- `manual` uses the repository's existing `--run-manual` option. Missing declared
  modules or insufficient GPUs produce explicit pytest skips. A GPU profile needs
  at least one device. Manual prerequisites are supplied by the operator.

## Coverage and limits

| README | Default smoke scenarios | Opt-in integration scenarios | Exclusions |
| --- | --- | --- | --- |
| `llm_qat` | Recipes; QAT/QAD/QLoRA; simple Llama demo; three Python APIs; FSDP2 layer override; config copy; CLI learning-rate override; dataset cache | Installation; vLLM adapter serving; MMLU/WikiText evaluation | None |
| `megatron_bridge` | PTQ plus export; mock-data distillation; all-iteration export; manual pruning; distillation/pruning CLI help | Login; checkpoint import; MLflow PTQ; real-data distillation/QAD; three NAS targets; VLM pruning; vLLM generation | Interactive Docker launch; incomplete VLM/inline-export/single-iteration templates; illustrative output tree |

QAT fixtures copy real scripts/configs and generate two-layer Qwen3 or Llama
models, with hidden/intermediate size 512 and 20 local text records. Training
configs use two optimizer steps, 128-token sequences, and one or two ranks via
`MODELOPT_DOC_TEST_GPUS=1|2`. QLoRA uses DDP; QAT/QAD use FSDP2. The simple demo
keeps its visible defaults. Python fragments receive real model, Trainer and data
objects. Dataset-cache and config-copy tests run without GPU execution.

Megatron smoke tests use a local two-layer Qwen3, four calibration records,
32-token sequences, two training steps and **two actual torchrun ranks**. Public
shell variables override inputs and scale while preserving full-scale defaults.
Export prerequisites are built by executing the README's actual training fence
inside the same timeout/process group. Pruning verifies the loaded model's reduced
intermediate dimension; multi-iteration export loads both saved HF checkpoints.

Checks cover quantization state, training progress/finite losses, changed QAT/QAD
weights, packed NVFP4 exports, adapter artifacts, dataset contents and loadable
Bridge outputs. These tests establish command/API composition; they do not establish
full-model accuracy, convergence, memory requirements or serving performance.
A copied FSDP2 config test checks the copy, not the subsequent human edit.

## Run and collect

Run inside the appropriate example environment: PyTorch 26.07 with the QAT
requirements, or NeMo 26.08 for Megatron Bridge, as in the example CI workflow.
Keep the container's matching Torch/FlashAttention binaries when installing test
dependencies. Install the checkout and its `dev-test` dependencies first.

```bash
# Parser/runner tests: no Torch, GPU or repository-wide conftest required.
python -m pytest tests/unit/doc_tests --confcutdir=tests/unit/doc_tests -o addopts='' -q

# Collect and inspect skip reasons in the normal test environment.
python -m pytest tests/examples/llm_qat/test_readme.py --collect-only --no-cov
python -m pytest tests/examples/megatron_bridge/test_megatron_bridge_readme.py --collect-only --no-cov

# Bounded defaults; manual integrations are skipped.
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/examples/llm_qat/test_readme.py --no-cov -rs
CUDA_VISIBLE_DEVICES=0,1 MODELOPT_DOC_TEST_GPUS=2 \
  python -m pytest tests/examples/llm_qat/test_readme.py --no-cov -k quickstart -rs
CUDA_VISIBLE_DEVICES=0,1 \
  python -m pytest tests/examples/megatron_bridge/test_megatron_bridge_readme.py --no-cov -rs

# Select ONE manual integration with its prerequisites already supplied.
# MODELOPT_DOC_QLORA_CHECKPOINT: exported adapter directory containing base_model/.
python -m pytest tests/examples/llm_qat/test_readme.py --no-cov --run-manual -k llm-qlora-serve
# MODELOPT_DOC_EVAL_CHECKPOINT: fake-quantized QAT checkpoint; lm_eval and dataset access required.
python -m pytest tests/examples/llm_qat/test_readme.py --no-cov --run-manual -k llm-qat-evaluation
```

Installation downloads packages into a temporary virtualenv and may compile GPU
extensions. Bridge manual tests need the full models, gated-data access and the
GPU count declared in their scenario. Login requires `HF_TOKEN` and writes into a
temporary `HF_HOME`; no token belongs in Markdown. Real-data training needs
`DATA_PREFIX_1` and `DATA_PREFIX_2`; QAD also needs `QUANTIZED_CHECKPOINT`. MLflow
requires `MLFLOW_TRACKING_URI`; generation requires `PRUNED_CHECKPOINT`.
Manual tests can run for hours and are never enabled by the example CI command.

## CI triggers

Both README test modules are collected by the existing example jobs. The
workflow runs nightly, on demand, and on pushes to copied PR branches
`pull-request/<number>` after unit tests pass. README-only changes explicitly
activate their respective Torch or Megatron lane despite the general Markdown
exclusion. Changes to shared test utilities activate every lane. Two-GPU Bridge
smokes skip on the single-GPU PR runner and run on the two-GPU nightly runner;
CLI-help scenarios still execute on the single-GPU runner.

## File responsibilities

| File | Purpose in this extension |
| --- | --- |
| `parser.py` | Shared support for Python, quoted/list fences, background services, resource metadata and explicit exclusions. |
| `runner.py` | Execute Python in the scenario namespace and keep servers alive through verification, under the existing timeout and cleanup. |
| `pytest_utils.py` | Common collection, manual markers and resource checks, independent of the example. |
| `fixtures/llm_qat.py` | Reduced QAT/QAD/QLoRA/Llama inputs, real prerequisite quantization, Python API inputs and artifact assertions. |
| `fixtures/megatron_bridge.py` | Local Bridge inputs, real distillation prerequisites and quantization/pruning/export assertions. |
| `fixtures/installation.py` | Disposable package-installation environment and import verification. |
| `fixtures/serving.py` | Adapter-server prerequisites, readiness and a real completion request. |
| `tests/unit/doc_tests/test_markdown.py` | CPU regression tests for the shared execution contract and complete fence accounting in both READMEs. |
| `tests/examples/llm_qat/test_readme.py` | Collect QAT scenarios through the common pytest adapter. |
| `tests/examples/megatron_bridge/test_megatron_bridge_readme.py` | Collect Bridge scenarios through the same adapter. |
| `examples/llm_qat/README.md` | Annotate the remaining snippets; correct QAD teacher construction and incomplete training commands. |
| `examples/megatron_bridge/README.md` | Annotate smoke/manual scenarios, expose input/scale overrides and account for non-executable templates. |
| `.github/workflows/example_tests.yml` | Trigger the Megatron lane for README-only changes. |
| `tests/examples/README.md` | Point example authors to the shared shell/Python test mechanism. |
| `tests/_test_utils/doc_tests/README.md` | Document the contract, scope, prerequisites, limits and CI behavior. |

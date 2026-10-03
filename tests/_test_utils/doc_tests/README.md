# Executable Markdown tests

Only explicitly marked shell fences run. The pilot is `examples/llm_qat/README.md`,
from Quick Start through QAT; installation and QAD fences are unmarked.
`tests/examples/llm_qat/test_readme.py` collects its scenarios in the existing example lane.
The runner prepends the checkout and test utilities to `PYTHONPATH`, so commands
launched from a temporary workspace still import the checkout.
These are trusted repository tests, with the same execution privileges as pytest.

## Directive contract

Directives occupy their own lines outside code fences. Metadata uses TOML with
exactly three required fields: a unique lowercase/hyphenated `id`, `profile`
(`cpu` or `gpu`), and a positive integer `timeout_seconds`.

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

- Setup Python precedes all executable fences; verification Python follows them.
  Both share a Python namespace with `ctx.repo`, `ctx.tmp`, `ctx.cwd` (absolute
  paths) and `ctx.env` (a mutable environment dictionary).
- Each `run` opts in **one** immediately following `bash` or `sh` fence, allowing
  blank lines only. Backtick and tilde fences are supported. Unmarked fences are
  skipped even inside a scenario. Nested scenarios and duplicate IDs are errors.
- All selected fences execute in one Bash shell with `set -Eeuo pipefail`.
  Directory changes, shell variables and exported environment persist between
  fences. Verification sees the final `ctx.cwd` and exported `ctx.env`.
  Python must use these explicit context fields; its own working directory and
  `os.environ` are not synchronized with the shell. Commands must not disable
  error checking, replace/exit the shell, or detach into new process groups.
- The deadline covers setup, shell commands and verification together. All child
  processes in the scenario's process group are killed on timeout, failure or
  success. Internal control files are removed; pytest retains workspace artifacts
  under its normal temporary-directory policy. Output is captured and returned on
  success or included in failures, with Markdown file/line locations.
- Profiles describe requirements; the pytest integration skips GPU scenarios
  when CUDA is unavailable. It applies the scenario timeout plus 30 seconds for
  pytest cleanup, while the subprocess deadline remains exactly the declared value.

## CPU development and Markdown collection

From the repository root in an environment with pytest, pytest-timeout and
`tomli` on Python 3.10:

```bash
# No Torch import, GPU, model download or repository-wide conftest required.
python -m pytest tests/unit/doc_tests --confcutdir=tests/unit/doc_tests -o addopts='' -q

# In the regular ModelOpt test environment: parse/compile directives, execute nothing.
python -m pytest tests/examples/llm_qat/test_readme.py --collect-only --no-cov
pre-commit run markdownlint-cli2 --files examples/llm_qat/README.md tests/_test_utils/doc_tests/README.md tests/examples/README.md
```

The CPU tests also parse the annotated pilot and execute its recipe-listing
scenario. The regular `python -m pytest tests/unit/doc_tests --no-cov` command
runs the same tests with repository-wide fixtures enabled.

## GPU setup and acceptance

Use the example CI image, `nvcr.io/nvidia/pytorch:26.07-py3`, with the checkout
mounted and the repository root as the working directory. For a local Docker host:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD:/workspace/Model-Optimizer" -w /workspace/Model-Optimizer \
  nvcr.io/nvidia/pytorch:26.07-py3 bash

# Inside the container: preserve its matching Torch/FlashAttention binaries.
/usr/bin/python3 -c 'import importlib.metadata as m; print("\n".join(f"{p}=={m.version(p)}" for p in ("torch", "torchvision", "flash-attn")))' > /tmp/modelopt-doc-constraints.txt
/usr/bin/python3 -m venv --system-site-packages /tmp/modelopt-doc-gpu
source /tmp/modelopt-doc-gpu/bin/activate
python -m pip install -c /tmp/modelopt-doc-constraints.txt \
  -e '.[hf,dev-test]' -r examples/llm_qat/requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Single-GPU smoke (default resource profile).
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/examples/llm_qat/test_readme.py --no-cov -s

# Two-rank FSDP2 acceptance.
CUDA_VISIBLE_DEVICES=0,1 MODELOPT_DOC_TEST_GPUS=2 \
  python -m pytest tests/examples/llm_qat/test_readme.py --no-cov -s

# Existing backend baseline, retained separately.
python -m pytest 'tests/examples/llm_qat/test_llm_qat.py::test_qwen3_qat_nvfp4[fsdp2]' --no-cov -s
```

On a Slurm host use its interactive container launcher instead of Docker, then
run the same inside-container commands. Installation fences are deliberately
excluded from scenarios: executing `pip install -U nvidia-modelopt[hf]` would
replace the checkout under test. This pilot does not validate package installation.

The fixture copies the real scripts and writable YAML configs into a temporary
repository layout and aliases a locally generated two-layer Qwen3 checkpoint to
`Qwen/Qwen3-8B`. Hidden setup supplies 20 synthetic text records, 16 train/4 eval
samples, two optimizer steps, 128-token training sequences and one or two FSDP2
ranks. PTQ retains the visible command's default 8,192-token length but calibrates
on only 16 samples. FlashAttention, Liger, FSDP2 and the NVFP4 recipe remain enabled.
No quantization, training or export implementation is mocked or substituted.

Verification requires saved ModelOpt state, two completed optimizer steps,
finite losses, changed trained weights, and packed exported weights with
`quant_algo: NVFP4`. This checks that the documented CLI sequence composes and
produces real artifacts. It does not establish Qwen3-8B accuracy, full-model memory
requirements, convergence, deployment compatibility or inference performance.
Only the two-GPU run establishes distributed coverage. Existing backend tests
remain because the README pilot does not cover DDP or DeepSpeed.

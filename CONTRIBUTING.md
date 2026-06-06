# Contributing to Model Optimizer

Thanks for your interest in contributing to Model Optimizer (ModelOpt)!

> [!NOTE]
> Any contributions to this repository are only accepted under the Apache 2.0 license.

## 🛠️ Setting up your environment

Ensure that Model Optimizer (ModelOpt) is installed in editable mode and that all `dev` optional requirements are installed:

```bash
pip install -e ".[dev]"
```

If you are working on features that require dependencies like TensorRT-LLM or Megatron-Core, consider using a docker container to simplify the setup process.
Visit our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) for more information.

## 🧹 Code linting and formatting

- All code (Python, C++, Markdown, etc.) is automatically checked to adhere to the coding standards upon commit (see below for more information).
- See [`.pre-commit-config.yaml`](.pre-commit-config.yaml) for details about each tool.
- For VSCode or Cursor, we provide default workspace settings to integrate the linting tools into your IDE: see [workspace settings](./.vscode/settings.json).

### Pre-commit hooks

Enable pre-commit hooks to automatically check and fix code quality before committing:

```bash
pre-commit install
```

If you want to make a temporary commit that skips checks, use the `-n` flag when committing:

```bash
git commit -m "temporary commit" -n
```

To run the pre-commit hooks without committing, use:

```bash
pre-commit run --all-files
```

## 📐 Coding standards

Guidelines for production code in ModelOpt. Key values: simplicity, modularity,
and conciseness.

### Principles

- **Prefer simple, surgical changes.** Touch only what the task requires. Avoid speculative
  refactors, broad rewrites, and "while we're here" cleanups.
- **Design for simplicity and readability.** Choose the design that is easiest to understand and maintain.
  Code is read top to bottom: put high-level behavior first, hide lower-level details behind well-named helpers,
  and treat heavy branching as a signal to reconsider the design.
- **Prefer modular, composable solutions.** Avoid input-specific or case-specific hard-coding.
  Use existing extension points when they fit. If none fit, add a simple, focused helper,
  class, or plugin that cleanly captures the new behavior. Keep scope limited to known cases.
- **Respect inheritance boundaries.** Parent abstractions should define shared contracts and
  shared behavior, not child-specific special cases.
- **Don't repeat yourself; keep a single source of truth.** Consolidate repeated logic or intent with a shared helper, API,
  or abstraction when doing so keeps the design simpler. Avoid duplication that can drift out of sync.
- **Comment cautiously.** Comments should add context, not translate code into English.
  Prefer making the code self-explanatory first. Use comments only for non-obvious
  intent or constraints that remain unclear from the code. Apply this guidance to new
  comments only; do not rewrite or delete existing comments just for style.
- **Document public APIs.** Public and higher-level APIs should have docstrings, including examples when useful.
  Internal helpers should usually be self-documenting through clear names and structure.
- **Fix the bug cause, not the side effect.** For bug fixes, find the root cause instead of patching for its side effect.
- **Validate external input once.** Check types and values at the interface boundary. Internal code can trust those
  checks and avoid redundant assertions.
- **Remove dead code.** Delete unused imports, unreachable branches, and obsolete helpers.
- **Keep imports at the top of the file.** Place all imports at module top in both source
  and test files so import errors surface at module load time rather than at runtime or
  during a specific test. Put an import inside a function only when there is a concrete
  reason: resolving a circular import that cannot be restructured, guarding an optional
  dependency (e.g., TensorRT-LLM, Megatron-Core), or deferring an unusually heavy import
  with explicit justification. Add a brief comment in those cases naming the reason.
- **Define the public API with `__all__` and re-export via `from .module import *`.**
  Each module declares its public surface with `__all__ = [...]` at the top of the file.
  Package `__init__.py` files re-export submodules with `from .module import *`. This
  keeps the public API explicit at the source (next to the definitions), avoids
  hand-maintained import lists in `__init__.py` drifting out of sync, and makes
  star-imports safe by limiting them to the curated `__all__` names.

### Performant AI code

- **Keep tensor work on the GPU and avoid unnecessary CPU-GPU syncs.** Reading metadata such as `tensor.shape` is fine.
  Avoid Python scalar extraction and operators such as `tensor.item()`, `float(tensor)`, or `min(tensor)` because they
  can trigger CPU-GPU syncs. Use PyTorch tensor ops such as `tensor.min()` by default, and only extract Python scalars
  when the CPU needs the value. Tensor-value-based Python branching can also break CUDA graphs.
- **Develop with distributed processing in mind.** Examples: Use `print_rank_0` or `warn_rank_0`
  when possible to avoid noisy logs. Guard shared side effects, such as
  file writes or shared state updates, against race conditions between ranks.

### Compatibility

- **Preserve config and checkpoint backward compatibility.** ModelOpt checkpoints include serialized
  `ModeloptBaseConfig` instances such as `QuantizeConfig`. If these Pydantic-based configs change
  without backward compatibility handling, older checkpoints may no longer load. Make breaking changes
  explicit and intentional.

## Adding a new PIP dependency

Currently we have 2 places where we mention pip dependencies: [pyproject.toml](./pyproject.toml) for dependencies that are required for the ModelOpt library and `examples/<example-name>/requirements.txt` for dependencies that are required for the specific examples.

If adding a new PIP dependency to any of these, make sure to verify the LICENSE of the dependency. If its not a permissive license (e.g. MIT, Apache 2), you need to provide a justification for the use of the dependency in the PR and check with `@NVIDIA/modelopt-setup-codeowners` if its allowed or not.

## 🔒 Security coding practices

All contributors must follow the security coding practices documented in *Security Coding Practices for
Contributors* section of [SECURITY.md](./SECURITY.md#security-coding-practices-for-contributors) page.

Any security-sensitive exception requires review and approval from `@NVIDIA/modelopt-setup-codeowners`.

## 📋 Copying code from other sources

The utilization of third-party code requires authorization via the Open Source Review Board (OSRB) team and needs to follow proper guidance on contributing code.

If you are an external contributor, seek guidance from `@NVIDIA/modelopt-setup-codeowners` for next steps. For internal contributors, follow the steps below:

- **Update NVBug for details on use of open-source code:**
  Reopen NVBug 6046893 and add your use case in the table. Merging your PR with code copied from permissive licensed repositories (e.g. MIT, Apache 2) is generally fine but for other licenses, it is necessary to get expert guidance before merging your PR.
- **License header format:** The file which has code copied from another third-party GitHub repository should have the following in order:
  1. A reference link (with commit hash) to the source from which the code was copied.
  1. The original repository's Copyright / License.
  1. The NVIDIA Apache 2.0 Copyright / License header.
- **Update `SPDX-License-Identifier`:** If the third-party code uses a different license than Apache 2.0, update the `SPDX-License-Identifier` in the NVIDIA header to reflect both licenses using SPDX expression syntax. For example, for MIT-licensed source code:

  ```python
  # SPDX-License-Identifier: Apache-2.0 AND MIT
  ```

  If the third-party code is also Apache 2.0, no change is needed (`SPDX-License-Identifier: Apache-2.0` remains correct).
- **Update `LICENSE`:** Add the third-party copyright holder to the appropriate license section in the [`LICENSE`](./LICENSE) file under *Third-Party Software Notices*. If the third-party license is not already listed there, add a new section with the full license text.
- **Exclude from license pre-commit hook:** Exclude copied files from the license pre-commit hook so it doesn't auto-add the NVIDIA Apache 2.0 license on top of the file. Add the file path to the `exclude` list in the `insert-license` hook in [`.pre-commit-config.yaml`](./.pre-commit-config.yaml).

See [`modelopt/torch/quantization/utils/calib_utils.py`](./modelopt/torch/quantization/utils/calib_utils.py) for an example of the correct license header format.

## 📝 Writing and running tests

We use [pytest](https://docs.pytest.org/) for all tests. For any new features / examples, make sure to add tests and that the coverage check in your PR passes. The tests are organized into the following directories:

- `tests/unit`: Fast cpu-based unit tests for the core ModelOpt library. They should not take more than a few seconds to run.
- `tests/gpu`: Fast GPU-based unit tests for the core ModelOpt library. In most cases, they should not take more than a few seconds to run.
- `tests/gpu_megatron`: Fast GPU-based unit tests for the core ModelOpt library for Megatron-Core features. In most cases, they should not take more than a few seconds to run.
- `tests/gpu_trtllm`: Fast GPU-based unit tests for the core ModelOpt library for TensorRT-LLM features. In most cases, they should not take more than a few seconds to run.
- `tests/gpu_vllm`: Fast GPU-based unit tests for the core ModelOpt library for vLLM features. In most cases, they should not take more than a few seconds to run.
- `tests/examples`: Integration tests for ModelOpt examples. They should not take more than a few minutes to run. Please refer to [example test README](./tests/examples/README.md) for more details.

For lightweight focused local validation, run `pytest` directly on the relevant test path. For example:

```bash
pytest tests/unit/torch/quantization
```

For broader repo validation and dependency setup, use [noxfile.py](./noxfile.py). Run `nox -l` to list available sessions, then run the matching session with `nox -s <session>`. The `unit-3.12(torch_211, tf_latest)` session runs `tests/unit` with a specific Torch and Transformers combination:

```bash
nox -s "unit-3.12(torch_211, tf_latest)"
```

### Test design principles

- **Develop with focused tests.** During development, write as many focused tests as needed, including lower-level
  unit tests or internal probes, to understand and harden behavior.
- **Curate production tests and keep them lean.** Before staging or committing, decide which tests should be checked
  in. Checked-in tests should document expected behavior, protect against regressions, or flag backward-incompatible
  behavior changes. Remove redundant lower-level tests when a higher-level test already covers the same behavior,
  keeping CI/CD fast and lean.
- **Keep `tests/unit` offline — no HuggingFace Hub access.** Unit tests must be hermetic so they never flake on
  network/timeout issues. Do not call `from_pretrained("<org>/<model>")`, `load_dataset("<hub-id>")`,
  `snapshot_download(...)`, etc. with Hub IDs. Instead build dummy models, tokenizers, configs, and datasets locally —
  e.g. the `create_tiny_*` helpers and `get_tiny_tokenizer()` in `tests/_test_utils/`, or a small on-disk dataset
  directory written with `datasets.Dataset.from_dict(...).to_parquet(...)`.
- **Respect the per-test timeout.** `tests/conftest.py` applies a default per-test call timeout by directory; override a
  single slow test with `@pytest.mark.timeout(<seconds>)`, and register any new top-level `tests/<group>/` in that
  mapping (collection errors until you do).

## ✍️ Signing your work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original
  work, or you have rights to submit it under the same license, or a compatible license.

- You need to cryptographically sign-off your commits as well using an SSH/GPG key which is different than the one used for authentication. See [GitHub docs](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits) for more details. Note that setting up the SSH key is much simpler than the GPG key hence recommended to use SSH signing key following the steps below (requires `git>=2.34`).

  - Generate a new SSH key as per steps [in this doc](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key). For example:

    ```bash
    ssh-keygen -t ed25519 -f "${HOME}/.ssh/id_ed25519_git_signing" -P ""
    ```

  - Upload the public key (`cat "${HOME}/.ssh/id_ed25519_git_signing.pub"`) as a new SSH key in your [GitHub settings](https://github.com/settings/ssh/new) with an appropriate title and select key type as `Signing Key`.

  - Configure your local `git` to use the new SSH key for signing commits:

    ```bash
    git config --global user.signingkey "${HOME}/.ssh/id_ed25519_git_signing.pub"
    git config --global gpg.format ssh
    git config --global commit.gpgsign true
    ```

- **Any contribution which contains commits that are not Signed-Off will not be accepted**.

- Once you have set up your SSH/GPG key, to sign off on a commit you simply use the `--signoff --gpg-sign` (or `-s -S`) option when committing your changes:

  ```bash
  git commit -s -S -m "Add cool feature."
  ```

  > *TIP: To enable this for committing in VSCode, you can enable `git.alwaysSignOff` and `git.enableCommitSigning` in your VSCode settings (`Ctrl/Cmd + ,`).*

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the Developer Certificate of Origin (DCO):

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Submitting your code

- Submit a pull request and let auto-assigned reviewers (based on [CODEOWNERS](./.github/CODEOWNERS)) review your PR.
- If any CI/CD checks fail, fix the issues and push again.
- Once your PR is approved and all checks pass, one of the reviewers will merge the PR.

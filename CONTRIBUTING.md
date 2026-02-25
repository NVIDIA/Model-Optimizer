# Contributing to Model Optimizer

Thanks for your interest in contributing to Model Optimizer (ModelOpt)!

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

## 🔒 Security coding practices

All contributors must follow the security coding practices documented in *Security Coding Practices for
Contributors* section of [SECURITY.md](./SECURITY.md#security-coding-practices-for-contributors) page.

Any security-sensitive exception requires review and approval from `@NVIDIA/modelopt-setup-codeowners`.

## 📋 Copying code from other sources

The utilization of third-party code requires authorization via the Open Source Review Board (OSRB) team and needs to follow proper guidance on contributing code.

If you are an external contributor, seek guidance from `@NVIDIA/modelopt-setup-codeowners` for next steps. For internal contributors, follow the steps below:

- **File NVBug for use of open-source code:**
  Clone NVBug 2885977 and add your use case. Copying code from permissive licensed repositories (e.g. MIT, Apache 2) is generally self-checkout but for other licenses, it is necessary to get expert guidance before merging your PR.
- **License header format:** The file which has code copied from another third-party GitHub repository should have the following in order:
  1. A reference link (with commit hash) to the source from which the code was copied.
  1. The original repository's Copyright / License.
  1. The NVIDIA Apache 2.0 Copyright / License header.

  See [`modelopt/torch/speculative/eagle/utils.py`](./modelopt/torch/speculative/eagle/utils.py)
  for an example of the correct license header format.
- **Exclude from license pre-commit hook:** Exclude copied files from the license pre-commit hook so it doesn't auto-add the NVIDIA Apache 2.0 license on top of the file. Add the file path to the `exclude` list in the `insert-license` hook in [`.pre-commit-config.yaml`](./.pre-commit-config.yaml).

## 📝 Writing tests

We use [pytest](https://docs.pytest.org/) for all tests. For any new features / examples, make sure to add tests and that the coverage check in your PR passes. The tests are organized into the following directories:

- `tests/unit`: Fast cpu-based unit tests for the core ModelOpt library. They should not take more than a few seconds to run.
- `tests/gpu`: Fast GPU-based unit tests for the core ModelOpt library. In most cases, they should not take more than a few seconds to run.
- `tests/gpu_megatron`: Fast GPU-based unit tests for the core ModelOpt library for Megatron-Core features. In most cases, they should not take more than a few seconds to run.
- `tests/examples`: Integration tests for ModelOpt examples. They should not take more than a few minutes to run. Please refer to [example test README](./tests/examples/README.md) for more details.

Please refer to [tox.ini](./tox.ini) for more details on how to run the tests and their dependencies.

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

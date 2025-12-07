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

## 📝 Writing tests

We use [pytest](https://docs.pytest.org/) for all tests. The tests are organized into the following directories:

- `tests/unit`: Fast cpu-based unit tests for the core ModelOpt library. They should not take more than a few seconds to run.
- `tests/gpu`: Fast GPU-based unit tests for the core ModelOpt library. In most cases, they should not take more than a few seconds to run.
- `tests/examples`: Integration tests for ModelOpt examples. They should not take more than a few minutes to run. Please refer to [example test README](./tests/examples/README.md) for more details.

Please refer to [tox.ini](./tox.ini) for more details on how to run the tests and their dependencies.

### Code Coverage

For any new features / examples, make sure to they are covered by the tests and that the Codecov coverage check in your PR passes.

## Submitting your code

- If you are an external contributor, create a fork of the repository.
- Rebase (not merge) your code to the most recent commit of the `main` branch. We want to ensure a linear history;
  see [Merge vs Rebase](https://www.atlassian.com/git/tutorials/merging-vs-rebasing). Remember to test again locally after rebasing to catch any new issues before pushing to your PR.

```bash
git pull
git rebase origin/main
git push origin <branch> --force-with-lease
```

- When pushing the rebased (or any) branch, use `git push --force-with-lease` instead of `git push --force`.
- Submit a pull request and let auto-assigned reviewers (based on [CODEOWNERS](./.github/CODEOWNERS)) review your PR.
- If any CI/CD checks fail, fix the issues and push again.
- Once your PR is approved and all checks pass, one of the reviewers will merge the PR.

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

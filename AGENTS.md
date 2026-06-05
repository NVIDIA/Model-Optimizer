# Agent Instructions for ModelOpt

These instructions apply to AI-assisted work in this repository.

## Repository orientation

- Start with `README.md` for project overview and install.
- Use `modelopt/` for source, `tests/` for focused test coverage, and
  `examples/` or `docs/` for usage patterns.
- **Agent skills and shared config live under `.agents/`** — the canonical,
  agent-agnostic source of truth (`.agents/skills/<name>/SKILL.md`,
  `.agents/scripts/`, `.agents/clusters.yaml.example`). Claude Code's
  `.claude/skills`, `.claude/scripts`, and `.claude/clusters.yaml.example` are
  relative symlinks into `.agents/`. Always edit files under `.agents/`, not the
  symlink path. See `.agents/README.md` for the convention.

## Coding guidelines

- **Coding guide:** Code development and review require reading and following
  the [coding standards in CONTRIBUTING.md](CONTRIBUTING.md#-coding-standards);
  do not skip this step.
- **Use relative paths** from the repo root in commands and file references.

## Iterative development

- **Running tests:** Follow the
  [writing and running tests](CONTRIBUTING.md#-writing-and-running-tests)
  instructions. For fast initial iteration, choose focused tests for the
  changed area from `tests/`.
- **Running pre-commit:** Follow the
  [pre-commit hook instructions](CONTRIBUTING.md#pre-commit-hooks). Hooks may
  modify files; review and re-stage those changes before committing.
- **Signed commit:** Use `git commit -s -S -m "<message>"` for commits so they
  follow the [signing your work](CONTRIBUTING.md#-signing-your-work)
  requirements.
- **Never `git push` without explicit approval in the current turn.** Commit
  locally is fine; publishing to a remote is not.
- After `git commit`, stop and wait for the user to say "push", "publish",
  "ship", or equivalent before running `git push`, `gh pr create`, or any
  push-option flags like `-o merge_request.create`.

## Contributing and PR readiness

- Before opening or marking a PR ready for review, read the
  [submitting your code](CONTRIBUTING.md#submitting-your-code) guidance.
- Read `.github/PULL_REQUEST_TEMPLATE.md` and satisfy the checklist.

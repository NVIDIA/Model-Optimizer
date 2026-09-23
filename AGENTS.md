# Agent Instructions for ModelOpt

These instructions apply to AI-assisted work in this repository.

## Repository orientation

- Start with `README.md` for project overview and install.
- Use `modelopt/` for source, `tests/` for focused test coverage, and
  `examples/` or `docs/` for usage patterns.
- **Agent skills live under `plugins/modelopt/skills/`**, the installable
  plugin's canonical skill tree. `.agents/skills` and `.claude/skills` expose
  those skills through relative symlinks. Shared agent config and scripts
  remain under `.agents/`. See `.agents/README.md` for the convention.

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

## Updating skills

- **Keep skill edits concise.** Skills are loaded into agent context, so every
  line costs tokens on each use. Add only what changes agent behavior, and
  prefer tightening existing text over appending new text.
- **Compress before opening the PR.** Make a final pass over the skill diff:
  cut redundancy, merge overlapping guidance, and drop examples the agent
  doesn't need.

## Sizing and splitting PRs

- **Keep each PR that goes up for review under ~500 changed lines of source.**
  Large PRs stall in review; exceed the budget only when the change genuinely
  cannot be split — a mechanical rename, generated files, or a self-contained
  drop such as a new example or a new model/backend that has no working
  intermediate state. Check the size with `git diff --stat <base>...HEAD`
  before opening.
- **Propose the split before opening an oversized PR.** When the work in flight
  is already over budget, offer a series of smaller PRs and, once the user
  agrees, do the split — don't open the big one and ask afterwards.
- **Split on structure first, features second.** Look for a file, directory, or
  module boundary that carves the change into independent pieces. Only when no
  such boundary exists, split by feature: land the enabling refactor or
  plumbing first, then one PR per behavior it unlocks.
- **Keep the series acyclic and linearly ordered.** A sub-PR may depend only on
  ones earlier in the series; if two pieces need each other, they belong in the
  same PR, or their shared part belongs in an earlier one. State the merge
  order in each description.
- **Prefix titles with `[x/N]`** — e.g. `[2/4] Add NVFP4 export path` — so
  reviewers know it is one slice of a planned split, and link the sibling PRs.
- **Every sub-PR stands on its own:** it builds, it carries unit tests for the
  code it introduces, and CI passes on it without the later PRs.
- **When a split makes the whole hard to follow, the full change may also go
  up as a draft PR** — for reference only, never as a second review request.
  Keep it in draft, say in its description that it is not for merge, link it
  from each sub-PR, and list the sub-PRs in it. Rebase or close it as the
  series lands.

## Contributing and PR readiness

- Before opening or marking a PR ready for review, read the
  [submitting your code](CONTRIBUTING.md#submitting-your-code) guidance.
- Read `.github/PULL_REQUEST_TEMPLATE.md` and satisfy the checklist.
- **PR description:** fill the template sections — what changed and why, a usage
  snippet if it adds an API or flag, and what you actually ran under Testing.
  Root cause, benchmark numbers, and design rationale belong here. Don't restate
  the diff file by file.
- **Only changelog-worthy changes get a `CHANGELOG.rst` entry:** new features,
  backward breaking changes, deprecations, and fixes for critical or known bugs
  from a previous release. Skip bugs introduced and fixed within the same
  unreleased cycle.
- **Keep each entry to one or two sentences** written for external users: what
  changed and what they need to do. No internal bug numbers (e.g. NVBug IDs),
  root-cause analysis, or implementation detail — that belongs in the PR
  description. File features under the matching `**New Features**` sub-section
  used by recent releases (e.g. `*Quantization*`, `*Speculative Decoding*`,
  `*Megatron Framework (M-LM / M-Bridge)*`, `*Misc*`) rather than relabeling
  existing ones.

## Responding to PR review feedback

- **Judge each comment on its merits before acting.** Check it against the
  current code — reviewers comment on stale diffs, and bot findings (CodeRabbit,
  Claude) are claims to verify, not instructions. Weight CODEOWNERS reviewers
  above bots; if a reviewer reaffirms after your pushback, that settles it.
- **Pick one outcome per thread:** address it in a commit, push back citing the
  code that shows the comment is wrong, or postpone it as out of scope. Report
  which threads got which when you ask for push approval.
- **Reply in every thread the pushed commits addressed** — a sentence on what
  changed and where. Those replies need no extra approval; pushback and postpone
  replies do, since no commit backs them. Never resolve threads: that is the
  reviewer's call.

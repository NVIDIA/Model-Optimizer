# Agent Tooling Notes

These notes are for humans maintaining repository agent setup. They are not part
of the always-loaded agent instructions.

## Shared Instructions

Update `.agents/README.md` for repository-wide agent instructions. The root
`AGENTS.md` and `CLAUDE.md` files are symlinked to `.agents/README.md`, so
changes there apply to both Codex and Claude Code.

## Local Overrides

For private local instructions, use the tool-specific override file:

- Claude Code: `CLAUDE.local.md` is additive; it is read after `CLAUDE.md`.
- Codex: `AGENTS.override.md` replaces `AGENTS.md` in the same directory, so it
  is not additive. Restate any shared instructions that should still apply.

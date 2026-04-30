# `.agents/` — agent-agnostic source of truth

This directory is the canonical location for assets shared by AI coding agents
working in this repository (Claude Code, Codex, Cursor, …).

## Layout

```text
.agents/
├── skills/                 # SKILL.md files (canonical)
│   └── <skill-name>/SKILL.md
├── scripts/                # shared helper scripts (sync-upstream-skills.sh, …)
└── clusters.yaml.example   # remote-cluster config template
```

## Why this exists

Different agents look for skills/config in vendor-specific directories:

| Agent       | Default location              |
|-------------|-------------------------------|
| Claude Code | `.claude/skills/`             |
| Codex       | `.codex/skills/`              |
| Cursor      | `.cursor/skills/`             |

Maintaining N copies of the same skill is a non-starter. Instead, **`.agents/`
is the single source of truth**, and each vendor directory is a symlink:

```text
.claude/skills              -> ../.agents/skills
.claude/scripts             -> ../.agents/scripts
.claude/clusters.yaml.example -> ../.agents/clusters.yaml.example
```

To add support for a new agent, create a directory with the symlinks that
agent expects, e.g.:

```bash
mkdir -p .codex
ln -s ../.agents/skills .codex/skills
git add .codex/skills
```

## Editing rules

- **Always edit files under `.agents/`**, never under the vendor symlink paths.
  Edits via the symlink work, but the diff will look like changes to
  `.agents/...` either way; editing the canonical path makes that explicit.
- Vendored-verbatim skills (`launching-evals`, `accessing-mlflow`) are managed
  by `.agents/scripts/sync-upstream-skills.sh` — do not modify by hand.
- New skills go in `.agents/skills/<skill-name>/SKILL.md` following the
  conventions of existing skills (e.g. `.agents/skills/monitor/SKILL.md`).

## Project-level cluster config

The remote-execution skills look for a `clusters.yaml` at, in order:

1. `~/.config/modelopt/clusters.yaml` (user-level, recommended)
2. `<repo-root>/.agents/clusters.yaml` (project-level, canonical)
3. `<repo-root>/.claude/clusters.yaml` (project-level, back-compat)

See `clusters.yaml.example` for the schema.

## A note on Windows

Git stores symlinks portably, but Windows requires either Developer Mode or
`git config --global core.symlinks true` plus admin rights for them to
materialise correctly. If you're on Windows and skills aren't being picked
up under `.claude/skills/`, that's the most likely cause — `.agents/skills/`
will still work directly.

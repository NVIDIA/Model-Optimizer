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

Different agents look for skills/config in vendor-specific directories. Rather
than maintaining N copies that drift out of sync, **`.agents/` is the single
source of truth** — each agent's guidance or install mechanism points here
directly.

## How each agent finds these

Each agent points at `.agents/` through whatever mechanism it supports — never
a copy:

- **Claude Code** only auto-discovers skills under `.claude/skills/`, so
  `.claude/` holds relative in-repo symlinks back into `.agents/`:
  `.claude/skills → ../.agents/skills`, `.claude/scripts → ../.agents/scripts`,
  and `.claude/clusters.yaml.example → ../.agents/clusters.yaml.example`. These
  follow the same committed-symlink pattern already used elsewhere in this repo
  (e.g. `CLAUDE.md`, `tools/launcher/modules/Model-Optimizer`).
- **Future agents** (Codex, Cursor, …) add their own symlink or config pointing
  at `.agents/`.

## Editing rules

- **Always edit files under `.agents/`**.
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

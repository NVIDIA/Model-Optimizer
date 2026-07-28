# `.agents/` — agent compatibility and shared config

This directory exposes the ModelOpt plugin skills to repository-local agents
and holds shared configuration.

## Layout

```text
.agents/
├── skills → ../plugins/modelopt/skills
├── plugins/
│   └── marketplace.json   # Codex marketplace
├── scripts/                # shared helper scripts (sync-upstream-skills.sh, …)
└── clusters.yaml.example   # remote-cluster config template

plugins/modelopt/
├── .claude-plugin/
├── .codex-plugin/
└── skills/                 # canonical SKILL.md files
    ├── common/             # shared skill support files
    └── <skill-name>/SKILL.md
```

## How each agent finds these

Each agent points at `.agents/` through whatever mechanism it supports — never
a copy:

- **Claude Code** only auto-discovers skills under `.claude/skills/`, so
  `.claude/skills/` holds relative symlinks into `.agents/skills/`.
- **Repository agents** use `.agents/skills`, a relative symlink into the
  plugin.
- **Claude Code and Codex plugins** load `plugins/modelopt/skills` directly.

## Editing rules

- **Always edit skills under `plugins/modelopt/skills/`**.
- Vendored-verbatim skills (`launching-evals`, `accessing-mlflow`) are managed
  by `.agents/scripts/sync-upstream-skills.sh` — do not modify by hand.
- New skills go in `plugins/modelopt/skills/<skill-name>/SKILL.md`.
- Shared support files go in `plugins/modelopt/skills/common/`.

## Project-level cluster config

The remote-execution skills look for a `clusters.yaml` at, in order:

1. `~/.config/modelopt/clusters.yaml` (user-level, recommended)
2. `<repo-root>/.agents/clusters.yaml` (project-level, canonical)
3. `<repo-root>/.claude/clusters.yaml` (project-level, back-compat)

See `clusters.yaml.example` for the schema.

# `.agents/` — agent compatibility and shared config

This directory exposes the ModelOpt plugin skills to repository-local agents
and holds shared configuration.

## Layout

```text
.agents/
├── skills → ../skills
├── plugins/
│   └── marketplace.json   # Codex marketplace
├── scripts/                # shared helper scripts (sync-upstream-skills.sh, …)
└── clusters.yaml.example   # remote-cluster config template

.codex/
└── agents/                 # Codex project role definitions

.claude-plugin/plugin.json  # root Claude plugin manifest
.codex-plugin/plugin.json   # root Codex plugin manifest
.claude-plugin/mcp.json     # Claude plugin MCP configuration
.codex-plugin/mcp.json      # Codex plugin MCP configuration
.claude/agents/             # canonical Claude subagent definitions

skills/                     # canonical skill content
├── common/                 # shared support
└── <skill-name>/SKILL.md    # catalog workflows use modelopt-<name>
```

## How each agent finds these

Repository discovery shares files through the supported agent paths:

- **Claude Code** only auto-discovers skills under `.claude/skills/`, so
  `.claude/skills/` holds relative symlinks into `.agents/skills/`.
- **Repository agents** use `.agents/skills`, a relative symlink to root `skills/`.
- **Both marketplaces** select the repository root as the plugin source.
  The root manifests load the real `skills/` directory, without installer symlinks.
- **Claude Code** loads the individual `.claude/agents/*.md` paths listed in
  the root plugin manifest. These canonical files also serve repository users.
- **MCP servers** load through each plugin manifest. In a plain Claude checkout,
  use `claude --mcp-config .claude-plugin/mcp.json` or load the plugin with
  `claude --plugin-dir .`; nested MCP files are not auto-discovered.
- **Codex** discovers project roles directly under `.codex/agents/`. Codex
  plugins cannot currently install custom roles.

## Editing rules

- **Always edit skills under root `skills/`**.
- Claude Code subagents belong in `.claude/agents/` as real files.
- Codex custom roles belong in `.codex/agents/`.
- Upstream-derived skills (`modelopt-launching-evals`, `modelopt-accessing-mlflow`) are managed
  by `.agents/scripts/sync-upstream-skills.sh` — do not modify by hand.
- New skills go in `skills/<skill-name>/SKILL.md`.
- Shared support files go in `skills/common/`.

## Review baseline

[Code review baseline](review-guidelines/code-review.md) is the shared rubric
for future reviews. Read the version at the PR's base revision and report that
commit. Domain-specific rubrics, such as the
[technical blog guideline](review-guidelines/technical-blog.md), add checks to it.

## Project-level cluster config

The remote-execution skills look for a `clusters.yaml` at, in order:

1. `~/.config/modelopt/clusters.yaml` (user-level, recommended)
2. `<repo-root>/.agents/clusters.yaml` (project-level, canonical)
3. `<repo-root>/.claude/clusters.yaml` (project-level, back-compat)

See `clusters.yaml.example` for the schema.

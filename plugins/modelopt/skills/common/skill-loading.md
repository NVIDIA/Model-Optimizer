# Loading ModelOpt source and companion skills

Use an existing ModelOpt checkout when available; otherwise clone
`https://github.com/NVIDIA/Model-Optimizer.git` into the task workspace. Set
`MODELOPT_ROOT` to its absolute path and record `git -C "$MODELOPT_ROOT" rev-parse HEAD`.
Do not pull, switch branches, or discard local patches as part of setup.

```bash
export MODELOPT_SKILLS="$MODELOPT_ROOT/plugins/modelopt/skills"
test -f "$MODELOPT_ROOT/examples/hf_ptq/hf_ptq.py"
test -f "$MODELOPT_SKILLS/common/SKILL.md"
```

Keep `SKILL_DIR` pointed at the skill currently being executed, for its bundled
scripts and references. Resolve shared files explicitly:

- Common documents and `remote_exec.sh`: `$MODELOPT_SKILLS/common/`.
- A companion workflow such as `evaluation`, `monitor`, or `deployment`:
  `$MODELOPT_SKILLS/<name>/SKILL.md`, if it is not installed separately.
- Cross-skill reference files: `$MODELOPT_SKILLS/<name>/references/`.
- Repository code, recipes, and examples: `$MODELOPT_ROOT/`.

Catalog entries use the source skill name; plugin invocations use
`modelopt:<name>`. Resolve the named ModelOpt workflow by its entrypoint; do not
assume that all companions are installed or infer their paths from neighboring
catalog directories. When executing a companion's scripts, use that companion's
directory as `SKILL_DIR` and restore the caller's directory afterward.

Read only the companion or common file needed for the current step. Loading a
reference does not authorize its execution. Artifact analysis and configuration
review need neither a GPU nor a running ModelOpt or MLflow MCP server. If an
optional integration is unavailable, use supplied local artifacts or ask for
the missing configuration rather than inventing endpoints or credentials.

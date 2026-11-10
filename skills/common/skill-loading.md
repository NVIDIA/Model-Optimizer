# Loading ModelOpt skills

Reuse an existing ModelOpt checkout; otherwise clone
`https://github.com/NVIDIA/Model-Optimizer.git` into the task workspace.
Set `MODELOPT_ROOT` to its absolute path and `MODELOPT_SKILLS="$MODELOPT_ROOT/skills"`.
Record the checkout revision; do not pull, switch branches, or discard patches.

Resolve shared files from `$MODELOPT_SKILLS/common/` and missing companions from
`$MODELOPT_SKILLS/modelopt-<name>/SKILL.md`. Load only what the current step needs.
Keep `SKILL_DIR` at the executing skill for its bundled scripts and references;
when running a companion, use its directory. Do not assume sibling catalog
skills are installed.

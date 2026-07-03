---
name: puzzletron
description: "End-to-end workflow for model pruning and MIP-based optimization. Commands: mip, all. Usage: /puzzletron <command>"
license: Apache-2.0
---

# Puzzletron

## Routing

**STEP 1: Check args before doing anything else. This is MANDATORY.**

- If args are **empty**, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**
- If the first word of args does **not exactly match** `mip` or `all`, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**

---

**Puzzletron**: end-to-end workflow for model pruning and MIP-based optimization.

Available commands:
- `mip <nproc_per_node>`: Run the MIP step (nproc_per_node: number of GPUs per node)
- `mip progress`: Show live MIP progress with timing summary
- `all <nproc_per_node>`: Run the full Puzzletron pipeline (nproc_per_node: number of GPUs per node)
- `all progress`: Show live full pipeline progress with timing summary

Usage: `/puzzletron <command> [args]`

Setup requirements:
- Real `mip` and `all` runs require an allocated GPU/container environment. Do not run them on a login node or in a plain local shell.
- If the coding agent is already running inside the approved allocated GPU/container environment, execute the Bash payload there directly. Otherwise send it through a user-provided execution method to an existing target. If neither a valid current target nor an execution method and target are available, ask the user for one and **STOP**.
- This skill does not create an allocation, submit a scheduler job, or prescribe where the coding agent runs. Allocation and scheduler submission are separate operations.
- Complete the container setup from `examples/puzzletron/README.md` before running this skill. The editable install makes `PYTHONPATH` changes unnecessary.
- Set `MODELOPT_REPO_ROOT` to the repository path inside the execution target, or start the payload inside the checkout so `git rev-parse --show-toplevel` can derive it.
- Optionally set `MODELOPT_CONFIG_PATH` to an absolute or repository-relative config path. The selected config must reference model, dataset, and output paths available inside the execution target.
- Use the container Python and container `torchrun` for GPU/container work. Do not create a repo-local virtual environment inside the container. CPU-only local virtual environments are only for progress scripts or documentation checks.

---

**STEP 2: Only if the first word of args exactly matches a command name, execute it. Never reach this step if args were empty.**

## Command: all

Parse `nproc_per_node` from args using either positional or flag syntax:
- Positional: second word is a number, e.g. `all 2`
- Flag: `--nproc_per_node <value>` anywhere in args, e.g. `all --nproc_per_node 2`

- If the second word is exactly `progress`, execute the **all progress** sub-command below.
- If no `nproc_per_node` value can be found, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- If the value does not match `^[1-9][0-9]*$`, ask the user: "nproc_per_node must be a positive integer." and **STOP**.
- Otherwise use the parsed value and run the full pipeline.

### all \<nproc_per_node\>

Run the following Bash payload inside the approved allocated GPU/container environment, substituting `<nproc_per_node>` with the parsed value. Execute it directly when the coding agent is already in that target; otherwise use the user-provided execution method. Do not execute it on a login node or in a plain local workstation shell.

```bash
set -o pipefail
repo_root="${MODELOPT_REPO_ROOT:-}"
if [ -z "$repo_root" ]; then
  repo_root="$(git rev-parse --show-toplevel)" || exit 2
fi
cd "$repo_root" || exit 2
config_path="${MODELOPT_CONFIG_PATH:-examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml}"
if [ ! -f "$config_path" ]; then
  printf 'Puzzletron config not found: %s\n' "$config_path" >&2
  exit 2
fi
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config "$config_path" \
  2>&1 | tee ./log.txt | grep "Puzzletron Progress"
```

Stream output to the user as it arrives. When the command finishes, report the exit code.

### all progress

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/all_progress.py
```

## Command: mip

Parse `nproc_per_node` from args using either positional or flag syntax:
- Positional: second word is a number, e.g. `mip 2`
- Flag: `--nproc_per_node <value>` anywhere in args, e.g. `mip --nproc_per_node 2`

- If the second word is exactly `progress`, execute the **mip progress** sub-command below.
- If no `nproc_per_node` value can be found, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- If the value does not match `^[1-9][0-9]*$`, ask the user: "nproc_per_node must be a positive integer." and **STOP**.
- Otherwise use the parsed value and run the MIP step.

### mip \<nproc_per_node\>

Run the following Bash payload inside the approved allocated GPU/container environment, substituting `<nproc_per_node>` with the parsed value. Execute it directly when the coding agent is already in that target; otherwise use the user-provided execution method. Do not execute it on a login node or in a plain local workstation shell.

```bash
set -o pipefail
repo_root="${MODELOPT_REPO_ROOT:-}"
if [ -z "$repo_root" ]; then
  repo_root="$(git rev-parse --show-toplevel)" || exit 2
fi
cd "$repo_root" || exit 2
config_path="${MODELOPT_CONFIG_PATH:-examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml}"
if [ ! -f "$config_path" ]; then
  printf 'Puzzletron config not found: %s\n' "$config_path" >&2
  exit 2
fi
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config "$config_path" \
  --mip-only 2>&1 | tee ./log.txt | grep "Puzzletron Progress"
```

Stream output to the user as it arrives. When the command finishes, report the exit code.

### mip progress

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/mip_progress.py
```

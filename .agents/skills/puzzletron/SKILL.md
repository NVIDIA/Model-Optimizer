---
name: puzzletron
description: "End-to-end workflow for model pruning and MIP-based optimization. Commands: mip, all. Usage: /puzzletron <command>"
license: Apache-2.0
---

# Puzzletron

## Routing

**STEP 1 — Check args before doing anything else. This is MANDATORY.**

- If args are **empty**, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**
- If the first word of args does **not exactly match** `mip` or `all`, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**

---

**Puzzletron** — end-to-end workflow for model pruning and MIP-based optimization.

Available commands:
- `mip <nproc_per_node>` — Run the MIP step (nproc_per_node: number of GPUs per node)
- `mip progress` — Show live MIP progress with timing summary
- `all <nproc_per_node>` — Run the full Puzzletron pipeline (nproc_per_node: number of GPUs per node)
- `all progress` — Show live full pipeline progress with timing summary

Usage: `/puzzletron <command> [args]`

---

**STEP 2 — Only if the first word of args exactly matches a command name, execute it. Never reach this step if args were empty.**

## Command: all

Parse `nproc_per_node` from args using either positional or flag syntax:
- Positional: second word is a number, e.g. `all 2`
- Flag: `--nproc_per_node <value>` anywhere in args, e.g. `all --nproc_per_node 2`

- If the second word is exactly `progress`, execute the **all progress** sub-command below.
- If no `nproc_per_node` value can be found, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- If the value does not match `^[0-9]+$`, ask the user: "nproc_per_node must be a positive integer." and **STOP**.
- Otherwise use the parsed value and run the full pipeline.

### all \<nproc_per_node\>

Run the following Bash command, substituting `<nproc_per_node>` with the parsed value:

```bash
set -o pipefail && export PYTHONPATH=$PYTHONPATH:/workspace/Model-Optimizer && \
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml \
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
- If the value does not match `^[0-9]+$`, ask the user: "nproc_per_node must be a positive integer." and **STOP**.
- Otherwise use the parsed value and run the MIP step.

### mip \<nproc_per_node\>

Run the following Bash command, substituting `<nproc_per_node>` with the parsed value:

```bash
set -o pipefail && export PYTHONPATH=$PYTHONPATH:/workspace/Model-Optimizer && \
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml \
  --mip-only 2>&1 | tee ./log.txt | grep "Puzzletron Progress"
```

Stream output to the user as it arrives. When the command finishes, report the exit code.

### mip progress

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/mip_progress.py
```

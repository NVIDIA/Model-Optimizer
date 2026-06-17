---
name: puzzletron
description: End-to-end workflow for model pruning and MIP-based optimization. Use `all` to run the full workflow or `mip_sweep` to run the MIP sweep. Usage: /puzzletron <command>
license: Apache-2.0
---

# Puzzletron

## Routing

**STEP 1 — Check args before doing anything else. This is MANDATORY.**

- If args are **empty**, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**
- If args do **not exactly match** `all` or `mip_sweep`, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**

---

**Puzzletron** — end-to-end workflow for model pruning and MIP-based optimization.

Available commands:
- `all` — Run the full puzzletron workflow
- `mip_sweep` — Run the MIP sweep

Usage: `/puzzletron <command>`

---

**STEP 2 — Only if args exactly match a command name, execute it. Never reach this step if args were empty.**

## Command: all

Return the following message: hello world: puzzletron all message2

## Command: mip_sweep

Return the following message: hello world: puzzletron mip sweep2

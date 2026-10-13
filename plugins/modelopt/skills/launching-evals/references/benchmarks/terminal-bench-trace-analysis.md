# Terminal Bench: Agent Trace Analysis

Analyze agent traces from a terminal-bench evaluation run.

```
Trace analysis progress:
- [ ] Step 1: Locate artifacts
- [ ] Step 2: Analyze each task
- [ ] Step 3: Produce summary table
- [ ] Step 4: Episode-level deep dive (optional)
```

## Step 1: Locate artifacts

- Agent logs: `artifacts/terminal-bench/agent_logs/default/tasks.jsonl`
- Per-task artifacts: `artifacts/terminal-bench/{task_name}/{trial_name}/`
  - `results.json` - test results and metadata
  - `panes/post-agent.txt` - terminal state after agent finished
  - `panes/post-test.txt` - terminal state after tests ran

## Step 2: Analyze each task

For each task, extract:

**Metadata:** Task ID, status (success/failure/error), duration (convert to hours and minutes), token usage (input/output), test results breakdown (passed/failed counts).

**Agent behavior:**
1. **Approach:** What strategy did the agent use? (read-then-write, iterative debugging, single-shot, etc.)
2. **Key Commands:** Summarize the critical shell commands executed from post-agent.txt
3. **Reasoning Quality:** Was the plan coherent? Did it address the task requirements?

**For successful tasks:** What made the approach work? Was it efficient or did it take unnecessary steps? Key success factors (domain knowledge, clean implementation, etc.)

**For failed tasks:**
- **Failure Mode:** Categorize as: environment/setup issues, algorithm/logic errors, timeout/resource limits, or task misunderstanding
- **Stuck Loops:** Did the agent repeat failed attempts without adapting?
- **Root Cause:** Single-sentence summary of why it failed
- **Missed Opportunities:** What should the agent have done differently?

Present each task using this format:

```
## [Task Name]

**Status:** PASS/FAIL (X/Y tests) | **Duration:** Hh MMmin | **Tokens:** Xk in / Xk out
**Task:** One-sentence description of what the task required
**Agent Approach:**
1. Step 1
2. Step 2
...

**[For failures only] Why It Failed:**
* Bullet points with specific errors/issues from the logs

**[For successes] Key Success Factors / [For failures] Root Cause:**
* Summary
```

## Step 3: Produce summary table

| Task | Status | Duration | Tokens | Failure Mode |
|------|--------|----------|--------|--------------|
| ... | PASS/FAIL | Hh MMmin | Xk | - or category |

## Step 4: Episode-level deep dive (optional)

When you need to trace exactly where the agent went wrong, check:

`artifacts/terminal-bench/{task}/*/agent-logs/episode-N/`

- `response.txt` - Agent's explicit reasoning (`analysis`, `plan` fields)
- `prompt.txt` - What terminal state the agent saw before acting

Use cases: identify the specific episode where the agent made a wrong decision, check if the plan was reasonable but execution failed, debug loops where the agent repeated the same failing approach.

Skip this for general pass/fail summaries and performance comparisons.

## Examples

### Successful: cross-entropy-method

```
Status: PASSED (22/22 tests) | Duration: 28 min | Tokens: 38.5k in / 3.3k out
Task: Implement three core RL methods: PointEnv.step(), CrossEntropyMethod.optimize(), and evaluate_plans_memoized() with caching.
Agent Approach:
1. Examined existing code structure (ls -la, cat cross_entropy.py)
2. Wrote complete implementations using a heredoc (cat > cross_entropy.py << 'EOF')
3. Implemented step() with position clipping and goal distance check
4. Implemented memoization with prefix caching (tuple keys for hashability)
5. Implemented cross-entropy optimization with elite selection
6. Ran tests which all passed

Key Success Factors:
* Clear algorithmic understanding (cross-entropy method, memoization)
* Clean single-shot implementation without debugging loops
* Proper numpy handling (clipping, distance calculation)
```

### Successful: oom (cache HuggingFace model)

```
Status: PASSED (1/1 test) | Duration: 9 min | Tokens: 4k in / 432 out
Task: Cache the albert/albert-base-v2 model for offline use.
Agent Approach:
1. Ran huggingface-cli download albert/albert-base-v2
2. Downloaded 12 files (~270MB total)
3. Task complete - straightforward execution

Key Success Factors:
* Simple task with direct solution
* Minimal steps required (single command)
```

### Failed: lean4-proof

```
Status: FAILED (6/11 tests) | Duration: 3h 20min | Tokens: 521k in / 15.8k out
Task: Install Lean v4.21.0 with Mathlib and complete 3 formal proofs.
Agent Failures:
1. Version Mismatch Hell: Agent tried to pin Mathlib to lean-4.21 but branch doesn't exist
2. Toolchain Override: Mathlib auto-updated to v4.27.0-rc1, breaking v4.21.0 requirement
3. Incompatible Linter Options: error: Unknown option `linter.unusedTactic`
4. Git Authentication Failures: Multiple failed clones requiring auth
5. Import Ordering Errors: set_option inserted before import

Root Cause: Agent couldn't reconcile Lean v4.21.0 requirement with Mathlib4's latest versions. Spent 3+ hours in a loop trying various revision formats without success.
```

### Failed: feal-differential-cryptanalysis

```
Status: FAILED (0/1 test) | Duration: 42 min | Tokens: 7.4k in / 1.6k out
Task: Implement a differential cryptanalysis attack to recover key[5] from a FEAL-like cipher.
Agent Approach:
1. Read feal.py to understand the cipher (4-round Feistel)
2. Wrote attack.py with basic differential attack logic

Why It Failed:
* The differential characteristic chosen was likely incorrect for this FEAL variant
* Attack logic assumed key[5] maps to round 4 key directly, but the cipher uses key[round+2] indexing
* No iterative refinement or multi-round differential propagation analysis
* Agent stopped after single implementation attempt without testing/debugging

Root Cause: Cryptanalysis tasks require precise differential trail analysis. The agent's heuristic approach didn't account for the specific F-function and key schedule.
```

# Skill Benchmark: modelopt-evaluation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `modelopt-evaluation`
- Validation status: `passed`
- Evaluation date: 2026-11-24
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 18 evaluation tasks (18 positive)
- Dataset digest: `sha256:416b0a4e352e772ba16af5e2e790b3004c32f68dd84d263ca9910cefe7cebcba` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| modelopt-evaluation | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-evaluation | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-evaluation | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-evaluation | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-evaluation | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `modelopt-evaluation`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.8% — baseline ran, but no comparable score was available; uplift unavailable | 89.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 97.2% → 100.0% (+2.8 points) |
| Correctness | 54.4% → 91.1% (+36.7 points) | 60.0% → 96.7% (+36.7 points) |
| Discoverability | 96.7% — baseline ran, but no comparable score was available; uplift unavailable | 93.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 25.7% → 70.8% (+45.1 points) | 28.1% → 69.1% (+41.0 points) |
| Efficiency | 85.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 92.4% (-5.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.4% → 88.5% (+2.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,162,349 | 1,518,627 | +1,643,722 | +108.24% | skill 18/18; base 18/18 |
| claude-code | bounded-failures-canary-blockers | 159,071 | 108,011 | +51,060 | +47.27% | skill 1/1; base 1/1 |
| claude-code | bounded-failures-overlap-and-resume | 165,455 | 36,877 | +128,578 | +348.67% | skill 1/1; base 1/1 |
| claude-code | bounded-failures-pre-full-run-canaries | 159,910 | 108,243 | +51,667 | +47.73% | skill 1/1; base 1/1 |
| claude-code | bounded-failures-scored-zeros-versus-missing | 119,835 | 36,639 | +83,196 | +227.07% | skill 1/1; base 1/1 |
| claude-code | bounded-failures-strict-boundary | 119,918 | 108,445 | +11,473 | +10.58% | skill 1/1; base 1/1 |
| claude-code | doom-loop-repetition-stop | 158,151 | 108,671 | +49,480 | +45.53% | skill 1/1; base 1/1 |
| claude-code | existing-config-serving-handoff | 160,141 | 71,910 | +88,231 | +122.70% | skill 1/1; base 1/1 |
| claude-code | explicit-external-model-target | 201,388 | 36,368 | +165,020 | +453.75% | skill 1/1; base 1/1 |
| claude-code | livecodebench-judge-pin | 204,350 | 107,773 | +96,577 | +89.61% | skill 1/1; base 1/1 |
| claude-code | nemotron3-nano-bf16-reasoning | 283,966 | 143,619 | +140,347 | +97.72% | skill 1/1; base 1/1 |
| claude-code | plan-fp8-mmlu-pro-accuracy | 241,485 | 37,111 | +204,374 | +550.71% | skill 1/1; base 1/1 |
| claude-code | quantized-checkpoint-local-vllm | 214,803 | 36,825 | +177,978 | +483.31% | skill 1/1; base 1/1 |
| claude-code | retained-smoke-server-is-not-eval-authorization | 119,163 | 36,386 | +82,777 | +227.50% | skill 1/1; base 1/1 |
| claude-code | scicode-eight-run-average | 161,803 | 107,821 | +53,982 | +50.07% | skill 1/1; base 1/1 |
| claude-code | slurm-quantized-model | 309,639 | 143,159 | +166,480 | +116.29% | skill 1/1; base 1/1 |
| claude-code | swebench-output-policy-precedence | 186,453 | 145,251 | +41,202 | +28.37% | skill 1/1; base 1/1 |
| claude-code | tb21-health-before-score | 122,616 | 36,382 | +86,234 | +237.02% | skill 1/1; base 1/1 |
| claude-code | terminalbench-output-policy-precedence | 74,202 | 109,136 | -34,934 | -32.01% | skill 1/1; base 1/1 |
| codex | All cases | 1,659,661 | 1,010,355 | +649,306 | +64.27% | skill 18/18; base 18/18 |
| codex | bounded-failures-canary-blockers | 48,093 | 55,448 | -7,355 | -13.26% | skill 1/1; base 1/1 |
| codex | bounded-failures-overlap-and-resume | 48,420 | 14,195 | +34,225 | +241.11% | skill 1/1; base 1/1 |
| codex | bounded-failures-pre-full-run-canaries | 67,463 | 70,572 | -3,109 | -4.41% | skill 1/1; base 1/1 |
| codex | bounded-failures-scored-zeros-versus-missing | 86,733 | 55,255 | +31,478 | +56.97% | skill 1/1; base 1/1 |
| codex | bounded-failures-strict-boundary | 67,189 | 14,036 | +53,153 | +378.69% | skill 1/1; base 1/1 |
| codex | doom-loop-repetition-stop | 100,700 | 97,898 | +2,802 | +2.86% | skill 1/1; base 1/1 |
| codex | existing-config-serving-handoff | 86,624 | 18,110 | +68,514 | +378.32% | skill 1/1; base 1/1 |
| codex | explicit-external-model-target | 120,320 | 18,212 | +102,108 | +560.66% | skill 1/1; base 1/1 |
| codex | livecodebench-judge-pin | 87,438 | 122,165 | -34,727 | -28.43% | skill 1/1; base 1/1 |
| codex | nemotron3-nano-bf16-reasoning | 95,134 | 60,261 | +34,873 | +57.87% | skill 1/1; base 1/1 |
| codex | plan-fp8-mmlu-pro-accuracy | 95,922 | 34,736 | +61,186 | +176.15% | skill 1/1; base 1/1 |
| codex | quantized-checkpoint-local-vllm | 93,876 | 30,632 | +63,244 | +206.46% | skill 1/1; base 1/1 |
| codex | retained-smoke-server-is-not-eval-authorization | 123,725 | 22,554 | +101,171 | +448.57% | skill 1/1; base 1/1 |
| codex | scicode-eight-run-average | 89,532 | 44,229 | +45,303 | +102.43% | skill 1/1; base 1/1 |
| codex | slurm-quantized-model | 124,428 | 91,177 | +33,251 | +36.47% | skill 1/1; base 1/1 |
| codex | swebench-output-policy-precedence | 140,750 | 152,104 | -11,354 | -7.46% | skill 1/1; base 1/1 |
| codex | tb21-health-before-score | 85,983 | 13,531 | +72,452 | +535.45% | skill 1/1; base 1/1 |
| codex | terminalbench-output-policy-precedence | 97,331 | 95,240 | +2,091 | +2.20% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,822,010 | 2,528,982 | +2,293,028 | +90.67% | skill 36/36; base 36/36 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 54 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across recipes/tasks/aa/gpqa_diamond.md and recipes/tasks/aa/hle.md and recipes/tasks/aa/ifbench.md and recipes/tasks/aa/lcr.md and recipes/tasks/aa/mmmu_pro.md and recipes/tasks/aa/omniscience.md and recipes/tasks/aa/scicode.md and recipes/tasks/aa/tau2_bench_telecom.md and recipes/tasks/aime_2025.md and recipes/tasks/livecodebench.md and recipes/tasks/mmlu_pro.md:
  "## Task Details" in recipes/tasks/aa/gpqa_diamond.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/aa/hle.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/aa/ifbench.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/aa/lcr.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/aa/mmmu_pro.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/aa/omniscience.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/aa/scicode.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/aa/tau2_bench_telecom.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/aime_2025.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/livecodebench.md (lines 3-6)
  vs "## Task Details" in recipes/tasks/mmlu_pro.md (lines 3-6) (`recipes/tasks/aa/gpqa_diamond.md:3`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across recipes/tasks/aa/hle.md and recipes/tasks/aa/lcr.md and recipes/tasks/aa/omniscience.md and recipes/tasks/aa/tau2_bench_telecom.md and recipes/tasks/aime_2025.md:
  "## Params" in recipes/tasks/aa/hle.md (lines 7-22)
  vs "## Params" in recipes/tasks/aa/lcr.md (lines 7-31)
  vs "## Params" in recipes/tasks/aa/omniscience.md (lines 7-18)
  vs "## Params" in recipes/tasks/aa/tau2_bench_telecom.md (lines 7-27)
  vs "## Params" in recipes/tasks/aime_2025.md (lines 7-18) (`recipes/tasks/aa/hle.md:7`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across recipes/tasks/aa_next/swebench_verified.md and recipes/tasks/aa_next/terminal_bench_2_1.md:
  "### Sharding" in recipes/tasks/aa_next/swebench_verified.md (lines 133-140)
  vs "## Task-specific values" in recipes/tasks/aa_next/terminal_bench_2_1.md (lines 75-80) (`recipes/tasks/aa_next/swebench_verified.md:133`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across recipes/tasks/gym/mrcr.md and references/gym.md:
  "## Canary" in recipes/tasks/gym/mrcr.md (lines 164-191)
  vs "## Failure modes to check at canary" in references/gym.md (lines 166-185) (`recipes/tasks/gym/mrcr.md:164`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/modelopt-evaluation/SKILL.md`)
- 53 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.

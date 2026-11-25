# Skill Benchmark: modelopt-day0-release

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `modelopt-day0-release`
- Validation status: `passed`
- Evaluation date: 2026-11-18
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 9 evaluation tasks (7 positive, 2 negative)
- Dataset digest: `sha256:de5befa070b07ebde59acd26a41d4369f15d99b63c0d8cd672f39261d1593040` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| modelopt-day0-release | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-day0-release | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-day0-release | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-day0-release | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-day0-release | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `modelopt-day0-release`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 88.6% — baseline ran, but no comparable score was available; uplift unavailable | 89.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 83.3% → 83.3% (±0.0 points) | 66.7% → 83.3% (+16.6 points) |
| Correctness | 62.2% → 97.8% (+35.6 points) | 60.0% → 91.1% (+31.1 points) |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 62.7% → 84.0% (+21.3 points) | 53.0% → 89.1% (+36.1 points) |
| Efficiency | 92.8% — baseline ran, but no comparable score was available; uplift unavailable | 90.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.4% → 95.4% (-3.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.9% → 90.6% (-0.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,217,736 | 972,734 | +245,002 | +25.19% | skill 9/9; base 9/9 |
| claude-code | evaluate-only-does-not-trigger-day0 | 145,520 | 143,834 | +1,686 | +1.17% | skill 1/1; base 1/1 |
| claude-code | exploratory-stopping-and-selection | 79,624 | 36,153 | +43,471 | +120.24% | skill 1/1; base 1/1 |
| claude-code | full-day0-release-triggers | 166,625 | 144,198 | +22,427 | +15.55% | skill 1/1; base 1/1 |
| claude-code | ptq-gate-blocks-on-coverage-miss | 222,546 | 218,004 | +4,542 | +2.08% | skill 1/1; base 1/1 |
| claude-code | ptq-smoke-test-cleanup | 214,100 | 180,037 | +34,063 | +18.92% | skill 1/1; base 1/1 |
| claude-code | quantize-only-does-not-trigger-day0 | 108,462 | 71,347 | +37,115 | +52.02% | skill 1/1; base 1/1 |
| claude-code | regression-reports-and-stops-in-v1 | 165,180 | 35,670 | +129,510 | +363.08% | skill 1/1; base 1/1 |
| claude-code | unresolved-uncertainty-blocks-release | 36,644 | 36,221 | +423 | +1.17% | skill 1/1; base 1/1 |
| claude-code | within-margin-needs-verbosity | 79,035 | 107,270 | -28,235 | -26.32% | skill 1/1; base 1/1 |
| codex | All cases | 798,089 | 555,582 | +242,507 | +43.65% | skill 9/9; base 9/9 |
| codex | evaluate-only-does-not-trigger-day0 | 157,698 | 254,276 | -96,578 | -37.98% | skill 1/1; base 1/1 |
| codex | exploratory-stopping-and-selection | 68,975 | 21,707 | +47,268 | +217.75% | skill 1/1; base 1/1 |
| codex | full-day0-release-triggers | 70,185 | 46,241 | +23,944 | +51.78% | skill 1/1; base 1/1 |
| codex | ptq-gate-blocks-on-coverage-miss | 70,432 | 59,599 | +10,833 | +18.18% | skill 1/1; base 1/1 |
| codex | ptq-smoke-test-cleanup | 190,639 | 74,295 | +116,344 | +156.60% | skill 1/1; base 1/1 |
| codex | quantize-only-does-not-trigger-day0 | 60,871 | 41,320 | +19,551 | +47.32% | skill 1/1; base 1/1 |
| codex | regression-reports-and-stops-in-v1 | 65,949 | 13,708 | +52,241 | +381.10% | skill 1/1; base 1/1 |
| codex | unresolved-uncertainty-blocks-release | 65,610 | 13,620 | +51,990 | +381.72% | skill 1/1; base 1/1 |
| codex | within-margin-needs-verbosity | 47,730 | 30,816 | +16,914 | +54.89% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,015,825 | 1,528,316 | +487,509 | +31.90% | skill 18/18; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 28 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within tests/test_gates.py:
  "test_ptq_waiver_requires_canonical_values()" in tests/test_gates.py (lines 415-421)
  vs "test_ptq_waiver_rejects_non_canonical_values()" in tests/test_gates.py (lines 745-750) (`tests/test_gates.py:415`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/modelopt-day0-release/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/modelopt-day0-release/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/modelopt-day0-release/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/modelopt-day0-release/SKILL.md`)
- 24 additional finding(s) are available in the full evaluation artifacts.

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

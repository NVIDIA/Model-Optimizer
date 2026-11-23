# Skill Benchmark: modelopt-deployment

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `modelopt-deployment`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 3 evaluation tasks (3 positive)
- Dataset digest: `sha256:2ef2f841f88d11b3ed986bc581b49963d23a5b223387e3d5845bf8ee22c94bcb` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| modelopt-deployment | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-deployment | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-deployment | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-deployment | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| modelopt-deployment | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `modelopt-deployment`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 97.1% — baseline ran, but no comparable score was available; uplift unavailable | 90.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 93.3% → 100.0% (+6.7 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 80.0% → 95.8% (+15.8 points) | 67.5% → 87.5% (+20.0 points) |
| Efficiency | 89.4% — baseline ran, but no comparable score was available; uplift unavailable | 69.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 95.5% (-2.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.2% → 87.9% (+0.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 276,753 | 111,461 | +165,292 | +148.30% | skill 3/3; base 3/3 |
| claude-code | remote-slurm-deployment | 120,206 | 37,460 | +82,746 | +220.89% | skill 1/1; base 1/1 |
| claude-code | unquantized-hf-model | 78,235 | 36,951 | +41,284 | +111.73% | skill 1/1; base 1/1 |
| claude-code | vllm-fp8-local | 78,312 | 37,050 | +41,262 | +111.37% | skill 1/1; base 1/1 |
| codex | All cases | 172,728 | 88,554 | +84,174 | +95.05% | skill 3/3; base 3/3 |
| codex | remote-slurm-deployment | 57,605 | 34,893 | +22,712 | +65.09% | skill 1/1; base 1/1 |
| codex | unquantized-hf-model | 57,517 | 27,016 | +30,501 | +112.90% | skill 1/1; base 1/1 |
| codex | vllm-fp8-local | 57,606 | 26,645 | +30,961 | +116.20% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 449,481 | 200,015 | +249,466 | +124.72% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 43 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/setup.md and references/sglang.md and references/trtllm.md and references/vllm.md:
  "#### TRT-LLM (direct)" in SKILL.md (lines 161-168)
  vs "### TRT-LLM" in references/setup.md (lines 21-36)
  vs "### As Python API" in references/sglang.md (lines 45-50)
  vs "### From HuggingFace Hub" in references/sglang.md (lines 60-68)
  vs "## Requirements" in references/trtllm.md (lines 3-8)
  vs "### Python API" in references/trtllm.md (lines 11-16)
  vs "### From HuggingFace Hub" in references/trtllm.md (lines 26-34)
  vs "### As Python API" in references/vllm.md (lines 25-30)
  vs "### From HuggingFace Hub" in references/vllm.md (lines 40-48) (`SKILL.md:161`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/sglang.md and references/vllm.md and scripts/deploy.sh:
  "# Serve as OpenAI-compatible endpoint" in SKILL.md (lines 112-138)
  vs "#### SGLang" in SKILL.md (lines 139-160)
  vs "### As OpenAI-compatible server" in references/sglang.md (lines 33-44)
  vs "### As OpenAI-compatible server" in references/vllm.md (lines 12-24)
  vs "start_sglang()" in scripts/deploy.sh (lines 306-331) (`SKILL.md:112`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/modelopt-deployment/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/modelopt-deployment/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/modelopt-deployment/SKILL.md`)
- 40 additional finding(s) are available in the full evaluation artifacts.

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

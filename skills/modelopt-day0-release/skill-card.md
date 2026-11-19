## Description: <br>
Deterministic end-to-end driver for day-0 quantized-checkpoint releases — chains PTQ → evaluation → comparison with enforced gates between stages (the evaluation stage deploys the checkpoint itself), and returns a publish decision (ACCEPT / REGRESSION / INDETERMINATE / ANOMALOUS / INFEASIBLE). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers releasing quantized model checkpoints with NVIDIA Model Optimizer use this skill to drive a pretrained model through PTQ, serving canary, baseline and quantized evaluation, accuracy and verbosity comparison gates, and obtain a publish / do-not-publish recommendation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [cluster access and service credentials per the common skill's credentials.md]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NVIDIA Model Optimizer documentation](https://nvidia.github.io/Model-Optimizer) <br>
- [Compare-results assessment](../modelopt-compare-results/references/assess-results.md) <br>
- [Evaluation run validation](../modelopt-evaluation/references/run-validation.md) <br>
- [PTQ checkpoint validation](../modelopt-ptq/references/checkpoint-validation.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Files] <br>
**Output Format:** [Markdown report with JSON gate results and inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Returns a decision (ACCEPT / REGRESSION / INDETERMINATE / ANOMALOUS / INFEASIBLE) with workspace path and MLflow run IDs for traceability.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
9 evaluation tasks (7 positive, 2 negative); 1 attempt per task, each in its own isolated sandbox pod (`k8s-sandbox`), compared with and without the skill. Dataset digest `sha256:de5befa070b07ebde59acd26a41d4369f15d99b63c0d8cd672f39261d1593040` (skill-evaluator-dataset-snapshot/1); evaluator version `1.6.10`; evaluation date 2026-11-18. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

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

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.4% → 95.4% (-3.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.9% → 90.6% (-0.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 28 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Skill Version(s): <br>
8e04597a (source: git SHA, committed 2026-11-18) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>

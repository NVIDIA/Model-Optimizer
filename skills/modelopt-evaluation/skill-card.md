# Skill Card: evaluation

## Description

Configure, run and validate model accuracy evaluations with NeMo Evaluator.

Publication preparation only: live agent evaluation and NVIDIA signing are pending.

## Owner

Proposed owner: NVIDIA Model Optimizer maintainers. Team/contact and release
clearance require owning-team confirmation before publication.
Contact: <https://github.com/NVIDIA/Model-Optimizer/issues>

## License/Terms of Use

Apache-2.0, as declared by the source repository. Dataset and checkpoint licenses
remain the user's responsibility; this card does not certify release clearance.

## Use Case

Model developers using the `modelopt-evaluation` workflow in a ModelOpt plugin or a standalone
catalog installation. Preserve the user's requested scope and execution limits.

## Deployment Geography for Use

Uses the user's selected compute and endpoint locations; no hosted service or
additional geography entitlement is provided by this skill.

## Requirements / Dependencies

A supported evaluator runtime and target endpoint or deployment resources; selected tasks may need gated datasets, judges or user-provisioned cloud sandboxes.
Common files and missing companion skills resolve from a ModelOpt checkout.
Credentials are optional for artifact-only work and task-dependent for execution:
Hugging Face tokens, remote/registry access, or configured API/cloud credentials.
Do not place secret values in prompts, tracked files or reports.

## Known Risks and Mitigations

GPU/cloud cost and data sent to configured model/judge endpoints; dry-run and canary gates, secret-safe configuration, and run validation precede score claims.

## References

- [Workflow](SKILL.md)
- [Evaluation cases](evals/evals.json)
- <https://github.com/NVIDIA/Model-Optimizer>

## Skill Output

Evaluation configs, logs, per-sample outputs, metrics and a validation verdict.

## Skill Version

Use the source directory's Git revision. Evaluation reports and the signing
identifier must refer to that exact content; neither exists for this draft yet.

## Ethical Considerations

Use checkpoints, datasets, model outputs and external services under their
applicable terms; validation evidence does not establish suitability for every
application or replace the owning team's release review.

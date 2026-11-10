# Skill Card: modelopt-launching-evals

## Description

Run, resume, inspect, and debug NEL evaluations.
Draft: live evaluation and NVIDIA signing are pending.

## Owner

Proposed owner: NVIDIA Model Optimizer maintainers.
Contact: <https://github.com/NVIDIA/Model-Optimizer/issues>

## License/Terms of Use

Apache-2.0. Model and dataset terms remain the user's responsibility;
publication clearance requires owning-team confirmation.

## Use Case

ModelOpt plugin and independent catalog installations.

## Deployment Geography for Use

User-selected compute and service locations.

## Requirements / Dependencies

NeMo Evaluator Launcher and user-provided compute, credentials, or local run artifacts.
Shared helpers and missing companions resolve from a ModelOpt checkout.

## Known Risks and Mitigations

Evaluation consumes compute and may submit jobs. Preserve requested execution limits and validate completed runs before reporting scores.

## References

- [Workflow](SKILL.md)
- [Evaluation cases](evals/evals.json)

## Skill Output

Invocation IDs, logs, artifacts, and validated evaluation findings.

## Skill Version

Use the exact source Git revision for evaluation and signing.

## Ethical Considerations

Use models, data, and services under their applicable terms.

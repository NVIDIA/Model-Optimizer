# Skill Card: modelopt-accessing-mlflow

## Description

Find evaluation runs, configs, metrics, and artifacts in MLflow.
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

An accessible MLflow service with configured MCP integration, or supplied exported artifacts for offline review.
Shared helpers and missing companions resolve from a ModelOpt checkout.

## Known Risks and Mitigations

Service credentials and evaluation artifacts may be sensitive. Never expose secrets or fabricate missing results.

## References

- [Workflow](SKILL.md)
- [Evaluation cases](evals/evals.json)

## Skill Output

Run metadata, artifact paths, and provenance.

## Skill Version

Use the exact source Git revision for evaluation and signing.

## Ethical Considerations

Use models, data, and services under their applicable terms.

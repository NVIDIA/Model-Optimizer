# Skill Card: modelopt-compare-results

## Description

Compare validated baseline and candidate accuracy results.

Publication preparation only: live agent evaluation and NVIDIA signing are pending.

## Owner

Proposed owner: NVIDIA Model Optimizer maintainers. Team/contact and release
clearance require owning-team confirmation before publication.
Contact: <https://github.com/NVIDIA/Model-Optimizer/issues>

## License/Terms of Use

Apache-2.0, as declared by the source repository. Dataset and checkpoint licenses
remain the user's responsibility; this card does not certify release clearance.

## Use Case

Model developers using the `modelopt-compare-results` workflow in a ModelOpt plugin or a standalone
catalog installation. Preserve the user's requested scope and execution limits.

## Deployment Geography for Use

Uses the user's selected compute and endpoint locations; no hosted service or
additional geography entitlement is provided by this skill.

## Requirements / Dependencies

ModelOpt task recipes and validation guidance; local artifacts or optional user-configured MLflow access.
Common files and missing companion skills resolve from a ModelOpt checkout.
Credentials are optional for artifact-only work and task-dependent for execution:
Hugging Face tokens, remote/registry access, or configured API/cloud credentials.
Do not place secret values in prompts, tracked files or reports.

## Known Risks and Mitigations

Invalid or mismatched runs can produce misleading acceptance decisions; check protocol, sample accounting, external baseline sanity and uncertainty first.

## References

- [Workflow](SKILL.md)
- [Evaluation cases](evals/evals.json)
- <https://github.com/NVIDIA/Model-Optimizer>

## Skill Output

Per-task deltas, uncertainty, protocol comparability and quantization-feasibility verdicts.

## Skill Version

Use the source directory's Git revision. Evaluation reports and the signing
identifier must refer to that exact content; neither exists for this draft yet.

## Ethical Considerations

Use checkpoints, datasets, model outputs and external services under their
applicable terms; validation evidence does not establish suitability for every
application or replace the owning team's release review.

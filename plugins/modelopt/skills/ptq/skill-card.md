# Skill Card: ptq

## Description

Calibrate and export a ModelOpt-quantized checkpoint.

Publication preparation only: live agent evaluation and NVIDIA signing are pending.

## Owner

Proposed owner: NVIDIA Model Optimizer maintainers. Team/contact and release
clearance require owning-team confirmation before publication.
Contact: <https://github.com/NVIDIA/Model-Optimizer/issues>

## License/Terms of Use

Apache-2.0, as declared by the source repository. Dataset and checkpoint licenses
remain the user's responsibility; this card does not certify release clearance.

## Use Case

Model developers using the `ptq` workflow in a ModelOpt plugin or a standalone
catalog installation. Preserve the user's requested scope and execution limits.

## Deployment Geography for Use

Uses the user's selected compute and endpoint locations; no hosted service or
additional geography entitlement is provided by this skill.

## Requirements / Dependencies

ModelOpt source and its HF dependencies; a CUDA GPU for calibration; deployment support for the serving canary.
Common files and missing companion skills resolve from a ModelOpt checkout.
Credentials are optional for artifact-only work and task-dependent for execution:
Hugging Face tokens, remote/registry access, or configured API/cloud credentials.
Do not place secret values in prompts, tracked files or reports.

## Known Risks and Mitigations

GPU time, checkpoint writes and downloads; export coverage and metadata plus a serving canary must pass before handoff.

## References

- [Workflow](SKILL.md)
- [Evaluation cases](evals/evals.json)
- <https://github.com/NVIDIA/Model-Optimizer>

## Skill Output

Checkpoint config/tokenizer/safetensors files and a validation report.

## Skill Version

Use the source directory's Git revision. Evaluation reports and the signing
identifier must refer to that exact content; neither exists for this draft yet.

## Ethical Considerations

Use checkpoints, datasets, model outputs and external services under their
applicable terms; validation evidence does not establish suitability for every
application or replace the owning team's release review.

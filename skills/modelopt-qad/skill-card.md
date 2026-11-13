# Skill Card: modelopt-qad

## Description

Recover a measured PTQ accuracy gap with explicitly authorized Quantization-Aware Distillation.
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

ModelOpt checkout, Megatron Bridge or HF Transformers, suitable GPUs, training data, and an explicit QAD budget.
Shared helpers and missing companions resolve from a ModelOpt checkout.

## Known Risks and Mitigations

Training consumes GPUs and writes checkpoints. Require explicit authorization and a budget; validate comparable results and exported checkpoints.

## References

- [Workflow](SKILL.md)
- [Evaluation cases](evals/evals.json)

## Skill Output

Training plan, checkpoint and recovery evidence.

## Skill Version

Use the exact source Git revision for evaluation and signing.

## Ethical Considerations

Use models, data, and services under their applicable terms.

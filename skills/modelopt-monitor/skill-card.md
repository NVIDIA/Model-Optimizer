# Skill Card: modelopt-monitor

## Description

Track submitted Slurm and NEL jobs in a per-session registry.
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

Scheduler or NEL access for live status; snapshots suffice for offline review.
Shared helpers and missing companions resolve from a ModelOpt checkout.

## Known Risks and Mitigations

Stale status or cross-session interference. Keep registries separate and report observed state.

## References

- [Workflow](SKILL.md)
- [Evaluation cases](evals/evals.json)

## Skill Output

Job state updates and session registry.

## Skill Version

Use the exact source Git revision for evaluation and signing.

## Ethical Considerations

Use models, data, and services under their applicable terms.

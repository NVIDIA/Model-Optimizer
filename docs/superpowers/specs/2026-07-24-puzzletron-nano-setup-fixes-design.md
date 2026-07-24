# Puzzletron Nano Setup Fixes Design

## Goal

Make the normal Puzzletron setup wizard render Nemotron Nano bundles
successfully, allocate AIPerf workers predictably, and avoid running the
Super-only latent-MoE width-importance pass on Nano.

## Symbolic Batch Alignment

The bundle renderer recursively aligns concrete model-stage batch sizes to the
inherited pipeline and data-parallel scheduling unit. Some composed family/base
configuration fields are Hydra references, such as:

```yaml
realize_model:
  micro_batch_size: ${replacement_scoring.micro_batch_size}
```

These references already resolve to an aligned concrete stage value later.
The recursive aligner must preserve interpolation strings instead of coercing
them with `int()`. Concrete integer values continue to be rounded up exactly as
they are today. Other invalid nonnumeric values remain errors rather than being
silently accepted.

## AIPerf Worker Allocation

The normal wizard currently records a hidden AIPerf worker count equal to twice
the GPUs per node. This bypasses the visible “Workers for sharded stages”
answer. For the reported Nano setup, it creates 16 AIPerf instances; with a
two-GPU serving topology this allocates 32 GPUs across four nodes.

AIPerf is a sharded per-candidate stage and will use the configured sharded
worker count. The hidden AIPerf override will no longer be written or consumed.
Candidate-count caps remain unchanged, so fewer candidates still reduce the
instance count. The reported setup will therefore use four AIPerf instances,
two GPUs per instance, and one eight-GPU node.

Existing saved normal-mode answer files that contain the old `workers.aiperf`
field will also render using `workers.sharded`; users do not need to repeat the
wizard.

## Nemotron Nano and Super Activation Passes

The shared Nemotron family configuration declares the latent-MoE activation
pass needed by Nemotron Super. Nano does not expose `moe_latent_size`, so model
inspection correctly omits the `moe_latent_dim` search axis, but the bundle
renderer currently copies the static pass unchanged.

The renderer will filter the `moe_latent` activation pass by capability:

- retain it when the detected search axes contain `moe_latent_dim` (Super);
- omit it when the axis is absent (Nano).

All other Nemotron activation passes remain unchanged. This uses detected model
geometry rather than repository-name matching.

## Verification and Current Campaign

Focused regression coverage will prove:

- Hydra interpolation batch values survive recursive alignment while concrete
  values are still aligned;
- AIPerf execution and resource summaries use the sharded worker count;
- a Nano-like Nemotron state excludes `moe_latent`;
- a Super-like Nemotron state retains `moe_latent`.

After the focused tests pass, regenerate both bundles from:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/puzzle_runs/nano/answers.yaml
```

The generated production experiment must render without error, contain no
Nano latent-MoE activation pass, and the production execution plan must assign
four two-GPU AIPerf instances.

# Model and Axis Validation

## Inventory and Admission

For every candidate axis, verify:

1. semantic capacity removed and tensors changed;
2. legal discrete values with alignment/grouping constraints;
3. dynamic slicing or masking;
4. physical materialization/export;
5. state-dict conversion and reload;
6. total/active parameter, memory, and runtime accounting;
7. required HF, AutoModel, and vLLM compatibility;
8. physical-versus-dynamic equivalence.

Treat norms, rotary encodings, grouped heads, tied weights, residual
projections, recurrent state, and multimodal projectors as coupled systems.
Zeroing output channels is not equivalent to physically reducing a normalized
input. Reject an invalid axis with evidence and continue with other axes.

## MoE Contracts

Inspect the actual latent contract. If `moe_latent_size` is absent or `None`,
expose no latent-MoE axis and rank/slice intermediate channels independently per
expert. If a latent projection exists, validate it as a coupled axis. Never
infer latent MoE from another model in the same family.

Preserve native router semantics. Correction bias may affect selection without
becoming a route weight. Grouped top-k may require fixed group membership. With
multiple groups, sort and compact experts within each original group unless
source inspection proves whole-group permutation is a symmetry. Run selection
on the physically compact vector and map compact IDs back only for resident-
teacher execution. Apply identical indices to router rows, correction bias,
auxiliary tensors, and expert weights.

Sort routed-expert intermediate channels independently inside each expert
before sorting expert identities. Keep shared-expert channel sorting separate.
Never reuse stale inner-channel scores after an outer identity permutation.

## Attention and Mamba/SSM Contracts

For grouped attention, sort query heads inside each KV group before sorting KV
groups. Compose that group-major order consistently across Q/K/V rows, output
columns, biases, rotary state, and auxiliary tensors.

For Mamba/SSM axes, preserve fixed group topology. Sort channels within the
head representation, then sort heads within fixed groups, and apply one
group-preserving composition to every coupled tensor. A physical reduction must
compact projected inputs, convolution tensors, state parameters, normalization
weight, and output-projection columns passed to the fused kernel. Teacher-sized
masking plus compensation is not sufficient BF16 evidence.

## Activation Hooks

Define the exact module boundary, measured tensor, reductions, statistic,
valid-token/media mask, packed-document boundaries, accumulation dtype,
numerical guards, shard ownership, and durable schema.

- DP ranks may see different samples; combine only commutative statistics.
- CP ranks own sequence shards; exclude padding and reduce correctly.
- TP ranks may own or replicate features; avoid double counting.
- PP stages observe local modules; use globally unique layer identities.
- EP ranks observe local experts; retain expert and routing identity.

The manifest includes model/data/config hashes, topology, expected/completed
shards, and a completion marker. Resume missing shards only.

## Sorting, Width Quality, and Physical Slicing

Keep three separate checks:

- **Sort sanity:** original, sorted, and reverse-sorted full-width teachers must
  agree within dtype-aware tolerance. Reverse sorting is an equivalence control.
- **Width sanity:** compare activation-sorted, unsorted/random, and reverse at
  reduced width. Poor ranking is a visible quality warning.
- **Slicing sanity:** compare dynamic sorted, dynamic unsorted, dynamic reverse,
  and physically materialized sorted against the same original. Dynamic versus
  physical disagreement is a correctness failure.

Verify every permutation is bijective and every coupled tensor receives the
same composition. By default, test three representative layers and two legal
targets per axis in production. For global embedding/residual width, use about
seven-eighths and three-quarters of teacher width when aligned. A smoke may be
smaller only while covering every layer implementation and axis.

Require exact config changes, parameter reduction, strict state-dict reload,
and all required backend forward/export paths. For native fused Mamba/SSM,
compare compact runtime arguments with physical export and require bitwise or
tightly bounded kernel-output parity.

## Multimodal and MTP

Use processor-native real examples for each modality. Preserve modality order,
placeholder tokens, grids/timestamps, cross-attention masks, labels, and packed
boundaries. Classify each tower/projector as prunable, immutable, or coupled to
language width, and include its costs even when not searchable.

For MTP, inspect depth, shifted targets, ties/sharing, projectors/norms, backend
support, and topology. Targets must not cross padding, document, or modality
boundaries. Report `main_ce`, `mtp_ce`, `main_kd`, and `mtp_kd` separately plus
their weighted total. Include nonzero MTP CE and KD in smoke when supported.


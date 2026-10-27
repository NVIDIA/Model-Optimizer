# QAD topology and pipeline layout

## Parallelism

- **TP=1** for recipes with static-block NVFP4 weight scales: their amax cannot
  be sharded across tensor-parallel ranks. Scale with PP and DP instead.
- Use the smallest node count where the static memory fits: student weights,
  fp32 gradients, the student's distributed optimizer state (sharded over DP),
  and the teacher weights. Add CP for long sequences on small models, and EP
  (with `--ep_size`) for MoE.
- Interleaved pipelining (VPP) is not supported for distillation, so fill and
  drain cost `(PP - 1) / (M + PP - 1)` of each iteration, where `M` is the
  number of micro-batches per pipeline (`GBS / (MBS * DP)`).

## Why the default layout is unbalanced

With an even layer split and uniform full recompute:

- The last stage also runs the teacher and student output layers, MTP, and the
  KD loss, so it is the slowest stage and the others wait on it.
- Under 1F1B, stage `k` holds `min(PP - k, M)` micro-batches of activations,
  so later stages have spare memory that full recompute leaves unused.

Fix both with `distill.py`'s `--num_layers_in_first_pipeline_stage`,
`--num_layers_in_last_pipeline_stage`, and `--recompute_num_layers_per_stage`
(with `--recompute_granularity full --recompute_method block`): fewer layers on
the last stage, and fewer recomputed layers on later stages.

## Find the layout by measuring

The right values depend on the model, sequence length, and GPU memory, so
measure on the target node count, with mock data and 4-5 iterations each:

1. Run `../day0-release/scripts/profile_distill.py` (same CLI as `distill.py`, from
   `examples/megatron_bridge`, with `PROFILE_OUT=<dir>`) twice with the default
   even split: once with `--recompute_granularity full --recompute_method
   uniform --recompute_num_layers 1`, once with `--recompute_granularity full
   --recompute_method block --recompute_num_layers <about half the layers per
   stage>`.
2. Plan:

   ```bash
   python "$SKILL_DIR/../day0-release/scripts/plan_layout.py" --full <dir1> --partial <dir2>
   ```

   It prints the fitted unit costs, the predicted iteration time, per-stage
   peak memory, and the `distill.py` flags. The default memory cap is 80% of
   the device, leaving room for fragmentation, validation, and checkpoint
   saves.
3. **Confirm** with 2-3 measured iterations of the proposed layout against the
   default on the real node count, and keep it only if it is faster. Then make
   sure the first validation pass and checkpoint save of the real run succeed.

At 32K, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the last stage
allocates one large fp32 logits-gradient buffer that otherwise fails on
fragmentation even when enough memory is free.

## Measured example: Qwen3.8-27B NVFP4 QAD

64 layers, 4 nodes x 4 GB300, TP1/PP4/DP4, MBS 1:

| Sequence, GBS | Layout (layers / block recompute per stage) | s/iter | Peak GiB |
| --- | --- | --- | --- |
| 16K, 512 | 16/16/16/16, uniform full recompute | 228.4 | 221 |
| 16K, 512 | 17/18/18/11, recompute 11/9/4/4 | 162.8 | 221 |
| 32K, 256 | 16/16/16/16, uniform full recompute | 233.9 | 253 |
| 32K, 256 | 17/18/18/11, recompute 14/13/10/11 | 185.8 | 236 |

On 16K profiles taken on one node (optimizer state offloaded to match DP4 memory), `plan_layout.py`
predicted the measured iteration time and peak memory of tuned layouts within 1-2%.

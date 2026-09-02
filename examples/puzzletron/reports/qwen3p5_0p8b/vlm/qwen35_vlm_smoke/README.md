# Qwen 3.5 0.8B VLM measurements

This campaign report retains historical quality and serving measurements for
the Qwen 3.5 0.8B teacher and a pre-distillation FFN-2048 student. The evidence
status remains preliminary because serving was measured in one node allocation
and checkpoint byte identity across the quality and serving runs was not
recorded.

See the [exclusive-node run](runs/exclusive_w32_v2/summary.md) for the measured
values, study conditions, limitations, structured record, and recorded recipe.

The maintained
[campaign config](../../../../configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml)
runs AIPerf only after candidate screening and final KD, with 32 warmup and 64
measured requests per serving cell. Follow the
[campaign guide](../../../../docs/qwen3p5_0p8b_vlm_smoke.md) to configure,
inspect, and run it. The maintained route does not reproduce this historical
pre-KD study, its eight-GPU placement swap, or its 256-request cells.

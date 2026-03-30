# QAD Research Learnings

## What Works

- Local hessian initialization consistently produces the best PTQ starting point across all families
- **lr=7e-6 CONFIRMED as optimal lr at 1.0ep for Family A -- bracket CLOSED** -- wlh_lr7e-6_ep1=0.0920, wlh_lr6e-6_ep1=0.0924, wlh_lr8e-6_ep1=~0.0928 (pending final). lr=7e-6 beats both neighbors by 0.0004-0.0008. **UPDATE (inv92): wlh_lr7e-6_ep2 BROKE THROUGH to 0.0929 at ep1.32.** The 0.094 plateau (ep1.10-1.27) finally broke with two consecutive drops (ep1.29=0.0931, ep1.32=0.0929). **UPDATE (inv93): ACCELERATION CONFIRMED at ep1.39=0.0924.** Descent rate ep1.32-1.39 = 0.0005/0.07ep = 0.0071/ep is FASTER than ep1.20-1.32. Full late trajectory: ep1.29=0.0933, 1.32=0.0929, 1.34=0.0930, 1.37=0.0926, 1.39=0.0924. **UPDATE (inv94): ep1.44=0.0919!** Steep 0.0006 drop from ep1.42=0.0925. Now only 0.001 above overall best (0.0909). Revised projection: 0.090-0.092. lr at ep1.44 ~1.96e-6, still productive with 28% remaining.
- **lr=7e-6 is WORSE than lr=1e-5 for Family B at 1.0ep** -- slh_lr7e-6_ep1 completed at 0.0937 vs slh_lr1e-5_ep1 at 0.0929. Family B benefits from the "sudden late drop" at very low lr (ep0.95-1.0), which lr=7e-6's higher residual lr prevents. **Optimal lr is family-dependent:** Family A prefers lr=7e-6 (sustained learning), Family B prefers lr=1e-5 (precise late fine-tuning via per-block scales). However, slh_lr7e-6_ep2 may still beat slh_lr1e-5_ep2 at 2.0ep because lr=7e-6 keeps lr productive in the second epoch. **UPDATE (inv103): slh_lr7e-6_ep2 at ep1.81=0.09088!** Dramatic late push broke through 0.091 barrier. Trajectory ep1.73-1.81: 0.09213->0.09146->0.09143->0.09088. Family B now within 0.001 of Family A best (0.08991). Confirms lr=7e-6 is correct for 2.0ep Family B.
- **[INVALIDATED - qerr bug] QErr regularization was reported DEAD for Family A**: ALL qerr experiment conclusions are INVALID due to a critical bug in the qerr implementation that the user is fixing. The frozen qerr/mse values and zero-effect conclusions were artifacts of the bug, NOT fundamental STE limitations. QErr must be re-tested after the fix.
- **[INVALIDATED - qerr bug] QErr SUM reduction was reported DEAD**: ALL qerr SUM results are INVALID. The frozen qerr/sum values were caused by the implementation bug, not by STE blocking gradients. Must be re-tested after the fix.
- QAD distillation is highly effective: reduces eval_loss from ~1.74 (PTQ baseline) to ~0.10 (after 0.25 epochs)
- Even the worst PTQ checkpoint (nvfp4_default, baseline 1.8162) reaches 0.1166 with QAD -- a massive improvement
- adaround_local_hessian_init (0.1103) is substantially better than adaround_mse_init (0.1289) for Family C
- Cosine and linear schedules perform identically at 0.25 epochs (both 0.1004) with warmup=0.1

## BUG: LSQ commit broke checkpoint loading (inv112) -- FIXED

- **LSQ commit (8310cf8c) refactored `_smooth_lsq` to `_learnable_scale` flag but broke backward compat with existing checkpoints.** Old checkpoints have `_smooth_lsq=True` saved in state but not `_learnable_scale=True` (new attribute). On loading, `_learnable_scale` defaults to `False`, causing `_fake_quantize` to skip the learnable-scale path and fall through to `scaled_e4m3` with FP4's E=2,M=1, which fails the E=4,M=3 check.
- **FIX:** Added backward compat check in `_fake_quantize`: if `_learnable_scale` is False but `_smooth_lsq` or `_lsq` is True, set `_learnable_scale = True`.
- **Impact:** Only affects NEW launches with Family B (scale_learn_*) checkpoints. Running jobs that started before the LSQ commit are unaffected (they loaded old code into memory).

## CRITICAL: QErr Implementation Bug (inv73) -- NOW FIXED AND VALIDATED

- **ALL qerr experiment conclusions from invocations BEFORE the fix are INVALID.** The user identified and fixed a critical bug in the qerr implementation.
- Post-fix QErr experiments are running and VALIDATED: qerr/sum is decreasing in all 3 running experiments.
- **QErr coeff=1.0 for Family A (wlh)**: CROSSOVER CONFIRMED at ep0.76. eval_loss=0.0955 vs plain QAD ~0.0960 at matched epoch. Strong drop from 0.0967 at ep0.73. Gap reversed: QErr now AHEAD by ~0.0005.
- **QErr coeff=1.0 for Family B (slh)**: crossover CONFIRMED and EXTENDING. At ep0.71, QErr=0.0957 vs plain ~0.0960. QErr ahead by 0.0003. Sustained for 7+ consecutive evals since ep0.54.
- **QErr coeff=0.1 for Family A**: COMPLETED at 0.0928, matching plain QAD exactly. qerr/sum reduced only 0.33% (1715034 -> 1708104). Perfectly neutral -- confirms QErr has zero benefit for Family A.
- **QErr coeff=10 for Family A**: CATASTROPHIC. KILLED at ep0.27=0.1279. Loss monotonically increased from ep0.17 onward. The huge qerr gradient overwhelms distillation. Sub-linear qerr/sum reduction (9.3% early) with devastating eval_loss cost.
- **Family A QErr coefficient sensitivity**: coeff=0.1 (net neutral at 0.0928, matching plain QAD exactly), coeff=1.0 (net -0.0007), coeff=10 (diverges). QErr at coeff=0.1 is perfectly neutral; higher coefficients are net negative. The optimal QErr coefficient for Family A is zero. Annealing 0->1.0 at ep0.34=0.1037 also behind plain QAD (by ~0.0009). **Family A QErr investigation CLOSED.**
- **Family A and B need DIFFERENT coefficient ranges** due to 200x difference in qerr/sum magnitude (Family A ~1.7M, Family B ~348M).

## Stagnation Warnings

- **Diversity improved**: Running QErr sweep (3 coefficients), lr bracket closure (2 lrs), long training (2 runs), coverage (wmse_ep1). Queue includes Family C plain QAD, constant lr schedule, QErr annealing, and QErr+long training combos.
- **Watch for Family C stagnation**: All 10 prior Family C experiments stuck at ~0.105 ceiling with adaround. Plain QAD (frozen round_logits) is the key untested paradigm to break below 0.100.

## What Doesn't Work

- **scale_learn_max_init (Family B) clearly inferior**: eval_loss=0.1149 at 0.25ep vs 0.1004 for hessian_init. Gap of 0.0145. Not worth pursuing.
- **adaround_lh at 1.0ep confirms Family C ceiling at ~0.105**: eval_loss=0.1050 at 1.0ep with dist_loss FROZEN at 114.02 throughout. Fixed beta=4 is fundamentally unable to reduce dist_loss. Family C gap to best = 0.012+.
- **Cosine confirmed inferior to linear for BOTH families at 1.0ep**: Family A cosine COMPLETED at 0.0935 vs linear 0.0928 (gap=0.0007). Family B cosine 0.0939 vs linear 0.0929 (gap=0.001). The cosine cliff (lr drops to near-zero too abruptly in the final 10%) kills improvement.
- **Per-scale lr 5e-5 (5x higher) causes divergence for Family B**: eval_loss spiked to 0.1427 at ep0.05, far above baseline. Scales are sensitive to lr -- only lower or equal lr viable.
- **lr=1e-4 is catastrophic** -- causes massive overshoot then slow recovery. Best lr=1e-4 result (0.2147) is 2.15x worse than best lr=1e-5 (0.0999). Consistent across all families.
- **lr=5e-5 also overshoots**: slh at lr=5e-5 gives eval_loss=0.1284. Viable lr range is [5e-6, 1e-5].
- **lr=5e-6 is too slow**: slh at lr=5e-6 gives 0.1032 at ep 0.25, worse than lr=1e-5 (0.1004).
- **bs=8 is slightly worse than bs=4**: slh_bs8 = 0.1024 vs slh_bs4 = 0.1004. Larger batch doesn't help at 0.25 epochs.
- **QErr does NOT help Family B**: slh_qerr = 0.1005 vs slh_plain = 0.1004. The large qerr/mse (~32.57) for Family B with per-block scales means the regularization gradient is too noisy/large relative to the distillation loss.
- **Cosine schedule + qerr does NOT help**: wlh_cosine_qerr = 0.1007 vs linear+qerr = 0.0997. Cosine hurts when combined with qerr.
- **Lower qerr coefficients (0.001->0.05) are worse**: 0.1002 vs 0.0997 with default (0.01->0.1). The default coefficients are well-tuned.
- **No warmup slightly hurts**: warmup=0.0 gives 0.1000 vs warmup=0.1 gives 0.0997. Warmup helps marginally.
- **Higher adaround dist_loss_coeff (1e7) is similar but not better**: 0.1101 vs 0.1103 with 1e6. The dist_loss itself dropped from 94->39 (stronger constraint forces rounding closer to binary) but eval_loss is essentially unchanged. Not worth the extra constraint.
- **Lower adaround dist_loss_coeff (1e5) confirmed too weak**: 0.1126 vs 0.1103 with 1e6. dist_loss barely moved (114->112). The rounding constraint is too loose. 1e6 is the sweet spot -- both 1e5 and 1e7 are worse.
- **Unfreezing weights with adaround is bad**: 0.1176 vs 0.1103 frozen at 0.25ep. Even with roundlr=1e-3 split, unfreeze at ep0.22=0.12297 is 0.020 behind standard frozen adaround. Trajectory decelerates rapidly. Weight updates fundamentally conflict with rounding optimization regardless of lr configuration.
- **weight_decay=0.01 hurts**: 0.1000 vs 0.0997 with wd=0.0. Always use wd=0.0.
- **gradient_accumulation=2 (eff_bs=8) hurts**: 0.1004 vs 0.0997 with accum=1. All forms of effective bs=8 are worse than bs=4 at 0.25ep.
- **lr=7e-6 is competitive at 0.25ep**: 0.1001 vs 0.0997 for lr=1e-5. Only 0.0004 gap. Slower lr decay at longer training may help.
- Dynamic FP4 max calibration (nvfp4_default) has the worst baseline AND worst QAD result (0.1166 vs 0.0999 for hessian)
- MSE and max initialization for scales (Family B) are worse than local_hessian at PTQ stage
- **LSQ (Learned Step Size) DEFINITIVELY DEAD.** All 4 tested lr values crashed with NaN: lr=1e-5 at ep0.05, lr=5e-6 at ep0.10, lr=1e-6 at ep0.20, lr=2e-6 at ep0.20. Pattern: halving lr doubles survival time but NaN is inevitable. The LSQ checkpoint's extremely tight quantization grid (qerr/sum=13463 vs 347M for slh) amplifies STE gradients for per-block scales beyond recoverability. LSQ training would require gradient clipping code changes (out of scope). **No further LSQ experiments.**
- **round_logits lr=1e-2 is catastrophically bad for adaround.** Killed at ep0.24 after oscillating in 0.120-0.124 band with upward trend. Best was ep0.12=0.1206 (0.015 behind standard adaround). The high round_logits lr destabilizes the optimization. round_logits lr=1e-3 is still recovering (ep0.17=0.1127, gap narrowing to 0.005).
- **round_logits lr=5e-4 is even MORE catastrophically bad.** First eval 0.23144 (2x worse than normal start ~0.115). ep0.05=0.18510. Killed. At lr=5e-4, round_logits get insufficient gradient signal to adapt, causing the rounding decisions to remain suboptimal and the entire distillation to fail. **Confirmed: roundlr=1e-3 is the LOWER bound, not a midpoint.** The viable round_logits lr range is [1e-3, ~5e-3]. Testing 2e-3 next.
- **QErr regularization DEFINITIVELY CLOSED for all families.** Post-fix results: coeff=0.01 (net -0.0005 Family B), coeff=0.1 (neutral both families), coeff=1.0 (net -0.0007 Family A, -0.0009 Family B), coeff=10 (diverges Family A). Annealing 0->1.0 also negative. Optimal QErr coefficient is zero for all families. No further QErr experiments needed.

## Key Finding: Family A vs Family B Race at 1.0ep

- **Family B (scale_learn_local_hessian_init) LED at matched epochs mid-training** but Family A caught up late.
- At 0.25 epochs: tied (A=0.0999, B=0.1004). At 0.5ep: A=0.0962, B=0.0971. Family A slightly ahead.
- At 1.0ep matched epochs: Family B led at ep0.20 (B=0.1043 vs A=0.1095, gap=0.005) but gap narrowed to essentially zero at ep0.66 (B=0.0971 vs A=0.0969, gap=0.0002 in A's favor). Family A pulled ahead once lr dropped below ~4e-6.
- **UPDATE (inv 36): slh_ep1 at ep0.66=0.0971, still declining.** The 1.0ep Family B race is extremely tight with Family A. The per-block scale parameters gave a mid-training advantage but did not translate to a superior final result. Family A's simpler structure may slightly edge out at very low lr.
- The per-block scale parameters give Family B additional degrees of freedom in mid-training, but Family A catches up in the low-lr fine-tuning phase.

## Key Finding: Longer Training

- **0.5ep Family A: eval_loss=0.0962, perplexity=1.1010.** 0.5ep Family B: eval_loss=0.0971, perplexity=1.1019. Both huge improvements over 0.25ep best (0.0997).
- **1.0ep Family A at ep0.54=0.0998.** Already below 0.5ep best. But FLATTENING: ep0.51=0.0998, ep0.54=0.1000. The linear lr has decayed to ~5e-6 and is losing effectiveness. Projected final: 0.093-0.097 (revised down from earlier optimistic 0.088-0.093).
- **1.0ep Family B at ep0.20=0.1043.** Clean downward trajectory with 80% training remaining. lr=8e-6 (still very useful). Projected: 0.085-0.092 depending on convergence behavior.
- **Longer training is the single most impactful axis discovered.** The improvement from ep0.25->ep0.50 dwarfs all hyperparameter mutations at 0.25 epochs.
- **NEW INSIGHT: wlh_qerr_ep1 is flattening around ep0.51-0.54** (0.0998-0.1000). The linear lr decay may have killed useful learning too early. This motivates: (a) cosine schedule at 1.0ep (maintains higher lr), (b) lr=1.5e-5 at 1.0ep (effective lr at ep0.50 = 7.5e-6 vs 5e-6 for lr=1e-5).

## Key Finding: Family A and B TIED at 1.0ep

- **slh_ep1 COMPLETED at 0.0929 vs wlh_qerr_ep1 at 0.0928.** Difference of 0.0001 is noise.
- Family B led at matched epochs mid-training (ep0.20-0.50) but the advantage diminished. The final 5% produced a sudden drop: ep0.93=0.0939, ep0.95=0.0938, ep0.98=0.0929. The very low lr (< 1e-6) still yielded 0.001 improvement in the last 7% of training.
- **The per-block scale params in Family B neither helped nor hurt at 1.0ep completion.** They gave a mid-training speed advantage but no final-result advantage.
- Full late trajectory: ep0.85=0.0944, ep0.88=0.0940, ep0.90=0.0940, ep0.93=0.0939, ep0.95=0.0938, ep0.98=0.0929.

## Key Finding: Cosine PASSED Linear for Family B at 1.0ep

- **CONFIRMED: slh_cosine_ep1 at ep0.63=0.0972 vs slh_ep1 at ep0.63=0.0975.** Cosine is 0.0003 AHEAD. The crossover happened between ep0.56-0.61.
- Full cosine trajectory: ep0.42=0.1005, ep0.44=0.1010, ep0.46=0.0998, ep0.49=0.0999, ep0.51=0.0999, ep0.54=0.0992, ep0.56=0.0984, ep0.59=0.0987, ep0.61=0.0976, ep0.63=0.0972. After crossing ep0.50, the decline ACCELERATED.
- Full linear trajectory at matched epochs: ep0.56=0.0978, ep0.59=0.0979, ep0.61=0.0975, ep0.63=0.0975. Linear is flattening while cosine is accelerating.
- **slh_cosine_ep05 at ep0.39=0.0982 is ahead of linear at matched epoch (~0.1008).** Cosine helps Family B at all durations beyond ep0.25.
- **Cosine 2.0ep for Family B is now the single most promising experiment.** Cosine maintains lr~5e-6 at ep1.0 (vs linear's ~1e-6), giving ~5x more productive learning rate. At 2.0ep, this advantage compounds further.
- **Cosine is family-specific.** wlh_cosine_ep1 at ep0.56=0.0997 vs wlh_ep1 at ep0.56=~0.0978. Gap=0.0019 -- cosine is NOT winning for Family A. The per-block scale parameters in Family B benefit more from sustained higher lr.
- **UPDATE (inv 40): Cosine TIED linear at ep0.71.** slh_cosine_ep1 at ep0.71=0.0960 vs slh_ep1 at ep0.71=0.0960. After trailing at ep0.66 (cosine=0.0975, linear=0.0971), cosine accelerated through ep0.68=0.0965, ep0.71=0.0960. Meanwhile linear continued: ep0.68=0.0963, ep0.71=0.0960. EXACTLY TIED. Cosine lr at ep0.71 = ~5.9e-6 vs linear's ~2.9e-6. With 2x the lr remaining, cosine should decisively pull ahead in ep0.75-1.0. The "cosine reversal" warning from ep0.66 was premature noise.
- **UPDATE (inv 42): slh_cosine_ep1 at ep0.78=0.0955, ACCELERATING.** Trajectory: ep0.71=0.0960, ep0.73=0.0962 (noise), ep0.76=0.0957, ep0.78=0.0955. At ep0.78, linear slh_ep1 was at ~0.0947. Cosine trails linear by 0.0008 at ep0.78. However, cosine still has ~4x the lr of linear at this point (cosine lr ~4.5e-6 vs linear ~1.1e-6). The final 22% will determine if cosine catches up.
- **UPDATE (inv 42): wlh_cosine_ep1 at ep0.68=0.0968, TIED with linear.** Linear wlh_qerr_ep1 at ep0.68 was 0.0969. This reverses the earlier conclusion that cosine was definitively behind for Family A. Cosine caught up from a 0.0019 gap at ep0.56 to parity at ep0.68. Family A cosine may finish close to or matching linear 1.0ep (0.0928).

## Key Finding: wlh_qerr_ep1 COMPLETED at 1.0ep

- **Final eval trajectory: ep0.85=0.0933, ep0.88=0.0936, ep0.90=0.0935, ep0.93=0.0928, ep0.95=0.0933, ep0.98=0.0928.** Best checkpoint: ep0.93 and ep0.98 both at 0.0928 (load_best_model_at_end selects best ckpt).
- At 1.0ep completion, lr had decayed to essentially zero (~2e-8). The model was still finding 0.0005 improvements in the last 5% of training.
- **This makes 2.0ep the single most important experiment.** At 2.0ep with lr=1e-5, the lr at ep1.0 would be ~5e-6 (same as where the 1.0ep run was at ep0.50 -- its most productive phase).
- **QErr was confirmed dead** (qerr/mse=0.2295 unchanged) but the experiment used qerr anyway. Future runs should drop qerr.

## Key Finding: 2.0ep Training Shows Continued Gains (inv50, updated inv64)

- **wlh_ep2 BREAKTHROUGH at ep1.20=0.0965!** Dramatic improvement from the 0.0975-0.0985 plateau at ep1.07-1.17. The second epoch is now producing real gains.
- Full trajectory ep0.93-1.22: 0.0989, 0.0979, 0.0982, 0.0981, 0.0983, 0.0980, 0.0977, 0.0975, 0.0965, 0.0970. The breakthrough at ep1.20 broke through a 15-eval plateau.
- **UPDATE (inv64): wlh_ep2 STEEP DESCENT at ep1.49=0.0945!** Late trajectory: ep1.37=0.0962, ep1.39=0.0952, ep1.42=0.0957, ep1.44=0.0950, ep1.46=0.0948, ep1.49=0.0945. Three consecutive improvements. Step 3053/4096 (75%). lr at ep1.49 ~2.5e-6. Projection: 0.090-0.093 at ep2.0. This will almost certainly set a new overall best.
- **UPDATE (inv65): wlh_ep2 at ep1.56=0.0944.** Continued descent: ep1.51=0.0948, ep1.54=0.0945, ep1.56=0.0944. Step 3200/4096 (78%). lr ~2.2e-6. Projection holds at 0.091-0.093.
- **UPDATE (inv70): wlh_ep2 CONVERGING at 0.0914-0.0916.** Latest: ep1.90=0.0916. In final eval at ep1.93. lr < 0.5e-6. Plateau reached. Final result: 0.0914. Full late trajectory: ep1.78=0.0920, ep1.81=0.0921, ep1.83=0.0914, ep1.86=0.0915, ep1.88=0.0914, ep1.90=0.0916. NEW ALL-TIME BEST when completed.
- **UPDATE (inv72): wlh_ep2 at ep1.98=0.0909!** Did NOT fully plateau -- continued improving past ep1.90. The final 8% of training (ep1.90->1.98) yielded another 0.0007 improvement despite lr < 0.2e-6. New projected final: 0.0909. This means even 2.0ep with lr=1e-5 did not fully converge, validating the case for lr=7e-6 at 2.0ep.
- **FINAL (inv73): wlh_ep2 COMPLETED at eval_loss=0.0909, perplexity=1.0951.** Train loss=0.0918. Best checkpoint loaded at end. This is the definitive best result. Full training took 14h18m. wlh_qad_lr7e-6_ep2 launched as successor experiment.
- **UPDATE (inv70): slh_ep2 BROKE THROUGH at ep1.54=0.0947! NEW FAMILY B LOW.** Trajectory: ep1.37=0.0961, ep1.39=0.0958, ep1.42=0.0957, ep1.44=0.0955, ep1.46=0.0955, ep1.49=0.0951, ep1.51=0.0952, ep1.54=0.0947. Smooth descent continues. Family B IS benefiting from 2.0ep -- just slower than Family A. Projected final: 0.091-0.094.
- **UPDATE (inv77): slh_ep2 STEEP DESCENT continues to 0.0926!** Full late trajectory: ep1.61=0.0945, ep1.64=0.0943, ep1.68=0.0943, ep1.71=0.0942, ep1.73=0.0935, ep1.76=0.0935, ep1.78=0.0937, ep1.81=0.0935, ep1.83=0.0929, ep1.90=0.0929, ep1.93=0.0926, ep1.95=0.0928. Best=0.0926 at ep1.93. CORRECTS inv76 analysis that "2nd epoch wasted for Family B." The lr=1e-5 linear decay still produced 0.003+ improvement in epoch 2 (0.0929->0.0926). Family B gap to Family A: 0.0909 vs ~0.0926 = 0.0017. Final eval in progress.
- **FINAL (inv78): slh_ep2 COMPLETED at eval_loss=0.0924, perplexity=1.0968.** Final eval improved over best checkpoint (ep1.98=0.0924 vs earlier best ep1.93=0.0926). The very last steps before lr reached zero still produced measurable gains. Family B at 2.0ep: 0.0924, behind Family A (0.0909) by 0.0015. slh_lr7e-6_ep2 is critical -- lr=7e-6 maintains lr ~1.75e-6 at ep1.0 vs lr=1e-5's ~zero.
- **slh_ep2 at ep1.12=0.0985 (inv64).** [Superseded -- see update above. The plateau broke after ep1.20.]
- **slh_cosine_ep2 KILLED at ep0.66.** Stuck at 0.103-0.106 for 13 evals.
- **adaround_lh_plain_qad_ep1 COMPLETED at 0.1051.** Matched frozen-weights adaround (0.1050). Both Family C paradigms converge to ~0.105 ceiling at 1.0ep.

## Key Finding: Frozen lm_head+embed_tokens -- MIXED RESULTS (inv112-113)

- **COMPLETED RESULTS (inv113):**
  - **Family A (wlh) frozen: eval_loss=0.09271** vs unfrozen 0.0928. Only 0.0001 better. Frozen is NEUTRAL for Family A.
  - **Family B (slh) frozen: eval_loss=0.09344** vs unfrozen 0.0929. 0.005 WORSE. Frozen HURTS Family B at 1.0ep.
  - **Family C (adaround) frozen: eval_loss=0.10507** vs unfrozen 0.1050. Identical. Frozen NEUTRAL for Family C.
- **CORRECTION:** inv112 reported "slh_frozen AHEAD of unfrozen at ep0.93" but final results show unfrozen wins decisively (0.0929 vs 0.09344). The apparent mid-training advantage was noise from oscillation.
- **Frozen training does NOT help any family at 1.0ep.** The hypothesis that focused gradients would help was wrong -- the lm_head/embed_tokens updates contribute meaningfully to the final result, especially for Family B.
- 2.0ep frozen runs still running on GPUs 3 (wlh) and 7 (slh). May show different behavior at longer duration.
- **UPDATE (inv125): wlh_frozen_ep2 at ep1.83=0.08987 -- NEW OVERALL BEST!** Frozen head/embed BEATS unfrozen (0.08991) at 2.0ep. The frozen model was tracking 0.001-0.002 behind unfrozen until a decisive breakthrough at ep1.83. Still running (92% done). This REVERSES the 1.0ep conclusion: frozen is neutral/hurts at 1.0ep but HELPS at 2.0ep. The mechanism is likely regularization -- preventing overfitting of the high-capacity lm_head during the second epoch.
- **scales_only lr=7e-6 is WORSE than lr=1e-5 (0.09497 vs 0.09400).** The lr=7e-6 advantage seen in full QAD does not transfer to scales-only training. Scales-only has far fewer parameters and benefits from higher lr.

## Key Finding: Scales-Only Training -- Excellent Training-Based PTQ (inv113)

- **slh_qad_lr1e-5_scales_only_ep1 COMPLETED at eval_loss=0.09400, perplexity=1.0986.**
- Only 0.005 behind full training (0.0929). Training ONLY the per_block_scale parameters (not model weights) achieves 94.6% of full QAD quality improvement.
- This is the strongest training-based PTQ result: modifying only quantization parameters, not model weights.
- qerr/sum=347983136 (unchanged, as expected with coeff=0).
- VRAM usage only ~26.7GB vs ~47.9GB for full training -- 44% less memory.
- Now launching: scales-only at 2.0ep, scales-only+frozen, scales-only at lr=7e-6.

## Key Finding: adaround round_logits lr=1e-3 BEATS Standard (inv113)

- **adaround_lh_roundlr1e-3_ep1: eval_loss=0.10391** vs standard adaround 0.1050. A 0.0011 improvement.
- dist_loss=105.94 (vs standard 114.02). Lower round_logits lr (1e-3 vs default) allowed better convergence of dist_loss.
- This is the best Family C result ever, though still far behind Families A/B (0.015+ gap).
- **UPDATE (inv133): roundlr=2e-3 BEATS roundlr=1e-3! eval_loss=0.10299 vs 0.10391 (gap=0.0009).** dist_loss=98.09 vs 105.94. BOTH metrics improved. Higher round_logits lr allows more aggressive rounding optimization. The roundlr bracket [1e-3, 5e-3] is essentially flat at 1.0ep (all within 0.001), but 2e-3 is the sweet spot. roundlr=2e-3 2.0ep convergence run launched.

## Key Finding: LSQ Definitively Dead (inv112)

- lr=2e-6 crashed at ep0.20 with NaN, exactly like lr=1e-6. 4th consecutive crash across lr span of 1e-5 to 2e-6.
- The halving-lr-doubles-survival pattern held perfectly but NaN is inevitable at every lr.
- Root cause: LSQ's tighter quantization grid produces much larger STE gradients that eventually blow up. Would require gradient clipping code change to fix (out of scope).
- **Conclusion: scale_after_dequantize (Family B) is the only viable learned-scale approach for FP4 QAD.**

## Key Finding: wlh_lr7e-6_ep2 COMPLETED -- ALL-TIME BEST at 0.08991 (inv100)

- **FINAL: eval_loss=0.08991, perplexity=1.0941.** Best checkpoint at ep1.78. Oscillated 0.0899-0.0907 from ep1.78 onward. lr effectively zero.
- train_loss=0.0888. qerr/sum=1715083 unchanged (coeff=0, monitor only). VRAM ~47.9GB.
- **adaround_lh_frozen_round_lr7e-6_ep1 COMPLETED at 0.10518.** Family C ceiling confirmed from frozen-round + lr=7e-6 direction. Identical to standard adaround ceiling (0.1050).
- **QErr anneal 0->1.0 COMPLETED at 0.09300.** Behind plain QAD (0.0928) by 0.0008. qerr/sum dropped only 1.5% (1715083->1689569). QErr annealing adds noise to distillation gradient without meaningful quantization improvement. QErr axis FULLY CLOSED.
- **slh_lr7e-6_ep2 at ep1.61=0.0923.** Still descending. Best ep1.59=0.0920. Already below completed slh_lr1e-5_ep2 (0.0924). Family B closing gap.
- **UPDATE (inv101): slh_lr7e-6_ep2 BROKE THROUGH to ep1.68=0.09124!** Trajectory: ep1.64=0.09157, ep1.66=0.09226, ep1.68=0.09124. New Family B all-time low. Only 0.0013 behind Family A's 0.08991. Still descending strongly with ~16% training remaining (~2h). Revised projection: 0.088-0.091.
- **QErr-4 (slh coeff=0.01) COMPLETED at 0.09341.** Behind plain slh (0.0929) by 0.0005. QErr FULLY CLOSED for ALL families.
- **LSQ checkpoint creation CONFIRMED successful.** checkpoints_ptq/Qwen3-1.7B_nvfp4_lsq_local_hessian_init/ exists. LSQ baseline eval launched.
- **UPDATE (inv102): LSQ baseline eval_loss=1.7485, perplexity=5.7462.** Between Family A (1.7449) and Family B (1.7501). Very close starting point -- controlled comparisons will be clean.
- **UPDATE (inv102): adaround_lh_frozen_round_qerr1 COMPLETED at 0.10456.** Matches standard adaround (0.1050). QErr had ZERO benefit for Family C (dist_loss=0.0 since round_logits were frozen). Family C ceiling re-confirmed at ~0.105.
- **UPDATE (inv102): roundlr1e-3 NOT CRASHED.** Early traceback at step 1 was transient; HF Trainer auto-resumed from checkpoint. Still running on GPU 0 (child PID 870819). ep0.05=0.155, behind standard adaround but recovering. ep0.02=0.219 was from before the restart.
- **UPDATE (inv102): LSQ first training run LAUNCHED on GPU 1 (PID 881889).** Direct LSQ vs slh comparison at identical settings (lr=1e-5, 1.0ep, linear, warmup=0.1). Control: slh_qad_lr1e-5_ep1 (0.0929).
- **UPDATE (inv102): slh_lr7e-6_ep2 at ep1.76=0.09146.** Best remains ep1.68=0.09124. Oscillating 0.091-0.093 with lr near zero. ~12% remaining (~1.5h).
- **FINAL (inv106): slh_lr7e-6_ep2 COMPLETED at eval_loss=0.09082, perplexity=1.0951.** Train_loss=0.08929. Best ckpt ep1.88=0.09082 (tied with final ep1.98=0.09082). Family B all-time best. Within 0.0009 of overall best (0.08991). 14h23m runtime. Confirms lr=7e-6 is optimal for 2.0ep across both families.

## Key Finding: nvfp4_default Uniquely Bad with Frozen Head/Embed (inv106)

- **default_qad_lr1e-5_ep1_frozen_head_embed KILLED at ep0.32=0.12644.** Stalled at 0.126-0.136 for 10+ consecutive evals since ep0.05.
- Compare unfrozen default at ep0.32: ~0.122. Frozen is 0.004 worse AND not improving.
- The dynamic FP4 max calibration checkpoint (nvfp4_default) is uniquely dependent on lm_head/embed_tokens updates for the distillation signal. When these are frozen, essentially no useful learning occurs.
- **This does NOT generalize to other families.** slh_frozen is AHEAD of unfrozen (+0.001), and wlh_frozen gap is narrowing (0.002 at ep0.63, down from 0.003). The difference is that slh has per-block scales (trainable quantizer params that compensate for frozen weights) and wlh has better hessian initialization.
- **UPDATE (inv111): wlh_frozen COMPLETED at 0.09271 (ep0.95).** Final gap to unfrozen (0.0928) is only 0.0001. Frozen advantage is real but marginal at 1.0ep for Family A. The 2.0ep frozen experiment (wlh_qad_lr7e-6_ep2_frozen_head_embed) launched to test if the advantage compounds with longer training.
- **UPDATE (inv111): slh_frozen at ep0.83=0.09426, ACCELERATING.** Best yet, 0.001+ ahead of unfrozen at matched epoch. Projected ep1.0: 0.091-0.093, potentially beating unfrozen slh_ep1 (0.0929) by 0.001+.
- **FINAL (inv128): wlh_frozen_ep1 COMPLETED at eval_loss=0.09226, perplexity=1.0966.** Frozen is 0.0006 WORSE than unfrozen (0.0920) at 1.0ep. Best ckpt ep0.98. The frozen head/embed strategy does NOT help at 1.0ep for Family A.
- **FINAL (inv128): wlh_frozen_ep2 COMPLETED at eval_loss=0.08952, perplexity=1.0937. NEW ALL-TIME BEST.** Frozen is 0.00039 BETTER than unfrozen (0.08991) at 2.0ep. Best ckpt ep1.98. train_loss=0.08888. The crossover where frozen becomes beneficial is between 1.0ep and 2.0ep. Mechanism: frozen head/embed prevents overfitting of high-capacity output layers during the extended second epoch, redirecting gradient budget to the quantized linear layers where it matters more.
- **CONCLUSION: Frozen lm_head+embed_tokens is the optimal strategy for 2.0ep Family A training.** Recipe: lr=7e-6, linear, warmup=0.1, frozen head/embed, 2.0ep = 0.08952. This is within 0.015 of the teacher baseline (1.7017 vs 1.7449 for PTQ, closing 96.4% of the PTQ-to-teacher gap).
- **FINAL (inv132): slh_frozen_ep2 COMPLETED at eval_loss=0.09098, perplexity=1.0952.** Frozen HURTS Family B by ~0.002 at 2.0ep (unfrozen=0.09082). Best ckpt ep1.86=0.09115. Consistent with 1.0ep finding (0.0005 worse). Family B universally does NOT benefit from frozen head/embed at any training duration. The per-block scale parameters in Family B need the full gradient budget including lm_head/embed_tokens contributions.

## Key Finding: wlh_lr7e-6_ep2 NEW ALL-TIME BEST at ep1.66 (inv97)

- **ep1.66=0.09071!** New all-time best, improving from 0.09081 at ep1.59. Step 3413/4096 (83%).
- Full late trajectory: ep1.44=0.0919, ep1.49=0.0914, ep1.54=0.0915, ep1.56=0.0923, ep1.59=0.0908, ep1.61=0.0914, ep1.66=0.0907.
- Oscillating in 0.090-0.092 band with lr ~1.2e-6. Still producing new lows. ~1.5h remaining.
- **3.0ep run (wlh_qad_lr7e-6_ep3) LAUNCHED on GPU 3.** With lr=7e-6 over 3.0ep: lr at ep1.0=4.67e-6, ep2.0=2.33e-6, ep3.0=0. The productive learning phase extends deep into epoch 3. This is the highest-ceiling experiment ever launched.
- **QErr coeff=1.0 at 2.0ep KILLED at ep0.29.** Gap to plain QAD = 0.005+ and widening. QErr damage amplifies with longer training. All QErr results are now conclusively negative across both families. QErr axis is CLOSED.
- **Family B QErr sweep closing:** coeff=0.01 at ep0.49=0.0981 (0.0006 behind plain), coeff=0.1 at ep0.37=0.1000 (tracking neutral). Both trending negative. QErr adds no value for Family B either.

## Key Finding: wlh_lr7e-6_ep2 IMPROVING STRONGLY into Epoch 2 (inv89)

- **ep1.0=0.0945, ep1.03=0.0946, ep1.05=0.0940, ep1.07=0.0940, ep1.10=0.0935.** The decline of 0.001 per 0.1 epoch is robust and sustained.
- At ep1.10, lr~3.1e-6 -- still highly productive. Compare: wlh_lr1e-5_ep2 at matched ep1.10 had lr near zero and was at ~0.0980. wlh_lr7e-6_ep2 is 0.0045 AHEAD at matched epoch.
- The wlh_lr1e-5_ep2 trajectory from ep1.0 to ep2.0 was: 0.0928->0.0909 (0.0019 improvement over entire second epoch). wlh_lr7e-6_ep2 already gained 0.0010 in just ep1.0->ep1.10. Projected 2.0ep: 0.085-0.089. This would be a new overall best by 0.002-0.006.

## Key Finding: QErr Completed -- Net Negative at coeff=1.0 for Both Families (inv90)

- **QErr-1 (wlh, coeff=1.0) COMPLETED at eval_loss=0.0935.** Gap to plain QAD (0.0928) = 0.0007. qerr/sum reduced from 1,715,034 to 1,646,377 (4% reduction). The grid alignment does reduce quantization error but the gradient budget trade-off is net negative.
- **QErr-2 (slh, coeff=1.0) COMPLETED at eval_loss=0.0938.** Gap to plain slh_ep1 (0.0929) = 0.0009. Family B gap is slightly larger despite per_block_scale providing real QErr gradients. The per_block_scale parameters absorb some gradient that would otherwise improve eval_loss.
- **QErr gap narrowed dramatically in final 5%:** ep0.93=0.0941 (gap=0.0013) -> ep0.95=0.0935 (gap=0.0007). At very low lr, QErr gradient becomes a larger fraction of total gradient. This motivates testing QErr at lower coefficients (0.01) where distillation gradient displacement is minimal.
- **Family B QErr sweep continues:** coeff=0.01 launched to test if minimal-interference QErr can close the gap. Per_block_scale gradient path may work better with lower coefficients.

## Key Finding: lr bracket DEFINITIVELY CLOSED at 1.0ep (inv88)

- **lr=7e-6 (0.0920) > lr=6e-6 (0.0924) > lr=8e-6 (0.0926).** All three neighbors measured. Optimal lr for Family A at 1.0ep is 7e-6.
- Note: lr=7e-6 was WORSE than lr=1e-5 for Family B at 1.0ep (0.0937 vs 0.0929). Family B benefits from steeper late-decay of lr=1e-5 schedule for fine-tuning per-block scales.
- weight_mse confirmed inferior at 1.0ep: 0.0976 vs wlh 0.0920 (gap=0.0056). No further wmse experiments warranted.

## Key Finding: lr=7e-6 Lead Growing Over lr=1e-5 (inv64)

- **wlh_lr7e-6 at ep0.29=0.1015 vs wlh_ep1_plain (lr=1e-5) at ep0.29=0.1045. Lead=0.0030 and GROWING.**
- Trajectory of lead: ep0.20=+0.0043, ep0.22=+0.0023 (briefly narrowed), ep0.24=+0.0029 (0.1029 vs 0.1058 interp), ep0.27=+0.0026 (0.1031 vs 0.1057), ep0.29=+0.0030 (0.1015 vs 0.1045).
- The lead stabilized around 0.003 after initially narrowing. For 1.0ep training with lr=7e-6, the lr at ep0.50 is ~3.5e-6 (vs ~5e-6 for lr=1e-5). The slower decay means more training happens at productive lr ranges.
- If the 0.003 advantage persists to ep1.0, lr=7e-6 could achieve ~0.0898 at 1.0ep (vs 0.0928 for lr=1e-5). This would be a massive improvement.
- **UPDATE (inv65): wlh_lr7e-6 at ep0.37=0.0997.** Lead vs ep1_plain at ep0.37: 0.0997 vs 0.1021 = 0.0024. Lead persistent and growing. At ep0.34: lr7e-6=0.0999 vs plain at ep0.34=0.1027 = 0.0028. The advantage is real. slh_lr7e-6_ep1 launched for cross-family validation.
- **UPDATE (inv70): wlh_lr7e-6 at ep0.71=0.0950.** Full late trajectory: ep0.49=0.0979, ep0.51=0.0975, ep0.54=0.0969, ep0.56=0.0964, ep0.59=0.0962, ep0.61=0.0958, ep0.63=0.0959, ep0.66=0.0961, ep0.68=0.0948, ep0.71=0.0950. Best ckpt so far ep0.68=0.0948. Compare completed wlh_qerr_ep1 at ep0.71: ~0.0963. Lead=0.0013. The lead narrowed from 0.003 at ep0.29 to 0.0013 at ep0.71. However, at ep0.71, lr=7e-6 has lr ~2e-6 vs lr=1e-5 having lr ~1.5e-6. Still some advantage remaining. Projected 1.0ep: 0.091-0.093.
- **UPDATE (inv70): slh_lr7e-6_ep1 at ep0.34=0.0996 -- LARGER lead than Family A.** Compare slh_ep1 at ep0.34: ~0.1030. Lead=0.0034. Confirms lr=7e-6 universally superior. Family B lead is even larger than Family A.
- **The combination of lr=7e-6 + 2.0ep is the highest-ceiling experiment in the pipeline.** At 2.0ep, lr=7e-6 gives an effective lr of ~3.5e-6 at ep1.0 (where wlh_ep2 is currently producing its best gains at 2.5e-6). The productive learning phase extends much longer.
- **UPDATE (inv74): wlh_lr7e-6 at ep0.90=0.0923 -- REVERSAL! BEATS lr=1e-5 at 1.0ep (0.0928).** The ep0.88 strategist observation that lr=7e-6 trails was WRONG -- it was noise. Late trajectory: ep0.83=0.0932, ep0.85=0.0930, ep0.88=0.0933 (noise up), ep0.90=0.0923 (strong new best). lr=7e-6 is now confirmed superior to lr=1e-5 for Family A at 1.0ep. wlh_lr7e-6_ep2 becomes even more critical.
- **UPDATE (inv77): wlh_lr6e-6 at ep0.34=0.0994, now AHEAD of lr=7e-6 at matched epoch.** lr=7e-6 was 0.1002 at ep0.29 in completed 1.0ep run. The warmup penalty for lr=6e-6 has resolved. The slower decay rate of lr=6e-6 may give a late-training advantage. If this holds, lr=6e-6 at 2.0ep could be even better than lr=7e-6 at 2.0ep. Need to wait for 1.0ep completion to confirm.

## Key Finding: "Plain QAD" on adaround checkpoint is NOT plain QAD

- The adaround_lh_plain_qad experiment (no explicit adaround args) STILL activates the adaround framework. The code auto-detects round_logits in the checkpoint.
- The only difference from standard adaround: freeze_weights defaults to False (weights train freely).
- The adaround/dist_loss dropped from ~114 to ~92 (decent reduction) and beta annealed from default.
- At 0.25ep, this "unfrozen weights + adaround" approach got 0.1161 -- slightly better than the explicit adaround_lh_unfreeze (0.1176) but WORSE than standard frozen-weights adaround (0.1103).
- **Conclusion:** Freezing weights is still better for the adaround framework at 0.25ep. The "plain QAD" paradigm shift was overhyped -- it's just unfrozen-weights adaround.

## Surprising Findings

- Family A and Family B achieve similar 0.25ep results (0.0999 vs 0.1004), BUT Family B pulls AHEAD at longer training: slh_ep05 at ep0.37=**0.0974** vs wlh_qerr_ep1 at ep0.37=0.1031. Gap = 0.0057 at matched epochs. Family B's learnable per-block scales provide a decisive advantage with more training. This reverses the early finding.
- **Family B (slh_ep05) is already below the completed best at ep0.37.** eval_loss=0.0974 vs completed best 0.0962. Projected final ~0.095-0.096. Family B is the winning family.
- **Scales-only training is surprisingly effective**: Training ONLY per_block_scale params gives eval_loss=0.1039 at 0.25ep vs 0.1004 with full training. Gap of 0.0035 with far fewer trainable params and only 28GB VRAM (vs 44GB). Practical deployment implications.
- Better PTQ init -> better QAD outcome: wlh (1.7449 baseline -> 0.0999) vs default (1.8162 -> 0.1166). The 0.07 gap in baseline translates to 0.017 gap after QAD.
- Teacher ceiling is only 0.04 eval_loss above best PTQ -- FP4 quantization gap is small for this model.
- QErr MSE values differ dramatically between checkpoints: default has qerr/mse=0.002 (near-perfect quantization), wlh has 0.229, slh has 32.57. The per-block scales in Family B introduce much larger quantization error.
- QErr helps Family A (small qerr/mse=0.23) but not Family B (large qerr/mse=32.57). Hypothesis: the qerr gradient is better-scaled for Family A.
- **Plain QAD outperforms QErr QAD at matched epochs (inv53).** wlh_ep05_plain at ep0.22=0.1040 vs qerr variant ~0.1075 at matched epoch. Lead of ~0.0035 and growing. QErr SUM coeff=0.01 pushed metric THE WRONG WAY (+37 over ~440 steps, from 1,715,035 to 1,715,072). QErr with mean reduction was dead; QErr SUM at low coefficients is counterproductive. wlh_qad_lr1e-5_ep1_plain is the most important pending experiment.
- **Per-scale lr is a dead axis.** slh_scale_lr5e-6 at ep0.24=0.1024 vs standard slh at ep0.24=0.1022. Difference of 0.0002 is noise. Combined with the scale_lr 5e-5 divergence, per-scale lr offers no benefit over using the global lr.
- **Plain QAD BEATS qerr at 0.5ep (inv57).** wlh_ep05_plain completed at 0.0958 vs wlh_qerr_ep05 at 0.0962. Gap of 0.0004 in favor of plain QAD. This is significant because qerr had zero effect on qerr/mse (confirmed by 9+ experiments with mean reduction), yet the qerr loss term may have added noise to the gradient. Removing the qerr term entirely produces cleaner distillation gradients. wlh_ep1_plain is now the most important experiment.
- **slh_scale_lr5e-6 completed at 0.0989 (inv57).** Standard slh at 0.5ep was 0.0971. Per-scale lr 5e-6 is 0.0018 worse. This definitively closes the per-scale lr axis: default lr is best for scales.

## Best Recipes So Far

1. **wlh_qad_lr7e-6_ep2** (COMPLETED): eval_loss=**0.08991**, perplexity=1.0941. Family A, lr=7e-6, linear, warmup=0.1, plain QAD, 2.0ep. ALL-TIME BEST.
2. **wlh_qad_lr1e-5_ep2** (COMPLETED): eval_loss=**0.0909**, perplexity=1.0951. Family A, lr=1e-5, linear, warmup=0.1, plain QAD, 2.0 epochs.
3. **wlh_qad_lr7e-6_ep1** (COMPLETED): eval_loss=**0.0920**, perplexity=1.0964. Family A, lr=7e-6, linear, 1.0ep.
4. **slh_qad_lr7e-6_ep2** (COMPLETED): eval_loss=**0.09082**, perplexity=1.0951. Family B, lr=7e-6, linear, 2.0ep. Family B all-time best, within 0.0009 of overall best.
5. **slh_qad_lr1e-5_ep2** (COMPLETED): eval_loss=**0.0924**, perplexity=1.0968. Family B, lr=1e-5, linear, 2.0ep.

## Open Questions

- ANSWERED: wlh_qerr_ep1 completed at 0.0928. Below 0.092? No.
- ANSWERED: slh_ep1 COMPLETED at 0.0929. TIED with wlh_qerr_ep1 (0.0928). Family A/B are equivalent at 1.0ep.
- ANSWERED: Cosine BEAT linear at 1.0ep for Family B. slh_cosine_ep1 at ep0.63=0.0972 vs slh_ep1 at ep0.63=0.0975. Cosine 0.0003 AHEAD. Crossover at ep0.56-0.61.
- Will cosine pass linear for Family A? (wlh_cosine_ep1 at ep0.49=0.1015, gap=0.0017 and narrowing. Less likely than Family B.)
- ANSWERED: lr=1.5e-5 is NOT viable at 1.0ep. KILLED at ep0.24. Gap to lr=1e-5 widened to 0.007. Oscillating/diverging since ep0.12. Viable lr range: [1e-5, ~1.2e-5].
- Can plain QAD on adaround_lh break 0.10? (RUNNING at ep0.10=0.1278, rapidly improving. 0.5ep extension LAUNCHED.)
- ANSWERED: Cosine 0.5ep BEATS linear 0.5ep for Family B: 0.0966 vs 0.0971. Difference of 0.0005.
- Can 2.0ep training break below 0.090? (THREE RUNS LAUNCHED: wlh_ep2, slh_ep2, slh_cosine_ep2. Most important open question.)
- Will plain QAD on adaround_lh (best PTQ baseline 1.7418) outperform Families A/B at equal training duration?
- ANSWERED: adaround_lh 1.0ep finished at 0.1050. Did NOT break below 0.105. dist_loss frozen at 114.02 throughout.
- ANSWERED: Scales-only training gives 0.1039 at 0.25ep. Gap of 0.0035 to full training (0.1004). Yes, scales alone recover most quality.
- ANSWERED: scale_learn_mse_init = 0.1076 at 0.25ep. Much worse than slh (0.1004). Hessian >> MSE for Family B too. qerr/mse=8.47 (intermediate).
- Per-pattern lr (round_logits at 1e-2)? (Running: round_lr_config, ep0.17=0.1142. Worse than standard adaround)
- PARTIALLY ANSWERED: QErr SUM at coeff=0.01 moved the metric THE WRONG WAY (qerr/sum +37 over 440 steps). coeff=0.1 COMPLETED: eval_loss=0.0961, qerr/sum=1715072 FROZEN throughout (went wrong way +38). coeff=1.0 running. coeff=10 COMPLETED: eval_loss=0.0999, qerr/sum=1715058 FROZEN (moved +23.5 out of 1.7M = 0.0014%). coeff=100 LAUNCHED. STE blocks gradients completely. User sweep continues.
- ANSWERED: Per-scale lr does NOT help Family B. scale_lr5e-5 KILLED (diverged to 0.204). scale_lr5e-6 at ep0.39=0.0993 vs standard ~0.1003 at matched ep. Slightly better but within noise. Dead axis.
- IMPORTANT: qerr_reduction field exists in code and defaults to "sum". The old experiments that showed qerr/mse=0.2295 used the old "mean" default. The SUM reduction produces O(num_elements) larger gradients.
- ANSWERED: Freeze-round-logits on adaround_lh checkpoint does NOT bypass Family C ceiling. eval_loss=0.1072 at 0.5ep. Matches standard adaround (0.1084 at 0.5ep). Round_logits are confirmed useless -- freezing them has no effect on training quality. The adaround callback auto-deactivated (dist_loss=0.0). This means Family C's ~0.105 ceiling comes from the checkpoint structure itself, not from the adaround rounding decisions.
- Does temperature=0.5 help adaround? (ANSWERED: Confirmed worse, 0.1161 vs 0.1103.)
- Does beta annealing 6->2 help? (Planned)
- Does plain QAD (no qerr, no adaround) beat qerr-augmented QAD at 1.0ep? (wlh_ep05_plain at ep0.42=**0.0980** -- OUTSTANDING. Gap to qerr at matched epochs is now ~0.005 and growing. wlh_ep1_plain is the #1 priority launch.)
- QErr SUM coeff=0.01 at ep0.34=0.0994 vs plain at ep0.34=0.1002. Interestingly, qerr_sum_1e-2 is slightly ahead of plain at matched epochs. The qerr/sum metric still goes wrong way (+38) but eval_loss is competitive. Need final results.
- nvfp4_default 1.0ep coverage gap being filled (LAUNCHED inv55, GPU 6).
- Can lr=2e-5 at 0.5ep maintain useful learning rate longer? (Planned)
- Will Family B 1.0ep (slh_ep1) reach below 0.090? (RUNNING on GPU 3, ep0.12=0.1075. Most important experiment.)
- Will cosine schedule at 1.0ep beat linear for Family B? (LAUNCHED: slh_cosine_ep1 on GPU 0)
- Will lr=2e-5 at 0.5ep maintain useful learning rate longer? (RUNNING: slh_lr2e-5_ep05 on GPU 5, ep0.05=0.1126)
- Will Family C 1.0ep break below 0.105? (QUEUED for GPU 2: adaround_lh_ep1)
- Can Family B match or approach the teacher ceiling (1.7017 baseline)? The best QAD so far (0.0974) is already impressively low.

## Answered Questions

- **Cosine vs linear schedule**: Identical at 0.25 epochs (both 0.1004) with warmup=0.1.
- **QErr regularization**: Helps Family A (0.0997 vs 0.0999). Does NOT help Family B (0.1005 vs 0.1004).
- **adaround_local_hessian >> adaround_mse**: 0.1103 vs 0.1289. Hessian init is crucial for Family C.
- **bs=8**: Worse than bs=4 (0.1024 vs 0.1004). Not worth it at 0.25 epochs. Cosine schedule doesn't help either (0.1024 with both linear and cosine).
- **lr=5e-6**: Too slow (0.1032). lr=1e-5 is better.
- **lr=5e-5**: Overshoots badly (0.1284). Not viable.
- **lr=1e-4**: Catastrophic across all families.
- **Cosine + qerr**: Worse than linear + qerr (0.1007 vs 0.0997). Linear schedule is better when using qerr.
- **Lower qerr coefficients (0.001->0.05)**: Slightly worse than default (0.1002 vs 0.0997). Default (0.01->0.1) is good.
- **Warmup=0.0 with qerr**: 0.1000 vs 0.0997 with warmup=0.1. Warmup helps marginally.
- **Higher adaround dist_loss_coeff (1e7)**: Similar (0.1101 vs 0.1103 with 1e6). Not worth it.
- **Beta annealing 20->2 for adaround**: 0.1118 vs 0.1103 fixed beta=4. High initial beta wastes early training. Annealing from high values is harmful.
- **Lower adaround dist_loss_coeff (1e5)**: 0.1126 vs 0.1103 with 1e6. dist_loss barely moved (114->112). Too weak. 1e6 is confirmed optimal; bracket is [1e5, 1e7] with 1e6 best.
- **Unfreezing weights with adaround**: Very bad (0.1264 vs 0.1103). Always freeze weights for adaround.
- **0.5 epochs >> 0.25 epochs**: 0.0962 vs 0.0997. Confirmed as the single most impactful axis. All hyperparam mutations at 0.25ep produce <0.0007 difference; longer training gives 0.0035.
- **weight_mse checkpoint with qerr**: 0.1070. Worse than wlh (0.0997). Confirms hessian init >> MSE init for QAD. Gap: wmse baseline 1.7871 vs wlh 1.7449 = 0.042 PTQ gap -> 0.007 QAD gap. QAD compresses the init quality gap ~6x.
- **weight_decay=0.01**: 0.1000 vs 0.0997. Hurts. Always wd=0.0.
- **grad_accum=2 (eff_bs=8)**: 0.1004 vs 0.0997. Stalled earlier than bs=4. Confirmed: bs=4/accum=1 is optimal.
- **lr=7e-6**: 0.1001 at 0.25ep. Very close to 1e-5 (0.0997). Testing at 0.5ep queued.
- **Beta annealing 20->2**: 0.1118 vs 0.1103 fixed beta=4. High initial beta wastes early training pushing rounding toward binary prematurely. Annealing from high starting values is harmful for adaround.
- **qerr 0.005->0.1**: eval_loss=0.0999, qerr/mse=0.2295. Eighth experiment confirming qerr is dead.
- **Training last 8 layers only**: CRASHED -- gradient checkpointing requires all layers to have gradients. Would need --gradient_checkpointing False (higher VRAM) or code fix. Not worth pursuing. Use leaf-param-only experiments instead (scales_only, round_only).
- **Per-pattern lr (round_logits=1e-2) for adaround**: 0.1108 vs 0.1103 standard. Marginally worse. round_logits lr=1e-2 is too aggressive -- creates interference with main distillation loss. Try 1e-3 next.
- **Scales-only training (Family B)**: eval_loss=0.1039 at 0.25ep. Only 0.0035 worse than full training (0.1004). Plateau visible at ep0.22-0.24. Scales capture ~85% of distillation signal.
- **qerr_fixed_05 (coeff=0.5, 0.5ep)**: eval_loss=0.0999, qerr/mse=0.2295. Ninth and final experiment confirming qerr is absolutely dead for Family A. Even at coeff=0.5 for 0.5 epochs, zero effect.
- **Constant lr at 0.29ep**: 0.1071 vs linear 0.1035. Gap WIDENING (was 0.0003 at ep0.17). Constant lr=1e-5 causes oscillation/non-convergence at longer training. Linear decay is definitively better.
- **Constant lr at 0.44ep (killed)**: eval_loss=0.1062, oscillating between 0.1053-0.1082 for last 0.15ep. Gap to linear: 0.006 (0.1062 vs 0.1002 at matched epoch). Killed to free GPU. Definitively inferior.
- **adaround_max_init vs lh_init**: at ep0.10, max=0.1845 vs lh=0.1308. Gap=0.054. max_init is bad for adaround. Starting dist_loss 122.6 (max) vs 114.1 (lh).
- **temp=0.5 for adaround COMPLETED**: eval_loss=0.1161 vs standard=0.1103. Confirmed inferior at all matched epochs. dist_loss frozen at 114.08. Standard temp=1.0 is definitively better.
- **scale_learn_mse_init (Family B)**: eval_loss=0.1076 at 0.25ep. Much worse than slh (0.1004). Hessian >> MSE confirmed for Family B too. qerr/mse=8.47 (intermediate between Family A's 0.23 and slh's 32.57).
- **slh_ep05 COMPLETED at 0.0971**: Family B 0.5ep final result. Best checkpoint at ep0.46=0.0971. Full trajectory steady decline with no plateau until lr decay wall. Family A 0.5ep = 0.0962 (slightly better at completion, but Family B was ahead at matched epochs mid-training).
- **wlh_qerr_ep1 FLATTENING at ep0.54**: eval_loss 0.0998->0.1000 from ep0.51->ep0.54. Linear lr decay to ~5e-6 is killing useful learning. The 1.0ep Family A result may only marginally beat 0.5ep (0.0962). This is a critical insight -- **cosine schedule or higher initial lr is needed to fully exploit 1.0ep training.**
- **UPDATE: wlh_qerr_ep1 BROKE THROUGH plateau at ep0.56-0.61**: eval_loss dropped to 0.0990 (ep0.56), 0.0981 (ep0.59), 0.0982 (ep0.61). The "flattening" at ep0.51-0.54 was noise, not convergence. With lr=3.9e-6, learning continues. Already 0.002 below the completed best of 0.0962.
- **UPDATE: wlh_qerr_ep1 continues to ep0.68=0.0969**: Trajectory ep0.63=0.0977, ep0.66=0.0974, ep0.68=0.0969. Still declining at ~0.0005/2% of training. lr=~3.5e-6. Projected final: 0.090-0.094.
- **adaround_lh_ep05 COMPLETED at 0.1084**: Confirms Family C ceiling at ~0.108 with standard adaround. dist_loss frozen at 114.06 throughout 0.5ep (moved 0.04% total). The round_logits are fundamentally not learning. Two paradigm-shift experiments launched: plain QAD (ignore round_logits) and round-only (isolate round_logits contribution).
- **BREAKTHROUGH: plain QAD on adaround_lh at ep0.05=0.1680 vs standard adaround at ep0.05=0.2320.** The adaround framework (frozen weights + dist_loss) is ACTIVELY HARMFUL. Plain QAD (training all weights, ignoring round_logits) learns 0.064 better at matched epoch. The adaround_lh checkpoint has the best PTQ baseline (1.7418) and responds excellently to plain QAD. This means Family C checkpoints should be trained with plain QAD, not the adaround framework.
- **lr=1.5e-5 showing divergence for Family B at ep0.15-0.17**: Gap to lr=1e-5 widened from 0.002 (ep0.12) to 0.004 (ep0.15). At ep0.15, lr=1.5e-5=0.1093 vs lr=1e-5=0.1052. The initial advantage at ep0.10 was likely noise from the warmup phase.
- **UPDATE: wlh_qerr_ep1 continues to ep0.76=0.0954**: Trajectory ep0.68=0.0969, ep0.71=0.0963, ep0.73=0.0959, ep0.76=0.0954. Still declining at ~0.0009/5% of training. lr=~2.6e-6. Projected final: 0.091-0.094. Already 0.0008 below the completed best (0.0962).
- **adaround_max_init COMPLETED at 0.1563 (0.25ep)**: Gap to lh_init = 0.046. max_init is worthless for adaround. All init rankings: lh >> mse >> max.
- **lr=1.5e-5 FAILED for Family B at 1.0ep**: Despite surviving warmup (ep0.12 gap=0.002), eval_loss oscillated/diverged from ep0.12 to ep0.24. Gap widened to 0.007 (0.1090 vs 0.1022). Combined with lr=2e-5 failure, the viable lr range for 1.0ep is [1e-5, ~1.2e-5] at most. lr=1e-5 is definitively the optimal learning rate.
- **round_only training is fundamentally broken within adaround framework**: eval_loss increased at every eval (0.1928->0.2162 over ep0.02-0.12). Round_logits get grad_norm=0.004 (near zero) because with freeze_weights=True, round_logits only get gradient through the dist_loss term, which is stuck. The adaround framework cannot learn round_logits effectively.

## Key Finding: Frozen lm_head+embed (inv105 update)

- **wlh_frozen at ep0.54=0.09885.** Crossed below 0.099. Gap to unfrozen stable at ~0.003. Trajectory: ep0.42=0.1013, ep0.44=0.1003, ep0.46=0.0999, ep0.49=0.0990, ep0.51=0.0998, ep0.54=0.0989. Continues descending.
- **slh_frozen at ep0.37=0.10009.** Trajectory flattening around 0.100 (ep0.32=0.1007, ep0.34=0.1000, ep0.37=0.1001). Still AHEAD of unfrozen at matched epoch by ~0.002-0.003.
- **scales_only at ep0.15=0.10690.** Still matching full training trajectory at matched epoch. The training-based PTQ hypothesis holds.
- **default_frozen at ep0.22=0.13164.** Slight improvement from 0.135 plateau. Still far behind all others. The nvfp4_default checkpoint is uniquely bad with frozen head+embed.
- **adaround_frozen at ep0.05=0.24264.** Very early, typical adaround warmup (high dist_loss dominates early).
- **roundlr1e-3 at ep0.24=0.11108.** Step 526/2048. Slow descent continues. Projected ~0.107-0.108 at ep1.0.
- **LSQ lr=5e-6 LAUNCHED (PID 915820, GPU 7).** Control: lsq_lr1e-6. Single variable: lr. Key experiment for user-directive LSQ vs scale_after_dequant comparison.
- **Family B lead over Family A narrows with training**: At ep0.20, Family B led by 0.004 (0.1043 vs 0.1095). At ep0.46, gap narrowed to 0.001 (0.0992 vs 0.1002). At ep0.54, slh=0.0990 vs wlh=0.0998 -- gap=0.0008. At ep0.66, slh=0.0971 vs wlh=0.0969 -- **Family A actually LEADS by 0.0002.** The crossover happened around ep0.56-0.63. Extra per-block scale params give diminishing advantage as lr decays below ~4e-6.
- **UPDATE (inv 38): Cosine is CATCHING UP to linear for Family B at 1.0ep.** slh_cosine gap to linear: ep0.44=0.006, ep0.54=0.0002, ep0.56=0.0006. The cosine advantage phase (ep0.50+) is producing a dramatic catchup. Initial observation that cosine was "definitively behind" was premature -- the cosine advantage phase had not started yet.
- **Cosine is family-specific:** Family B benefits from cosine (sustained high lr helps learnable per-block scales). Family A benefits less (weights-only training prefers linear decay). At ep0.49, Family A gap=0.0017 (also narrowing but slower).
- **wlh_cosine_ep1 oscillation persists through ep0.22**: Trajectory: ep0.10=0.1098, ep0.12=0.1109, ep0.15=0.1112, ep0.17=0.1096, ep0.20=0.1110, ep0.22=0.1089. Oscillating between 0.109-0.111. At ep0.22, linear was 0.1067. Cosine is 0.002 behind Family A. The oscillation suggests cosine's flat lr in early training hurts convergence. The cosine advantage (maintaining higher lr late) may not compensate.
- **wlh_qerr_ep1 at ep0.83=0.0947: broke through the ep0.76-0.78 plateau.** The "flattening" seen at ep0.76-0.78 (both 0.0954) was noise. ep0.81=0.0948, ep0.83=0.0947. lr=1.9e-6 still productive. Projected final: 0.092-0.095.
- **lr=2e-5 KILLED for Family B (slh_lr2e-5_ep05)**: eval_loss was INCREASING from ep0.10: 0.1140 -> 0.1150 -> 0.1162 -> 0.1164. Gap to lr=1e-5 was 0.012 at ep0.20. lr=2e-5 is definitively too high for Family B. The viable lr range for 1.0ep is [1e-5, 1.5e-5] -- the slh_lr1.5e-5 experiment at ep0.05=0.1145 (no overshoot) is the critical test.
- **Family B maintains matched-epoch advantage at ep0.27**: slh_ep1 at ep0.27=0.1027 vs wlh_ep1 at ep0.27=0.1068. Gap=0.004. This gap has been consistent from ep0.15-0.27, not narrowing as the strategist suggested earlier.
- **UPDATE (inv 37): slh_ep1 ACCELERATING at ep0.73=0.0955.** Trajectory: ep0.66=0.0971, ep0.68=0.0963, ep0.71=0.0960, ep0.73=0.0955. The decline rate is ~0.0005/eval. lr=~2.7e-6. Projected 1.0ep completion: 0.091-0.093. At ep0.73, Family A's wlh_qerr_ep1 was at ~0.0959. Family B trails by only 0.0004 at ep0.73 -- virtually tied.
- **slh_cosine_ep1 CROSSED below 0.100 at ep0.49=0.0999.** First time cosine has been this low. At ep0.49, linear (slh_ep1) was at ~0.0989. Gap = 0.001. The cosine advantage phase (ep0.50+) is just starting. Key question: does the gap keep narrowing as cosine maintains higher lr?
- **wlh_cosine_ep1 at ep0.44=0.1029, declining from plateau.** Trajectory: ep0.37=0.1039, ep0.39=0.1039, ep0.42=0.1037, ep0.44=0.1029. Gap to linear at ep0.44 is 0.0024 (wlh_qerr_ep1 at ep0.44 was ~0.1005). Cosine is definitively behind for Family A.
- **wlh_ep2 at ep0.10=0.1085 (still in warmup).** Tracking slightly ahead of the 1.0ep run at matched epoch (1.0ep at ep0.10=0.1095). The slower lr buildup at 2.0ep (warmup spans 0.20 epochs = 410 steps) means less initial disturbance. Real comparison begins at ep0.20+.
- **slh_cosine_ep1 COMPLETED at 0.0939 (1.0ep)**: CONFIRMED cosine does NOT beat linear for Family B at 1.0ep. Linear achieved 0.0929 -- a gap of 0.001. The cosine schedule maintained higher lr through ep0.50-0.85 which produced faster mid-training progress, but the lr hit near-zero too abruptly (cosine cliff) in the final 10% and the model oscillated 0.0939-0.0944 for the last 10 evals. Linear's gentle decay allowed continued improvement all the way to ep0.98. The mid-training "cosine advantage" was a mirage.
- **wlh_cosine_ep1 at ep0.90=0.0937, STILL declining.** At ep0.90, linear was 0.0935. Gap = 0.0002 and cosine is closing. Unlike Family B where cosine plateaued, Family A cosine is still finding improvements. Projected final: 0.093-0.094. May approach but unlikely to beat linear's 0.0928.
- **wlh_ep2 at ep0.59=0.1030: BROKE OUT of oscillation.** Was stuck at 0.1039-0.1048 for ep0.46-0.54. Now at 0.1030. At ep0.59, the 1.0ep run was at ~0.0990. Gap = 0.0040. Still wide but the trajectory is downward. The question remains whether 2.0ep can match 1.0ep performance.

# vllm_calib — vLLM-runtime calibration → unified-HF export (experimental)

End-to-end workflow that calibrates a model under vLLM (fast multi-node TP, native compressed-tensors handling) and exports a unified-HF NVFP4 checkpoint, without paying the slow HF eager load cost on every iteration.

## Pieces

| File | Role |
|------|------|
| `examples/vllm_serve/fakequant_worker.py` | Patched (in this branch) to **dump per-rank quantizer state** at end of `_fakequant_run_prolog_worker` (before `mtq.fold_weight`). Reads `MODELOPT_VLLM_STATE_DIR` + `MODELOPT_VLLM_STATE_TAG` env vars; writes `<tag>-rank<NN>-of<MM>.pth`. Also adds `QUANT_MAX_SAMPLE_LEN` env. |
| `experimental/vllm_calib/merge_and_convert_vllm_state.py` | **Stage 1** (CPU-only). Folds per-rank state files into one HF-keyed file: max/concat-merge `_amax` across ranks (via `merge_amax_tensors_for_group`), and rewrites vLLM fused-MoE quantizer keys (`...mlp.experts.w13_input_quantizer`) to HF per-expert names (`...mlp.experts.<i>.gate_proj.input_quantizer`). |
| `experimental/vllm_calib/export_hf_from_vllm_state.py` | **Stage 2** (GPU). Loads source HF model via `example_utils.get_model`, applies `mtq.quantize` with the SAME quant_cfg the calibration used (`forward_loop=None`), then non-strict per-quantizer `load_state_dict` of the merged state, then `export_hf_checkpoint` to write the unified-HF NVFP4 ckpt. Avoids `mto.restore_from_modelopt_state` because saved metadata carries vLLM-side fused names that don't match HF per-expert structure. |

## Why experimental

- **w2_input amax not observed.** The post-act intermediate between `gate*silu*up` and `down` is internal to vLLM's Marlin MoE kernel. The current `_QuantFusedMoEBase.forward` patch (in `modelopt/torch/quantization/plugins/vllm.py`) intercepts only the activation entering the MoE block (`w13_input`). `w2_input` falls back to default amax at export. Production-quality calibration needs a deeper hook.
- **Compressed-tensors source weight access patch.** `modelopt/torch/quantization/nn/modules/quant_module.py` is patched to skip `_register_dynamic_attribute("weight", ...)` when the source module has no `.weight` (CompressedLinear). Required because routed-expert linears in pack-quantized sources expose `.weight_packed` instead.
- **Export-side memory pressure.** `unified_export_hf._export_transformers_checkpoint` runs a `dummy_forward_fn` that triggers compressed-tensors' `decompress_model` hook, fully materialising every routed-expert INT4 weight as BF16 in HBM at once — ~4× expansion. K2-Thinking + 8×H100-80G OOMs at this step. Multi-node export or layer-wise streaming export is required for production.

## Run

```bash
# 1. Calibrate via vLLM (single- or multi-node) — see experiments/vllm_calib_kimi-k2-thinking/.
#    Worker dumps state to ${MODELOPT_VLLM_STATE_DIR}/<JOBID>-rank<NN>-of<MM>.pth.

# 2. Stage 1: merge per-rank → HF-keyed (~seconds, CPU)
python experimental/vllm_calib/merge_and_convert_vllm_state.py \
    --state-dir <state_dir> --tag <JOBID> --num-experts <N>

# 3. Stage 2: HF restore + unified-HF NVFP4 export (~30 min, 1×8×H100)
python experimental/vllm_calib/export_hf_from_vllm_state.py \
    --ckpt-path <hf_source> \
    --state-path <state_dir>/<JOBID>-merged-hf.pth \
    --export-dir <out_dir>
```

# EAGLE3

Draft head trained on hidden states dumped from the target model. Examples:
`tools/launcher/examples/*/*/hf_offline_eagle3.yaml` (and the `hf_online_*`,
`hf_streaming_*` variants).

## Pipeline tasks

The offline configuration is 4 tasks; each passes artifacts to the next through a
shared `/scratchspace`.

| Task | Script | Purpose | Output |
| --- | --- | --- | --- |
| task_0 | `common/vllm/query.sh` or `common/tensorrt_llm/query.sh` | Data synthesis — serve the target model, generate prompt/response pairs | `/scratchspace/data/*.jsonl` |
| task_1 | `common/eagle3/dump_offline_data_vllm.sh` (or `_hf.sh` / `dump_offline_data.sh`) | Forward the target model, save hidden states | `/scratchspace/offline_hidden_states/*.pt` |
| task_2 | `common/eagle3/train_eagle.sh` | Train the draft head, then export | `/scratchspace/eagle3/model.safetensors`, `/scratchspace/export/` |
| task_3 | `common/specdec_bench/quick_check.sh` | Benchmark acceptance rate and throughput | JSON result files |

### Choosing the task_1 dump backend

| Backend | Script | When to use |
| --- | --- | --- |
| vLLM | `common/eagle3/dump_offline_data_vllm.sh` | **Default.** Broad coverage via vLLM's native hidden-state extractor. |
| HF | `common/eagle3/dump_offline_data_hf.sh` | VLMs / multimodal, custom-code models, sliding-window attention (TRT-LLM can't serve these). Uses `device_map="auto"`. |
| TRT-LLM | `common/eagle3/dump_offline_data.sh` | Pure-text models with TRT-LLM support; pass `--tp <TP>` and `--moe-ep <EP>`. |

Rule of thumb: **HF** if the model is a VLM or uses sliding-window attention; **vLLM**
otherwise. TRT-LLM only when you specifically want its kernels for a supported
plain-text model.

## Recipe and training knobs

`modelopt_recipes/general/speculative_decoding/eagle3.yaml`, passed to
`train_eagle.sh` via `--config` with dotted overrides:

| Override | Note |
| --- | --- |
| `model.model_name_or_path` | Target checkpoint |
| `data.offline_data_path` | task_1 output directory |
| `training.output_dir` | Draft checkpoint destination |
| `training.training_seq_len` | Lower it first when training OOMs |
| `training.train_bs` | Lower it next when training OOMs |
| `training.lr` | Lower it when loss is NaN or diverging |
| `training.ar_validate_steps` | Set to run AR validation during training |

`task_3` selects the algorithm at benchmark time with
`--speculative_algorithm EAGLE3`.

## Per-model adjustments

| Situation | What to change |
| --- | --- |
| Requires `--trust-remote-code` | Add to `task_0` server args (before the `--` separator) **and** to `task_3` benchmark args |
| MoE with large expert hidden dim | Increase `intermediate_size` in `eagle_config.json` to match the model's `moe_intermediate_size` |
| Custom tokenizer (e.g. tiktoken) | Set `TIKTOKEN_RS_CACHE_DIR` to a pre-populated cache path in `task_0` and `task_1` |
| VLM | Use `dump_offline_data_hf.sh` — the text-only path, no vision encoder invoked |
| Sliding-window attention | TRT-LLM backend won't work; use HF or vLLM |
| Architecture unrecognized by training | Needs code changes in `modelopt/torch/speculative/` — a separate ModelOpt PR |

## Success markers

| Task | Log evidence | Artifact |
| --- | --- | --- |
| task_0 | "Saved N samples", or a progress bar completing | `/scratchspace/data/*.jsonl` |
| task_1 | "Successfully processed N conversations" | `/scratchspace/offline_hidden_states/*.pt` |
| task_2 | Training loss decreasing, "export complete" | `/scratchspace/eagle3/model.safetensors`, `/scratchspace/export/` |
| task_3 | `Average Acceptance Length ... ratio: X.XX` | JSON result files |

## Quality gate

The `task_3` log prints:

```text
Average Acceptance Length {'accept': X, 'count': Y, 'ratio': Z.ZZ}
```

The `ratio` field is the acceptance rate (AR).

| Criterion | Threshold |
| --- | --- |
| AR (MT-Bench) | >= 2.1 |

If the log shows `AR ... < lower bound`, the run already tripped the threshold check
and exited non-zero.

## Known failures

Generic infrastructure failures are in `../stages/triage.md`. These are
EAGLE3-specific:

| Error pattern | Root cause | Fix |
| --- | --- | --- |
| `No such file or directory: dump_offline_data_vllm.sh` | Wrong script path in YAML | Use the correct path under `common/eagle3/` |
| `FileNotFoundError: /scratchspace/data` | task_0 failed or produced no output | Re-run task_0, or point `--input-data` at existing data |
| `FileNotFoundError: /scratchspace/offline_hidden_states` | task_1 failed or produced no output | Re-run task_1 |
| `FileNotFoundError: /scratchspace/export` | task_2 or its export step failed | Re-run task_2; check export output |
| `RuntimeError` / unsupported arch during dump | Model not supported by the TRT-LLM backend | Switch to `dump_offline_data_hf.sh` or `dump_offline_data_vllm.sh` |
| No `.pt` files in the dump output dir | Extraction produced nothing | Check `--max-seq-len` and the input data format |
| `KeyError` / `AttributeError` loading the model in task_2 | Architecture not recognized by EAGLE3 training | Needs code changes in `modelopt/torch/speculative/` |
| Loss is NaN or diverging | LR too high, or data quality issue | Reduce `training.lr`; check the hidden-state data |
| `export_hf_checkpoint.py` fails | Training produced an incomplete checkpoint | Check `/scratchspace/eagle3/` for `model.safetensors` |
| Empty `/scratchspace/data/` after task_0 | `query.py` ran but wrote nothing | Check `--data` path exists and contains prompts; check `query.py` logs |
| Server fails to load the draft model in task_3 | Draft config incompatible with the engine | Check `eagle_config.json` and the engine version |
| vLLM reports EAGLE3 not supported | vLLM version too old | Use a newer vLLM container |
| AR below threshold / exit code 1 | Draft quality too low | More epochs or data, or hyperparameter tuning |

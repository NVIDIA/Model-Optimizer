# IQ Bypass

Bypass repairs an existing ModelOpt IQ student on frozen teacher inputs at each selected feed-forward boundary. It updates native scale fields, preserves element codes, and returns the same finalized student. Teacher weights, routing gates, and other formats stay frozen.

```python
import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto

# student is your existing ModelOpt PTQ result; teacher has separate original weights.
student = mtd.bypass(student, teacher, train_batches)
mto.save(student, "student.pt")
```

The default recipe is 512 global updates, Lion with betas `(0.9, 0.99)` and zero weight decay, global batch 8, microbatch 1, 32,768-token inputs, peak `max_lr=0.001`, 5% warmup, cosine decay to zero, gradient clipping at 1, and seed 42. Warmup is `ceil(num_steps * warmup_fraction)`. Each boundary uses MSE normalized over all valid feature elements in the complete global batch. Finalization keeps the final state.

The Transformers integration requires a **stable 5.23.x** release and is qualified in its own CI environment. ModelOpt's existing `[hf]` extra and general CI matrix retain their current Transformers range. Ordinary callable Torch models can use Bypass without Transformers.

## Run the tutorial

Install from this checkout in a separate environment with a compatible Torch/CUDA build:

```bash
pip install -e . -r examples/bypass/requirements.txt
```

The example installs its HF dependencies directly. Combining it with `[hf]` or `[all]` would request incompatible Transformers versions. ModelOpt's general version notice may still appear on 5.23; this stack qualifies the Bypass paths separately.

Prepare data once with the target model's tokenizer. The default source is `nvidia/Nemotron-Cascade-2-SFT-Data`, using its eight configurations and native chat templates, including reasoning and tools.

```bash
python -m examples.bypass.prepare_data \
  --tokenizer YOUR_MODEL --output packed-cascade

python -m examples.bypass.bypass \
  --model YOUR_FLOATING_MODEL --data packed-cascade \
  --format iq2_xxs --output repaired
```

The first command prepares 4,096 unique training packs plus separate calibration, validation, and diagnostic packs. It pins source revisions and tokenizer provenance, keeps conversation splits stable across training budgets, and refuses insufficient data. A source is never cycled or silently dropped from the blend. Relative token weights are:

| Source | Weight |
| --- | ---: |
| chat | 5620 |
| math | 2110 |
| science | 1090 |
| instruction_following | 330 |
| conversational_agent | 330 |
| terminal_agent | 330 |
| swe | 180 |
| safety | 2 |

The second command loads a complete floating teacher, runs ordinary ModelOpt PTQ on the selected feed-forward weights, calls `mtd.bypass`, and writes `student.pt` and `diagnostics.json`. Supply `--quant-config ptq.json` to use another ordinary ModelOpt quantization configuration. The tutorial's default PTQ algorithm is `max`; Bypass itself does not choose or rerun PTQ. The CLI uses one device; the native parallel API recipe is in [ADVANCED.md](ADVANCED.md).

Diagnostics report full-vocabulary `KL(teacher || student)`, next-token cross entropy/perplexity, exact token counts, projection mode, time, and peak allocated GPU memory. KL covers all valid input tokens, including prompts and document-final tokens. Perplexity covers only within-document next-token targets. Standard heads use vocabulary and token chunks; custom or quantized heads keep their real forward with token chunks.

Restore the artifact with the original architecture:

```python
import modelopt.torch.opt as mto

student = mto.restore(fresh_original_architecture, "repaired/student.pt")
```

Supported unsharded deployment/export remains the existing ModelOpt workflow. The saved student contains its quantization state and native payloads; the teacher and temporary Bypass fields are absent.

## Select layers and formats

By default, Bypass selects all eligible, disjoint MLP and MoE boundaries containing IQ weights. A model may mix dense and MoE blocks. Shared experts belong to their enclosing MoE. Routing gates stay frozen.

```python
mtd.bypass(student, teacher, train_batches, {"layer_indices": list(range(k))})
mtd.bypass(student, teacher, train_batches, {"modules": ["model.layers.0.mlp"]})
```

`layer_indices` selects logical decoder positions. `modules` uses exact module paths. When supplied together, the selectors intersect. Shared weights, overlapping owners, or student storage aliased with the teacher are refused before updates.

Supported trainable formats are `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, and `iq2_s`. Assign formats using ordinary ModelOpt quantizer rules. Frozen FP8 and NVFP4 projections preserve input gradients; Bypass never adds them to Lion. See the mixed-format example, trimming, native parallelism, resume, and extension contracts in [ADVANCED.md](ADVANCED.md), and source/loading qualifications in [MODEL_COVERAGE.md](MODEL_COVERAGE.md).

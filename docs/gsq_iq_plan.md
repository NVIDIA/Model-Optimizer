# GSQ implementation plan for IQ quantization

Implement a bounded Gumbel Softmax quantization (GSQ) refinement of the existing GPTQ IQ quantization path. Use GPTQ as the sole initialization, preserve every native IQ representation, and compare the selected result with its paired GPTQ baseline on Qwen/Qwen3.8-27B. The work will establish implementation correctness and measure perplexity; it does not presume an accuracy improvement.

## Scope and fixed settings

The reference model is the official BF16 Qwen/Qwen3.8-27B checkpoint. Restrict quantization to the linears already eligible under the existing IQ scope, currently the MLP linears. Do not expand the scope to attention, vision, embeddings, or the output head. Run calibration, refinement, validation, and final evaluation on CUDA GPUs.

| Format | Native candidate bundle | Candidate limit |
| --- | ---: | ---: |
| IQ1_S | 32 weights | K <= 5 |
| IQ1_M | 16 weights | K <= 5 |
| IQ2_XXS | 32 weights | K <= 5 |
| IQ2_XS | 16 weights | K <= 5 |
| IQ2_S | 16 weights | K <= 5 |

Bundle size follows the native format's shared fields. It is fixed per format and is not a search dimension. Keep other settings fixed within a format; compare the default 200-update run with at most one 500-update alternative.

## Native candidates and shared scale

Run the existing GPTQ initializer once per eligible linear and format. Save its exact packed representation and dequantized weights as the baseline. Do not add RTN, random, or other initialization branches.

Each candidate is a complete legal native bundle of 16 or 32 weights. It includes all native field types needed by that format: codebook or grid selections, signs where encoded, and every format-specific scale or auxiliary field. Construct a fixed set of at most five candidates per bundle, including the exact GPTQ bundle. Keep the candidate set fixed during optimization. Deduplicate identical native encodings and permit fewer than five candidates.

Preserve the native superblock layout, field widths, sharing relationships, and packing rules. Train the shared FP16 scale d as a relative adjustment to its GPTQ value, at the native unit that owns d. Do not introduce independent scales where the format shares one. For hard scoring and export, materialize d in its native FP16 representation and reconstruct weights with the production dequantizer.

The exported model contains only legal native IQ fields and the existing shared FP16 d. Logits, soft mixtures, optimizer state, and candidate tables are training artifacts and do not change the stored quantization format or its effective bit rate.

## Objective and optimization

Optimize each eligible linear independently. For input activation rows X, form the original undamped covariance C = X transpose X / N. For reference BF16 weights W and reconstructed weights W_hat, minimize the covariance-weighted reconstruction error:

    L = trace((W_hat - W) C (W_hat - W) transpose) / n_out

This is the mean squared output error over the calibration activations, with a consistent normalization. Use the undamped covariance for GSQ loss and hard checkpoint selection. Keep it separate from any damping, factorization, or other numerical treatment used internally by the GPTQ initializer.

Use Gumbel Softmax to train the discrete candidate choices and Lion to update the candidate logits and relative scale parameters. Fix the temperature schedule, learning rates, random seed, and other optimizer settings for a format before comparing update budgets. Record those settings with the run.

At step 0, every 25 optimizer updates, and the final update, materialize hard candidate choices and native FP16 d, then score the same undamped objective. Bound memory by processing independent output-row chunks and retaining the best evaluated hard checkpoint for each chunk. Step 0 is the exact GPTQ representation. Handle ties deterministically and keep the earlier state.

Assemble the selected chunks into one complete legal native state for each linear. Output rows share neither candidate choices nor native scales, and the covariance objective is the sum of their reconstruction errors. Therefore the assembled state has exactly the sum of the selected chunk errors, and its error cannot exceed the sum of their GPTQ starting errors. Different chunks may select different optimizer steps; no common iteration is required. Record the resulting complete-linear error with output-row normalization, and restore this assembled state for native checkpoint saving and evaluation.

This safeguard bounds the selected local objective against its stored GPTQ starting state, subject to numerical tolerance. It does not guarantee lower model perplexity, since independent local reconstruction objectives do not determine end-to-end accuracy.

## Data amount and optimizer updates

| Quantity | Fixed value | Purpose |
| --- | --- | --- |
| Calibration source | WikiText 2 training split | Capture activations and covariances |
| Calibration sequences | 128 sequences | Fixed sample set |
| Sequence length | 2,048 tokens | Fixed calibration window |
| Calibration token total | 262,144 tokens | 128 times 2,048 |
| Default refinement budget | 200 optimizer updates | Update choices and relative scales |
| One optional alternative | 500 total optimizer updates | Compare a longer run |
| Validation source | WikiText 2 validation split | Select update budget per format |
| Final evaluation | Complete WikiText 2 test split at 2,048 tokens | Paired perplexity comparison |

The 128 calibration sequences describe the data amount. The 200 or 500 optimizer updates describe repeated optimization over the captured statistics; they do not mean 200 or 500 new calibration sequences. The longer alternative has 500 total updates, not 500 additional updates after 200. Use the same calibration samples, tokenizer, and covariance construction for the two budgets.

Choose between 200 and the optional 500-update result independently for each IQ format using WikiText 2 validation perplexity. Keep the test split out of candidate generation, optimization, checkpoint selection, and budget selection. Use the same deterministic tie rule for budget selection and prefer 200 updates when validation perplexity is tied within the stated tolerance.

## Paired evaluation and reporting

Keep model and dataset revisions, calibration data, and scoring settings fixed across each paired comparison.

Evaluate the exact GPTQ baseline and the validation-selected GSQ checkpoint for each of the five formats on the full WikiText 2 test split using 2,048-token evaluation windows. Match tokenization, split revision, windowing, token coverage, precision, model scope, and evaluation code between each pair. Record the number of scored tokens and treatment of the final partial window so full-test coverage is auditable.

Report absolute baseline and GSQ test perplexity, the paired difference, validation perplexity used for selection, selected update budget, local reconstruction loss, and runtime. Also record the model and dataset revisions, checkout commit, hardware, environment versions, quantization scope, calibration seed, and optimizer settings. Do not add other models, benchmarks, broad sweeps, or an open-ended search for an accuracy gain.

## Implementation and verification

1. Inspect the current IQ candidate construction, dequantization, packing, and eligible-linear selection before changing behavior.
2. Add native candidate adapters for the five formats. Keep the existing GPTQ path available as the paired baseline and exact step 0. Share common GSQ logic while keeping native field ownership explicit.
3. Implement per-linear undamped covariance loss, Gumbel Softmax choices, Lion updates, relative d training, hard scoring, and best-state restoration. Expose only the settings needed for this bounded experiment.
4. Add focused tests for legal native candidates and exact step 0, covariance loss versus explicit output MSE, native export/reload equivalence, and restoration of the best evaluated hard checkpoint.
5. Run targeted GPU smoke checks on CUDA GPUs for every format, then the bounded calibration and validation runs. Verify finite losses, legal exported values, and export/reload equivalence before final evaluation.
6. Review the change against this specification and repository standards. Resolve actionable findings, run relevant repository checks, and commit the implementation, tests, and reproducible run configuration in the isolated checkout.

## Completion criteria

- All five native IQ formats pass correctness checks with unchanged quantization scope and exactly recoverable GPTQ step 0.
- Candidates and exported checkpoints obey native field ownership, the five-candidate limit, and the undamped objective defined above.
- Calibration and selection follow the recorded split, token, and update limits. Paired complete WikiText 2 test results at 2,048 tokens are recorded.
- Focused tests, review, required checks, and a commit are complete, with measured results and limitations recorded.

An accuracy gain is an experimental outcome, not a completion guarantee. A neutral or worse test result must be reported honestly and does not justify expanding this bounded plan.

## Reproduce the experiment

Run on a CUDA allocation with ModelOpt and the HF PTQ example dependencies available:

    python examples/hf_ptq/gsq_iq.py \
      --model-path /path/to/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
      --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
      --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3 \
      --num-steps 200 500 --output-dir /path/to/experiment

Use `--dry-run` first to verify the official cached snapshot, hash the actual source weights,
and prepare the fixed data without loading the model. The full command runs all five formats;
`--formats iq2_s` runs one format. Reusing the same output directory resumes only when verified
source artifacts, data, settings, and implementation identity match.

The library entrypoint is `modelopt.torch.quantization.ggml.gsq.quantize_iq_gsq`. It uses an
existing IQ GPTQ recipe, returns native states for each requested budget, and leaves the model
at GPTQ. `result.activate(model, 200)` activates GSQ; `result.activate(model, 0)` restores GPTQ.
The driver saves these native states in `result.pt` and paired measurements in `results.json`.

## Model reference

- Official checkpoint: <https://huggingface.co/Qwen/Qwen3.8-27B>
- Official architecture configuration: <https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json>

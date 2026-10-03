# Softmax reparameterization for output-head quantization

An experimental implementation of [Kadav et al., *Softmax Reparameterization for
Output-Head Quantization*](https://arxiv.org/abs/2609.31291).

For a vocabulary-by-hidden weight matrix `W`, each candidate is
`W_t = W - t * W.mean(dim=0)`. Every vocabulary row receives the same vector
subtraction, preserving the full-precision softmax distribution algebraically.
Quantizing these equivalent heads produces different residuals. We fit the same
ModelOpt quantizer independently for each candidate and select the coefficient
with the lowest forward KL to the **original** head on validation states.

## Usage

Install ModelOpt from this checkout, then run the synthetic example:

```bash
python -m experimental.softmax_reparameterization.example --device cpu
python -m experimental.softmax_reparameterization.example --device cuda --algorithm gptq
```

For an existing head, capture the final hidden states **entering the output
projection**. Use articles disjoint across quantizer fitting, coefficient
validation, and final test evaluation; exclude padding tokens. The caller must
ensure these splits are disjoint. Each tensor has shape `[tokens, hidden_width]`
and may reside on CPU to limit accelerator memory usage.

```python
import copy

import torch

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from experimental.softmax_reparameterization import head_kl, search_head

# source_head is an unquantized torch.nn.Linear immediately preceding softmax.
config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
result = search_head(source_head, fit_states, validation_states, config)
print(result.coefficient, result.validation_kl)
print("Test KL:", head_kl(source_head, result.model, test_states))
mto.save(result.model, "selected_head.pt")
restored = mto.restore(torch.nn.Sequential(copy.deepcopy(source_head)), "selected_head.pt")
```

The returned model is a one-layer `Sequential`, with the head at index `0`; quantizer name patterns refer
to this container, not to the original language model. The default blockwise INT4
preset applies to it without changing ModelOpt's default LM-head exclusions.
To use GPTQ, set `config["algorithm"] = "gptq"` before search. This is ModelOpt's
calibration implementation; its rounding and fitting conventions need not match
the paper's quantizers.

The ordered default grid is `[-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8]`.
Zero is required; one provides the fixed mean-centering comparison. Exact ties
keep the first candidate. The full validation curve is returned, and the selected
calibrated candidate is retained without refitting. Including zero guarantees
non-increasing **measured validation KL**, not improvement on unseen data.

## Scope and numerical behavior

- Supports plain, unsharded `torch.nn.Linear` heads, with optional unchanged bias,
  on CPU or a single CUDA device. Inputs and weights use FP32, FP16, or BF16;
  log-softmax and KL are computed in FP32 and accumulated in FP64.
- Supports weight-only fake quantization with `max` or `gptq` calibration.
  Activation quantization, composite quantizers, nonlinear logit transformations
  (including soft-capping), and grouped coefficient search are outside this PR.
- Every candidate has independent weights and quantizer state. The source head
  and any tied input embedding remain untouched, including on failure. Using the
  selected head in a tied model requires a separate output weight allocation and
  disabling weight tying when saving/reloading the full model.
- Equivalence holds in exact arithmetic. Casting shifted weights and low-precision
  matrix products can introduce small differences even before quantization.
- The search runs the head only and does not retrain or rerun the decoder. Peak
  memory includes the source, candidate, selected head, and temporary folded-weight
  copies; GPTQ additionally needs a hidden-width-squared Hessian.

## Model support and deployment

This first integration tests synthetic linear heads. It does not claim that the
paper's model-level results have been reproduced with ModelOpt. Capture and
full-model adapters are intentionally left to the caller, who must verify a
shift-compatible logit path. Replacing a Hugging Face child module also needs a
model-level ModelOpt conversion/export workflow; the standalone head container
must not be mistaken for an export-ready language-model checkpoint.

The shift is stored in weights and requires no extra inference operator for a
linear-softmax head. This implementation evaluates fake-quantized heads and tests
ModelOpt head save/restore. Packed export and TensorRT-LLM/vLLM/SGLang serving
are **not validated** by this prototype; no latency benefit is claimed here.

## Tests

```bash
pytest tests/unit/torch/quantization/test_softmax_reparameterization.py -q
pytest tests/gpu/torch/quantization/test_softmax_reparameterization_cuda.py -q
```

CPU tests compare selection with independent ModelOpt candidates, check softmax
equivalence, source/tied-embedding isolation, tie handling, invalid inputs, and
save/restore. CUDA tests exercise BF16 heads with INT4 max and GPTQ calibration.

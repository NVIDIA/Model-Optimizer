# GDN/KDA recurrent-state fake quantization in vLLM

The existing `FakeQuantWorker` adds a ModelOpt `TensorQuantizer` immediately before
native vLLM prefill and decode for `Qwen3NextGatedDeltaNet` and
`KimiDeltaAttention`. Native attention kernels, projections, convolution, gates,
outputs, and cache management remain in use.

## Runtime and launch

The initial adapter targets vLLM 0.15.x V1 with key-first
`[slot, head, Dk, Dv]` FP32 recurrent state. Runtime checks reject other layouts
and versions. Qualification uses source revision `930288170` with Torch
2.9.1/CUDA 12.8 on RTX A6000 GPUs.

```bash
PYTHONPATH=.:examples/vllm_serve \
RECIPE_PATH=examples/vllm_serve/linear_attention_state_int8.yaml \
python examples/vllm_serve/vllm_serve_fakequant.py /path/to/model \
  --tensor-parallel-size 2 --enforce-eager --no-async-scheduling \
  --no-enable-prefix-caching --mamba-cache-dtype float32
```

The example enables signed symmetric dynamic INT8 state QDQ. Set
`num_bits: [4, 3]` in each state quantizer to select FP8 E4M3. Each invocation
receives `[active_sequence, local_head, Dk, Dv]`; `axis: [0, 1]` retains one
scale per sequence and local head, reducing over the whole state matrix.
State tensors remain FP32 after dequantization.

Dynamic scales need no calibration dataset; `algorithm: null` skips dataset
loading. Weight/activation calibration can use the worker's existing recipe
and calibration loop. Static state calibration is unsupported in this adapter.

Use `MODELOPT_STATE_PATH=/path/to/modelopt_state.pt` instead of a recipe to
restore quantizer configuration. Reload maps recurrent-state quantizer names
through the model's HF-to-vLLM mapper. Checkpoint weights must already match
the vLLM model architecture.

## Quantization boundaries

- Prefill: vLLM gathers initial states and zeros fresh requests. The wrapper
  applies `TensorQuantizer` to this tensor and calls the original prefill kernel.
- Decode: the wrapper gathers active native cache slots, quantizes them, writes
  them back, and calls the original decode kernel. Other slots are untouched.
- Native kernels compute outputs and update recurrent state normally. No
  additional rounding is applied to the kernel's final-state write.
- The next call quantizes that state before reading it. A fresh zero state is
  unchanged; prompt-to-decode rounding happens before the first decode call.
- Scheduler-level chunked prefill creates one QDQ boundary per invocation.
  Internal kernel chunks do not create extra state-quantization boundaries.
  Results can therefore depend on scheduler prompt-chunk sizes.
- Requests, cache-slot reuse, and preemption remain managed by vLLM. The plugin
  allocates temporary gathered states, with no additional persistent state cache.
- TP ranks quantize their local heads independently, without scale all-reduce.

This state-only adapter accepts the default `linear_attention` execution policy
and state quantizer configuration. Enabled W/operand quantizers and nondefault
training, replay, decay, or solve policies are rejected rather than reinterpreted.
The previous experimental replay-serving recipe is replaced by
`linear_attention_state_int8.yaml`; its saved decode policies are unsupported.

The qualified scope is eager synchronous execution with TP=1/2 and PP=DP=CP=1.
Speculative decoding, prefix caching, state transfer, and CUDA graphs require
separate integration. This is floating-point numerical emulation; model-quality,
capacity, and performance claims require separate measurements.

## Later prefill GEMM support

Prefill operand QDQ will be a separate change after an optimized fused kernel
is available. It will reuse the eight numerical sites from the training prefill
implementation and preserve this native-cache wrapper. The PyTorch materialized
backend is not exposed in vLLM by this state-only adapter. State QDQ cadence and
prefill operand QDQ remain independent numerical policies.

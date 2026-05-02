# Custom MoE Quantization Support

This guide explains how to add quantization support for a custom Mixture-of-Experts (MoE)
architecture from a HuggingFace checkpoint. ModelOpt already supports several MoE patterns
out of the box. If your model uses a novel MoE design, you can follow the patterns below
to patch `modelopt/torch/quantization/plugins/huggingface.py`.

## Background

ModelOpt quantizes models by wrapping modules with *quantized* counterparts. For standard
`nn.Linear` layers this happens automatically. MoE layers, however, use fused weight tensors
(e.g., `[num_experts, out_dim, in_dim]`) or batched matrix multiplications (`torch.bmm`) instead
of individual `nn.Linear` calls, so ModelOpt cannot insert quantizers automatically. Each MoE
variant therefore needs a dedicated `QuantModule` subclass that places `TensorQuantizer` nodes
at the right points in the forward pass.

There are two broad strategies used in the existing plugins:

| Strategy | When to use | Examples |
|----------|------------|----------|
| **Expand fused weights into per-expert `nn.Linear`** | The MoE stores weights in a single fused tensor and dispatches tokens per-expert in a loop. Expanding lets each expert get its own `input_quantizer` / `weight_quantizer` automatically. | Qwen3VL MoE, Qwen3.5 MoE, DBRX |
| **Quantize around `torch.bmm` / fused ops** | The MoE uses batched matmuls across all experts simultaneously. Expanding to per-expert linears would break the batched execution. Instead, add explicit `TensorQuantizer` nodes for inputs and weights around the `bmm` calls. | Llama4 / GPT-OSS |

## Existing MoE Patterns

### 1. Default HF SparseMoE (`_QuantSparseMoe`)

**Applies to:** Mixtral, Qwen3Moe, Qwen2Moe, NemotronH, and any HF MoE block that follows
the standard `gate` + `experts` structural pattern.

**How it works:** This is the most generic wrapper. It does **not** modify the forward pass of
the MoE block itself. Instead, it provides:

- **Expert amax synchronization** (`layer_sync_moe_local_experts_amax`): After calibration,
  syncs the `input_quantizer` amax across all experts so they share the same quantization
  scale. This is important because different experts see different subsets of tokens.
- **Calibration coverage boost** (`_moe_calib_experts_ratio`): Optionally increases
  `top_k` during calibration so more experts see tokens, improving calibration quality.
- **Expert token counting**: Tracks how many tokens each expert receives to detect
  under-calibrated experts.

Because the individual experts in these models are regular `nn.Linear` modules, ModelOpt
quantizes them automatically. `_QuantSparseMoe` only wraps the outer MoE block for the
coordination features above.

**Auto-detection:** `register_sparse_moe_on_the_fly` walks the model tree and registers any
module that has `gate` (with `top_k` and `num_experts` attributes) and `experts` sub-modules.

```python
# Simplified registration logic
def _is_sparse_moe_block(module):
    if not hasattr(module, "experts"):
        return False
    if hasattr(module, "gate"):
        gate = module.gate
        if hasattr(gate, "top_k") and hasattr(gate, "num_experts"):
            return True
    return False
```

**When your model fits this pattern:** If your MoE block has a `gate` with `top_k`/`num_experts`
and an `experts` `ModuleList` of `nn.Linear`-based FFN modules, it will be auto-detected.
No code changes needed.

### 2. Llama4-Style BMM MoE (`_QuantLlama4TextExperts`)

**Applies to:** `Llama4TextExperts` — a fused MoE where expert weights are stored as 3D
tensors and the forward pass uses `torch.bmm`.

**Key challenge:** The expert weights have shape `(num_experts, in_dim, out_dim)` — note
the **transposed** layout compared to `nn.Linear` convention `(out_features, in_features)`.
ModelOpt's per-channel/per-block quantization assumes `in_dim` is the last dimension, so
these weights need **transposed quantization**.

**How it works:**

```python
class _QuantLlama4TextExperts(QuantModule):
    def _setup(self):
        # Four quantizers: input + weight for each of the two matmuls
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.gate_up_proj_weight_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizer = TensorQuantizer()

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        # Quantize input, then quantize weight with transpose trick
        gate_up = torch.bmm(
            self.gate_up_proj_input_quantizer(hidden_states),
            _transposed_quantize(self.gate_up_proj, self.gate_up_proj_weight_quantizer),
        )
        gate, up = gate_up.chunk(2, dim=-1)
        next_states = torch.bmm(
            self.down_proj_input_quantizer(up * self.act_fn(gate)),
            _transposed_quantize(self.down_proj, self.down_proj_weight_quantizer),
        )
        next_states = next_states.view(-1, self.hidden_size)
        return next_states
```

The `_transposed_quantize` helper transposes the weight before quantization and transposes
back, so that per-channel scales are computed along the correct axis.

**Registration:**

```python
from transformers.models.llama4.modeling_llama4 import Llama4TextExperts
QuantModuleRegistry.register({Llama4TextExperts: "hf.Llama4TextExperts"})(_QuantLlama4TextExperts)
```

### 3. GPT-OSS BMM MoE (`_QuantGptOssExperts`)

**Applies to:** `GptOssExperts` — similar to Llama4 but uses the original HF forward pass
(which may use `torch.bmm` or `torch.Tensor.__matmul__`). Instead of rewriting forward,
this approach **intercepts** the matmul calls.

**How it works:** Extends `_QuantFunctionalMixin`, which monkey-patches functions during
`forward()`.

1. **Weight quantization** via dynamic attributes: `_register_dynamic_attribute` intercepts
   access to `self.gate_up_proj` and `self.down_proj`, returning quantized (transposed) versions.
   The quantized weight is cached per forward pass to avoid redundant computation.

2. **Activation quantization** via function replacement: Replaces `torch.bmm` and
   `torch.Tensor.__matmul__` with versions that quantize the input on every second call
   (first call = `gate_up_proj` input, second call = `down_proj` input).

```python
class _QuantGptOssExperts(_QuantFunctionalMixin):
    def _setup(self):
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.gate_up_proj_weight_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizer = TensorQuantizer()
        # Register dynamic attributes for weight interception
        self._setup_for_weight_quantization()

    @property
    def functionals_to_replace(self):
        def _quantized_bmm(batch1, batch2, *, out=None):
            batch1 = self.down_proj_input_quantizer(batch1) if self._down_proj_mul else batch1
            self._down_proj_mul = not self._down_proj_mul
            return torch.ops.aten.bmm(batch1, batch2)
        return [
            (torch, "bmm", _quantized_bmm),
            (torch.Tensor, "__matmul__", _tensor_matmul),
        ]

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        hidden_states = self.gate_up_proj_input_quantizer(hidden_states)
        with self.quantize_weight():
            return super().forward(hidden_states, router_indices, routing_weights)
```

**When to use this pattern:** When you want to keep the original forward pass unchanged and
intercept the underlying ops. This is useful when the forward logic is complex or uses
external kernels.

### 4. DBRX MoE (`_QuantDbrxExperts` + `_QuantDbrxExpertGLU`)

**Applies to:** DBRX models that store expert weights as fused 2D tensors
(e.g., `w1` of shape `[num_experts * ffn_hidden_size, hidden_size]`).

**How it works:** Uses the **expand-to-per-expert-linears** strategy at two levels:

- `_QuantDbrxExperts`: Rewrites the forward to loop over experts and dispatch tokens,
  calling `self.mlp(expert_tokens, expert_idx)` per expert.
- `_QuantDbrxExpertGLU`: In `_setup`, splits the fused weight tensors (`w1`, `v1`, `w2`)
  into per-expert `nn.ModuleList` of `nn.Linear` layers. Each linear is then auto-quantized.

```python
class _QuantDbrxExpertGLU(QuantModule):
    def _setup(self):
        # Split fused w1 into per-expert nn.Linear modules
        self.w1_linear = nn.ModuleList([
            nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
            for _ in range(self.moe_num_experts)
        ])
        # Copy weights from fused tensor to per-expert linears
        for expert_idx, module in enumerate(self.w1_linear):
            module.weight.copy_(
                self.w1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[expert_idx]
            )
        delattr(self, "w1")
        # ... same for v1 and w2

    def forward(self, x, expert_idx):
        x1 = self.w1_linear[expert_idx](x)
        x2 = self.v1_linear[expert_idx](x)
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        return self.w2_linear[expert_idx](x1)
```

**Also registers** `_QuantDbrxFFN` (extending `_QuantSparseMoe`) for the outer MoE block
with properties adapting `num_experts` and `top_k` to the DBRX router attribute names.

### 5. Qwen3VL MoE (`_QuantQwen3VLMoeTextExperts`)

**Applies to:** `Qwen3VLMoeTextExperts` — fused expert weights stored as 3D tensors
`gate_up_proj[num_experts, hidden_size, 2*intermediate_size]` and
`down_proj[num_experts, intermediate_size, hidden_size]`.

**How it works:** Expands fused weights into per-expert `nn.Linear` modules. Note the
weight transposition: the fused tensor has shape `(hidden_size, intermediate_size)` but
`nn.Linear` expects `(intermediate_size, hidden_size)`, so `.T` is applied during copy.

```python
class _QuantQwen3VLMoeTextExperts(QuantModule):
    def _setup(self):
        # Split gate_up_proj into separate gate_proj and up_proj per expert
        gate_proj = nn.ModuleList([
            nn.Linear(self.hidden_size, expert_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        up_proj = nn.ModuleList([...])
        down_proj = nn.ModuleList([...])

        for idx in range(self.num_experts):
            _copy_weight(gate_proj[idx], self.gate_up_proj[idx, :, :expert_dim].T)
            _copy_weight(up_proj[idx], self.gate_up_proj[idx, :, expert_dim:].T)
            _copy_weight(down_proj[idx], self.down_proj[idx, :].T)

        delattr(self, "gate_up_proj")
        delattr(self, "down_proj")

    def forward(self, hidden_states, routing_weights, router_indices):
        # Per-expert dispatch loop
        for expert_idx in expert_hit:
            current_state = hidden_states[token_idx]
            gate = self.gate_proj[expert_idx](current_state)
            up = self.up_proj[expert_idx](current_state)
            out = self.down_proj[expert_idx](self.act_fn(gate) * up)
            next_states.index_add_(0, token_idx, out * routing_weights[...])
        return next_states
```

### 6. Qwen3.5 MoE (`_QuantQwen35MoeExperts`)

**Applies to:** `Qwen3_5MoeExperts` — similar fused weight layout to Qwen3VL but with a
different tensor shape convention: `gate_up_proj[num_experts, 2*intermediate_dim, hidden_dim]`
(already in `(out_features, in_features)` format, no transpose needed).

**How it works:** Expands into per-expert `_Qwen35MoeExpertModule` containers (each holding
`gate_proj`, `up_proj`, `down_proj` as `nn.Linear`). Uses `add_module(str(idx), ...)` for
naming consistency (`experts.0.gate_proj.weight`, `experts.1.gate_proj.weight`, etc.).

Also implements `__len__`, `__iter__`, and `__getitem__` so the expanded experts behave like
a `ModuleList` — this is important because the outer `_QuantSparseMoe` wrapper (auto-detected
via `register_sparse_moe_on_the_fly`) iterates over `self.experts`.

```python
class _Qwen35MoeExpertModule(nn.Module):
    """Container for a single expert's linear layers."""
    def __init__(self, hidden_dim, expert_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.down_proj = nn.Linear(expert_dim, hidden_dim, bias=False)

class _QuantQwen35MoeExperts(QuantModule):
    def _setup(self):
        # No transpose needed — weights already (out_features, in_features)
        for idx in range(self.num_experts):
            _copy_weight(expert_modules[idx].gate_proj, self.gate_up_proj[idx, :expert_dim, :])
            _copy_weight(expert_modules[idx].up_proj, self.gate_up_proj[idx, expert_dim:, :])
            _copy_weight(expert_modules[idx].down_proj, self.down_proj[idx])
```

### 7. Stepfun Step3p5 MoE (`_QuantMoELinear`)

**Applies to:** `Step3p5ForCausalLM` / `Step3p5Model` — a `trust_remote_code` model where
each expert FFN layer (`up_proj`, `gate_proj`, `down_proj`) is a custom `MoELinear` class
with weight shape `[num_experts, out_features, in_features]` and a `forward(x, expert_id)`
interface that selects a single expert per call.

**Key challenge:** Unlike the Qwen/DBRX models where the outer MoE block dispatches tokens,
here the dispatch logic lives *outside* the expert weight module. The `MoELinear` itself
just does `linear(x, weight[expert_id])`. Since the class is loaded dynamically via
`trust_remote_code`, it cannot be imported at module level — registration must happen
on-the-fly.

**How it works:** Expands the fused 3D weight into per-expert `nn.Linear` modules. Each
expert linear is then auto-wrapped with `input_quantizer` and `weight_quantizer` by
ModelOpt's standard `_QuantLinear`. The forward selects the expert by `expert_id` and
casts input/output dtypes to match the original `MoELinear` behavior.

```python
class _QuantMoELinear(QuantModule):
    def _setup(self):
        # Weight shape: [num_experts, out_features, in_features]
        # Already in nn.Linear convention — no transpose needed
        experts = nn.ModuleList([
            nn.Linear(self.in_features, self.out_features, bias=False)
            for _ in range(self.num_experts)
        ])
        for i in range(self.num_experts):
            experts[i].weight.data = self.weight[i].detach()
        delattr(self, "weight")
        self.experts = experts

    def forward(self, x, expert_id):
        expert = self.experts[expert_id]
        x = x.to(expert.weight.dtype)
        return expert(x).float()
```

**Export reconstruction:** A unique aspect of this pattern is that downstream serving
engines (e.g., vLLM) expect the original 3D weight layout with stacked scaling factors.
After calibration and quantization, `_reconstruct_fused_moe_linear()` stacks the per-expert
weights and scales back into the original format:

```python
def _reconstruct_fused_moe_linear(model):
    for name, module in model.named_modules():
        if type(module).__name__ != "QuantMoELinear":
            continue
        n = module.num_experts
        experts = module.experts
        # Stack per-expert weights back to [num_experts, out_features, in_features]
        module.weight = nn.Parameter(
            torch.stack([experts[i].weight.data for i in range(n)])
        )
        # Stack per-expert scales (weight_scale, weight_scale_2, input_scale)
        for attr in ("weight_scale", "weight_scale_2", "input_scale"):
            if all(hasattr(experts[i], attr) for i in range(n)):
                module.register_buffer(attr, torch.stack([getattr(experts[i], attr) for i in range(n)]))
        del module.experts
```

This expand-then-reconstruct approach (rather than `add_module()` as in Qwen3.5) is
specifically chosen because vLLM requires stacked 3D scaling factors and does not accept
per-expert expanded keys.

**On-the-fly registration:** Because the `MoELinear` class is only available at runtime
(loaded via `trust_remote_code`), registration detects it by model class name and grabs
the type from the first MoE layer:

```python
def register_step3p5_moe_on_the_fly(model):
    if type(model).__name__ not in ("Step3p5ForCausalLM", "Step3p5Model"):
        return
    for module in model.modules():
        if type(module).__name__ == "Step3p5MoEMLP":
            moe_linear_type = type(module.up_proj)
            if QuantModuleRegistry.get(moe_linear_type) is None:
                QuantModuleRegistry.register({moe_linear_type: f"hf.{moe_linear_type.__name__}"})(
                    _QuantMoELinear
                )
            break
```

## How to Add Support for Your Custom MoE

### Step 1: Understand Your MoE Architecture

Identify the following about your model's MoE implementation:

1. **What class implements the expert computation?** (e.g., `MyModelExperts`)
2. **How are weights stored?** Fused 3D tensor `[num_experts, ...]` or per-expert modules?
3. **What is the weight layout?** `(num_experts, out_dim, in_dim)` or `(num_experts, in_dim, out_dim)`?
4. **How does the forward pass work?** Per-expert loop with indexing, or batched `torch.bmm`?

### Step 2: Choose a Strategy

| Your MoE design | Recommended strategy | Reference implementation |
|-----------------|---------------------|--------------------------|
| Per-expert `nn.Linear` modules in a `ModuleList` | No changes needed — auto-detected by `register_sparse_moe_on_the_fly` | `_QuantSparseMoe` |
| Fused weights + per-expert dispatch loop | Expand to per-expert `nn.Linear` in `_setup` | `_QuantQwen3VLMoeTextExperts`, `_QuantQwen35MoeExperts` |
| Fused weights + `torch.bmm` | Add explicit `TensorQuantizer` around bmm calls | `_QuantLlama4TextExperts` |
| Complex forward with bmm/matmul ops | Intercept ops via `_QuantFunctionalMixin` | `_QuantGptOssExperts` |
| Fused weights + per-expert `forward(x, expert_id)` interface | Expand to per-expert linears + reconstruct on export | `_QuantMoELinear` (Step3p5) |

### Step 3: Implement the QuantModule

Here is a template for the **expand-to-per-expert-linears** strategy (most common):

```python
class _QuantMyModelExperts(QuantModule):
    def _setup(self):
        """Expand fused expert weights into per-expert nn.Linear modules."""
        from accelerate import init_empty_weights

        dtype, device = self.fused_weight.dtype, self.fused_weight.device
        num_experts = self.num_experts
        hidden_size = ...   # extract from model attributes
        expert_dim = ...    # extract from model attributes

        with init_empty_weights():
            gate_proj = nn.ModuleList([
                nn.Linear(hidden_size, expert_dim, bias=False)
                for _ in range(num_experts)
            ])
            # ... similarly for up_proj, down_proj

        for idx in range(num_experts):
            gate_proj[idx].to_empty(device=device)
            with torch.no_grad():
                # IMPORTANT: Check your weight layout!
                # If weights are (num_experts, in_dim, out_dim), transpose with .T
                # If weights are (num_experts, out_dim, in_dim), no transpose needed
                gate_proj[idx].weight.data = self.fused_weight[idx, ...].detach().to(dtype=dtype)

        # Remove the original fused weight and replace with expanded modules
        delattr(self, "fused_weight")
        self.gate_proj = gate_proj
        # ...

    def forward(self, hidden_states, routing_weights, router_indices):
        """Per-expert dispatch loop."""
        next_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            _, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current = hidden_states[token_idx]
            # Each nn.Linear is auto-wrapped with input_quantizer + weight_quantizer
            gate = self.gate_proj[expert_idx](current)
            up = self.up_proj[expert_idx](current)
            out = self.down_proj[expert_idx](self.act_fn(gate) * up)
            next_states.index_add_(0, token_idx, out * routing_weights[token_idx, expert_idx, None])

        return next_states
```

Here is a template for the **BMM quantization** strategy:

```python
class _QuantMyBmmExperts(QuantModule):
    def _setup(self):
        """Add quantizers for each matmul in the forward pass."""
        self.proj_input_quantizer = TensorQuantizer()
        self.proj_weight_quantizer = TensorQuantizer()
        # Add more quantizers for each matmul operation

    def forward(self, hidden_states):
        # Quantize input and weight around each bmm call
        out = torch.bmm(
            self.proj_input_quantizer(hidden_states),
            # Use _transposed_quantize if weight layout is (num_experts, in_dim, out_dim)
            _transposed_quantize(self.weight, self.proj_weight_quantizer),
        )
        return out
```

### Step 4: Register the QuantModule

Add the registration at the bottom of `huggingface.py`:

```python
try:
    from transformers.models.my_model.modeling_my_model import MyModelExperts

    if MyModelExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register({MyModelExperts: "hf.MyModelExperts"})(_QuantMyModelExperts)
except ImportError:
    pass
```

If your model uses `trust_remote_code=True` (class not available in `transformers` at import
time), add an on-the-fly registration function instead:

```python
def register_my_model_moe_on_the_fly(model):
    if type(model).__name__ != "MyModelForCausalLM":
        return
    for module in model.modules():
        if type(module).__name__ == "MyModelExperts":
            moe_type = type(module)
            if QuantModuleRegistry.get(moe_type) is None:
                QuantModuleRegistry.register({moe_type: f"hf.{moe_type.__name__}"})(_QuantMyModelExperts)
            break

# Add to the plugin set
CUSTOM_MODEL_PLUGINS.add(register_my_model_moe_on_the_fly)
```

### Step 5: Handle the Outer MoE Block

If your model's outer MoE block (the module containing `gate` + `experts`) follows the
standard HF pattern, `register_sparse_moe_on_the_fly` will auto-detect it. If not, you may
need to register a `_QuantSparseMoe` subclass with property adapters for `num_experts` and
`top_k` (see `_QuantDbrxFFN` for an example).

## Key Considerations

- **Weight layout matters.** Check whether your fused weights are `(out_features, in_features)`
  or `(in_features, out_features)`. Use `_transposed_quantize` or `.T` during weight copy
  as needed.
- **Use `accelerate.init_empty_weights()`** when creating per-expert `nn.Linear` modules to
  avoid allocating memory twice.
- **Expert amax sync** is handled by `_QuantSparseMoe.layer_sync_moe_local_experts_amax()`.
  If your custom experts wrapper is used inside a `_QuantSparseMoe`-wrapped block, this
  sync happens automatically.
- **Calibration coverage**: MoE models often have experts that receive very few tokens
  during calibration. The `_moe_calib_experts_ratio` feature in `_QuantSparseMoe` can
  help by routing tokens to more experts during calibration.

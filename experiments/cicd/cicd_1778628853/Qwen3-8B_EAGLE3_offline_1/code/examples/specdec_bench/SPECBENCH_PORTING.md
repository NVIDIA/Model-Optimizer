# Porting Spec-Bench Inference Runners to specdec_bench

This guide explains how to convert any `inference_*.py` runner from [Spec-Bench](https://github.com/hemingkx/Spec-Bench) to a model class compatible with `specdec_bench`.

## Overview

Spec-Bench inference runners follow a pattern where:

1. A `*_forward()` function handles the speculative decoding logic
2. The `run_eval()` function orchestrates evaluation with tokenized inputs
3. Models are loaded in `__main__` and passed to `run_eval()`

In contrast, `specdec_bench` uses a class-based approach where:

1. Models inherit from the `Model` base class
2. `__init__()` handles model loading
3. `run()` is an async method that processes single requests
4. `stop()` handles cleanup

## The specdec_bench Model Interface

```python
class Model:
    def __init__(self, model_dir, tokenizer, max_draft_length):
        raise NotImplementedError
    
    async def run(self, prompt_ids, sampling_params, request_id, turn_id):
        """
        prompt_ids: list of token IDs (not a tensor!)
        Returns dict with:
            - output_ids: list of list of token chunks per step [[chunk1, chunk2, ...]]
            - output_logits: optional logits (usually None)
            - token_times: list of timestamps per decoding step
        """
        raise NotImplementedError

    def stop(self):
        pass
```

## Step-by-Step Porting Guide

### Step 1: Identify the Key Components in Spec-Bench

Look at the `inference_*.py` file and identify:

1. **The forward function** (e.g., `medusa_forward`, `ea_forward`)
   - This contains the core speculative decoding loop
   - Signature: `forward_func(inputs, model, tokenizer, max_new_tokens, **kwargs)`
   - Returns: `(output_ids, new_token_count, num_steps, accept_length_list)`

2. **The model class** (e.g., `MedusaModel`, `EaModel`)
   - Found in `model/<method>/` directory
   - Has a `from_pretrained()` class method

3. **Required utilities** from the method's module:
   - Buffer generation (e.g., `generate_medusa_buffers`)
   - Initialization functions (e.g., `initialize_medusa`, `initialize_past_key_values`)
   - Decoding functions (e.g., `tree_decoding`, `generate_candidates`)
   - State update functions (e.g., `update_inference_inputs`)

4. **Method-specific choices/configs** (e.g., `mc_sim_7b_63` for Medusa)

### Step 2: Create the specdec_bench Model Class

```python
# specdec_bench/specdec_bench/models/specbench_<method>.py

from .base import Model
import asyncio
import time
import torch

# Import dependencies from Spec-Bench
try:
    import sys
    import os
    spec_bench_path = os.path.join(os.getcwd(), "Spec-Bench")
    sys.path.insert(0, spec_bench_path)
    from model.<method>.<model_file> import <ModelClass>
    from model.<method>.kv_cache import initialize_past_key_values
    from model.<method>.utils import (
        # Import all required utilities
    )
    from model.<method>.<choices_file> import <default_choices>
except ImportError as e:
    print(f"<Method> dependencies not found: {e}")
    <ModelClass> = None


class SpecBench<Method>Model(Model):
    def __init__(self, model_dir, max_concurrent_requests, sampling_kwargs, **kwargs):
        # 1. Validate dependencies
        if <ModelClass> is None:
            raise ImportError("<Method> dependencies not found.")
        
        # 2. Extract configuration from kwargs
        self.dtype = kwargs.get("dtype", "float16")
        self.max_steps = kwargs.get("max_steps", 512)
        self.temperature = sampling_kwargs.get("temperature", 0.0)
        # ... other method-specific parameters
        
        # 3. Set up device (avoid device_map="auto" for multi-GPU issues)
        self.device = torch.device(kwargs.get("device", "cuda:0"))
        
        # 4. Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)
        
        # 5. Load the model
        self.model = <ModelClass>.from_pretrained(
            model_dir,
            # ... other args from Spec-Bench's __main__
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)
        
        self.sampling_kwargs = sampling_kwargs
```

### Step 3: Port the Forward Function

Convert the standalone `*_forward()` function to an internal method:

```python
    def _forward(self, input_ids, max_new_tokens, end_id):
        """
        Port of the original *_forward function.
        
        Key changes from Spec-Bench:
        1. input_ids is already a tensor (converted in run())
        2. Add timing list to track per-step timestamps
        3. Use self.device instead of model.base_model.device
        4. Return timing along with other outputs
        """
        accept_length_list = []
        timing = [time.perf_counter()]  # ADD: Track timing
        
        # === COPY THE FORWARD LOGIC FROM SPEC-BENCH ===
        # Replace: device=model.base_model.device
        # With:    device=self.device
        
        # Initialize buffers...
        # Initialize KV cache...
        # Main decoding loop...
        
        for idx in range(self.max_steps):
            # Generate candidates...
            # Tree decoding...
            # Evaluate posterior...
            # Update inputs...
            
            timing.append(time.perf_counter())  # ADD: Record time per step
            
            # Check for EOS
            if end_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
        
        return input_ids, new_token, idx + 1, accept_length_list, timing  # ADD timing
```

### Step 4: Implement the run() Method

```python
    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        """
        Async interface for specdec_bench.
        
        Args:
            prompt_ids: List of input token IDs (NOT a tensor)
            max_length: Maximum new tokens to generate
            end_id: EOS token ID
            request_id: Request identifier
            turn_id: Turn identifier
        
        Returns:
            dict with output_ids, output_logits, token_times
        """
        output_dict = {}
        
        # Convert prompt_ids list to tensor
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.device
        )
        
        # Run forward pass (use asyncio.to_thread for sync code)
        result = await asyncio.to_thread(
            self._forward, input_ids, max_length, end_id
        )
        input_ids_out, new_token, num_steps, accept_length_list, timing = result
        
        # Extract generated tokens (excluding prompt)
        original_len = len(prompt_ids)
        generated_tokens = input_ids_out[0, original_len:].tolist()
        
        # Remove EOS token if present
        if end_id in generated_tokens:
            eos_idx = generated_tokens.index(end_id)
            generated_tokens = generated_tokens[:eos_idx]
        
        # Format output_ids as list of token chunks per step
        # This matches specdec_bench's expected format
        reformatted_output_ids = [[]]
        start = 0
        for accept_len in accept_length_list:
            if accept_len > 0 and start < len(generated_tokens):
                chunk = generated_tokens[start:start + accept_len]
                if chunk:
                    reformatted_output_ids[0].append(chunk)
                start += accept_len
        
        # Handle remaining tokens
        if start < len(generated_tokens):
            reformatted_output_ids[0].append(generated_tokens[start:])
        
        output_dict['output_ids'] = reformatted_output_ids
        output_dict['output_logits'] = None
        output_dict['token_times'] = timing
        
        return output_dict
```

### Step 5: Implement stop() for Cleanup

```python
    def stop(self):
        """Clean up resources."""
        # Clear any cached states
        if hasattr(self.model, "past_key_values"):
            del self.model.past_key_values
            del self.model.past_key_values_data
            del self.model.current_length_data
        
        # Clear method-specific buffers
        if hasattr(self.model, "<method>_buffers"):
            del self.model.<method>_buffers
        
        # Free GPU memory
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            torch.cuda.empty_cache()
```

### Step 6: Register the Model (Optional)

Add to `specdec_bench/specdec_bench/models/__init__.py`:

```python
from .specbench_<method> import SpecBench<Method>Model
```

## Key Differences Summary

| Aspect | Spec-Bench | specdec_bench |
|--------|-----------|---------------|
| Input format | `inputs.input_ids` (tensor from tokenizer) | `prompt_ids` (list of ints) |
| Output format | `(output_ids, new_token, steps, accept_lengths)` | `dict` with `output_ids`, `output_logits`, `token_times` |
| Output IDs | Full sequence tensor | List of token chunks per step |
| Timing | External (in `run_eval`) | Internal (in `run()`) |
| Device | `device_map="auto"` | Explicit single device |
| Interface | Function-based | Class-based with async `run()` |

## Common Pitfalls

1. **Device Mismatch**: Avoid `device_map="auto"` which spreads model across GPUs. Use explicit `.to(device)`.

2. **Tensor vs List**: `prompt_ids` in specdec_bench is a Python list, not a tensor. Convert it in `run()`.

3. **Output Format**: specdec_bench expects `output_ids` as `[[chunk1, chunk2, ...]]` (list of lists of lists for beam_width=1).

4. **Timing**: Add `time.perf_counter()` calls to track per-step latency.

5. **EOS Handling**: Strip EOS tokens from output before formatting.

6. **Async Wrapper**: Use `asyncio.to_thread()` to wrap synchronous forward passes.

## Example: Mapping Spec-Bench Methods

| Spec-Bench File | Model Class | Forward Function | Key Utils |
|-----------------|-------------|------------------|-----------|
| `inference_medusa.py` | `MedusaModel` | `medusa_forward` | `generate_medusa_buffers`, `initialize_medusa` |
| `inference_eagle.py` | `EaModel` | `ea_forward` | `generate_tree_buffers`, `initialize_tree` |
| `inference_eagle2.py` | `EaModel` | `ea_forward` | Same as EAGLE |
| `inference_hydra.py` | `HydraModel` | `hydra_forward` | `generate_hydra_buffers`, `initialize_hydra` |
| `inference_lookahead.py` | `LookaheadModel` | `lookahead_forward` | Lookahead-specific utils |

## Testing Your Port

```python
import asyncio

async def test():
    model = SpecBench<Method>Model(
        model_dir="/path/to/model",
        max_concurrent_requests=1,
        sampling_kwargs={"temperature": 0.0},
        # method-specific kwargs...
    )
    
    result = await model.run(
        prompt_ids=[1, 2, 3, 4, 5],  # Example token IDs
        max_length=100,
        end_id=2,  # EOS token
        request_id="test",
        turn_id=0
    )
    
    print("Output chunks:", result['output_ids'])
    print("Timing:", result['token_times'])
    
    model.stop()

asyncio.run(test())
```

Adjust the vicuna chat template to be in the tokenizer_config to be

Insert to tokenizer_config (for vicuna)

```json
"chat_template": "{% set ns = namespace(system='') %}{% for m in messages %}{% if m['role'] == 'system' %}{% set ns.system = m['content'] %}{% endif %}{% endfor %}{{ ns.system | trim }}{% if ns.system | trim != '' %} {% endif %}{% for m in messages %}{% if m['role'] == 'user' %}USER: {{ m['content'] | trim }} ASSISTANT:{% elif m['role'] == 'assistant' %}{{ m['content'] | trim }}{% endif %}{% endfor %}"
```

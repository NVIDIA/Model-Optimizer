# DFlash Checkpoint Structure — Qwen3-8B-DFlash-b16

Source: https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16
Captured: 2026-05-21 17:00 UTC

## Files in repo
- config.json
- model.safetensors (single file, no index file in repo)
- dflash.py
- modeling_dflash.py
- utils.py
- README.md

## config.json (key fields)
- model_type: "qwen3"
- architectures: ["DFlashDraftModel"]
- auto_map: {"AutoModel": "dflash.DFlashDraftModel"}
- hidden_size: 4096
- intermediate_size: 12288
- num_hidden_layers: 5
- num_attention_heads: 32
- num_key_value_heads: 8
- head_dim: 128
- max_position_embeddings: 40960
- rope_theta: 1000000
- vocab_size: 151936
- dtype: "bfloat16"
- block_size: 16
- num_target_layers: 36
- layer_types: ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"]
- dflash_config: {"mask_token_id": 151669, "target_layer_ids": [1, 9, 17, 25, 33]}
- other_keys: attention_bias, attention_dropout, bos_token_id, eos_token_id, hidden_act, initializer_range, max_window_layers, rms_norm_eps, rope_scaling, sliding_window, tie_word_embeddings, transformers_version, use_cache, use_sliding_window

## model.safetensors
- tensor_count: 58
- dtype: BF16 (per safetensors header)
- bias_tensors: none

### Tensor list (name: shape)
- fc.weight: [4096, 20480]
- hidden_norm.weight: [4096]
- layers.0.input_layernorm.weight: [4096]
- layers.0.mlp.down_proj.weight: [4096, 12288]
- layers.0.mlp.gate_proj.weight: [12288, 4096]
- layers.0.mlp.up_proj.weight: [12288, 4096]
- layers.0.post_attention_layernorm.weight: [4096]
- layers.0.self_attn.k_norm.weight: [128]
- layers.0.self_attn.k_proj.weight: [1024, 4096]
- layers.0.self_attn.o_proj.weight: [4096, 4096]
- layers.0.self_attn.q_norm.weight: [128]
- layers.0.self_attn.q_proj.weight: [4096, 4096]
- layers.0.self_attn.v_proj.weight: [1024, 4096]
- layers.1.input_layernorm.weight: [4096]
- layers.1.mlp.down_proj.weight: [4096, 12288]
- layers.1.mlp.gate_proj.weight: [12288, 4096]
- layers.1.mlp.up_proj.weight: [12288, 4096]
- layers.1.post_attention_layernorm.weight: [4096]
- layers.1.self_attn.k_norm.weight: [128]
- layers.1.self_attn.k_proj.weight: [1024, 4096]
- layers.1.self_attn.o_proj.weight: [4096, 4096]
- layers.1.self_attn.q_norm.weight: [128]
- layers.1.self_attn.q_proj.weight: [4096, 4096]
- layers.1.self_attn.v_proj.weight: [1024, 4096]
- layers.2.input_layernorm.weight: [4096]
- layers.2.mlp.down_proj.weight: [4096, 12288]
- layers.2.mlp.gate_proj.weight: [12288, 4096]
- layers.2.mlp.up_proj.weight: [12288, 4096]
- layers.2.post_attention_layernorm.weight: [4096]
- layers.2.self_attn.k_norm.weight: [128]
- layers.2.self_attn.k_proj.weight: [1024, 4096]
- layers.2.self_attn.o_proj.weight: [4096, 4096]
- layers.2.self_attn.q_norm.weight: [128]
- layers.2.self_attn.q_proj.weight: [4096, 4096]
- layers.2.self_attn.v_proj.weight: [1024, 4096]
- layers.3.input_layernorm.weight: [4096]
- layers.3.mlp.down_proj.weight: [4096, 12288]
- layers.3.mlp.gate_proj.weight: [12288, 4096]
- layers.3.mlp.up_proj.weight: [12288, 4096]
- layers.3.post_attention_layernorm.weight: [4096]
- layers.3.self_attn.k_norm.weight: [128]
- layers.3.self_attn.k_proj.weight: [1024, 4096]
- layers.3.self_attn.o_proj.weight: [4096, 4096]
- layers.3.self_attn.q_norm.weight: [128]
- layers.3.self_attn.q_proj.weight: [4096, 4096]
- layers.3.self_attn.v_proj.weight: [1024, 4096]
- layers.4.input_layernorm.weight: [4096]
- layers.4.mlp.down_proj.weight: [4096, 12288]
- layers.4.mlp.gate_proj.weight: [12288, 4096]
- layers.4.mlp.up_proj.weight: [12288, 4096]
- layers.4.post_attention_layernorm.weight: [4096]
- layers.4.self_attn.k_norm.weight: [128]
- layers.4.self_attn.k_proj.weight: [1024, 4096]
- layers.4.self_attn.o_proj.weight: [4096, 4096]
- layers.4.self_attn.q_norm.weight: [128]
- layers.4.self_attn.q_proj.weight: [4096, 4096]
- layers.4.self_attn.v_proj.weight: [1024, 4096]
- norm.weight: [4096]

#!/usr/bin/env python3
"""
Create a HuggingFace checkpoint with MXFP4 MoE weights from the original gpt-oss-120b model.

This script:
1. Copies non-MoE weights from the student model (trained attention, embeddings, etc.)
2. Extracts MoE expert weights from the original gpt-oss-120b in MXFP4 format
3. Either loads experts_to_keep.json or deduces expert mappings by comparing weights
4. Outputs a new checkpoint in decihf format with PACKED MXFP4 expert weights
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Any, Tuple, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from  transformers.integrations.mxfp4 import convert_moe_packed_tensors


def deduce_experts_for_layer(
    layer: int,
    original_path: str,
    original_index: Dict,
    student_path: str,
) -> Tuple[List[int], int, int]:
    """
    Deduce which original experts match the student experts by comparing weights.
    
    Compares dequantized MXFP4 weights from the original model against the student
    model's BF16 weights using L2 distance. Finds the best 1-to-1 matching.
    
    Args:
        layer: Layer index
        original_path: Path to original model
        original_index: Original model's safetensors index
        student_path: Path to student model
        num_student_experts: Number of experts in student model (if None, auto-detect)
    
    Returns:
        Tuple of (expert_indices, num_student_experts, num_original_experts)
    """
    # Load original tensors
    orig_tensors = load_layer_tensors(original_path, layer, original_index)
    mlp1_blocks = orig_tensors[f'model.layers.{layer}.mlp.experts.gate_up_proj_blocks']
    mlp1_scales = orig_tensors[f'model.layers.{layer}.mlp.experts.gate_up_proj_scales']
    mlp2_blocks = orig_tensors[f'model.layers.{layer}.mlp.experts.down_proj_blocks']
    mlp2_scales = orig_tensors[f'model.layers.{layer}.mlp.experts.down_proj_scales']
    
    num_original_experts = mlp1_blocks.shape[0]
    
    # Load student tensors
    student_subblocks = os.path.join(student_path, 'subblocks_safetensors')
    student_ffn = os.path.join(student_subblocks, f'block_{layer}_ffn.safetensors')
    if not os.path.exists(student_ffn):
        print(f"FFN file not found at {student_ffn} - fallback to no_op")
        return [], 0, num_original_experts
    
    student_experts = {}
    with safe_open(student_ffn, framework='pt') as f:
        for key in f.keys():
            if 'experts' in key or 'router' in key:
                student_experts[key] = f.get_tensor(key)
    
    # Auto-detect number of student experts
    num_student_experts = student_experts[f'model.layers.{layer}.mlp.experts.gate_up_proj'].size(0)
    print(f"  Layer {layer}: Comparing {num_student_experts} student experts against {num_original_experts} original experts")
    
    # Pre-dequantize all original experts once (optimization)
    print(f"    Pre-dequantizing {num_original_experts} original experts...")
    deqexpert_mlp1 = convert_moe_packed_tensors(mlp1_blocks, mlp1_scales).cpu()        
    deqexpert_mlp2 = convert_moe_packed_tensors(mlp2_blocks, mlp2_scales).cpu()
    original_experts_dequant = []
    for orig_idx in range(num_original_experts):
        
        original_experts_dequant.append({
            'up': deqexpert_mlp1[orig_idx],
            'down': deqexpert_mlp2[orig_idx]
        })
    
    # For each student expert, find best matching original expert
    experts_to_keep = []
    used_original_indices = set()
    
    # Number of values to use for quick comparison (tune this)
    quick_compare_size = 8
    # Number of candidates to keep for full comparison
    top_k_candidates = min(10, num_original_experts)
    
    for student_idx in range(num_student_experts):
        # Get student expert weights
        prefix = f'model.layers.{layer}.mlp'
        student_up = student_experts.get(f'{prefix}.experts.gate_up_proj')[student_idx]
        student_down = student_experts.get(f'{prefix}.experts.down_proj')[student_idx]
        
        # if student_gate is None or student_up is None or student_down is None:
        if student_up is None or student_down is None:
            raise ValueError(f"Missing student expert weights for layer {layer} expert {student_idx}")
        
        # Step 1: Quick filtering using first N values
        candidate_scores = []
        for orig_idx in range(num_original_experts):
            if orig_idx in used_original_indices:
                continue
            
            orig_expert = original_experts_dequant[orig_idx]
            
            up_quick = (orig_expert['up'].flatten()[:quick_compare_size] - 
                       student_up.float().flatten()[:quick_compare_size]).pow(2).mean().sqrt()
            down_quick = (orig_expert['down'].flatten()[:quick_compare_size] - 
                         student_down.float().flatten()[:quick_compare_size]).pow(2).mean().sqrt()
            
            quick_score = (up_quick + down_quick) / 2.0
            candidate_scores.append((orig_idx, quick_score.item()))
        
        # Step 2: Get top-k candidates based on quick comparison
        candidate_scores.sort(key=lambda x: x[1])
        top_candidates = [idx for idx, _ in candidate_scores[:top_k_candidates]]
        
        # Step 3: Full comparison only on top candidates
        best_match_idx = None
        best_match_score = float('inf')
        
        for orig_idx in top_candidates:
            orig_expert = original_experts_dequant[orig_idx]
            
            # Full comparison across all values
            up_diff = (orig_expert['up'] - student_up.float()).pow(2).mean().sqrt()
            down_diff = (orig_expert['down'] - student_down.float()).pow(2).mean().sqrt()
            
            score = (up_diff + down_diff) / 2.0
            
            if score < best_match_score:
                best_match_score = score
                best_match_idx = orig_idx
        
        if best_match_idx is None:
            raise ValueError(f"Could not find match for student expert {student_idx} in layer {layer}")
        
        experts_to_keep.append(best_match_idx)
        used_original_indices.add(best_match_idx)
        print(f"    Student expert {student_idx} -> Original expert {best_match_idx} (RMSE: {best_match_score:.6f})")
    
    return experts_to_keep, num_student_experts, num_original_experts


def load_original_index(path: str) -> Dict[str, Any]:
    """Load the original model's safetensors index."""
    with open(path, 'r') as f:
        return json.load(f)


def load_layer_tensors(original_path: str, layer: int, index: Dict) -> Dict[str, torch.Tensor]:
    """Load all MoE-related tensors for a layer, potentially from multiple files."""
    keys_to_load = [
        f'model.layers.{layer}.mlp.experts.gate_up_proj_blocks',
        f'model.layers.{layer}.mlp.experts.gate_up_proj_scales',
        f'model.layers.{layer}.mlp.experts.gate_up_proj_bias',
        f'model.layers.{layer}.mlp.experts.down_proj_blocks',
        f'model.layers.{layer}.mlp.experts.down_proj_scales',
        f'model.layers.{layer}.mlp.experts.down_proj_bias',
        f'model.layers.{layer}.mlp.router.weight',  # Router weight
        f'model.layers.{layer}.mlp.router.bias',    # Router bias
    ]
    
    # Group by file
    file_to_keys = {}
    for key in keys_to_load:
        if key in index['weight_map']:
            filename = index['weight_map'][key]
            if filename not in file_to_keys:
                file_to_keys[filename] = []
            file_to_keys[filename].append(key)
    
    # Load from each file
    tensors = {}
    for filename, keys in file_to_keys.items():
        filepath = os.path.join(original_path, filename)
        with safe_open(filepath, framework='pt') as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
    
    return tensors


def copy_non_moe_weights(
    student_path: str,
    output_path: str,
    num_layers: int
) -> Dict[str, str]:
    """
    Copy non-MoE weights from student model.
    Returns weight_map for the new index.
    """
    weight_map = {}
    subblocks_dir = os.path.join(output_path, 'subblocks_safetensors')
    os.makedirs(subblocks_dir, exist_ok=True)
    
    student_subblocks = os.path.join(student_path, 'subblocks_safetensors')
    
    # Copy embeddings
    src_emb = os.path.join(student_subblocks, 'embeddings.safetensors')
    dst_emb = os.path.join(subblocks_dir, 'embeddings.safetensors')
    shutil.copy2(src_emb, dst_emb)
    with safe_open(src_emb, framework='pt') as f:
        for key in f.keys():
            weight_map[key] = 'subblocks_safetensors/embeddings.safetensors'
    
    # Copy lm_head
    src_head = os.path.join(student_subblocks, 'lm_head.safetensors')
    dst_head = os.path.join(subblocks_dir, 'lm_head.safetensors')
    shutil.copy2(src_head, dst_head)
    with safe_open(src_head, framework='pt') as f:
        for key in f.keys():
            weight_map[key] = 'subblocks_safetensors/lm_head.safetensors'
    
    # Copy attention blocks
    for layer in range(num_layers):
        src_attn = os.path.join(student_subblocks, f'block_{layer}_attention.safetensors')
        dst_attn = os.path.join(subblocks_dir, f'block_{layer}_attention.safetensors')
        shutil.copy2(src_attn, dst_attn)
        with safe_open(src_attn, framework='pt') as f:
            for key in f.keys():
                weight_map[key] = f'subblocks_safetensors/block_{layer}_attention.safetensors'
    
    return weight_map




def process_single_layer(
    layer: int,
    original_path: str,
    original_index: Dict,
    student_path: str,
    output_path: str,
    experts_to_keep: List[int],
) -> Tuple[Dict[str, str], List[str]]:
    """
    Process a single layer - loads tensors from potentially multiple files.
    Returns (weight_map, verification_errors).
    """
    weight_map = {}
    verification_errors = []
    subblocks_dir = os.path.join(output_path, 'subblocks_safetensors')
    student_subblocks = os.path.join(student_path, 'subblocks_safetensors')
    
    # Load all tensors for this layer (may come from multiple files)
    orig_tensors = load_layer_tensors(original_path, layer, original_index)
    
    # Load student FFN file
    student_ffn = os.path.join(student_subblocks, f'block_{layer}_ffn.safetensors')
    
    tensors_to_save = {}
    student_tensors = {}
    
    with safe_open(student_ffn, framework='pt') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if 'experts' not in key and 'router' not in key:
                # Copy norm weights
                tensors_to_save[key] = tensor
    
    # Get router from original model, sliced to kept experts
    orig_router_weight = orig_tensors[f'model.layers.{layer}.mlp.router.weight']
    orig_router_bias = orig_tensors[f'model.layers.{layer}.mlp.router.bias']

    kept_indices_tensor = torch.tensor(experts_to_keep, dtype=torch.long)
    sliced_router_weight = orig_router_weight[kept_indices_tensor]
    sliced_router_bias = orig_router_bias[kept_indices_tensor]
    
    tensors_to_save[f'model.layers.{layer}.mlp.router.weight'] = sliced_router_weight
    tensors_to_save[f'model.layers.{layer}.mlp.router.bias'] = sliced_router_bias
    
    # Get MoE tensors
    mlp1_blocks = orig_tensors[f'model.layers.{layer}.mlp.experts.gate_up_proj_blocks']
    mlp1_scales = orig_tensors[f'model.layers.{layer}.mlp.experts.gate_up_proj_scales']
    mlp2_blocks = orig_tensors[f'model.layers.{layer}.mlp.experts.down_proj_blocks']
    mlp2_scales = orig_tensors[f'model.layers.{layer}.mlp.experts.down_proj_scales']  
    mlp1_bias = orig_tensors[f'model.layers.{layer}.mlp.experts.gate_up_proj_bias']
    mlp2_bias = orig_tensors[f'model.layers.{layer}.mlp.experts.down_proj_bias']

    tensors_to_save[f'model.layers.{layer}.mlp.experts.gate_up_proj_blocks'] = mlp1_blocks[kept_indices_tensor]
    tensors_to_save[f'model.layers.{layer}.mlp.experts.gate_up_proj_scales'] = mlp1_scales[kept_indices_tensor]
    tensors_to_save[f'model.layers.{layer}.mlp.experts.gate_up_proj_bias'] = mlp1_bias[kept_indices_tensor]
    
    tensors_to_save[f'model.layers.{layer}.mlp.experts.down_proj_blocks'] = mlp2_blocks[kept_indices_tensor]
    tensors_to_save[f'model.layers.{layer}.mlp.experts.down_proj_scales'] = mlp2_scales[kept_indices_tensor]
    tensors_to_save[f'model.layers.{layer}.mlp.experts.down_proj_bias'] = mlp2_bias[kept_indices_tensor]
    
    # Save the FFN file
    output_file = os.path.join(subblocks_dir, f'block_{layer}_ffn.safetensors')
    save_file(tensors_to_save, output_file)
    
    # Build weight map
    for key in tensors_to_save.keys():
        weight_map[key] = f'subblocks_safetensors/block_{layer}_ffn.safetensors'
    
    return weight_map, verification_errors


def copy_config_files(student_path: str, output_path: str):
    """Copy configuration files from student model and update config.json."""
    files_to_copy = [
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'chat_template.jinja',
    ]
    
    # Also copy transformers compatibility files
    if os.path.exists(student_path):
        for f in os.listdir(student_path):
            if f.startswith('transformers_'):
                files_to_copy.append(f)
    
    for filename in files_to_copy:
        src = os.path.join(student_path, filename)
        dst = os.path.join(output_path, filename)
        
        # Try student path first
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
                continue
            except PermissionError:
                pass
        
        # If we get here, file doesn't exist or permission denied
        if not os.path.exists(dst):
            print(f"  Warning: Could not copy {filename}")
            
    # Update config.json for DeciGptOssForCausalLM with MXFP4
    src_config = os.path.join(student_path, 'config.json')
    if not os.path.exists(src_config):
        raise FileNotFoundError(f"config.json not found at {src_config}")
    
    with open(src_config, 'r') as f:
        config = json.load(f)
 
    # Set architecture to DeciGptOssForCausalLM for MXFP4 support
    config['architectures'] = ['DeciGptOssForCausalLM']
    
    # Add quantization_config so vllm calls _load_weights_mxfp4
    config['quantization_config'] = {
        "quant_method": "mxfp4",
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ]
    }
    
    dst_config = os.path.join(output_path, 'config.json')
    with open(dst_config, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Create MXFP4 checkpoint from student model')
    parser.add_argument(
        '--student-path',
        type=str,
        required=True,
        help='Path to student model checkpoint'
    )
    parser.add_argument(
        '--original-path',
        type=str,
        required=True,
        help='Path to original gpt-oss-120b model with MXFP4 weights'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Output path for the new checkpoint'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=36,
        help='Number of transformer layers'
    )
    args = parser.parse_args()
    
    print(f"Creating MXFP4 checkpoint...")
    print(f"  Student model: {args.student_path}")
    print(f"  Original model: {args.original_path}")
    print(f"  Output: {args.output_path}")
    
   
    # Load original model index
    original_index = load_original_index(
        os.path.join(args.original_path, 'model.safetensors.index.json')
    )
    
    print("\nDeducing expert mappings by comparing weights...")
    experts_to_keep = []
    layer_statistics = []  # Store (num_student, num_original) for each layer
    
    for layer in range(args.num_layers):
        layer_experts, num_student, num_original = deduce_experts_for_layer(
            layer,
            args.original_path,
            original_index,
            args.student_path,
        )
        experts_to_keep.append(layer_experts)
        layer_statistics.append((num_student, num_original))
    
    # Print statistics
    print(f"\n{'='*70}")
    print("EXPERT DEDUCTION STATISTICS")
    print(f"{'='*70}")
    print(f"{'Layer':<8} {'Student Experts':<18} {'Original Experts':<18} {'Kept %':<10}")
    print(f"{'-'*70}")
    
    total_student = 0
    total_original = 0
    for layer, (num_student, num_original) in enumerate(layer_statistics):
        percentage = (num_student / num_original * 100) if num_original > 0 else 0
        print(f"{layer:<8} {num_student:<18} {num_original:<18} {percentage:<10.2f}")
        total_student += num_student
        total_original += num_original
    
    print(f"{'-'*70}")
    avg_percentage = (total_student / total_original * 100) if total_original > 0 else 0
    print(f"{'TOTAL':<8} {total_student:<18} {total_original:<18} {avg_percentage:<10.2f}")
    print(f"{'='*70}")
    print(f"\n  Deduced experts_to_keep mapping for {len(experts_to_keep)} layers")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'subblocks_safetensors'), exist_ok=True)
    
    # Copy config files
    print("Copying configuration files...")
    copy_config_files(args.student_path, args.output_path)
    
    # Save experts_to_keep.json
    experts_to_keep_output = os.path.join(args.output_path, 'experts_to_keep.json')
    with open(experts_to_keep_output, 'w') as f:
        json.dump(experts_to_keep, f, indent=2)
    print(f"  Saved experts_to_keep mapping to {experts_to_keep_output}")
    
    # Copy non-MoE weights (embeddings, attention, lm_head)
    print("Copying non-MoE weights...")
    weight_map = copy_non_moe_weights(
        args.student_path,
        args.output_path,
        args.num_layers
    )
    
    # Load weights per layer (handles multi-file loading)
    print(f"Processing {args.num_layers} layers...")
    
    all_verification_errors = []
    
    # Process each layer
    for layer in tqdm(range(args.num_layers), desc="Processing layers"):
        if len(experts_to_keep[layer]) == 0:
            print(f"Layer {layer} has no experts to keep - ffn->no_op")
            continue
        layer_weight_map, layer_errors = process_single_layer(
            layer,
            args.original_path,
            original_index,
            args.student_path,
            args.output_path,
            experts_to_keep[layer],
        )
        weight_map.update(layer_weight_map)
        all_verification_errors.extend(layer_errors)
    
    # Calculate total size
    total_size = 0
    subblocks_dir = os.path.join(args.output_path, 'subblocks_safetensors')
    for filename in os.listdir(subblocks_dir):
        filepath = os.path.join(subblocks_dir, filename)
        total_size += os.path.getsize(filepath)
    
    # Create model.safetensors.index.json
    index = {
        'metadata': {
            'total_size': total_size
        },
        'weight_map': weight_map
    }
    
    index_path = os.path.join(args.output_path, 'model.safetensors.index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\nCheckpoint created successfully at: {args.output_path}")
    print(f"Total size: {total_size / 1e9:.2f} GB")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Download and preprocess NVIDIA Nemotron-Post-Training-Dataset-v2 for QAD training.

Each split is saved to its own folder for fine-grained control over datablends.

Splits available:
- stem: Science, reasoning, humanities (English)
- math: Step-by-step math solutions (English)
- code: Programming challenges (English)
- chat: Conversational tuning (English)
- multilingual_ja: Japanese
- multilingual_de: German
- multilingual_it: Italian
- multilingual_es: Spanish
- multilingual_fr: French

NOTE: This dataset is GATED. You need to:
1. Go to https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2
2. Request access and wait for approval
3. Login with: huggingface-cli login

Usage:
    # Download all English splits (each to separate folder)
    python download_nemotron_v2.py --sample-percent 30
    
    # Download specific splits
    python download_nemotron_v2.py --splits stem,math --sample-percent 50
    
    # Include multilingual
    python download_nemotron_v2.py --sample-percent 30 --include-multilingual

Output structure:
    nemotron_v2/
    ├── stem/
    │   ├── stem_30pct_train.jsonl
    │   ├── stem_30pct_validation.jsonl
    │   └── stem_30pct_test.jsonl
    ├── math/
    │   ├── math_30pct_train.jsonl
    │   └── ...
    └── ...

Datablend configs:
    datasets/
    ├── datablend_nemotron_v2_stem_30pct.json      # Per-split configs
    ├── datablend_nemotron_v2_math_30pct.json
    └── datablend_nemotron_v2_all_en_30pct.json    # Combined config
"""

import argparse
import json
import os
from datasets import load_dataset, get_dataset_config_names, load_dataset_builder
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = "/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/nemotron_v2"
DATABLEND_DIR = "/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets"
DATASET_NAME = "nvidia/Nemotron-Post-Training-Dataset-v2"

# Known splits (actual sizes will be fetched from HuggingFace)
ENGLISH_SPLITS = ["stem", "math", "code", "chat"]
MULTILINGUAL_SPLITS = ["multilingual_ja", "multilingual_de", "multilingual_it", 
                       "multilingual_es", "multilingual_fr"]
ALL_SPLIT_NAMES = ENGLISH_SPLITS + MULTILINGUAL_SPLITS


def get_split_sizes(splits_to_check: list) -> dict:
    """Fetch actual split sizes from HuggingFace dataset info."""
    print("\n📊 Fetching actual dataset sizes from HuggingFace...")
    
    split_sizes = {}
    
    for split_name in splits_to_check:
        try:
            # Try to get dataset info without downloading
            builder = load_dataset_builder(DATASET_NAME, split_name)
            info = builder.info
            
            # Get the split info
            if info.splits and split_name in info.splits:
                split_sizes[split_name] = info.splits[split_name].num_examples
                print(f"  ✓ {split_name}: {split_sizes[split_name]:,} samples")
            else:
                # If split info not available, try loading a small sample to estimate
                print(f"  ⚠ {split_name}: size not in metadata, will count during download")
                split_sizes[split_name] = None
                
        except Exception as e:
            if "gated" in str(e).lower() or "access" in str(e).lower():
                print(f"\n❌ ACCESS DENIED - Please request access at:")
                print(f"   https://huggingface.co/datasets/{DATASET_NAME}")
                print("   Then login with: huggingface-cli login")
                raise
            else:
                print(f"  ⚠ {split_name}: could not fetch size ({e})")
                split_sizes[split_name] = None
    
    return split_sizes

# Train/valid/test split ratios
TRAIN_RATIO = 0.95
VALID_RATIO = 0.025
TEST_RATIO = 0.025
RANDOM_SEED = 42

# Global tokenizer for chat template (initialized if --tokenizer is provided)
_TOKENIZER = None


def init_tokenizer(tokenizer_name: str):
    """Initialize tokenizer for chat template formatting."""
    global _TOKENIZER
    if tokenizer_name:
        from transformers import AutoTokenizer
        print(f"📝 Loading tokenizer for chat template: {tokenizer_name}")
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Show example
        example = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        formatted = _TOKENIZER.apply_chat_template(example, tokenize=False)
        print(f"   Example format:\n   {formatted[:200]}...")


def format_messages_to_text(messages: list, reasoning: str = None) -> str:
    """Convert messages format to text for QAD training.
    
    If a tokenizer is initialized, uses its chat template.
    Otherwise, uses simple role-based formatting.
    """
    global _TOKENIZER
    
    # Optionally prepend reasoning/chain-of-thought as thinking block
    if reasoning and reasoning.strip():
        # Insert thinking block before last assistant message
        messages_with_cot = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and i == len(messages) - 1:
                # Add thinking before final assistant response
                thinking_content = f"<think>\n{reasoning}\n</think>\n{msg.get('content', '')}"
                messages_with_cot.append({"role": "assistant", "content": thinking_content})
            else:
                messages_with_cot.append(msg)
        messages = messages_with_cot
    
    # Use chat template if tokenizer is available
    if _TOKENIZER is not None:
        try:
            return _TOKENIZER.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            print(f"Warning: Chat template failed, using simple format: {e}")
    
    # Fallback: simple role-based format
    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            text_parts.append(f"System: {content}")
        elif role == "user":
            text_parts.append(f"User: {content}")
        elif role == "assistant":
            text_parts.append(f"Assistant: {content}")
    
    return "\n\n".join(text_parts)


def download_split(split_name: str, max_samples: int, output_dir: str, 
                   pct_str: str, include_reasoning: bool = False,
                   sample_percent: float = None) -> dict:
    """Download a single split and save to its own folder.
    
    Args:
        split_name: Name of the split to download
        max_samples: Maximum samples to download (None = download all, then sample)
        output_dir: Output directory
        pct_str: Percentage string for filenames
        include_reasoning: Include chain-of-thought reasoning
        sample_percent: If max_samples is None, use this percentage after counting
    """
    
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    if max_samples is not None:
        print(f"\n📥 Loading split: {split_name} (target: {max_samples:,} samples)")
    else:
        print(f"\n📥 Loading split: {split_name} (downloading all, will sample {sample_percent}%)")
    
    examples = []
    
    try:
        # Load the specific split
        dataset = load_dataset(
            DATASET_NAME,
            split=split_name,
            streaming=True  # Use streaming for large datasets
        )
        
        count = 0
        for example in tqdm(dataset, desc=f"Processing {split_name}", total=max_samples):
            if max_samples is not None and count >= max_samples:
                break
            
            messages = example.get("messages", [])
            reasoning = example.get("reasoning", "") if include_reasoning else ""
            
            # Convert to text format
            text = format_messages_to_text(messages, reasoning)
            
            if text.strip():
                examples.append({
                    "text": text,
                    "category": example.get("category", split_name),
                    "source": "nemotron_v2",
                    "split": split_name,
                    "language": "multilingual" if "multilingual" in split_name else "en"
                })
                count += 1
        
        print(f"✓ Collected {count:,} examples from {split_name}")
        
        # If we downloaded all and need to sample
        if max_samples is None and sample_percent is not None:
            import random
            random.seed(RANDOM_SEED)
            target_samples = int(len(examples) * sample_percent / 100)
            if target_samples < len(examples):
                examples = random.sample(examples, target_samples)
                print(f"  Sampled {len(examples):,} examples ({sample_percent}% of {count:,})")
        
    except Exception as e:
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print(f"\n❌ ACCESS DENIED for {split_name}")
            print("   Please request access at:")
            print("   https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2")
            print("   Then login with: huggingface-cli login")
            return None
        else:
            print(f"Error loading {split_name}: {e}")
            return None
    
    if not examples:
        print(f"Warning: No examples collected from {split_name}")
        return None
    
    # Shuffle and split into train/valid/test
    import random
    random.seed(RANDOM_SEED)
    random.shuffle(examples)
    
    total_size = len(examples)
    train_end = int(total_size * TRAIN_RATIO)
    valid_end = train_end + int(total_size * VALID_RATIO)
    
    splits = {
        'train': examples[:train_end],
        'validation': examples[train_end:valid_end],
        'test': examples[valid_end:]
    }
    
    # Save each split
    saved_files = {}
    for data_split, data in splits.items():
        output_file = os.path.join(split_dir, f"{split_name}{pct_str}_{data_split}.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in data:
                json_line = json.dumps({"text": ex["text"]}, ensure_ascii=False)
                f.write(json_line + '\n')
        
        saved_files[data_split] = output_file
        print(f"  Saved {data_split}: {len(data):,} examples -> {output_file}")
    
    return {
        'split_name': split_name,
        'total': len(examples),
        'train': len(splits['train']),
        'validation': len(splits['validation']),
        'test': len(splits['test']),
        'files': saved_files
    }


def create_datablend_config(split_info: dict, output_dir: str, pct_str: str) -> str:
    """Create datablend config for a single split."""
    split_name = split_info['split_name']
    
    # Preprocessed path pattern
    preprocessed_dir = output_dir.replace("nemotron_v2", "nemotron_v2_preprocessed")
    split_preprocessed_dir = os.path.join(preprocessed_dir, split_name)
    
    blend_config = {
        "train": [1.0, f"{split_preprocessed_dir}/{split_name}{pct_str}_train_text_document"],
        "valid": [1.0, f"{split_preprocessed_dir}/{split_name}{pct_str}_validation_text_document"],
        "test": [1.0, f"{split_preprocessed_dir}/{split_name}{pct_str}_test_text_document"]
    }
    
    blend_file = os.path.join(DATABLEND_DIR, f"datablend_nemotron_v2_{split_name}{pct_str}.json")
    with open(blend_file, 'w') as f:
        json.dump(blend_config, f, indent=2)
    
    return blend_file


def create_combined_datablend(all_split_infos: list, output_dir: str, pct_str: str, 
                               suffix: str = "all_en") -> str:
    """Create combined datablend config for multiple splits with equal weighting."""
    
    preprocessed_dir = output_dir.replace("nemotron_v2", "nemotron_v2_preprocessed")
    
    # Calculate total samples for weighting
    total_train = sum(info['train'] for info in all_split_infos)
    
    train_blend = []
    valid_blend = []
    test_blend = []
    
    for info in all_split_infos:
        split_name = info['split_name']
        split_preprocessed_dir = os.path.join(preprocessed_dir, split_name)
        
        # Weight proportional to sample count
        weight = info['train'] / total_train if total_train > 0 else 1.0 / len(all_split_infos)
        
        train_blend.extend([weight, f"{split_preprocessed_dir}/{split_name}{pct_str}_train_text_document"])
        valid_blend.extend([weight, f"{split_preprocessed_dir}/{split_name}{pct_str}_validation_text_document"])
        test_blend.extend([weight, f"{split_preprocessed_dir}/{split_name}{pct_str}_test_text_document"])
    
    blend_config = {
        "train": train_blend,
        "valid": valid_blend,
        "test": test_blend
    }
    
    blend_file = os.path.join(DATABLEND_DIR, f"datablend_nemotron_v2_{suffix}{pct_str}.json")
    with open(blend_file, 'w') as f:
        json.dump(blend_config, f, indent=2)
    
    return blend_file


def main():
    parser = argparse.ArgumentParser(description="Download Nemotron-v2 for QAD (per-split folders)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for JSONL files")
    parser.add_argument("--splits", type=str, default="stem,math,code,chat",
                        help="Comma-separated list of English splits to download")
    parser.add_argument("--include-multilingual", action="store_true",
                        help="Include all multilingual splits (ja, de, it, es, fr)")
    parser.add_argument("--sample-percent", type=float, default=30.0,
                        help="Percentage of each split to use (1-100). Default: 30%%")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples per split (absolute cap)")
    parser.add_argument("--include-reasoning", action="store_true", default=True,
                        help="Include chain-of-thought reasoning in output (default: True)")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="Exclude chain-of-thought reasoning from output")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="HuggingFace tokenizer to use for chat template (e.g., Qwen/Qwen3-8B). "
                             "If not specified, uses simple role-based formatting.")
    args = parser.parse_args()
    
    # Handle reasoning flag (--no-reasoning overrides default)
    include_reasoning = args.include_reasoning and not args.no_reasoning
    
    # Initialize tokenizer if specified
    if args.tokenizer:
        init_tokenizer(args.tokenizer)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Build list of splits to download
    splits_to_download = [s.strip() for s in args.splits.split(",")]
    if args.include_multilingual:
        splits_to_download.extend(MULTILINGUAL_SPLITS.keys())
    
    # Remove duplicates while preserving order
    splits_to_download = list(dict.fromkeys(splits_to_download))
    
    pct_str = f"_{int(args.sample_percent)}pct"
    reasoning_str = "_cot" if include_reasoning else ""  # chain-of-thought suffix
    chat_str = "_chat" if args.tokenizer else ""  # chat template suffix
    
    print("=" * 70)
    print("Downloading NVIDIA Nemotron-Post-Training-Dataset-v2")
    print("=" * 70)
    print("⚠️  NOTE: This dataset requires HuggingFace access approval!")
    print("   If you get an access error, visit:")
    print("   https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2")
    print("=" * 70)
    print(f"Splits: {splits_to_download}")
    print(f"Sample percent: {args.sample_percent}%")
    print(f"Include reasoning: {include_reasoning}")
    print(f"Chat template: {args.tokenizer or 'Simple role-based format'}")
    print(f"Output directory: {output_dir}")
    print(f"Each split saved to: {output_dir}/<split_name>/")
    print("=" * 70)
    
    # Get actual split sizes from HuggingFace
    try:
        actual_sizes = get_split_sizes(splits_to_download)
    except Exception as e:
        print(f"\n❌ Failed to fetch dataset info: {e}")
        return
    
    # Calculate samples per split based on actual sizes
    print(f"\nTarget samples per split:")
    samples_per_split = {}
    for split_name in splits_to_download:
        available = actual_sizes.get(split_name)
        if available is None:
            print(f"  ⚠ {split_name}: size unknown, will download all and sample")
            # If size unknown, set a large number and we'll sample during download
            samples_per_split[split_name] = None  # Download all, then sample
            continue
        
        if args.max_samples is not None:
            samples_per_split[split_name] = min(available, args.max_samples)
        else:
            samples_per_split[split_name] = int(available * args.sample_percent / 100)
        pct = samples_per_split[split_name] / available * 100
        print(f"  {split_name}: {samples_per_split[split_name]:,} ({pct:.1f}% of {available:,})")
    
    print("=" * 70)
    
    # Download each split to its own folder
    all_split_infos = []
    
    for split_name in splits_to_download:
        if split_name not in samples_per_split:
            continue
        
        max_for_split = samples_per_split.get(split_name)
        split_info = download_split(
            split_name=split_name,
            max_samples=max_for_split,
            output_dir=output_dir,
            pct_str=pct_str + reasoning_str + chat_str,  # Include reasoning and chat template suffix
            include_reasoning=include_reasoning,
            sample_percent=args.sample_percent if max_for_split is None else None
        )
        
        if split_info:
            all_split_infos.append(split_info)
            
            # Create per-split datablend config
            blend_file = create_datablend_config(split_info, output_dir, pct_str + reasoning_str + chat_str)
            print(f"  📝 Datablend config: {blend_file}")
    
    if not all_split_infos:
        print("\n❌ Error: No splits were successfully downloaded!")
        return
    
    # Create combined datablend config
    print("\n" + "=" * 70)
    print("Creating combined datablend configs...")
    
    # English-only combined
    full_suffix = pct_str + reasoning_str + chat_str
    en_splits = [info for info in all_split_infos if "multilingual" not in info['split_name']]
    if en_splits:
        combined_file = create_combined_datablend(en_splits, output_dir, full_suffix, "all_en")
        print(f"📝 Combined English datablend: {combined_file}")
    
    # All splits combined (if multilingual included)
    if len(all_split_infos) > len(en_splits):
        combined_all_file = create_combined_datablend(all_split_infos, output_dir, full_suffix, "all_multilingual")
        print(f"📝 Combined all datablend: {combined_all_file}")
    
    # Save metadata JSON with sample counts
    total_samples = sum(info['total'] for info in all_split_infos)
    total_train = sum(info['train'] for info in all_split_infos)
    total_valid = sum(info['validation'] for info in all_split_infos)
    total_test = sum(info['test'] for info in all_split_infos)
    
    metadata = {
        "dataset": DATASET_NAME,
        "sample_percent": args.sample_percent,
        "include_reasoning": include_reasoning,
        "chat_template": args.tokenizer or "none (simple role format)",
        "download_date": __import__('datetime').datetime.now().isoformat(),
        "total_samples": total_samples,
        "total_train": total_train,
        "total_validation": total_valid,
        "total_test": total_test,
        "splits": {}
    }
    
    for info in all_split_infos:
        split_name = info['split_name']
        metadata["splits"][split_name] = {
            "available_in_dataset": actual_sizes.get(split_name),  # Actual HF count
            "downloaded": info['total'],
            "train": info['train'],
            "validation": info['validation'],
            "test": info['test'],
            "files": info['files']
        }
    
    metadata_file = os.path.join(output_dir, f"metadata{full_suffix}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📊 Metadata saved: {metadata_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ Download complete!")
    print("=" * 70)
    
    print(f"\nSummary:")
    print(f"  Total splits downloaded: {len(all_split_infos)}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total train samples: {total_train:,}")
    
    print(f"\nPer-split breakdown:")
    for info in all_split_infos:
        print(f"  {info['split_name']}:")
        print(f"    Total: {info['total']:,} | Train: {info['train']:,} | Valid: {info['validation']:,} | Test: {info['test']:,}")
    
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    for info in all_split_infos:
        print(f"  └── {info['split_name']}/")
        print(f"      ├── {info['split_name']}{full_suffix}_train.jsonl")
        print(f"      ├── {info['split_name']}{full_suffix}_validation.jsonl")
        print(f"      └── {info['split_name']}{full_suffix}_test.jsonl")
    
    print(f"\nNext steps:")
    print(f"1. Preprocess each split:")
    for info in all_split_infos:
        print(f"   bash process_nemotron_v2_qwen3-8B.sh {info['split_name']} {full_suffix.replace('_', '')}")
    print(f"\n2. Or use individual datablend configs:")
    for info in all_split_infos:
        print(f"   DATASET_NAME=nemotron_v2_{info['split_name']}{full_suffix}")
    print(f"\n3. Or use combined config:")
    print(f"   DATASET_NAME=nemotron_v2_all_en{full_suffix}")
    print("=" * 70)


if __name__ == "__main__":
    main()

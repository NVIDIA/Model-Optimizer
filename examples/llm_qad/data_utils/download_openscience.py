#!/usr/bin/env python3
"""
Download and preprocess NVIDIA OpenScience dataset for QAD training.

Usage:
    # Simple format (default)
    python download_openscience.py
    
    # With chat template (Qwen format)
    python download_openscience.py --tokenizer Qwen/Qwen3-8B
"""

import argparse
from datasets import load_dataset
import json
import os
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = None  # Must be specified via --output-dir

# Split configuration
TRAIN_RATIO = 0.95
VALID_RATIO = 0.025
TEST_RATIO = 0.025
RANDOM_SEED = 42

# Global tokenizer for chat template
_TOKENIZER = None


def init_tokenizer(tokenizer_name: str):
    """Initialize tokenizer for chat template formatting."""
    global _TOKENIZER
    if tokenizer_name:
        from transformers import AutoTokenizer
        print(f"📝 Loading tokenizer for chat template: {tokenizer_name}")
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def format_example(example: dict) -> str:
    """Format a single example to text."""
    global _TOKENIZER
    
    # OpenScience has input/output format
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    
    if _TOKENIZER is not None:
        # Use chat template
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        try:
            return _TOKENIZER.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            print(f"Warning: Chat template failed: {e}")
    
    # Simple format
    return f"User: {input_text}\n\nAssistant: {output_text}"


def main():
    parser = argparse.ArgumentParser(description="Download OpenScience dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory (required)")
    parser.add_argument("--datablend-dir", type=str, required=True,
                        help="Directory for datablend config files (required)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="HuggingFace tokenizer for chat template (e.g., Qwen/Qwen3-8B)")
    args = parser.parse_args()
    
    OUTPUT_DIR = args.output_dir
    DATABLEND_DIR = args.datablend_dir
    chat_suffix = "_chat" if args.tokenizer else ""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATABLEND_DIR, exist_ok=True)
    
    if args.tokenizer:
        init_tokenizer(args.tokenizer)
    
    print("Loading NVIDIA/OpenScience dataset...")
    try:
        dataset = load_dataset("nvidia/OpenScience", "OS-Q3-235B-4")
        
        # create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Handle different dataset structures
        if 'train' in dataset:
            full_data = dataset['train']
        else:
            # If no 'train' key, use the first available split
            split_name = list(dataset.keys())[0]
            print(f"No 'train' split found, using '{split_name}' split")
            full_data = dataset[split_name]
        
        print(f"Shuffling {len(full_data)} examples with seed {RANDOM_SEED}...")
        shuffled_data = full_data.shuffle(seed=RANDOM_SEED)
        
        total_size = len(shuffled_data)
        train_end = int(total_size * TRAIN_RATIO)
        valid_end = train_end + int(total_size * VALID_RATIO)
        
        splits_config = {
            'train': shuffled_data.select(range(0, train_end)),
            'validation': shuffled_data.select(range(train_end, valid_end)),
            'test': shuffled_data.select(range(valid_end, total_size))
        }
        
        print(f"\nCreated splits:")
        for name, data in splits_config.items():
            print(f"  {name}: {len(data)} examples ({len(data)/total_size*100:.2f}%)")
        
        print(f"\nFormat: {'Chat template' if args.tokenizer else 'Simple role format'}")
        
        # Save splits to JSONL
        for split_name, split_data in splits_config.items():
            output_file = os.path.join(OUTPUT_DIR, f"openscience{chat_suffix}_{split_name}.jsonl")
            print(f"\nWriting {output_file}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in tqdm(split_data, desc=split_name):
                    # Format using chat template or simple format
                    text = format_example(example)
                    
                    json_line = json.dumps({"text": text}, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            print(f"✓ Saved {len(split_data)} examples")
        
        # Create datablend config
        preprocessed_dir = OUTPUT_DIR.replace("openscience_splits", "openscience_splits_preprocessed")
        blend_file = os.path.join(DATABLEND_DIR, f"datablend_openscience{chat_suffix}.json")
        blend_config = {
            "train": [1.0, f"{preprocessed_dir}/openscience{chat_suffix}_train_text_document"],
            "valid": [1.0, f"{preprocessed_dir}/openscience{chat_suffix}_validation_text_document"],
            "test": [1.0, f"{preprocessed_dir}/openscience{chat_suffix}_test_text_document"]
        }
        with open(blend_file, 'w') as f:
            json.dump(blend_config, f, indent=2)
        print(f"📝 Created datablend config: {blend_file}")
        
        print("\n✓ Dataset splitting complete!")
        print(f"\nOutput files: openscience{chat_suffix}_*.jsonl")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative loading method...")
        raise


if __name__ == "__main__":
    main()

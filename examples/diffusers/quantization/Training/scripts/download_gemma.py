#!/usr/bin/env python3
"""
Script to download Gemma 3 12B model and tokenizer.

Note: The Gemma 3 12B model is gated and requires:
1. Accepting the model terms at https://huggingface.co/google/gemma-3-12b-it
2. Authenticating with Hugging Face using: huggingface-cli login
   OR setting the HF_TOKEN environment variable
"""

import argparse
import os
from pathlib import Path
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
from huggingface_hub import login

def download_gemma_model(token=None):
    """Download Gemma 3 12B model and tokenizer to gemma/model/ directory.

    Args:
        token: Hugging Face access token (optional, can also use HF_TOKEN env var)
    """

    # Model identifier for Gemma 3 12B
    model_name = "google/gemma-3-12b-it"

    # Target directory structure: gemma/model/
    base_dir = Path("gemma")
    model_dir = base_dir / "model"

    # Create directories if they don't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Gemma 3 12B model from {model_name}...")
    print(f"Target directory: {model_dir.absolute()}")

    # Check for HF token from command line, environment variable, or cached login
    hf_token = token or os.getenv("HF_TOKEN")
    if hf_token:
        print("\nAuthenticating with Hugging Face token...")
        login(token=hf_token)
    else:
        print("\nNote: Using cached Hugging Face credentials if available.")
        print("If authentication fails, provide token via:")
        print("  --token YOUR_TOKEN")
        print("  or export HF_TOKEN=YOUR_TOKEN")
        print("  or run: huggingface-cli login")

    try:
        # Download model
        print("\nDownloading model...")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=None,  # Download directly to target directory
            local_files_only=False,
        )

        # Save model to target directory
        print(f"\nSaving model to {model_dir}...")
        model.save_pretrained(str(model_dir))

        # Download and save tokenizer
        print("\nDownloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(model_dir))

        print(f"\n✓ Successfully downloaded Gemma 3 12B model and tokenizer to {model_dir.absolute()}")
        print(f"\nModel files:")
        for file in sorted(model_dir.iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")

    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower():
            print("\n" + "="*70)
            print("AUTHENTICATION REQUIRED")
            print("="*70)
            print("\nThe Gemma 3 12B model is gated and requires:")
            print("\n1. Accept the model terms:")
            print(f"   Visit: https://huggingface.co/{model_name}")
            print("   Click 'Agree and access repository'")
            print("\n2. Authenticate with Hugging Face:")
            print("   Option A: Set environment variable:")
            print("     export HF_TOKEN=your_huggingface_token")
            print("     python3 download_gemma.py")
            print("\n   Option B: Use huggingface-cli:")
            print("     huggingface-cli login")
            print("     python3 download_gemma.py")
            print("\n   To get a token, visit: https://huggingface.co/settings/tokens")
            print("\n" + "="*70)
        else:
            print(f"\n✗ Error downloading model: {e}")

        raise Exception("Authentication required. Please follow the instructions above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gemma 3 12B model and tokenizer")
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face access token (or set HF_TOKEN environment variable)",
    )
    args = parser.parse_args()

    download_gemma_model(token=args.token)

"""Extract MTP (Multi-Token Prediction) head weights from an FP8 checkpoint.

The GLM-5 model has 78 transformer layers (0-77) plus one MTP layer (layer 78)
used for speculative decoding. During NVFP4 quantization, only layers 0-77 are
quantized. This script extracts the MTP head from a separate FP8 checkpoint and
adds it to the NVFP4 output so the final checkpoint is complete.

Usage:
    python extract_mtp_head.py \
        --fp8_index /path/to/glm-5-fp8/model.safetensors.index.json \
        --fp8_dir /path/to/glm-5-fp8 \
        --nvfp4_dir /path/to/glm-5-nvfp4 \
        --mtp_layer 78
"""

import argparse
import json

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8_index", type=str, required=True,
                        help="Path to the FP8 model.safetensors.index.json")
    parser.add_argument("--fp8_dir", type=str, required=True,
                        help="Directory containing FP8 safetensors files")
    parser.add_argument("--nvfp4_dir", type=str, required=True,
                        help="NVFP4 output directory to add MTP head to")
    parser.add_argument("--mtp_layer", type=int, default=78,
                        help="Layer index of the MTP head (default: 78)")
    parser.add_argument("--output_name", type=str, default="mtp-fp8.safetensors",
                        help="Filename for the MTP safetensors file")
    args = parser.parse_args()

    mtp_prefix = f"model.layers.{args.mtp_layer}"

    # Find MTP keys in the FP8 index
    with open(args.fp8_index) as f:
        idx = json.load(f)

    mtp_keys_by_file = {}
    for key, filename in idx["weight_map"].items():
        if key.startswith(mtp_prefix):
            mtp_keys_by_file.setdefault(filename, []).append(key)

    if not mtp_keys_by_file:
        print(f"No MTP keys found with prefix '{mtp_prefix}'")
        return

    total_keys = sum(len(v) for v in mtp_keys_by_file.values())
    print(f"Found {total_keys} MTP keys across {len(mtp_keys_by_file)} files")

    # Extract MTP tensors
    mtp_tensors = {}
    for filename, keys in sorted(mtp_keys_by_file.items()):
        filepath = f"{args.fp8_dir}/{filename}"
        print(f"  Loading {len(keys)} keys from {filename}...")
        data = load_file(filepath, device="cpu")
        for k in keys:
            mtp_tensors[k] = data[k]
        del data

    # Save as single file
    out_path = f"{args.nvfp4_dir}/{args.output_name}"
    save_file(mtp_tensors, out_path)
    print(f"Saved {len(mtp_tensors)} MTP tensors to {out_path}")

    # Update the NVFP4 index
    nvfp4_index_path = f"{args.nvfp4_dir}/model.safetensors.index.json"
    with open(nvfp4_index_path) as f:
        nvfp4_idx = json.load(f)

    for k in mtp_tensors:
        nvfp4_idx["weight_map"][k] = args.output_name

    with open(nvfp4_index_path, "w") as f:
        json.dump(nvfp4_idx, f, indent=2)

    print(f"Updated {nvfp4_index_path} with MTP keys -> {args.output_name}")


if __name__ == "__main__":
    main()

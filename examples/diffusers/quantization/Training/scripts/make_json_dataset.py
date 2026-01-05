#!/usr/bin/env python3
"""Create a JSON dataset in the correct format from .feather files."""

import argparse
import json
import pandas as pd
from pathlib import Path

# ----------
# Read the .feather files for a dataset to get appropriate captions for each video.
# Create a file named "training_data.json" with the filename and caption for each video.
# This .json file can be passed to preprocess_dataset to precompute the dataset.
# Latents will be put in the ".precomputed" subdirectory of the video_dir.
#
# Usage:
#
# Create a directory for the dataset:
#
#    mkdir video_dataset
#    cd video_dataset
#
# Extract video files:
#
#    for f in *.tar; do
#        tar -xvf "$f" -k
#    done
#
# Run script, and pass it all of the feather files.  Does not require GPU.
#
#    python3 make_json_dataset.py /path/to/video_dataset path/to/visual_audio_concatenated_caption/*.feather
#
# Precompute dataset (on machine with GPU):
#
#    apt-get -y update
#    apt-get -y install ffmpeg
#    uv sync
#    . .venv/bin/activate
#
#    python3 scripts/preprocess_dataset.py \
#        /home/dhutchins/data/video_files/training_data.json \
#        --resolution-buckets '1280x736x25' \
#        --model-source /home/scavallari/ltx-qad/model/ltx-av-step-1933500-split-new-vae.safetensors \
#        --encoder-type gemma --text-encoder-path /home/dhutchins/models/gemma \
#        --with-audio \
#


def find_video_subdirectory_name(
    feather_filename: str,
    prefix="video_files"
) -> str:
    """Get the video subdirectory name for a given .feather filename."""
    # Given a .feather filename, e.g. something_something_003_0023.feather
    # find a matching video directory name, e.g. video_files_003_0023.
    stem = Path(feather_filename).stem  # e.g. "something_something_003_0023"
    parts = stem.split("_")
    # Take the last two parts (e.g. "003" and "0023")
    suffix = "_".join(parts[-2:])
    return f"{prefix}_{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Convert .feather files to .json format")
    parser.add_argument("video_dir", type=str, help="Directory containing video files")
    parser.add_argument("feather_files", type=str, nargs="+", help="Path(s) to .feather file(s)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Video directory not found: {video_dir}")

    unique_keys = set()
    dataset_rows: list[dict] = []
    for feather_file in args.feather_files:
        feather_path = Path(feather_file)
        if not feather_path.exists():
            raise FileNotFoundError(f"Feather file not found: {feather_path}")

        print(f"Reading file {feather_path}.")
        df = pd.read_feather(feather_path)

        # Validate columns
        expected_columns = {"keys", "caption"}
        actual_columns = set(df.columns)
        if actual_columns != expected_columns:
            raise ValueError(
                f"File {feather_path} has unexpected columns. "
                f"Expected {expected_columns}, got {actual_columns}"
            )

        # Find the video subdirectory corresponding to the feather file.
        # [Disabled] -- seemed to confuse preprocess_dataset.py
        #
        # video_subdir_name = find_video_subdirectory_name(feather_file)
        # video_subdir = video_dir / video_subdir_name
        # if not video_subdir.exists():
        #     print(f"Error: cannot find video subdirectory {video_subdir}")

        # Read .feather file
        rows = df.to_dict(orient="records")

        # Add records to dataset_rows
        for row in rows:
            fkey = row["keys"]
            if fkey in unique_keys:
                print(f"Warning: skipping duplicate key {fkey} in {feather_file}.")
                continue
            unique_keys.add(fkey)

            filename = row["keys"] + ".mp4"
            # file_path = video_subdir / filename
            # media_path = video_subdir_name + "/" + filename
            file_path = video_dir / filename
            media_path = filename
            if file_path.exists():  # Only add records if we can find the video
                dataset_rows.append({
                    "media_path": media_path,
                    "caption": row["caption"]
                })

    # Write dataset to JSON file
    output_path = video_dir / "training_data.json"
    with open(output_path, "w") as f:
        json.dump(dataset_rows, f, indent=2)

    print(f"Total rows: {len(dataset_rows)}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
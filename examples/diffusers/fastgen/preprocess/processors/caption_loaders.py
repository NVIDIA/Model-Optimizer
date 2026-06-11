# Vendored from NVIDIA-NeMo/Automodel @ e42584e3 (Apache-2.0):
#   https://github.com/NVIDIA-NeMo/Automodel/blob/e42584e303397e9bd34643407b8a57d7def88ce9/tools/diffusion/processors/caption_loaders.py
# Vendored into the fastgen example so preprocessing runs against stock nemo_automodel
# without the un-packaged AutoModel ``tools/`` tree. Self-contained (no internal import changes).
# Original license below.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Caption loading strategies for preprocessing.

Provides multiple ways to load captions for media files:
- JSONSidecarCaptionLoader: video.mp4 -> video.json with {"caption": "..."}
- MetaJSONCaptionLoader: meta.json with [{"file_name": "...", "caption": "..."}]
- JSONLCaptionLoader: Existing JSONL format for images
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CaptionLoadingStats:
    """
    Statistics from caption loading operations.

    Provides detailed information about caption loading for debugging
    and progress reporting.
    """

    # Number of captions successfully loaded
    loaded_count: int = 0

    # Number of caption files found and parsed
    files_parsed: int = 0

    # Number of expected caption files that were missing
    files_missing: int = 0

    # Number of media files without captions (will use fallback)
    captions_missing: int = 0

    # Error messages encountered during loading
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"Loaded {self.loaded_count} captions from {self.files_parsed} files "
            f"({self.files_missing} missing, {self.captions_missing} using fallback)"
        )


class CaptionLoader(ABC):
    """
    Abstract base class for caption loading strategies.

    Different datasets organize captions in different ways:
    - Sidecar files (one JSON per media file)
    - Single metadata file (meta.json with all captions)
    - JSONL files (line-delimited JSON entries)
    """

    @abstractmethod
    def load_captions(
        self,
        media_files: list[Path],
        caption_field: str = "caption",
        verbose: bool = False,
    ) -> dict[str, str]:
        """
        Load captions for a list of media files.

        Args:
            media_files: List of media file paths
            caption_field: Field name containing the caption text
            verbose: If True, print progress information

        Returns:
            Dict mapping filename (not full path) to caption text
        """

    def load_captions_with_stats(
        self,
        media_files: list[Path],
        caption_field: str = "caption",
        verbose: bool = False,
    ) -> tuple[dict[str, str], CaptionLoadingStats]:
        """
        Load captions and return statistics.

        Args:
            media_files: List of media file paths
            caption_field: Field name containing the caption text
            verbose: If True, print progress information

        Returns:
            Tuple of (captions dict, loading statistics)
        """
        # Default implementation - subclasses can override for efficiency
        captions = self.load_captions(media_files, caption_field, verbose)
        stats = CaptionLoadingStats(
            loaded_count=len(captions),
            captions_missing=len(media_files) - len(captions),
        )
        return captions, stats

    @staticmethod
    def get_loader(format_name: str) -> "CaptionLoader":
        """
        Factory method to get the appropriate caption loader.

        Args:
            format_name: One of 'sidecar', 'meta_json', 'jsonl'

        Returns:
            CaptionLoader instance

        Raises:
            ValueError: If format_name is unknown
        """
        loaders = {
            "sidecar": JSONSidecarCaptionLoader,
            "meta_json": MetaJSONCaptionLoader,
            "jsonl": JSONLCaptionLoader,
        }
        if format_name not in loaders:
            available = ", ".join(sorted(loaders.keys()))
            raise ValueError(f"Unknown caption format: '{format_name}'. Available: {available}")
        return loaders[format_name]()


class JSONSidecarCaptionLoader(CaptionLoader):
    """
    Load captions from JSON sidecar files.

    Expects: video.mp4 -> video.json with content like:
        {"caption": "A video of..."}

    This is common for video datasets where each video has its own metadata file.
    """

    def load_captions(
        self,
        media_files: list[Path],
        caption_field: str = "caption",
        verbose: bool = False,
    ) -> dict[str, str]:
        """
        Load captions from sidecar JSON files.

        For each media file (e.g., video.mp4), looks for a corresponding
        JSON file (video.json) in the same directory.

        Args:
            media_files: List of media file paths
            caption_field: Field name containing the caption text
            verbose: If True, print progress information

        Returns:
            Dict mapping filename to caption text
        """
        captions, _ = self.load_captions_with_stats(media_files, caption_field, verbose)
        return captions

    def load_captions_with_stats(
        self,
        media_files: list[Path],
        caption_field: str = "caption",
        verbose: bool = False,
    ) -> tuple[dict[str, str], CaptionLoadingStats]:
        """
        Load captions from sidecar JSON files with statistics.

        Args:
            media_files: List of media file paths
            caption_field: Field name containing the caption text
            verbose: If True, print progress information

        Returns:
            Tuple of (captions dict, loading statistics)
        """
        captions = {}
        stats = CaptionLoadingStats()

        if verbose:
            logger.info("Loading captions from sidecar JSON files...")

        for media_path in media_files:
            # Look for sidecar JSON: video.mp4 -> video.json
            json_path = media_path.with_suffix(".json")

            if not json_path.exists():
                stats.files_missing += 1
                continue

            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)

                stats.files_parsed += 1
                caption = data.get(caption_field)
                if caption:
                    captions[media_path.name] = caption
                    stats.loaded_count += 1

            except json.JSONDecodeError as e:
                stats.errors.append(f"JSON error in {json_path}: {e}")
            except OSError as e:
                stats.errors.append(f"IO error reading {json_path}: {e}")

        stats.captions_missing = len(media_files) - stats.loaded_count

        if verbose:
            logger.info("  %s", stats)
            if stats.errors and len(stats.errors) <= 5:
                for err in stats.errors:
                    logger.warning("  %s", err)

        return captions, stats


class MetaJSONCaptionLoader(CaptionLoader):
    """
    Load captions from a centralized meta.json file.

    Expects: meta.json with content like:
        [
            {"file_name": "video1.mp4", "caption": "..."},
            {"file_name": "video2.mp4", "caption": "..."}
        ]
    or:
        {
            "items": [
                {"file_name": "video1.mp4", "caption": "..."},
                ...
            ]
        }

    This is common for curated datasets with a single metadata file.
    """

    def load_captions(
        self,
        media_files: list[Path],
        caption_field: str = "caption",
        verbose: bool = False,
    ) -> dict[str, str]:
        """
        Load captions from meta.json files.

        Looks for meta.json in each unique directory containing media files.

        Args:
            media_files: List of media file paths
            caption_field: Field name containing the caption text
            verbose: If True, print progress information

        Returns:
            Dict mapping filename to caption text
        """
        captions, _ = self.load_captions_with_stats(media_files, caption_field, verbose)
        return captions

    def load_captions_with_stats(
        self,
        media_files: list[Path],
        caption_field: str = "caption",
        verbose: bool = False,
    ) -> tuple[dict[str, str], CaptionLoadingStats]:
        """
        Load captions from meta.json files with statistics.

        Args:
            media_files: List of media file paths
            caption_field: Field name containing the caption text
            verbose: If True, print progress information

        Returns:
            Tuple of (captions dict, loading statistics)
        """
        captions = {}
        stats = CaptionLoadingStats()

        if verbose:
            logger.info("Loading captions from meta.json files...")

        # Group media files by directory to find meta.json files
        dirs = {p.parent for p in media_files}

        for directory in dirs:
            meta_path = directory / "meta.json"
            if not meta_path.exists():
                stats.files_missing += 1
                continue

            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)

                stats.files_parsed += 1

                # Handle both list format and dict with 'items' key
                if isinstance(data, dict):
                    items = data.get("items", data.get("data", []))
                else:
                    items = data

                for item in items:
                    if not isinstance(item, dict):
                        continue

                    file_name = item.get("file_name") or item.get("filename")
                    caption = item.get(caption_field)

                    if file_name and caption:
                        captions[file_name] = caption
                        stats.loaded_count += 1

            except json.JSONDecodeError as e:
                stats.errors.append(f"JSON error in {meta_path}: {e}")
            except OSError as e:
                stats.errors.append(f"IO error reading {meta_path}: {e}")

        stats.captions_missing = len(media_files) - stats.loaded_count

        if verbose:
            logger.info("  %s", stats)
            if stats.errors and len(stats.errors) <= 5:
                for err in stats.errors:
                    logger.warning("  %s", err)

        return captions, stats


class JSONLCaptionLoader(CaptionLoader):
    """
    Load captions from JSONL files.

    Expects: <prefix>_internvl.json (JSONL format) with content like:
        {"file_name": "image1.jpg", "internvl": "..."}
        {"file_name": "image2.jpg", "internvl": "..."}

    This is the existing format used for image preprocessing.
    """

    def __init__(self, jsonl_suffix: str = "_internvl.json"):
        """
        Args:
            jsonl_suffix: Suffix for JSONL files (default: '_internvl.json')
        """
        self.jsonl_suffix = jsonl_suffix

    def load_captions(
        self,
        media_files: list[Path],
        caption_field: str = "internvl",
        verbose: bool = False,
    ) -> dict[str, str]:
        """
        Load captions from JSONL files.

        For each media file, determines the associated JSONL file based on
        the filename pattern (prefix before '_sample' + suffix).

        Args:
            media_files: List of media file paths
            caption_field: Field name containing the caption text
            verbose: If True, print progress information

        Returns:
            Dict mapping filename to caption text
        """
        captions, _ = self.load_captions_with_stats(media_files, caption_field, verbose)
        return captions

    def load_captions_with_stats(
        self,
        media_files: list[Path],
        caption_field: str = "internvl",
        verbose: bool = False,
    ) -> tuple[dict[str, str], CaptionLoadingStats]:
        """
        Load captions from JSONL files with statistics.

        Args:
            media_files: List of media file paths
            caption_field: Field name containing the caption text
            verbose: If True, print progress information

        Returns:
            Tuple of (captions dict, loading statistics)
        """
        from collections import defaultdict

        captions = {}
        stats = CaptionLoadingStats()

        if verbose:
            logger.info("Loading captions from JSONL files...")

        # Group files by their JSONL file
        jsonl_to_files: dict[Path, list[str]] = defaultdict(list)

        for media_path in media_files:
            media_name = media_path.name

            # Extract prefix: everything before '_sample'
            if "_sample" in media_name:
                prefix = media_name.rsplit("_sample", 1)[0]
            else:
                prefix = media_path.stem

            json_path = media_path.parent / f"{prefix}{self.jsonl_suffix}"
            jsonl_to_files[json_path].append(media_name)

        # Load each JSONL file once
        for json_path, file_names in jsonl_to_files.items():
            if not json_path.exists():
                stats.files_missing += 1
                continue

            try:
                with open(json_path, encoding="utf-8") as f:
                    stats.files_parsed += 1
                    line_num = 0
                    for line in f:
                        line_num += 1
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            entry = json.loads(line)
                            file_name = entry.get("file_name")
                            caption = entry.get(caption_field)

                            if file_name and caption and file_name in file_names:
                                captions[file_name] = caption
                                stats.loaded_count += 1

                        except json.JSONDecodeError as e:
                            stats.errors.append(f"JSON error in {json_path} line {line_num}: {e}")

            except OSError as e:
                stats.errors.append(f"IO error reading {json_path}: {e}")

        stats.captions_missing = len(media_files) - stats.loaded_count

        if verbose:
            logger.info("  %s", stats)
            if stats.files_missing > 0:
                logger.info(
                    "  %d JSONL files not found (will use filename fallback)", stats.files_missing
                )
            if stats.errors and len(stats.errors) <= 5:
                for err in stats.errors:
                    logger.warning("  %s", err)

        return captions, stats


def get_caption_loader(format_name: str) -> CaptionLoader:
    """
    Convenience function to get a caption loader by format name.

    Args:
        format_name: One of 'sidecar', 'meta_json', 'jsonl'

    Returns:
        CaptionLoader instance
    """
    return CaptionLoader.get_loader(format_name)

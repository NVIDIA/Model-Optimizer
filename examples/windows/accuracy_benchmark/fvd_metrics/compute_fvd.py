# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Compute FVD (Fréchet Video Distance) between two sets of videos.

Uses a pre-trained I3D model (Kinetics-400, RGB) to extract 1024-dim features
from the final pooling layer, then computes the Fréchet distance between the
two feature distributions.

Reference: Unterthiner et al., "FVD: A New Metric for Video Generation", 2019.

Usage:
    python compute_fvd.py \
        --ref-dir /path/to/reference/videos \
        --gen-dir /path/to/generated/videos \
        --device cuda

    # With local weights and PCA
    python compute_fvd.py \
        --ref-dir ./real --gen-dir ./fake \
        --weights ./rgb_imagenet.pt \
        --clips-per-video 4 --pca-dim 64
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.linalg import svd
from scipy import linalg
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i3d_model import InceptionI3d, load_i3d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

CLIP_LENGTH = 16
SPATIAL_SIZE = 224
BATCH_SIZE = 8
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}


def positive_int(value: str) -> int:
    """Argparse type that enforces strictly positive integers."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return ivalue


def non_negative_int(value: str) -> int:
    """Argparse type that enforces non-negative integers."""
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return ivalue


WEIGHTS_URL = "https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt"
WEIGHTS_SHA256 = "2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538"
DEFAULT_CACHE = Path.home() / ".cache" / "fvd" / "rgb_imagenet.pt"


# ─── Video loading & preprocessing ──────────────────────────────────────────


def list_videos(folder: str) -> list:
    """Find all video files recursively under *folder*."""
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not paths:
        raise ValueError(f"No video files found in {folder}")
    return paths


def load_video_clip(path: Path, clip_length: int, start_frame: int = 0) -> np.ndarray | None:
    """Read *clip_length* consecutive frames starting at *start_frame*.

    Returns (T, H, W, 3) uint8 RGB or None on failure.
    Short videos are padded by repeating the last frame.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(clip_length):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None
    if len(frames) < clip_length:
        frames += [frames[-1]] * (clip_length - len(frames))

    return np.stack(frames, axis=0)


def preprocess_clip(frames: np.ndarray, spatial_size: int = SPATIAL_SIZE) -> torch.Tensor:
    """Resize shortest side to 256, center-crop, normalize to [-1, 1].

    Returns (3, T, H, W) float32.
    """
    n_frames, h, w, _ = frames.shape

    scale = 256 / min(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = np.stack(
        [
            cv2.resize(frames[t], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            for t in range(n_frames)
        ],
        axis=0,
    )

    cy, cx = resized.shape[1] // 2, resized.shape[2] // 2
    h2, w2 = spatial_size // 2, spatial_size // 2
    cropped = resized[:, cy - h2 : cy + h2, cx - w2 : cx + w2, :]

    tensor = torch.from_numpy(cropped).permute(3, 0, 1, 2).float()
    tensor = tensor / 127.5 - 1.0
    return tensor


def get_clips(
    video_paths: list,
    clip_length: int,
    clips_per_video: int,
    label: str,
) -> list:
    """Sample *clips_per_video* clips from each video."""
    clips = []
    for path in tqdm(video_paths, desc=f"Loading {label}", unit="video"):
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total < clip_length:
            starts = [0] * clips_per_video
        else:
            max_start = max(0, total - clip_length)
            starts = np.linspace(0, max_start, num=clips_per_video, dtype=int).tolist()

        for start in starts:
            raw = load_video_clip(path, clip_length, start_frame=start)
            if raw is None:
                log.warning(f"Failed to load {path.name} (start={start})")
                continue
            clips.append(preprocess_clip(raw))

    if not clips:
        raise ValueError(f"Could not load any clips from {label} videos.")
    return clips


# ─── Feature extraction ──────────────────────────────────────────────────────


@torch.no_grad()
def extract_features(
    clips: list,
    model: InceptionI3d,
    batch_size: int,
    device: torch.device,
    label: str,
) -> np.ndarray:
    """Run I3D on clips and return (N, 1024) features."""
    feats = []
    for i in tqdm(
        range(0, len(clips), batch_size),
        desc=f"Extracting {label} features",
    ):
        batch = torch.stack(clips[i : i + batch_size]).to(device)
        feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ─── PCA & Fréchet distance ──────────────────────────────────────────────────


def apply_pca(feats_a: np.ndarray, feats_b: np.ndarray, n_components: int) -> tuple:
    """Fit PCA on combined features, project both sets.

    Avoids rank-deficient covariance when n_samples < n_features.
    """
    n_components = min(n_components, feats_a.shape[0] - 1, feats_b.shape[0] - 1, feats_a.shape[1])
    combined = np.concatenate([feats_a, feats_b], axis=0)
    mu = combined.mean(axis=0)
    _, _, vt = svd(combined - mu, full_matrices=False)
    components = vt[:n_components]
    return (feats_a - mu) @ components.T, (feats_b - mu) @ components.T


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between N(mu1, sigma1) and N(mu2, sigma2)."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        log.warning("sqrtm produced non-finite values; adding epsilon to diagonal.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        max_imag = np.max(np.abs(covmean.imag))
        if max_imag > 1e-3:
            log.warning(f"Large imaginary component in sqrtm: {max_imag:.4f}")
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_fvd(feats_ref: np.ndarray, feats_gen: np.ndarray) -> float:
    """Compute FVD from two feature arrays."""
    if feats_ref.shape[0] < 2 or feats_gen.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 clips per set for FVD "
            f"(got ref={feats_ref.shape[0]}, gen={feats_gen.shape[0]})"
        )
    mu_r = feats_ref.mean(axis=0)
    sigma_r = np.atleast_2d(np.cov(feats_ref, rowvar=False))
    mu_g = feats_gen.mean(axis=0)
    sigma_g = np.atleast_2d(np.cov(feats_gen, rowvar=False))
    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


# ─── Weight download ─────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_weights(weights_arg: str | None) -> Path:
    """Return path to weights file, downloading and verifying if necessary."""
    if weights_arg:
        p = Path(weights_arg)
        if not p.exists():
            raise FileNotFoundError(f"Weights not found: {p}")
        return p

    if DEFAULT_CACHE.exists():
        return DEFAULT_CACHE

    log.info(f"Downloading I3D weights to {DEFAULT_CACHE} …")
    DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(WEIGHTS_URL, DEFAULT_CACHE)

    digest = _sha256(DEFAULT_CACHE)
    if digest != WEIGHTS_SHA256:
        DEFAULT_CACHE.unlink()
        raise RuntimeError(
            f"SHA-256 mismatch for downloaded I3D weights "
            f"(got {digest[:16]}…, expected {WEIGHTS_SHA256[:16]}…). "
            f"File deleted. Please retry or download manually."
        )

    log.info("Download complete (SHA-256 verified).")
    return DEFAULT_CACHE


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    """Parse arguments, load model, extract features, compute and report FVD."""
    parser = argparse.ArgumentParser(
        description="Compute FVD between two video sets (I3D, 1024-dim features).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ref-dir", required=True, help="Reference videos directory")
    parser.add_argument("--gen-dir", required=True, help="Generated videos directory")
    parser.add_argument(
        "--weights", default=None, help="Path to I3D weights (auto-downloaded if omitted)"
    )
    parser.add_argument("--device", default=None, help="torch device (auto-detected if omitted)")
    parser.add_argument(
        "--clip-length",
        type=positive_int,
        default=CLIP_LENGTH,
        help="Frames per clip (default: 16)",
    )
    parser.add_argument(
        "--clips-per-video",
        type=positive_int,
        default=1,
        help="Clips sampled per video (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=BATCH_SIZE,
        help="Batch size for I3D inference (default: 8)",
    )
    parser.add_argument(
        "--pca-dim",
        type=non_negative_int,
        default=None,
        help="PCA dimension (auto-selected when n_clips < 1024; 0 to disable)",
    )
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info(f"Device: {device}")

    weights_path = resolve_weights(args.weights)
    user_supplied_weights = args.weights is not None
    model = load_i3d(str(weights_path), device, allow_unsafe_pickle=not user_supplied_weights)
    log.info(f"I3D model loaded from {weights_path.name} (1024-dim features)")

    ref_paths = list_videos(args.ref_dir)
    gen_paths = list_videos(args.gen_dir)
    log.info(f"Reference videos: {len(ref_paths)}")
    log.info(f"Generated videos: {len(gen_paths)}")

    ref_clips = get_clips(ref_paths, args.clip_length, args.clips_per_video, "ref")
    gen_clips = get_clips(gen_paths, args.clip_length, args.clips_per_video, "gen")
    log.info(f"Total clips — ref: {len(ref_clips)}, gen: {len(gen_clips)}")

    if len(ref_clips) < 2 or len(gen_clips) < 2:
        sys.exit("Need at least 2 clips per set to compute covariance.")

    if min(len(ref_clips), len(gen_clips)) < 256:
        log.warning(
            f"Only {min(len(ref_clips), len(gen_clips))} clips — "
            "FVD estimates are noisy below ~256 clips. "
            "Consider --clips-per-video to increase sample count."
        )

    ref_feats = extract_features(ref_clips, model, args.batch_size, device, "ref")
    gen_feats = extract_features(gen_clips, model, args.batch_size, device, "gen")

    n_clips = min(len(ref_clips), len(gen_clips))
    feat_dim = ref_feats.shape[1]

    if args.pca_dim is None:
        pca_dim = min(n_clips - 1, 64) if n_clips < feat_dim else None
    elif args.pca_dim == 0:
        pca_dim = None
    else:
        pca_dim = args.pca_dim

    if pca_dim is not None:
        log.info(f"Applying PCA: {feat_dim}d → {pca_dim}d  (n_clips={n_clips})")
        ref_feats, gen_feats = apply_pca(ref_feats, gen_feats, pca_dim)

    fvd_score = compute_fvd(ref_feats, gen_feats)
    log.info(f"FVD = {fvd_score:.4f}")

    if args.output:
        result = {
            "fvd": fvd_score,
            "ref_dir": args.ref_dir,
            "gen_dir": args.gen_dir,
            "num_ref_clips": len(ref_clips),
            "num_gen_clips": len(gen_clips),
            "clip_length": args.clip_length,
            "clips_per_video": args.clips_per_video,
            "feature_dim": int(ref_feats.shape[1]),
            "pca_dim": pca_dim,
            "model": "I3D (Kinetics-400, 1024-dim pool)",
        }
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"Results saved to {args.output}")

    return fvd_score


if __name__ == "__main__":
    main()

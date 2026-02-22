"""Generate teacher (input, target) latent pairs for QAD distillation.

For each prompt, assigns a denoising step k (round-robin across NUM_STEPS),
runs the teacher transformer through steps 0..k, and saves:

    input_latent  = noisy latent BEFORE step k  (student input at train time)
    target_latent = denoised latent AFTER step k (ground truth for student)

Output format (saved as a dict):
    input_latents   : [N, 4096, 64]  bf16
    target_latents  : [N, 4096, 64]  bf16
    step_indices    : [N]            int     (which denoising step, 0..29)
    timestep_values : [N]            float   (the actual timestep passed to model)
    noise_seeds     : [N]            int     (for reproducibility)

Usage:
    python teacher_latents.py
"""

import os
import time
from typing import MutableSequence

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm

# ─── CLI args for multi-GPU sharding ─────────────────────────────────────────
import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--gpu", type=int, default=5)
_p.add_argument("--start", type=int, default=0, help="first sample index (inclusive)")
_p.add_argument("--end", type=int, default=-1, help="last sample index (exclusive), -1=all")
_p.add_argument("--tag", type=str, default="", help="output filename suffix")
_args = _p.parse_args()

# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE = f"cuda:{_args.gpu}"
NUM_STEPS = 30
BATCH_SIZE = 30
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")

PIXEL_H, PIXEL_W = 1024, 1024
VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
IMG_H = PIXEL_H // VAE_SCALE_FACTOR // PATCH_SIZE   # 64
IMG_W = PIXEL_W // VAE_SCALE_FACTOR // PATCH_SIZE   # 64
SEQ_LEN = IMG_H * IMG_W                             # 4096
IN_CHANNELS = 64

# img_shapes tells the transformer the spatial layout for RoPE computation.
# For 1024x1024: one frame, 64x64 patchified grid.
IMG_SHAPES_SINGLE = [(1, IMG_H, IMG_W)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calculate_mu(seq_len, base_seq=256, max_seq=8192, base_shift=0.5, max_shift=0.9):
    #returns, by default, 0.693548, for 1024x1024
    #m=y2-y1/x2-x1
    #b=y1-m*x1
    #return y=mx+b
    y2,y1 = max_shift, base_shift
    x, x2,x1 = seq_len, max_seq, base_seq
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b


# ─── Load model ──────────────────────────────────────────────────────────────
log("Loading teacher pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16,
)
transformer = pipe.transformer.to(DEVICE).eval()
scheduler = pipe.scheduler

log(f"Transformer: {transformer.__class__.__name__}, "
    f"params={sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B")

# ─── Set up denoising schedule (matches SGLang QwenImagePipeline) ────────────
raw_sigmas = np.linspace(1.0, 1.0 / NUM_STEPS, NUM_STEPS)
mu = calculate_mu(SEQ_LEN)
scheduler.set_timesteps(sigmas=raw_sigmas.tolist(), mu=mu, device=DEVICE)
timesteps = scheduler.timesteps.clone()
sigmas = scheduler.sigmas.clone()

log(f"Schedule: {NUM_STEPS} steps, mu={mu:.4f}")
log(f"Timesteps: [{', '.join(f'{t:.1f}' for t in timesteps[:5].tolist())} ... "
    f"{', '.join(f'{t:.1f}' for t in timesteps[-3:].tolist())}]")


# ─── Load prompt embeddings ──────────────────────────────────────────────────
# Format: dict {int_index: tensor [seq_len_i, 3584]} with variable-length prompts.
# We store them as a list and pad per-batch at runtime to avoid wasting memory.
def load_embed_list(data):
    """Convert saved format into a list of [seq_len_i, dim] tensors."""
    if isinstance(data, dict) and isinstance(next(iter(data.keys())), int):
        n = len(data)
        return [data[i] for i in range(n)]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unknown embed format: {type(data)}, keys={list(data.keys())[:5]}")


def pad_batch(embed_list, max_seq=1024):
    """Pad variable-length embeds into a batch with attention masks."""
    dim = embed_list[0].shape[-1]
    lengths = [min(e.shape[0], max_seq) for e in embed_list]
    max_len = max(lengths)
    B = len(embed_list)
    padded = torch.zeros(B, max_len, dim, dtype=embed_list[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, e in enumerate(embed_list):
        L = lengths[i]
        padded[i, :L] = e[:L]
        mask[i, :L] = 1
    return padded, mask, lengths


# ─── Core: generate teacher latent pairs ─────────────────────────────────────
@torch.no_grad()
def generate_teacher_latents(embed_list, split_name):
    N = len(embed_list)
    log(f"[{split_name}] {N} samples, batch_size={BATCH_SIZE}, "
        f"~{N // NUM_STEPS} samples/step")

    all_inputs = []
    all_targets = []
    all_step_idx = []
    all_ts_val = []
    all_seeds = []

    for batch_start in tqdm(range(0, N, BATCH_SIZE), desc=split_name):
        batch_end = min(batch_start + BATCH_SIZE, N)
        B = batch_end - batch_start

        step_assignments = [(batch_start + j) % NUM_STEPS for j in range(B)]
        max_step_needed = max(step_assignments)

        batch_embeds = embed_list[batch_start:batch_end]
        prompt_embeds, prompt_mask, txt_lengths = pad_batch(batch_embeds)
        prompt_embeds = prompt_embeds.to(device=DEVICE, dtype=torch.bfloat16)
        prompt_mask = prompt_mask.to(device=DEVICE)
        txt_seq_lens = txt_lengths
        img_shapes = [IMG_SHAPES_SINGLE] * B

        # deterministic noise per sample (seed = global sample index)
        g = torch.Generator(device="cpu")
        noise = torch.stack([
            torch.randn(SEQ_LEN, IN_CHANNELS, generator=g.manual_seed(batch_start + j),
                         dtype=torch.bfloat16)
            for j in range(B)
        ]).to(DEVICE)

        latents = noise.clone()
        saved_input = [None] * B
        saved_target = [None] * B

        # Reset scheduler step counter for this fresh trajectory
        scheduler.set_begin_index(0)

        for step_idx in range(max_step_needed + 1):
            # Before the transformer runs: snapshot input for assigned samples
            for j in range(B):
                if step_assignments[j] == step_idx:
                    saved_input[j] = latents[j].cpu().clone()

            t = timesteps[step_idx]
            # Pipeline divides by 1000 before passing to transformer
            t_normed = (t / 1000).expand(B).to(latents.dtype)

            noise_pred = transformer(
                hidden_states=latents,
                timestep=t_normed,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # After scheduler step: snapshot target for assigned samples
            for j in range(B):
                if step_assignments[j] == step_idx:
                    saved_target[j] = latents[j].cpu().clone()

        for j in range(B):
            k = step_assignments[j]
            all_inputs.append(saved_input[j])
            all_targets.append(saved_target[j])
            all_step_idx.append(k)
            all_ts_val.append(timesteps[k].item())
            all_seeds.append(batch_start + j)

    result = {
        "input_latents": torch.stack(all_inputs),
        "target_latents": torch.stack(all_targets),
        "step_indices": torch.tensor(all_step_idx, dtype=torch.long),
        "timestep_values": torch.tensor(all_ts_val, dtype=torch.float32),
        "noise_seeds": torch.tensor(all_seeds, dtype=torch.long),
    }
    dist = torch.bincount(result["step_indices"], minlength=NUM_STEPS)
    log(f"[{split_name}] Done — {result['input_latents'].shape}, "
        f"step distribution min={dist.min().item()} max={dist.max().item()}")
    return result


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    for split, embed_file in [("train", "train_embeds.pt"), ("val", "val_embeds.pt")]:
        path = os.path.join(SAVE_DIR, embed_file)
        if not os.path.exists(path):
            log(f"[{split}] {path} not found, skipping")
            continue

        log(f"[{split}] Loading {path}...")
        data = torch.load(path, weights_only=False)
        embed_list = load_embed_list(data)

        # Apply shard range
        total = len(embed_list)
        s, e = _args.start, (_args.end if _args.end > 0 else total)
        e = min(e, total)
        embed_list = embed_list[s:e]
        log(f"[{split}] shard [{s}:{e}) = {len(embed_list)} of {total} prompts")

        if len(embed_list) == 0:
            log(f"[{split}] nothing in this shard range, skipping")
            continue

        result = generate_teacher_latents(embed_list, split)

        # Global sample indices (so noise seeds stay consistent across shards)
        result["noise_seeds"] = result["noise_seeds"] + s

        suffix = f"_{_args.tag}" if _args.tag else ""
        out_path = os.path.join(SAVE_DIR, f"teacher_{split}_latents{suffix}.pt")
        torch.save(result, out_path)
        log(f"[{split}] Saved → {out_path}")

    log("All done!")

# get scheduler 
# calculate_mu
# pass mu to scheduler, let it adjust timesteps and sigmas based on that
# but apparently it doesn't
# we have to do so manually, adjust timesteps by passing sigmas and mus
# but why do i have to compute sigma myself? why can't the scheduler do it?
# then i 
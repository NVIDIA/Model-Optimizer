# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Minimal AutoQuantize example for Wan 2.2 5B (TI2V-5B).

Uses the loss-based ("gradient") AutoQuantize searcher to pick a per-layer
mixed-precision mapping between FP8 and NVFP4 under an ``effective_bits`` budget.

Calibration data (Option B): real prompts + random clean latents.
Loss: rectified-flow MSE ``target = noise - x_0``, ``x_t = (1 - t) * x_0 + t * noise``.

Example::

    python auto_quantize_wan22_5b.py \\
        --output wan22_5b_autoquant.pt \\
        --effective-bits 6.0 \\
        --num-samples 32
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import WanPipeline
from utils import load_calib_prompts

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DATASET = {
    "name": "Gustavosta/Stable-Diffusion-Prompts",
    "split": "train",
    "column": "Prompt",
}
FORMATS = {
    "fp8": mtq.FP8_DEFAULT_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
}


def _disable_wan_attention_quantization() -> None:
    """Keep Wan attention native so gradient-based AutoQuantize can run.

    Why: modelopt's diffusers plugin wraps ``WanAttention`` with
    ``_QuantAttentionModuleMixin``, whose forward replaces
    ``F.scaled_dot_product_attention`` with ``FP8SDPA.apply`` — a forward-only
    ``autograd.Function`` with no backward. That breaks ``loss.backward()`` in
    the gradient-based searcher. Unregistering leaves the q/k/v/out projections
    quantizable (still standard QuantLinear) and only skips SDPA itself.
    """
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModuleRegistry

    try:
        from diffusers.models.transformers.transformer_wan import WanAttention
    except ImportError:
        return
    if WanAttention in QuantModuleRegistry:
        QuantModuleRegistry.unregister(WanAttention)


def build_calibration_samples(
    pipe: WanPipeline,
    prompts: list[str],
    batch_size: int,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    gen_inference_steps: int,
) -> list[dict[str, torch.Tensor]]:
    """Build autoquant calibration batches using real generated latents.

    For each prompt we:

    1. Encode the prompt to ``text_emb``.
    2. Generate a clean latent ``x_0`` by running the full pipeline once with
       ``output_type="latent"`` (no VAE decode). This puts ``x_0`` on the
       transformer's own learned manifold — same role GT video latents play in
       training-style flow-matching loss.
    3. Sample ``noise`` and ``t``, form ``x_t = (1-t)*x_0 + t*noise``, and set
       ``target = noise - x_0`` (rectified-flow velocity).

    The autoquant searcher reads ``forward_step(transformer, batch)``'s output
    and ``loss_func = MSE(pred, target)``. Because ``target`` is anchored to a
    real manifold point, the gradient signal reflects actual training-style
    velocity prediction error — matching how Wan was trained.
    """
    num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    latent_h = height // pipe.vae_scale_factor_spatial
    latent_w = width // pipe.vae_scale_factor_spatial
    expand_ts = getattr(pipe.config, "expand_timesteps", False)
    num_train_ts = pipe.scheduler.config.num_train_timesteps

    # Cache the full timestep schedule so we can sample `t` after pipe() mutates
    # the scheduler state during each generation call.
    pipe.scheduler.set_timesteps(50, device=device)
    ts_schedule = pipe.scheduler.timesteps.detach().clone()

    samples: list[dict[str, torch.Tensor]] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch_prompts = prompts[bi * batch_size : (bi + 1) * batch_size]

        # Encode once; reused for both generation and the autoquant forward pass.
        with torch.no_grad():
            pos_emb, _ = pipe.encode_prompt(
                prompt=batch_prompts,
                do_classifier_free_guidance=False,
                max_sequence_length=226,
                device=device,
                dtype=dtype,
            )

        # Generate a clean latent on the model's manifold.
        # guidance_scale=1.0 disables CFG (2x faster, no negative prompt needed).
        print(
            f"  [{bi + 1}/{n_batches}] Generating x_0 ({gen_inference_steps} denoising steps) ..."
        )
        with torch.no_grad():
            x_0 = pipe(
                prompt_embeds=pos_emb,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=gen_inference_steps,
                guidance_scale=1.0,
                output_type="latent",
                return_dict=False,
            )[0].to(dtype=dtype)

        b = pos_emb.shape[0]
        noise = torch.randn_like(x_0)
        t_raw = ts_schedule[torch.randint(0, len(ts_schedule), (b,))].to(
            device=device, dtype=torch.float32
        )
        t_norm = (t_raw / num_train_ts).to(dtype).view(b, 1, 1, 1, 1)
        x_t = (1 - t_norm) * x_0 + t_norm * noise
        target = noise - x_0

        if expand_ts:
            seq_len = num_latent_frames * (latent_h // 2) * (latent_w // 2)
            t_for_model = t_raw.unsqueeze(1).expand(b, seq_len).to(dtype)
        else:
            t_for_model = t_raw.to(dtype)

        # Move to CPU to keep GPU memory free for the search itself.
        samples.append(
            {
                "x_t": x_t.cpu(),
                "t": t_for_model.cpu(),
                "text_emb": pos_emb.cpu(),
                "target": target.cpu(),
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoQuantize example for Wan 2.2 5B (loss-based per-layer search).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path", default=MODEL_ID, help="HuggingFace model id or local path."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Path to save the quantized backbone (.pt)."
    )
    parser.add_argument(
        "--quant-formats",
        nargs="+",
        default=["fp8", "nvfp4"],
        choices=list(FORMATS.keys()),
        help="Candidate per-layer quantization formats to search over.",
    )
    parser.add_argument(
        "--effective-bits",
        type=float,
        default=6.0,
        help="Target effective bits-per-weight averaged across the backbone.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=32, help="Number of calibration prompts."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts per calibration batch.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument(
        "--num-score-steps",
        type=int,
        default=32,
        help="Batches used for gradient-based sensitivity scoring.",
    )
    parser.add_argument(
        "--gen-inference-steps",
        type=int,
        default=20,
        help="Denoising steps used to generate each x_0 latent (fewer = faster).",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    _disable_wan_attention_quantization()

    print(f"Loading {args.model_path} ...")
    pipe = WanPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.to(device)
    assert pipe.config.boundary_ratio is None, (
        "This example targets Wan 2.2 5B (single transformer). For 14B (dual "
        "transformer + boundary_ratio), the search would need to be run per backbone."
    )

    print(f"Loading up to {args.num_samples} calibration prompts from {DATASET['name']} ...")
    batched = load_calib_prompts(
        args.batch_size, DATASET["name"], DATASET["split"], DATASET["column"]
    )
    flat_prompts = [p for batch in batched for p in batch][: args.num_samples]

    print("Generating reference latents and building calibration samples ...")
    samples = build_calibration_samples(
        pipe,
        flat_prompts,
        args.batch_size,
        args.height,
        args.width,
        args.num_frames,
        device,
        dtype,
        gen_inference_steps=args.gen_inference_steps,
    )
    print(f"Built {len(samples)} calibration batches.")

    # Free the text encoder and VAE — only the transformer participates in the search.
    pipe.text_encoder = None
    pipe.vae = None
    torch.cuda.empty_cache()

    def forward_step(transformer, batch):
        return transformer(
            hidden_states=batch["x_t"].to(device),
            timestep=batch["t"].to(device),
            encoder_hidden_states=batch["text_emb"].to(device),
            return_dict=False,
        )[0]

    def loss_func(output, batch):
        target = batch["target"].to(output.device, output.dtype)
        return F.mse_loss(output, target)

    print(
        f"Running auto_quantize (method=gradient, formats={args.quant_formats}, "
        f"effective_bits={args.effective_bits}) ..."
    )
    transformer, _ = mtq.auto_quantize(
        pipe.transformer,
        constraints={"effective_bits": args.effective_bits},
        data_loader=samples,
        forward_step=forward_step,
        loss_func=loss_func,
        quantization_formats=[FORMATS[f] for f in args.quant_formats],
        num_calib_steps=len(samples),
        num_score_steps=min(len(samples), args.num_score_steps),
        verbose=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving quantized backbone to {args.output}")
    mto.save(transformer, str(args.output))
    print("Done. Restore with: `quantize.py --model wan2.2-t2v-5b --restore-from <dir>`")


if __name__ == "__main__":
    main()

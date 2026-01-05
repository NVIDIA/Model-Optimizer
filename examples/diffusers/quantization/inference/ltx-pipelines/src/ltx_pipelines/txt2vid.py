import os
import argparse

import torch
from einops import rearrange
from tqdm import tqdm

from ltx_core.loader.primitives import LoraPathStrengthAndKeyOps
from ltx_core.loader.sd_keys_ops import (
    LTXV_LORA_COMFY_RENAMING_MAP,
    LTXV_MODEL_COMFY_RENAMING_MAP,
)
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae.audio_vae import Decoder as AudioDecoder
from ltx_core.model.audio_vae.model_configurator import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae.model_configurator import VAEDecoderConfigurator as AudioDecoderConfigurator
from ltx_core.model.audio_vae.vocoder import Vocoder
from ltx_core.model.clip.gemma.encoders.av_encoder import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModel,
    AVGemmaTextEncoderModelConfigurator,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
from ltx_core.model.video_vae.model_configurator import VAE_DECODER_COMFY_KEYS_FILTER
from ltx_core.model.video_vae.model_configurator import VAEDecoderConfigurator as VideoDecoderConfigurator
from ltx_core.model.video_vae.video_vae import Decoder as VideoDecoder
from ltx_core.pipeline.components.diffusion_steps import EulerDiffusionStep
from ltx_core.pipeline.components.guiders import CFGGuider
from ltx_core.pipeline.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.pipeline.components.schedulers import LTX2Scheduler
from ltx_core.pipeline.conditioning import AudioConditioningBuilder, VideoConditioningBuilder
from ltx_pipelines.media_io import encode_video


enable_vfly = os.environ.get("ENABLE_VFLY", "").lower() == "true"
if enable_vfly:
    import gc
    import vfly
    from vfly.layers import VflyLinear
    from vfly.utils import cudagraph_wrapper


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Txt2VidPipeline:
    def __init__(
        self, checkpoint_path: str, gemma_root: str, loras: list[LoraPathStrengthAndKeyOps], device: str = get_device()
    ):
        self.device = device

        transformer_builder: Builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_key_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        )
        for lora in loras:
            transformer_builder = transformer_builder.lora(lora.path, lora.strength, lora.sd_key_ops)
        self.transformer: LTXModel = transformer_builder.build(device=device)

        self.vae_decoder: VideoDecoder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_key_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=device)

        self.text_encoder: AVGemmaTextEncoderModel = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AVGemmaTextEncoderModelConfigurator.with_gemma_root_path(gemma_root),
            model_sd_key_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
        ).build(device=device)

        self.audio_decoder: AudioDecoder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_key_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=device)

        self.vocoder: Vocoder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_key_ops=VOCODER_COMFY_KEYS_FILTER,
        ).build(device=device)

    def __call__(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
    ) -> None:
        self.text_encoder.to(self.device)
        with torch.inference_mode():
            v_context_p, a_context_p, _ = self.text_encoder(prompt)
            v_context_n, a_context_n, _ = self.text_encoder(negative_prompt)
        self.text_encoder.to(torch.device("cpu"))

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        video_builder = VideoConditioningBuilder(
            patchifier=VideoLatentPatchifier(patch_size=1),
            batch=1,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=frame_rate,
        )
        video_input = video_builder.build(device=self.device, dtype=torch.bfloat16, generator=generator)

        audio_builder = AudioConditioningBuilder(
            patchifier=AudioPatchifier(patch_size=1),
            batch=1,
            duration=num_frames / frame_rate,
        )
        audio_input = audio_builder.build(device=self.device, dtype=torch.bfloat16, generator=generator)

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=num_inference_steps).to(self.device).float()
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        video = Modality(
            enabled=True,
            latent=video_input.latent,
            timesteps=video_input.denoise_mask,
            positions=video_input.positions,
            context=v_context_p,
            context_mask=None,
        )
        audio = Modality(
            enabled=True,
            latent=audio_input.latent,
            timesteps=audio_input.denoise_mask,
            positions=audio_input.positions,
            context=a_context_p,
            context_mask=None,
        )
        self.transformer.to(self.device)
        with torch.inference_mode():
            for step_idx, sigma in enumerate(tqdm(sigmas[:-1])):
                video.timesteps = sigma * video_input.denoise_mask
                audio.timesteps = sigma * audio_input.denoise_mask
                if cfg_guidance_scale != 1.0:
                    video_neg = Modality(
                        enabled=True,
                        latent=video.latent.clone(),
                        timesteps=video.timesteps,
                        positions=video.positions,
                        context=v_context_n,
                        context_mask=None,
                    )
                    audio_neg = Modality(
                        enabled=True,
                        latent=audio.latent.clone(),
                        timesteps=audio.timesteps,
                        positions=audio.positions,
                        context=a_context_n,
                        context_mask=None,
                    )
                    neg_video, neg_audio = self.transformer(video=video_neg, audio=audio_neg, perturbations=None)
                pos_video, pos_audio = self.transformer(video=video, audio=audio, perturbations=None)
                video_velocity = pos_video
                audio_velocity = pos_audio
                if cfg_guidance_scale != 1.0:
                    video_velocity += cfg_guider.delta(pos_video, neg_video)
                    audio_velocity += cfg_guider.delta(pos_audio, neg_audio)
                video.latent = stepper.step(
                    sample=video.latent, velocity=video_velocity, sigmas=sigmas, step_index=step_idx
                )
                audio.latent = stepper.step(
                    sample=audio.latent, velocity=audio_velocity, sigmas=sigmas, step_index=step_idx
                )

            latent_video = video_builder.unbuild(video.latent)
            self.transformer.to(torch.device("cpu"))
            self.vae_decoder.to(self.device)
            decoded_video = self.vae_decoder(latent_video[:1])
            decoded_video = (((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            self.vae_decoder.to(torch.device("cpu"))

            latent_audio = audio_builder.unbuild(audio.latent)
            self.audio_decoder.to(self.device)
            decoded_audio = self.audio_decoder(latent_audio)
            waveform = self.vocoder(decoded_audio)
            self.audio_decoder.to(torch.device("cpu"))

        encode_video(
            video=rearrange(decoded_video[0], "c f h w -> f h w c"),
            fps=frame_rate,
            audio=waveform.squeeze(0).float().cpu(),
            audio_sample_rate=self.vocoder.output_sample_rate,
            output_path=output_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--gemma_root", type=str, required=True)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
            "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
            "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
            "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
            "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
            "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
            "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
            "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
            "off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
            "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
            "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=25.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--lora", type=str, action="append", default=[])
    parser.add_argument("--lora_strength", type=float, action="append", default=[])
    parser.add_argument("--cfg_guidance_scale", type=float, default=1.0)
    # VFly args
    parser.add_argument(
        "--attn_type",
        type=str,
        default="default",
        choices=[
            "default",
            "auto",
            "fivx",
            "sage-attn",
            "sparse-videogen",
            "sparse-videogen2",
            "trtllm-attn",
            "flash-attn3",
            "flash-attn3-fp8",
            "flash-attn4",
            "te",
            "te-fp8",
        ],
        help="Attention type",
    )
    parser.add_argument(
        "--linear_type",
        type=str,
        default="default",
        choices=[
            "default",
            "auto",
            "trtllm-fp8-blockwise",
            "trtllm-fp8-per-tensor",
            "te-fp8-blockwise",
            "te-fp8-per-tensor",
            "trtllm-nvfp4",
            "torch-ao-fp8",
            "svd-nvfp4",
            "flashinfer-nvfp4-trtllm",
            "flashinfer-nvfp4-cudnn",
            "flashinfer-nvfp4-cutlass",
        ],
        help="Linear type",
    )
    parser.add_argument(
        "--linear_recipe",
        type=str,
        default="dynamic",
        choices=[
            "dynamic",
            "static",
        ],
    )
    parser.add_argument(
        "--cp",
        type=str,
        default="ulysses",
        choices=[
            "ulysses",
            "ring",
            "cp",
        ],
        help="Context parallel method",
    )
    args = parser.parse_args()

    if enable_vfly:
        cp_key_mapping = {
            "ulysses": "dit_ulysses_size",
            "ring": "dit_ring_size",
            "cp": "dit_cp_size",
        }
        vfly_configs = {
            "parallel": {
                cp_key_mapping[args.cp]: int(os.environ.get("WORLD_SIZE", "1")) # available cp methods: dit_cp_size, dit_ring_size, dit_ulysses_size
            },
            "attn": {
                "type": args.attn_type,
            },
            "linear": {
                "type": args.linear_type,
                "recipe": args.linear_recipe,
            },
        }
        vfly.setup_configs(**vfly_configs)

    lora_strengths = (args.lora_strength + [1.0] * len(args.lora))[: len(args.lora)]
    loras = [
        LoraPathStrengthAndKeyOps(lora, strength, LTXV_LORA_COMFY_RENAMING_MAP)
        for lora, strength in zip(args.lora, lora_strengths, strict=True)
    ]
    pipeline = Txt2VidPipeline(checkpoint_path=args.checkpoint_path, gemma_root=args.gemma_root, loras=loras)
    if enable_vfly:
        # with open("transformer.txt", "w") as f:
        #     f.write(str(pipeline.transformer))
        class VflyAttention(vfly.layers.VflyAttnProcessor):
            # modified from ltx-core/src/ltx_core/model/transformer/attention.py:PytorchAttention
            def __call__(
                self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
            ) -> torch.Tensor:
                b, _, dim_head = q.shape
                dim_head //= heads
                q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

                if mask is not None:
                    # add a batch dimension if there isn't already one
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    # add a heads dimension if there isn't already one
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(1)
                # replace the default attention with vfly attention
                out = self.vfly_attn(q, k, v, tensor_layout="NHD", attn_mask=mask, dropout_p=0.0, is_causal=False)
                out = out.reshape(b, -1, heads * dim_head)
                return out
        def apply_vfly_attention(transformer: LTXModel):
            for name, module in transformer.named_modules():
                if hasattr(module, "attention_function"):
                    module.attention_function = VflyAttention()
        apply_vfly_attention(pipeline.transformer)
        vfly.layers.apply_vfly_linear(pipeline.transformer, load_parameters=True, quantize_weights=True)
        vfly.layers.apply_vfly_norm(pipeline.transformer, load_parameters=True)
        # todo: torch compile has bug when enable multi-gpu
        # pipeline.transformer = torch.compile(pipeline.transformer, mode="default")
        # todo: fullgraph has accuracy issue
        pipeline.transformer.transformer_blocks.forward = cudagraph_wrapper(pipeline.transformer.transformer_blocks.forward)
        # with open("transformer_vfly.txt", "w") as f:
        #     f.write(str(pipeline.transformer))
        # warmup
        pipeline(
            prompt=args.prompt,
            output_path=args.output_path,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=2,
            cfg_guidance_scale=args.cfg_guidance_scale,
        )

    pipeline(
        prompt=args.prompt,
        output_path=args.output_path,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_guidance_scale=args.cfg_guidance_scale,
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()

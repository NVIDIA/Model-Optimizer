import argparse

import torch
from einops import rearrange
from tqdm import tqdm

from ltx_core.loader.primitives import LoraPathStrengthAndKeyOps
from ltx_core.loader.sd_keys_ops import LTXV_LORA_COMFY_RENAMING_MAP, LTXV_MODEL_COMFY_RENAMING_MAP
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
from ltx_core.model.video_vae.model_configurator import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VAEDecoderConfigurator,
    VAEEncoderConfigurator,
)
from ltx_core.model.video_vae.video_vae import Decoder, Encoder
from ltx_core.pipeline.components.diffusion_steps import EulerDiffusionStep
from ltx_core.pipeline.components.guiders import CFGGuider
from ltx_core.pipeline.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.pipeline.components.schedulers import LTX2Scheduler
from ltx_core.pipeline.conditioning import AudioConditioningBuilder, ConditioningMethod, VideoConditioningBuilder
from ltx_pipelines.media_io import decode_and_preprocess_image, encode_video


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Img2VidPipeline:
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

        self.vae_decoder: Decoder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VAEDecoderConfigurator,
            model_sd_key_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=device)

        self.vae_encoder: Encoder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VAEEncoderConfigurator,
            model_sd_key_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
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

    def __call__(  # noqa: PLR0913 PLR0915
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
        image_path: str,
    ) -> None:
        self.text_encoder.to(self.device)
        with torch.inference_mode():
            v_context_p, a_context_p, _ = self.text_encoder(prompt)
            v_context_n, a_context_n, _ = self.text_encoder(negative_prompt)
        self.text_encoder.to(torch.device("cpu"))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        image = rearrange(decode_and_preprocess_image(image_path=image_path, dtype=torch.float32, device=self.device), "h w c -> 1 c h w")
        image = torch.nn.functional.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)
        image = rearrange(image, "1 c h w -> 1 c 1 h w")
        image = image.to(device=self.device, dtype=torch.float32) / 127.5 - 1.0

        self.transformer.to(self.device)

        video_builder = VideoConditioningBuilder(
            patchifier=VideoLatentPatchifier(patch_size=1),
            batch=1,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=frame_rate,
        )

        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            self.vae_encoder.to(self.device)
            encoded_image = self.vae_encoder(image)
            self.vae_encoder.to(torch.device("cpu"))

        video_input = video_builder.with_single_frame(
            image_latent=encoded_image, strength=1.0, frame_idx=0, method=ConditioningMethod.REPLACE
        ).build(device=self.device, dtype=torch.bfloat16, generator=generator)

        audio_builder = AudioConditioningBuilder(
            patchifier=AudioPatchifier(patch_size=1),
            batch=1,
            duration=num_frames / frame_rate,
        )
        audio_input = audio_builder.build(device=self.device, dtype=torch.bfloat16, generator=generator)

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
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

        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                for step_idx, sigma in enumerate(tqdm(sigmas[:-1])):
                    video.timesteps = sigma * video_input.denoise_mask
                    audio.timesteps = sigma * audio_input.denoise_mask
                    if cfg_guidance_scale != 1.0:
                        video_neg = Modality(
                            enabled=True,
                            latent=video.latent,
                            timesteps=video.timesteps,
                            positions=video.positions,
                            context=v_context_n,
                            context_mask=None,
                        )
                        audio_neg = Modality(
                            enabled=True,
                            latent=audio.latent,
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

                    masked_video_velocity = video_velocity * video_input.denoise_mask.unsqueeze(-1)
                    masked_audio_velocity = audio_velocity * audio_input.denoise_mask.unsqueeze(-1)

                    video.latent = stepper.step(
                        sample=video.latent, velocity=masked_video_velocity, sigmas=sigmas, step_index=step_idx
                    )
                    audio.latent = stepper.step(
                        sample=audio.latent, velocity=masked_audio_velocity, sigmas=sigmas, step_index=step_idx
                    )

                    video_builder.apply_conditioning_(video.latent, video_input.denoise_mask)
                    audio_builder.apply_conditioning_(audio.latent, audio_input.denoise_mask)

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
    parser.add_argument("--prompt", type=str, required=True)
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
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--cfg_guidance_scale", type=float, default=1.0)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--lora", type=str, action="append", default=[])
    parser.add_argument("--lora_strength", type=float, action="append", default=[])
    args = parser.parse_args()
    lora_strengths = (args.lora_strength + [1.0] * len(args.lora))[: len(args.lora)]
    loras = [
        LoraPathStrengthAndKeyOps(lora, strength, LTXV_LORA_COMFY_RENAMING_MAP)
        for lora, strength in zip(args.lora, lora_strengths, strict=True)
    ]
    pipeline = Img2VidPipeline(checkpoint_path=args.checkpoint_path, gemma_root=args.gemma_root, loras=loras)
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
        image_path=args.image_path,
    )


if __name__ == "__main__":
    main()

import json
from dataclasses import dataclass
import math
import torch
import torchaudio

from ltx_core.models.ops.symmetric_patchifier import AudioPatchifier
from ltx_core.models.vae.causal_audio_autoencoder import (
    CausalityAxis,
    CausalAudioAutoencoder,
)
from ltx_core.models.vocoders.vocoder import Vocoder

LATENT_DOWNSAMPLE_FACTOR = 4


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
@dataclass(frozen=True)
class AudioVAEComponentConfig:
    """Container for model component configuration extracted from metadata."""

    autoencoder: dict
    vocoder: dict

    @classmethod
    def from_metadata(cls, metadata: dict) -> "AudioVAEComponentConfig":
        assert metadata is not None and "config" in metadata, "Metadata is required for audio VAE"

        raw_config = metadata["config"]
        if isinstance(raw_config, str):
            parsed_config = json.loads(raw_config)
        else:
            parsed_config = raw_config

        audio_config = parsed_config.get("audio_vae")
        vocoder_config = parsed_config.get("vocoder")

        assert audio_config is not None, "Audio VAE config is required for audio VAE"
        assert vocoder_config is not None, "Vocoder config is required for audio VAE"

        return cls(autoencoder=audio_config, vocoder=vocoder_config)



class AudioLatentNormalizer:
    """Applies per-channel statistics in patch space and restores original layout."""

    def __init__(self, patchfier: AudioPatchifier, statistics_processor: torch.nn.Module):
        self.patchifier = patchfier
        self.statistics = statistics_processor

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        channels = latents.shape[1]
        freq = latents.shape[3]
        patched, _ = self.patchifier.patchify(latents)
        normalized = self.statistics.normalize(patched)
        return self.patchifier.unpatchify(normalized, channels=channels, freq=freq)

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        channels = latents.shape[1]
        freq = latents.shape[3]
        patched, _ = self.patchifier.patchify(latents)
        denormalized = self.statistics.un_normalize(patched)
        return self.patchifier.unpatchify(denormalized, channels=channels, freq=freq)


class AudioPreprocessor:
    """Prepares raw waveforms for the autoencoder by matching training conditions."""

    def __init__(self, target_sample_rate: int, mel_bins: int, mel_hop_length: int, n_fft: int):
        self.target_sample_rate = target_sample_rate
        self.mel_bins = mel_bins
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft

    def resample(self, waveform: torch.Tensor, source_rate: int) -> torch.Tensor:
        if source_rate == self.target_sample_rate:
            return waveform
        return torchaudio.functional.resample(waveform, source_rate, self.target_sample_rate)

    @staticmethod
    def normalize_amplitude(
        waveform: torch.Tensor, max_amplitude: float = 0.5, eps: float = 1e-5
    ) -> torch.Tensor:
        waveform = waveform - waveform.mean(dim=2, keepdim=True)
        peak = torch.max(torch.abs(waveform)) + eps
        scale = peak.clamp(max=max_amplitude) / peak
        return waveform * scale

    def waveform_to_mel(
        self, waveform: torch.Tensor, waveform_sample_rate: int, device
    ) -> torch.Tensor:
        waveform = self.resample(waveform, waveform_sample_rate)
        waveform = self.normalize_amplitude(waveform)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.mel_hop_length,
            f_min=0.0,
            f_max=self.target_sample_rate / 2.0,
            n_mels=self.mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        ).to(device)

        mel = mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.permute(0, 1, 3, 2).contiguous()


class AudioVAE(torch.nn.Module):
    """High-level Audio VAE wrapper exposing encode and decode entry points."""

    def __init__(self, state_dict: dict, metadata: dict):
        super().__init__()

        component_config = AudioVAEComponentConfig.from_metadata(metadata)

        vae_sd = {k.replace("audio_vae.", ""): v for k, v in state_dict.items() if "audio_vae." in k}
        vocoder_sd = {k.replace("vocoder.", ""): v for k, v in state_dict.items() if "vocoder." in k}

        # Create models on meta device to avoid memory allocation during initialization
        with torch.device("meta"):
            self.autoencoder = CausalAudioAutoencoder(config=component_config.autoencoder)
            self.vocoder = Vocoder(config=component_config.vocoder)

        # Load weights (this will allocate memory and load the actual weights)
        self.autoencoder.load_state_dict(vae_sd, assign=True)
        self.vocoder.load_state_dict(vocoder_sd, assign=True)

        autoencoder_config = self.autoencoder.get_config()
        self.normalizer = AudioLatentNormalizer(
            AudioPatchifier(
                patch_size=1,
                audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
                sample_rate=autoencoder_config["sampling_rate"],
                hop_length=autoencoder_config["mel_hop_length"],
                is_causal=autoencoder_config["is_causal"],
            ),
            self.autoencoder.per_channel_statistics,
        )

        self.preprocessor = AudioPreprocessor(
            target_sample_rate=autoencoder_config["sampling_rate"],
            mel_bins=autoencoder_config["mel_bins"],
            mel_hop_length=autoencoder_config["mel_hop_length"],
            n_fft=autoencoder_config["n_fft"],
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def encode(self, audio: dict) -> torch.Tensor:
        """Encode a waveform dictionary into normalized latent tensors."""

        waveform = audio["waveform"]
        waveform_sample_rate = audio["sample_rate"]

        expected_channels = self.autoencoder.encoder.in_channels
        if waveform.shape[1] != expected_channels:
            raise ValueError(
                f"Input audio must have {expected_channels} channels, got {waveform.shape[1]}"
            )

        mel_spec = self.preprocessor.waveform_to_mel(
            waveform, waveform_sample_rate, device=self.device
        )

        latents = self.autoencoder.encode(mel_spec)
        posterior = DiagonalGaussianDistribution(latents)
        latent_mode = posterior.mode()

        return self.normalizer.normalize(latent_mode)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized latent tensors into an audio waveform."""
        original_shape = latents.shape

        latents = self.normalizer.denormalize(latents)

        target_shape = self.target_shape_from_latents(original_shape)
        mel_spec = self.autoencoder.decode(latents, target_shape=target_shape)

        waveform = self.run_vocoder(mel_spec)
        return waveform

    def target_shape_from_latents(self, latents_shape):
        batch, _, time, _ = latents_shape
        target_length = time * LATENT_DOWNSAMPLE_FACTOR
        if self.autoencoder.causality_axis != CausalityAxis.NONE:
            target_length -= LATENT_DOWNSAMPLE_FACTOR - 1
        return (
            batch,
            self.autoencoder.decoder.out_ch,
            target_length,
            self.autoencoder.mel_bins,
        )

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int:
        return math.ceil((float(frames_number) / frame_rate) * self.latents_per_second)

    def run_vocoder(self, mel_spec: torch.Tensor) -> torch.Tensor:
        audio_channels = self.autoencoder.decoder.out_ch
        vocoder_input = mel_spec.transpose(2, 3)

        if audio_channels == 1:
            vocoder_input = vocoder_input.squeeze(1)
        elif audio_channels != 2:
            raise ValueError(f"Unsupported audio_channels: {audio_channels}")

        return self.vocoder(vocoder_input)

    @property
    def sample_rate(self) -> int:
        return int(self.autoencoder.sampling_rate)

    @property
    def mel_hop_length(self) -> int:
        return int(self.autoencoder.mel_hop_length)

    @property
    def mel_bins(self) -> int:
        return int(self.autoencoder.mel_bins)

    @property
    def latent_channels(self) -> int:
        return int(self.autoencoder.decoder.z_channels)

    @property
    def latent_frequency_bins(self) -> int:
        return int(self.mel_bins // LATENT_DOWNSAMPLE_FACTOR)

    @property
    def latents_per_second(self) -> float:
        return self.sample_rate / self.mel_hop_length / LATENT_DOWNSAMPLE_FACTOR

    @property
    def output_sample_rate(self) -> int:
        output_rate = getattr(self.vocoder, "output_sample_rate", None)
        if output_rate is not None:
            return int(output_rate)
        upsample_factor = getattr(self.vocoder, "upsample_factor", None)
        if upsample_factor is None:
            raise AttributeError(
                "Vocoder is missing upsample_factor; cannot infer output sample rate"
            )
        return int(self.sample_rate * upsample_factor / self.mel_hop_length)

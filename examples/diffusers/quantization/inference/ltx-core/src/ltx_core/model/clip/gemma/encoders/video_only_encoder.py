from collections import namedtuple
from typing import Self

import torch
from transformers import Gemma3ForConditionalGeneration

from ltx_core.loader.sd_keys_ops import CompulsoryContent, ContentReplacement, SDKeyOps
from ltx_core.model.clip.gemma.embeddings_connector import (
    Embeddings1DConnector,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.model.clip.gemma.encoders.base_encoder import (
    GemmaTextEncoderModelBase,
    GemmaTextEncoderModelConfiguratorBase,
)
from ltx_core.model.clip.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from ltx_core.model.clip.gemma.tokenizer import LTXVGemmaTokenizer

VideoGemmaEncoderOutput = namedtuple("VideoGemmaEncoderOutput", ["video_encoding", "attention_mask"])


class VideoGemmaTextEncoderModel(GemmaTextEncoderModelBase):
    """
    Video Gemma Text Encoder Model.

    This class combines the tokenizer, Gemma model, feature extractor from base class and a
    video embeddings connector to provide a preprocessing for video only pipeline.
    """

    def __init__(
        self,
        tokenizer: LTXVGemmaTokenizer,
        model: Gemma3ForConditionalGeneration,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        embeddings_connector: Embeddings1DConnector,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(tokenizer, model, feature_extractor_linear, dtype)
        self.embeddings_connector = embeddings_connector.to(dtype=dtype)

    def _run_connector(
        self, encoded_input: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        connector_attention_mask = self._convert_to_additive_mask(attention_mask, encoded_input.dtype)

        encoded, encoded_connector_attention_mask = self.embeddings_connector(
            encoded_input,
            connector_attention_mask,
        )

        # restore the mask values to int64
        attention_mask = (encoded_connector_attention_mask < 0.000001).to(torch.int64)
        attention_mask = attention_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
        encoded = encoded * attention_mask

        return encoded, attention_mask.squeeze(-1)

    def forward(self, text: str, padding_side: str = "left") -> VideoGemmaEncoderOutput:
        encoded_inputs, attention_mask = self._preprocess_text(text, padding_side)
        video_encoding, attention_mask = self._run_connector(encoded_inputs, attention_mask)
        return VideoGemmaEncoderOutput(video_encoding, attention_mask)


class VideoGemmaTextEncoderModelConfigurator(GemmaTextEncoderModelConfiguratorBase):
    @classmethod
    def from_config(cls: type[Self], config: dict) -> Self:
        model = cls.gemma
        tokenizer = cls.tokenizer
        feature_extractor_linear = GemmaFeaturesExtractorProjLinear.from_config(config)
        embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        return VideoGemmaTextEncoderModel(
            tokenizer=tokenizer,
            model=model,
            feature_extractor_linear=feature_extractor_linear,
            embeddings_connector=embeddings_connector,
        )


VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS = SDKeyOps(
    name="VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS",
    mapping=(
        CompulsoryContent(prefix="text_embedding_projection."),
        CompulsoryContent(prefix="model.diffusion_model.embeddings_connector."),
        ContentReplacement(content="text_embedding_projection.", replacement="feature_extractor_linear."),
        ContentReplacement(content="model.diffusion_model.embeddings_connector.", replacement="embeddings_connector."),
    ),
)

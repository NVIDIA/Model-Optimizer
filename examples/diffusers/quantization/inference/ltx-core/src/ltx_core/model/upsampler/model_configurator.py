from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.upsampler.model import LatentUpsampler


class LatentUpsamplerConfigurator(ModelConfigurator[LatentUpsampler]):
    @classmethod
    def from_config(cls: type[LatentUpsampler], config: dict) -> LatentUpsampler:
        return LatentUpsampler(**config)

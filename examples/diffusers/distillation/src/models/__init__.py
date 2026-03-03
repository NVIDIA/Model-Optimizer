"""Model backend registry.

Maps model names to their (ModelLoader, adapter_factory, InferencePipeline) tuple.
Adapter is returned as a factory callable because it may need to be initialized
with model-specific components after the model is loaded.

Usage::

    @register_backend("wan")
    def _wan_backend(variant):
        ...

    loader, create_adapter, pipeline_cls = get_model_backend("wan", variant="ti2v-5B")
"""

from __future__ import annotations

from collections.abc import Callable

from ..interfaces import InferencePipeline, ModelLoader, TrainingForwardAdapter

BackendTuple = tuple[ModelLoader, Callable[..., TrainingForwardAdapter], Callable[..., InferencePipeline] | None]

_REGISTRY: dict[str, Callable[[str | None], BackendTuple]] = {}


def register_backend(name: str):
    """Decorator that registers a backend factory under *name*."""
    def decorator(factory: Callable[[str | None], BackendTuple]):
        if name in _REGISTRY:
            raise ValueError(f"Backend '{name}' is already registered.")
        _REGISTRY[name] = factory
        return factory
    return decorator


def get_model_backend(name: str, variant: str | None = None) -> BackendTuple:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Unknown model backend '{name}'. Available: {available}")
    return _REGISTRY[name](variant)


@register_backend("wan")
def _wan_backend(variant: str | None) -> BackendTuple:
    from .wan import WanInferencePipeline, WanModelLoader, create_wan_adapter

    return (
        WanModelLoader(variant),
        lambda: create_wan_adapter(variant),
        lambda: WanInferencePipeline(variant),
    )


@register_backend("ltx2")
def _ltx2_backend(variant: str | None) -> BackendTuple:
    from .ltx2 import LTX2InferencePipeline, LTX2ModelLoader, create_ltx2_adapter

    return LTX2ModelLoader(), create_ltx2_adapter, LTX2InferencePipeline

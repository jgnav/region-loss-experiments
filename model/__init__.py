from .head import DINOHead, iBOTHead
from .vision_transformer import (
    VisionTransformer,
    vit_base,
    vit_large,
    vit_small,
)


ARCHITECTURES = {
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
}


def create_model(architecture="vit_small", **kwargs):
    try:
        factory = ARCHITECTURES[architecture]
    except KeyError as error:
        choices = ", ".join(ARCHITECTURES)
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choose from: {choices}"
        ) from error
    return factory(**kwargs)


__all__ = [
    "ARCHITECTURES",
    "DINOHead",
    "VisionTransformer",
    "create_model",
    "iBOTHead",
    "vit_base",
    "vit_large",
    "vit_small",
]

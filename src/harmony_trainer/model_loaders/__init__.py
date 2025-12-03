"""Model weight loader subsystem for third-party ML models."""

from harmony_trainer.model_loaders.bs_roformer import (
    get_bs_roformer_checkpoint_path,
    load_bs_roformer_model,
)
from harmony_trainer.model_loaders.medleyvox import (
    get_medleyvox_checkpoint_path,
    load_medleyvox_model,
)

__all__ = [
    "get_bs_roformer_checkpoint_path",
    "get_medleyvox_checkpoint_path",
    "load_bs_roformer_model",
    "load_medleyvox_model",
]

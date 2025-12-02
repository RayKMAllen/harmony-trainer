"""Model weight loader subsystem for third-party ML models."""

from harmony_trainer.model_loaders.medleyvox import (
    get_medleyvox_checkpoint_path,
    load_medleyvox_model,
)

__all__ = ["get_medleyvox_checkpoint_path", "load_medleyvox_model"]

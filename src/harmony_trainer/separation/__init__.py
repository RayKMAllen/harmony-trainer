"""Audio separation module for vocal/instrumental splitting."""

from harmony_trainer.separation.demucs import DemucsWrapper, separate_audio
from harmony_trainer.separation.svoice import (
    SeparationError,
    SVoiceWrapper,
    separate_vocals,
)

__all__ = [
    "DemucsWrapper",
    "SeparationError",
    "SVoiceWrapper",
    "separate_audio",
    "separate_vocals",
]

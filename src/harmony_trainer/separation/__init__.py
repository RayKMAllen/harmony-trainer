"""Audio separation module for vocal/instrumental splitting."""

from harmony_trainer.separation.bs_roformer import (
    BSRoFormerWrapper,
    separate_vocals_bs_roformer,
)
from harmony_trainer.separation.demucs import DemucsWrapper, separate_audio
from harmony_trainer.separation.medleyvox import (
    MedleyVoxWrapper,
    separate_vocals_medleyvox,
)
from harmony_trainer.separation.svoice import (
    SeparationError,
    SVoiceWrapper,
    separate_vocals,
)

__all__ = [
    "BSRoFormerWrapper",
    "DemucsWrapper",
    "MedleyVoxWrapper",
    "SeparationError",
    "SVoiceWrapper",
    "separate_audio",
    "separate_vocals",
    "separate_vocals_bs_roformer",
    "separate_vocals_medleyvox",
]

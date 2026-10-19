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
from harmony_trainer.separation.errors import SeparationError

__all__ = [
    "BSRoFormerWrapper",
    "DemucsWrapper",
    "MedleyVoxWrapper",
    "SeparationError",
    "separate_audio",
    "separate_vocals_bs_roformer",
    "separate_vocals_medleyvox",
]

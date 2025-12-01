"""Audio separation module for vocal/instrumental splitting."""

from harmony_trainer.separation.demucs import DemucsWrapper, separate_audio

__all__ = ["DemucsWrapper", "separate_audio"]

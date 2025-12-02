"""MedleyVox wrapper for lead/backing vocal separation."""

from pathlib import Path
from typing import Any
from uuid import UUID

import torch
import torchaudio  # type: ignore[import-untyped]

from harmony_trainer.audio_io import load_audio, save_wav
from harmony_trainer.model_loaders.medleyvox import (
    CheckpointNotFoundError,
    get_medleyvox_checkpoint_path,
    load_medleyvox_model,
)
from harmony_trainer.separation.svoice import SeparationError


class MedleyVoxWrapper:
    """Wrapper for MedleyVox audio separation model.

    MedleyVox is designed for singing voice separation, specifically
    for separating lead vocals from backing vocals in music.

    Note: MedleyVox requires a pretrained checkpoint. You must obtain
    or train a model checkpoint. See: https://github.com/jeonchangbin49/MedleyVox
    """

    def __init__(self, checkpoint_path: Path | str | None = None):
        """Initialize MedleyVox wrapper.

        Args:
            checkpoint_path: Path to pretrained MedleyVox checkpoint.
                If None, uses the loader subsystem with fallback hierarchy.
        """
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model: dict[str, Any] | None = None

    @property
    def checkpoint_path(self) -> Path:
        """Get checkpoint path using the loader subsystem."""
        if self._checkpoint_path is None:
            try:
                self._checkpoint_path = get_medleyvox_checkpoint_path()
            except CheckpointNotFoundError as e:
                raise SeparationError(str(e)) from e
        if not self._checkpoint_path.exists():
            raise SeparationError(f"Checkpoint not found: {self._checkpoint_path}")
        return self._checkpoint_path

    @property
    def model(self) -> dict[str, Any]:
        if self._model is None:
            try:
                self._model = load_medleyvox_model(self.checkpoint_path)
            except (CheckpointNotFoundError, RuntimeError) as e:
                raise SeparationError(f"Failed to load MedleyVox model: {e}") from e
        return self._model

    def separate(
        self,
        vocals_path: Path,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Separate vocals into lead and backing.

        Returns paths to (lead.wav, backing.wav).
        Raises SeparationError on failure.
        """
        return self._do_separate(vocals_path, output_dir)

    def _do_separate(
        self,
        vocals_path: Path,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Internal separation logic."""
        # Load model (this will raise if checkpoint not found)
        _ = self.model

        # Load audio
        waveform, sample_rate = load_audio(vocals_path)

        # MedleyVox typically expects 44.1kHz stereo input
        target_sr = 44100
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        # Convert mono to stereo if needed
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # MedleyVox model inference would go here
        # Since we don't have the actual model class, we simulate the interface
        # The actual implementation would use the MedleyVox model architecture
        raise SeparationError(
            "MedleyVox model inference not implemented. "
            "Requires MedleyVox model class from https://github.com/jeonchangbin49/MedleyVox"
        )

    def _save_outputs(
        self,
        lead: torch.Tensor,
        backing: torch.Tensor,
        output_dir: Path,
        sample_rate: int,
    ) -> tuple[Path, Path]:
        """Save separated outputs to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        lead_path = output_dir / "lead.wav"
        backing_path = output_dir / "backing.wav"

        save_wav(lead_path, lead.cpu(), sample_rate)
        save_wav(backing_path, backing.cpu(), sample_rate)

        return lead_path, backing_path


def separate_vocals_medleyvox(
    vocals_path: Path,
    song_id: UUID,
    data_dir: Path = Path("data"),
) -> tuple[Path, Path]:
    """Convenience function to separate vocals using MedleyVox.

    Returns paths to (lead.wav, backing.wav).
    Raises SeparationError on failure.
    """
    wrapper = MedleyVoxWrapper()
    output_dir = data_dir / "stems" / str(song_id)
    return wrapper.separate(vocals_path, output_dir)

"""MedleyVox wrapper for lead/backing vocal separation."""

from pathlib import Path
from typing import Any
from uuid import UUID

import torch
import torchaudio  # type: ignore[import-untyped]

from harmony_trainer.audio_io import load_audio, save_wav
from harmony_trainer.separation.svoice import SeparationError


def _get_cache_dir() -> Path:
    """Get cache directory for MedleyVox checkpoints."""
    cache_dir = Path.home() / ".cache" / "harmony_trainer" / "medleyvox"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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
                If None, will look for checkpoint in cache directory.
        """
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model: Any | None = None

    @property
    def checkpoint_path(self) -> Path:
        """Get checkpoint path, checking cache if not explicitly provided."""
        if self._checkpoint_path is None:
            cache_dir = _get_cache_dir()
            # Look for any .pth or .pt file in cache
            checkpoints = list(cache_dir.glob("*.pth")) + list(cache_dir.glob("*.pt"))
            if checkpoints:
                self._checkpoint_path = checkpoints[0]
            else:
                raise SeparationError(
                    "No MedleyVox checkpoint found. Please provide checkpoint_path. "
                    "See https://github.com/jeonchangbin49/MedleyVox for model downloads."
                )
        if not self._checkpoint_path.exists():
            raise SeparationError(f"Checkpoint not found: {self._checkpoint_path}")
        return self._checkpoint_path

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                pkg = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                # MedleyVox typically stores model state_dict under 'model' or 'state_dict'
                if isinstance(pkg, dict):
                    if "model" in pkg:
                        state_dict = pkg["model"]
                    elif "state_dict" in pkg:
                        state_dict = pkg["state_dict"]
                    else:
                        state_dict = pkg
                else:
                    state_dict = pkg

                # MedleyVox uses a UNet-style architecture
                # For now, store the state_dict; actual model instantiation
                # requires the MedleyVox model class
                self._model = state_dict
            except Exception as e:
                raise SeparationError(f"Failed to load MedleyVox model: {e}") from e
        return self._model

    def separate(
        self,
        vocals_path: Path,
        output_dir: Path,
    ) -> tuple[Path | None, Path | None]:
        """Attempt to separate vocals into lead and backing.

        Returns paths to (lead.wav, backing.wav) or (None, None) on failure.
        Gracefully handles separation failures.
        """
        try:
            return self._do_separate(vocals_path, output_dir)
        except Exception:
            return None, None

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
) -> tuple[Path | None, Path | None]:
    """Convenience function to separate vocals using MedleyVox.

    Returns paths to (lead.wav, backing.wav) or (None, None) on failure.
    """
    wrapper = MedleyVoxWrapper()
    output_dir = data_dir / "stems" / str(song_id)
    return wrapper.separate(vocals_path, output_dir)

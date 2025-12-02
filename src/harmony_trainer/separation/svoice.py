"""SVoice wrapper for lead/backing vocal separation."""

from pathlib import Path
from typing import Any
from uuid import UUID

import torch
import torchaudio  # type: ignore[import-untyped]

from harmony_trainer.audio_io import load_audio, save_wav


class SeparationError(Exception):
    """Raised when vocal separation fails."""


def _get_cache_dir() -> Path:
    """Get cache directory for SVoice checkpoints."""
    cache_dir = Path.home() / ".cache" / "harmony_trainer" / "svoice"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class SVoiceWrapper:
    """Wrapper for SVoice audio separation model.

    SVoice is designed for speech separation but can be attempted
    for lead/backing vocal separation. Requires a pretrained checkpoint.

    Note: Facebook's pretrained checkpoints are no longer publicly available.
    You must train your own model or obtain a checkpoint from another source.
    See: https://github.com/facebookresearch/svoice
    """

    def __init__(self, checkpoint_path: Path | str | None = None):
        """Initialize SVoice wrapper.

        Args:
            checkpoint_path: Path to pretrained SVoice checkpoint (.th file).
                If None, will look for checkpoint in cache directory.
        """
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model: Any | None = None

    @property
    def checkpoint_path(self) -> Path:
        """Get checkpoint path, checking cache if not explicitly provided."""
        if self._checkpoint_path is None:
            cache_dir = _get_cache_dir()
            # Look for any .th file in cache
            checkpoints = list(cache_dir.glob("*.th"))
            if checkpoints:
                self._checkpoint_path = checkpoints[0]
            else:
                raise SeparationError(
                    "No SVoice checkpoint found. Please provide checkpoint_path. "
                    "Facebook's pretrained models are no longer publicly available. "
                    "See https://github.com/facebookresearch/svoice for training instructions."
                )
        if not self._checkpoint_path.exists():
            raise SeparationError(f"Checkpoint not found: {self._checkpoint_path}")
        return self._checkpoint_path

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from harmony_trainer.separation._svoice_model import deserialize_model

                pkg = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                model_data = pkg.get("model", pkg)
                self._model = deserialize_model(model_data)
                self._model.eval()
            except Exception as e:
                raise SeparationError(f"Failed to load SVoice model: {e}") from e
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
        model = self.model

        # Load audio
        waveform, sample_rate = load_audio(vocals_path)

        # SVoice expects mono input at 8kHz or 16kHz
        target_sr = 16000
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Add batch dimension
        waveform = waveform.unsqueeze(0)

        # Apply model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        waveform = waveform.to(device)

        with torch.no_grad():
            # SVoice model expects (batch, time) input, squeeze channel dim
            input_wav = waveform.squeeze(1)  # (batch, time)
            outputs = model(input_wav)

        # SVoice returns stacked outputs: (num_blocks, batch, num_sources, time)
        # Use the last block's output (most refined)
        if isinstance(outputs, torch.Tensor) and outputs.dim() == 4:
            separated = outputs[-1]  # (batch, num_sources, time)
        elif isinstance(outputs, torch.Tensor) and outputs.dim() == 3:
            separated = outputs  # (batch, num_sources, time)
        else:
            raise SeparationError(
                f"SVoice output unexpected type/shape: {type(outputs)}, "
                f"{outputs.shape if isinstance(outputs, torch.Tensor) else 'N/A'}"
            )

        if separated.shape[1] < 2:
            raise SeparationError(f"SVoice produced {separated.shape[1]} sources, need at least 2")

        lead = separated[0, 0]  # First source as lead
        backing = separated[0, 1]  # Second source as backing

        # Resample back to original sample rate if needed
        if target_sr != sample_rate:
            resampler_back = torchaudio.transforms.Resample(target_sr, sample_rate)
            lead = resampler_back(lead.unsqueeze(0)).squeeze(0)
            backing = resampler_back(backing.unsqueeze(0)).squeeze(0)
            output_sr = sample_rate
        else:
            output_sr = target_sr

        # Convert mono to stereo for consistency
        lead = lead.unsqueeze(0).repeat(2, 1) if lead.dim() == 1 else lead
        backing = backing.unsqueeze(0).repeat(2, 1) if backing.dim() == 1 else backing

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save outputs
        lead_path = output_dir / "lead.wav"
        backing_path = output_dir / "backing.wav"

        save_wav(lead_path, lead.cpu(), output_sr)
        save_wav(backing_path, backing.cpu(), output_sr)

        return lead_path, backing_path


def separate_vocals(
    vocals_path: Path,
    song_id: UUID,
    data_dir: Path = Path("data"),
) -> tuple[Path, Path]:
    """Convenience function to separate vocals for a song.

    Returns paths to (lead.wav, backing.wav).
    Raises SeparationError on failure.
    """
    wrapper = SVoiceWrapper()
    output_dir = data_dir / "stems" / str(song_id)
    return wrapper.separate(vocals_path, output_dir)

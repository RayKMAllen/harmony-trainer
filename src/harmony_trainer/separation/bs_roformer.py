"""BS-RoFormer wrapper for lead/backing vocal separation per ADR-0006."""

from pathlib import Path
from uuid import UUID

import torch
import torchaudio  # type: ignore[import-untyped]

from harmony_trainer.audio_io import load_audio, save_wav
from harmony_trainer.model_loaders.bs_roformer import (
    CheckpointNotFoundError,
    get_bs_roformer_checkpoint_path,
    load_bs_roformer_model,
)
from harmony_trainer.models.bs_roformer import BSRoformerNet
from harmony_trainer.separation.svoice import SeparationError


class BSRoFormerWrapper:
    """Wrapper for BS-RoFormer audio separation model.

    BS-RoFormer is a state-of-the-art music source separation model
    using band-split processing with rotary embeddings.
    """

    def __init__(self, checkpoint_path: Path | str | None = None):
        """Initialize BS-RoFormer wrapper.

        Args:
            checkpoint_path: Path to pretrained checkpoint.
                If None, uses the loader subsystem with fallback hierarchy.
        """
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model: BSRoformerNet | None = None

    @property
    def checkpoint_path(self) -> Path:
        """Get checkpoint path using the loader subsystem."""
        if self._checkpoint_path is None:
            try:
                self._checkpoint_path = get_bs_roformer_checkpoint_path()
            except CheckpointNotFoundError as e:
                raise SeparationError(str(e)) from e
        if not self._checkpoint_path.exists():
            raise SeparationError(f"Checkpoint not found: {self._checkpoint_path}")
        return self._checkpoint_path

    @property
    def model(self) -> BSRoformerNet:
        if self._model is None:
            try:
                self._model = load_bs_roformer_model(self.checkpoint_path)
            except (CheckpointNotFoundError, RuntimeError) as e:
                raise SeparationError(f"Failed to load BS-RoFormer model: {e}") from e
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

        # BS-RoFormer expects 44.1kHz stereo input
        target_sr = model.sample_rate
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        # Convert mono to stereo if needed (BSRoformerNet expects stereo)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Add batch dimension: (1, channels, samples)
        input_tensor = waveform.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            # Output shape: (batch, stems, channels, samples)
            output = model(input_tensor)

        # BS-RoFormer with num_stems=1 outputs the separated source
        # For lead/backing, we use the separated output as lead
        # and compute backing as original - lead
        lead = output[0, 0, :, :]  # (channels, samples)

        # Compute backing as residual
        backing = waveform - lead

        return self._save_outputs(lead, backing, output_dir, target_sr)

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


def separate_vocals_bs_roformer(
    vocals_path: Path,
    song_id: UUID,
    data_dir: Path = Path("data"),
) -> tuple[Path, Path]:
    """Convenience function to separate vocals using BS-RoFormer.

    Returns paths to (lead.wav, backing.wav).
    Raises SeparationError on failure.
    """
    wrapper = BSRoFormerWrapper()
    output_dir = data_dir / "stems" / str(song_id)
    return wrapper.separate(vocals_path, output_dir)

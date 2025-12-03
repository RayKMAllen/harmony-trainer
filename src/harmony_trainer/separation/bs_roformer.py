"""BS-RoFormer wrapper for lead/backing vocal separation per ADR-0006."""

import logging
import sys
from collections.abc import Callable
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

logger = logging.getLogger(__name__)


def default_progress_callback(current: int, total: int) -> None:
    """Default console progress display."""
    pct = current * 100 // total
    sys.stdout.write(f"\rSeparating: {pct}% ({current}/{total} chunks)")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


class BSRoFormerWrapper:
    """Wrapper for BS-RoFormer audio separation model.

    BS-RoFormer is a state-of-the-art music source separation model
    using band-split processing with rotary embeddings.
    """

    # Default chunk duration in seconds (per ADR-0007)
    DEFAULT_CHUNK_SECONDS: float = 30.0
    # Default overlap duration in seconds for cross-fade
    DEFAULT_OVERLAP_SECONDS: float = 3.0

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
    ):
        """Initialize BS-RoFormer wrapper.

        Args:
            checkpoint_path: Path to pretrained checkpoint.
                If None, uses the loader subsystem with fallback hierarchy.
            chunk_seconds: Duration of each chunk in seconds (default 30s).
            overlap_seconds: Overlap duration for cross-fade (default 3s).
        """
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model: BSRoformerNet | None = None
        self._chunk_seconds = chunk_seconds or self.DEFAULT_CHUNK_SECONDS
        self._overlap_seconds = overlap_seconds or self.DEFAULT_OVERLAP_SECONDS

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
        progress_callback: Callable[[int, int], None] | None = default_progress_callback,
    ) -> tuple[Path, Path]:
        """Separate vocals into lead and backing.

        Args:
            progress_callback: Optional callback(current, total) for progress updates.
                Defaults to console output. Pass None for silent operation.

        Returns paths to (lead.wav, backing.wav).
        Raises SeparationError on failure.
        """
        return self._do_separate(vocals_path, output_dir, progress_callback)

    def _do_separate(
        self,
        vocals_path: Path,
        output_dir: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[Path, Path]:
        """Internal separation logic with chunked inference per ADR-0007."""
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

        # Process with chunked inference
        lead = self._chunked_inference(model, waveform, target_sr, progress_callback)

        # Compute backing as residual
        backing = waveform - lead

        return self._save_outputs(lead, backing, output_dir, target_sr)

    def _chunked_inference(
        self,
        model: BSRoformerNet,
        waveform: torch.Tensor,
        sample_rate: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """Process audio in chunks with overlap-add reconstruction.

        Args:
            model: The BS-RoFormer model.
            waveform: Input audio tensor (channels, samples).
            sample_rate: Sample rate of the audio.
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            Separated lead audio tensor (channels, samples).
        """
        channels, total_samples = waveform.shape
        chunk_samples = int(self._chunk_seconds * sample_rate)
        overlap_samples = int(self._overlap_seconds * sample_rate)
        hop_samples = chunk_samples - overlap_samples

        # If audio is shorter than one chunk, process directly
        if total_samples <= chunk_samples:
            logger.info("Processing 100%% (single chunk)")
            if progress_callback:
                progress_callback(1, 1)
            with torch.no_grad():
                input_tensor = waveform.unsqueeze(0)
                output: torch.Tensor = model(input_tensor)
            return output[0, 0, :, :]

        # Calculate number of chunks
        num_chunks = max(1, (total_samples - overlap_samples + hop_samples - 1) // hop_samples)

        # Initialize output buffer and weight accumulator for overlap-add
        output_buffer = torch.zeros(channels, total_samples, dtype=waveform.dtype)
        weight_buffer = torch.zeros(total_samples, dtype=waveform.dtype)

        for i in range(num_chunks):
            start = i * hop_samples
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[:, start:end]

            # Pad if chunk is shorter than expected (last chunk)
            actual_len = chunk.shape[1]
            if actual_len < chunk_samples:
                padding = torch.zeros(channels, chunk_samples - actual_len, dtype=chunk.dtype)
                chunk = torch.cat([chunk, padding], dim=1)

            # Run inference on chunk
            with torch.no_grad():
                input_tensor = chunk.unsqueeze(0)
                chunk_output = model(input_tensor)
            chunk_lead = chunk_output[0, 0, :, :]  # (channels, chunk_samples)

            # Trim to actual length if padded
            chunk_lead = chunk_lead[:, :actual_len]

            # Build weight window for this chunk (linear cross-fade in overlap regions)
            chunk_weight = torch.ones(actual_len, dtype=waveform.dtype)

            # Apply fade-in weight at start (except for first chunk)
            if i > 0 and actual_len >= overlap_samples:
                fade_in = torch.linspace(0.0, 1.0, overlap_samples)
                chunk_weight[:overlap_samples] = fade_in

            # Apply fade-out weight at end (except for last chunk)
            if i < num_chunks - 1 and actual_len >= overlap_samples:
                fade_out = torch.linspace(1.0, 0.0, overlap_samples)
                chunk_weight[-overlap_samples:] = fade_out

            # Weighted accumulation into output buffer
            output_buffer[:, start : start + actual_len] += chunk_lead * chunk_weight
            weight_buffer[start : start + actual_len] += chunk_weight

            # Report progress
            if progress_callback:
                progress_callback(i + 1, num_chunks)
            progress = int((i + 1) / num_chunks * 100)
            logger.info("Processing %d%% (chunk %d/%d)", progress, i + 1, num_chunks)

        # Normalize by accumulated weights (avoid division by zero)
        weight_buffer = weight_buffer.clamp(min=1e-8)
        output_buffer /= weight_buffer.unsqueeze(0)

        return output_buffer

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

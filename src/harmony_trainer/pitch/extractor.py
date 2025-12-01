"""Pitch extraction using torchcrepe."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np
import torch
import torchaudio  # type: ignore[import-untyped]


@dataclass
class PitchContour:
    """Time-series pitch data."""

    times: np.ndarray  # Time in seconds
    frequencies: np.ndarray  # Pitch in Hz (0 = unvoiced)
    confidence: np.ndarray  # Confidence values [0, 1]
    sample_rate: int  # Original audio sample rate


class PitchExtractor:
    """Wrapper for torchcrepe pitch extraction."""

    def __init__(
        self,
        model: str = "full",
        hop_length: int = 512,
        fmin: float = 50.0,
        fmax: float = 2000.0,
    ):
        """Initialize pitch extractor.

        Args:
            model: torchcrepe model size ('tiny', 'small', 'medium', 'large', 'full')
            hop_length: Hop length in samples (at 16kHz)
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
        """
        self.model = model
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self._target_sr = 16000  # torchcrepe expects 16kHz

    def extract(self, audio_path: Path) -> PitchContour:
        """Extract pitch contour from mono audio file."""
        import torchcrepe  # type: ignore[import-untyped]

        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != self._target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self._target_sr)
            waveform = resampler(waveform)

        # Select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform = waveform.to(device)

        # Extract pitch using torchcrepe
        pitch, periodicity = torchcrepe.predict(
            waveform,
            self._target_sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            model=self.model,
            batch_size=2048,
            device=device,
            return_periodicity=True,
        )

        # Convert to numpy
        pitch_np: np.ndarray = pitch.squeeze().cpu().numpy()
        confidence_np: np.ndarray = periodicity.squeeze().cpu().numpy()

        # Generate time array
        hop_seconds = self.hop_length / self._target_sr
        times = np.arange(len(pitch_np)) * hop_seconds

        return PitchContour(
            times=times,
            frequencies=pitch_np,
            confidence=confidence_np,
            sample_rate=sample_rate,
        )


def extract_pitch(
    audio_path: Path,
    song_id: UUID | None = None,
    data_dir: Path = Path("data"),
    model: str = "full",
) -> PitchContour:
    """Convenience function to extract pitch from audio.

    If song_id is provided, saves results to data/analysis/{song_id}/.
    """
    extractor = PitchExtractor(model=model)
    contour = extractor.extract(audio_path)

    if song_id is not None:
        output_dir = data_dir / "analysis" / str(song_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pitch data as npz
        np.savez(
            output_dir / "pitch.npz",
            times=contour.times,
            frequencies=contour.frequencies,
            confidence=contour.confidence,
            sample_rate=contour.sample_rate,
        )

    return contour

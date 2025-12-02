"""Demucs wrapper for vocal/instrumental separation."""

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import torch
import torchaudio  # type: ignore[import-untyped]

from harmony_trainer.audio_io import load_audio, save_wav

if TYPE_CHECKING:
    from demucs.htdemucs import HTDemucs


class DemucsWrapper:
    """Wrapper for Demucs audio separation model."""

    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
        self._model: HTDemucs | None = None

    @property
    def model(self) -> "HTDemucs":
        if self._model is None:
            from demucs.pretrained import get_model

            self._model = get_model(self.model_name)
            self._model.eval()
        return self._model

    def separate(
        self,
        audio_path: Path,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Separate audio into vocals and instrumental.

        Returns paths to (vocals_all.wav, instrumental.wav).
        """
        from demucs.apply import apply_model

        model = self.model

        # Load audio
        waveform, sample_rate = load_audio(audio_path)

        # Resample if needed
        model_sr: int = model.samplerate
        if sample_rate != model_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, model_sr)
            waveform = resampler(waveform)

        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Add batch dimension
        waveform = waveform.unsqueeze(0)

        # Apply model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        waveform = waveform.to(device)

        with torch.no_grad():
            sources: torch.Tensor = apply_model(model, waveform, device=device)

        # Get source names and indices
        source_names: list[str] = model.sources
        vocals_idx = source_names.index("vocals")

        # Extract vocals and create instrumental (sum of non-vocal sources)
        vocals: torch.Tensor = sources[0, vocals_idx]
        instrumental: Any = sources[0, 0] * 0  # Start with zeros
        for i in range(len(source_names)):
            if i != vocals_idx:
                instrumental = instrumental + sources[0, i]

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save outputs
        vocals_path = output_dir / "vocals_all.wav"
        instrumental_path = output_dir / "instrumental.wav"

        save_wav(vocals_path, vocals.cpu(), model_sr)
        save_wav(instrumental_path, instrumental.cpu(), model_sr)

        return vocals_path, instrumental_path


def separate_audio(
    audio_path: Path,
    song_id: UUID,
    data_dir: Path = Path("data"),
    model_name: str = "htdemucs",
) -> tuple[Path, Path]:
    """Convenience function to separate audio for a song.

    Returns paths to (vocals_all.wav, instrumental.wav).
    """
    wrapper = DemucsWrapper(model_name=model_name)
    output_dir = data_dir / "stems" / str(song_id)
    return wrapper.separate(audio_path, output_dir)

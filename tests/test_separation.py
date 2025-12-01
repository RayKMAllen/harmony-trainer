"""Integration tests for audio separation."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch
import torchaudio

from harmony_trainer.separation.demucs import DemucsWrapper, separate_audio


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    """Create a short sample audio file for testing."""
    sample_rate = 44100
    duration = 2.0  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))

    # Create a simple stereo signal (sine waves at different frequencies)
    left = torch.sin(2 * np.pi * 440 * t)  # A4
    right = torch.sin(2 * np.pi * 554 * t)  # C#5
    waveform = torch.stack([left, right])

    audio_path = tmp_path / "test_audio.wav"
    torchaudio.save(str(audio_path), waveform, sample_rate)
    return audio_path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create output directory for stems."""
    out = tmp_path / "stems"
    out.mkdir()
    return out


class TestDemucsWrapper:
    def test_wrapper_initialization(self) -> None:
        wrapper = DemucsWrapper()
        assert wrapper.model_name == "htdemucs"
        assert wrapper._model is None

    def test_wrapper_custom_model(self) -> None:
        wrapper = DemucsWrapper(model_name="htdemucs_ft")
        assert wrapper.model_name == "htdemucs_ft"

    @pytest.mark.slow
    def test_separate_creates_output_files(self, sample_audio: Path, output_dir: Path) -> None:
        """Test that separation creates vocals and instrumental files."""
        wrapper = DemucsWrapper()
        vocals_path, instrumental_path = wrapper.separate(sample_audio, output_dir)

        assert vocals_path.exists()
        assert instrumental_path.exists()
        assert vocals_path.name == "vocals_all.wav"
        assert instrumental_path.name == "instrumental.wav"

    @pytest.mark.slow
    def test_separate_output_is_valid_audio(self, sample_audio: Path, output_dir: Path) -> None:
        """Test that output files are valid audio."""
        wrapper = DemucsWrapper()
        vocals_path, instrumental_path = wrapper.separate(sample_audio, output_dir)

        vocals, sr1 = torchaudio.load(str(vocals_path))
        instrumental, sr2 = torchaudio.load(str(instrumental_path))

        assert vocals.shape[0] == 2  # stereo
        assert instrumental.shape[0] == 2
        assert sr1 == sr2


class TestSeparateAudio:
    @pytest.mark.slow
    def test_separate_audio_convenience_function(self, sample_audio: Path, tmp_path: Path) -> None:
        """Test the convenience function creates files in correct location."""
        song_id = uuid4()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        vocals_path, instrumental_path = separate_audio(sample_audio, song_id, data_dir=data_dir)

        expected_dir = data_dir / "stems" / str(song_id)
        assert vocals_path.parent == expected_dir
        assert instrumental_path.parent == expected_dir
        assert vocals_path.exists()
        assert instrumental_path.exists()

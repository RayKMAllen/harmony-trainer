"""Integration tests for audio separation."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from harmony_trainer.audio_io import load_audio, save_wav
from harmony_trainer.separation.demucs import DemucsWrapper, separate_audio
from harmony_trainer.separation.svoice import (
    SeparationError,
    SVoiceWrapper,
    separate_vocals,
)


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
    save_wav(audio_path, waveform, sample_rate)
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

        vocals, sr1 = load_audio(vocals_path)
        instrumental, sr2 = load_audio(instrumental_path)

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


class TestSVoiceWrapper:
    def test_wrapper_initialization(self) -> None:
        wrapper = SVoiceWrapper()
        assert wrapper._checkpoint_path is None
        assert wrapper._model is None

    def test_wrapper_custom_checkpoint(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "model.th"
        wrapper = SVoiceWrapper(checkpoint_path=checkpoint)
        assert wrapper._checkpoint_path == checkpoint

    def test_separate_graceful_failure(self, sample_audio: Path, output_dir: Path) -> None:
        """Test that separation returns None on failure (model not available)."""
        wrapper = SVoiceWrapper()
        lead_path, backing_path = wrapper.separate(sample_audio, output_dir)

        # SVoice model may not be available, so we expect graceful failure
        # Either both are None (failure) or both exist (success)
        assert (lead_path is None and backing_path is None) or (
            lead_path is not None and backing_path is not None
        )

    @pytest.mark.slow
    def test_separate_creates_output_files_if_model_available(
        self, sample_audio: Path, output_dir: Path
    ) -> None:
        """Test that separation creates lead and backing files if model works."""
        wrapper = SVoiceWrapper()
        lead_path, backing_path = wrapper.separate(sample_audio, output_dir)

        if lead_path is not None and backing_path is not None:
            assert lead_path.exists()
            assert backing_path.exists()
            assert lead_path.name == "lead.wav"
            assert backing_path.name == "backing.wav"


class TestSeparateVocals:
    def test_separate_vocals_convenience_function(self, sample_audio: Path, tmp_path: Path) -> None:
        """Test the convenience function handles graceful failure."""
        song_id = uuid4()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        lead_path, backing_path = separate_vocals(sample_audio, song_id, data_dir=data_dir)

        # Either both are None (failure) or both exist in correct location
        if lead_path is not None and backing_path is not None:
            expected_dir = data_dir / "stems" / str(song_id)
            assert lead_path.parent == expected_dir
            assert backing_path.parent == expected_dir
            assert lead_path.exists()
            assert backing_path.exists()
        else:
            assert lead_path is None
            assert backing_path is None


class TestSeparationError:
    def test_separation_error_is_exception(self) -> None:
        """Test that SeparationError is a proper exception."""
        error = SeparationError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

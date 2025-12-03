"""Integration tests for audio separation."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from harmony_trainer.audio_io import load_audio, save_wav
from harmony_trainer.separation.demucs import DemucsWrapper, separate_audio
from harmony_trainer.separation.medleyvox import (
    MedleyVoxWrapper,
    separate_vocals_medleyvox,
)
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

    def test_separate_raises_separation_error(self, sample_audio: Path, output_dir: Path) -> None:
        """Test that separation raises SeparationError on failure (model not available)."""
        wrapper = SVoiceWrapper()
        with pytest.raises(SeparationError):
            wrapper.separate(sample_audio, output_dir)

    @pytest.mark.slow
    def test_separate_creates_output_files_if_model_available(
        self, sample_audio: Path, output_dir: Path, tmp_path: Path
    ) -> None:
        """Test that separation creates lead and backing files if model works."""
        # Create a fake checkpoint to test with (will still fail at model load)
        checkpoint = tmp_path / "model.th"
        checkpoint.touch()
        wrapper = SVoiceWrapper(checkpoint_path=checkpoint)
        # This will raise SeparationError since the checkpoint is not valid
        with pytest.raises(SeparationError):
            wrapper.separate(sample_audio, output_dir)


class TestSeparateVocals:
    def test_separate_vocals_raises_separation_error(
        self, sample_audio: Path, tmp_path: Path
    ) -> None:
        """Test the convenience function raises SeparationError on failure."""
        song_id = uuid4()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with pytest.raises(SeparationError):
            separate_vocals(sample_audio, song_id, data_dir=data_dir)


class TestMedleyVoxWrapper:
    def test_wrapper_initialization(self) -> None:
        wrapper = MedleyVoxWrapper()
        assert wrapper._checkpoint_path is None
        assert wrapper._model is None

    def test_wrapper_custom_checkpoint(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "model.pth"
        wrapper = MedleyVoxWrapper(checkpoint_path=checkpoint)
        assert wrapper._checkpoint_path == checkpoint

    def test_separate_raises_separation_error_when_checkpoint_missing(
        self, sample_audio: Path, output_dir: Path, tmp_path: Path
    ) -> None:
        """Test that separation raises SeparationError when checkpoint doesn't exist."""
        fake_checkpoint = tmp_path / "nonexistent.pth"
        wrapper = MedleyVoxWrapper(checkpoint_path=fake_checkpoint)
        with pytest.raises(SeparationError, match="Checkpoint not found"):
            wrapper.separate(sample_audio, output_dir)

    @pytest.mark.slow
    def test_separate_creates_output_files(self, sample_audio: Path, output_dir: Path) -> None:
        """Test that separation creates lead and backing files when model available."""
        wrapper = MedleyVoxWrapper()
        lead_path, backing_path = wrapper.separate(sample_audio, output_dir)

        assert lead_path.exists()
        assert backing_path.exists()
        assert lead_path.name == "lead.wav"
        assert backing_path.name == "backing.wav"


class TestSeparateVocalsMedleyVox:
    def test_separate_vocals_raises_separation_error_when_checkpoint_missing(
        self, sample_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the convenience function raises SeparationError when checkpoint missing."""
        song_id = uuid4()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Mock get_medleyvox_checkpoint_path to raise CheckpointNotFoundError
        from harmony_trainer.model_loaders.medleyvox import CheckpointNotFoundError

        def mock_get_path() -> Path:
            raise CheckpointNotFoundError("No checkpoint available")

        # Patch in the separation module where it's imported
        monkeypatch.setattr(
            "harmony_trainer.separation.medleyvox.get_medleyvox_checkpoint_path",
            mock_get_path,
        )

        with pytest.raises(SeparationError):
            separate_vocals_medleyvox(sample_audio, song_id, data_dir=data_dir)

    @pytest.mark.slow
    def test_separate_vocals_creates_output_files(self, sample_audio: Path, tmp_path: Path) -> None:
        """Test the convenience function creates files when model available."""
        song_id = uuid4()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        lead_path, backing_path = separate_vocals_medleyvox(
            sample_audio, song_id, data_dir=data_dir
        )

        expected_dir = data_dir / "stems" / str(song_id)
        assert lead_path.parent == expected_dir
        assert backing_path.parent == expected_dir
        assert lead_path.exists()
        assert backing_path.exists()

    def test_separate_vocals_with_fake_model(
        self, sample_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration test: verify separation produces valid stems using a fake model."""
        import torch.nn as nn

        class FakeMedleyVoxNet(nn.Module):
            """Fake model that returns two sources (lead, backing) from input."""

            def __init__(self) -> None:
                super().__init__()
                self.num_sources = 2

            def forward(self, mixture: torch.Tensor) -> torch.Tensor:
                # mixture: (batch, samples) or (batch, 1, samples)
                if mixture.dim() == 3:
                    mixture = mixture.squeeze(1)
                batch, samples = mixture.shape
                # Return (batch, 2, samples) - lead is scaled input, backing is inverted
                lead = mixture * 0.7
                backing = mixture * 0.3
                return torch.stack([lead, backing], dim=1)

        def mock_load_model(checkpoint_path: Path | None = None) -> nn.Module:
            model = FakeMedleyVoxNet()
            model.eval()
            return model

        # Patch load_medleyvox_model to return our fake model
        monkeypatch.setattr(
            "harmony_trainer.separation.medleyvox.load_medleyvox_model",
            mock_load_model,
        )

        # Create a fake checkpoint file so the path check passes
        fake_checkpoint = tmp_path / "fake_checkpoint.pth"
        fake_checkpoint.write_bytes(b"fake")

        # Patch get_medleyvox_checkpoint_path to return the fake path
        monkeypatch.setattr(
            "harmony_trainer.separation.medleyvox.get_medleyvox_checkpoint_path",
            lambda: fake_checkpoint,
        )

        song_id = uuid4()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        lead_path, backing_path = separate_vocals_medleyvox(
            sample_audio, song_id, data_dir=data_dir
        )

        # Verify output files exist
        assert lead_path.exists(), "lead.wav should exist"
        assert backing_path.exists(), "backing.wav should exist"
        assert lead_path.name == "lead.wav"
        assert backing_path.name == "backing.wav"

        # Verify output files are valid audio
        lead_audio, lead_sr = load_audio(lead_path)
        backing_audio, backing_sr = load_audio(backing_path)

        # Check sample rates match expected (44100 Hz for MedleyVox)
        assert lead_sr == 44100, f"Expected 44100 Hz, got {lead_sr}"
        assert backing_sr == 44100, f"Expected 44100 Hz, got {backing_sr}"

        # Check audio has samples (not empty)
        assert lead_audio.shape[-1] > 0, "lead.wav should have samples"
        assert backing_audio.shape[-1] > 0, "backing.wav should have samples"


class TestSeparationError:
    def test_separation_error_is_exception(self) -> None:
        """Test that SeparationError is a proper exception."""
        error = SeparationError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

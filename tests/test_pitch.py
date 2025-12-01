"""Integration tests for pitch extraction."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch
import torchaudio

from harmony_trainer.pitch.extractor import PitchContour, PitchExtractor, extract_pitch


@pytest.fixture
def sample_mono_audio(tmp_path: Path) -> Path:
    """Create a short mono audio file with known pitch for testing."""
    sample_rate = 16000
    duration = 1.0  # seconds
    frequency = 440.0  # A4
    t = torch.linspace(0, duration, int(sample_rate * duration))

    # Create a pure sine wave at 440 Hz
    waveform = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)

    audio_path = tmp_path / "test_mono.wav"
    torchaudio.save(str(audio_path), waveform, sample_rate)
    return audio_path


@pytest.fixture
def sample_stereo_audio(tmp_path: Path) -> Path:
    """Create a short stereo audio file for testing."""
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0
    t = torch.linspace(0, duration, int(sample_rate * duration))

    mono = torch.sin(2 * np.pi * frequency * t)
    waveform = torch.stack([mono, mono])

    audio_path = tmp_path / "test_stereo.wav"
    torchaudio.save(str(audio_path), waveform, sample_rate)
    return audio_path


class TestPitchExtractor:
    def test_extractor_initialization(self) -> None:
        extractor = PitchExtractor()
        assert extractor.model == "full"
        assert extractor.hop_length == 512
        assert extractor.fmin == 50.0
        assert extractor.fmax == 2000.0

    def test_extractor_custom_params(self) -> None:
        extractor = PitchExtractor(model="tiny", hop_length=256, fmin=100.0, fmax=1000.0)
        assert extractor.model == "tiny"
        assert extractor.hop_length == 256
        assert extractor.fmin == 100.0
        assert extractor.fmax == 1000.0

    @pytest.mark.slow
    def test_extract_returns_pitch_contour(self, sample_mono_audio: Path) -> None:
        """Test that extraction returns a valid PitchContour."""
        extractor = PitchExtractor(model="tiny")
        contour = extractor.extract(sample_mono_audio)

        assert isinstance(contour, PitchContour)
        assert len(contour.times) > 0
        assert len(contour.frequencies) == len(contour.times)
        assert len(contour.confidence) == len(contour.times)

    @pytest.mark.slow
    def test_extract_detects_known_pitch(self, sample_mono_audio: Path) -> None:
        """Test that extraction detects the known 440 Hz pitch."""
        extractor = PitchExtractor(model="tiny")
        contour = extractor.extract(sample_mono_audio)

        # Filter by confidence and check median pitch is close to 440 Hz
        confident_mask = contour.confidence > 0.5
        if confident_mask.any():
            median_pitch = np.median(contour.frequencies[confident_mask])
            # Allow 10% tolerance
            assert 396 < median_pitch < 484, f"Expected ~440 Hz, got {median_pitch}"

    @pytest.mark.slow
    def test_extract_handles_stereo(self, sample_stereo_audio: Path) -> None:
        """Test that extraction handles stereo audio (converts to mono)."""
        extractor = PitchExtractor(model="tiny")
        contour = extractor.extract(sample_stereo_audio)

        assert isinstance(contour, PitchContour)
        assert len(contour.times) > 0


class TestExtractPitch:
    @pytest.mark.slow
    def test_extract_pitch_convenience_function(self, sample_mono_audio: Path) -> None:
        """Test the convenience function returns valid contour."""
        contour = extract_pitch(sample_mono_audio, model="tiny")

        assert isinstance(contour, PitchContour)
        assert len(contour.times) > 0

    @pytest.mark.slow
    def test_extract_pitch_saves_to_analysis_dir(
        self, sample_mono_audio: Path, tmp_path: Path
    ) -> None:
        """Test that pitch data is saved when song_id is provided."""
        song_id = uuid4()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        contour = extract_pitch(sample_mono_audio, song_id=song_id, data_dir=data_dir, model="tiny")

        expected_file = data_dir / "analysis" / str(song_id) / "pitch.npz"
        assert expected_file.exists()

        # Verify saved data matches returned contour
        loaded = np.load(expected_file)
        np.testing.assert_array_equal(loaded["times"], contour.times)
        np.testing.assert_array_equal(loaded["frequencies"], contour.frequencies)
        np.testing.assert_array_equal(loaded["confidence"], contour.confidence)

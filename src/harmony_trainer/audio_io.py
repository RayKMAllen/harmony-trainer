"""Audio I/O utilities for WAV and MP3 file operations."""

from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.io import wavfile  # type: ignore

SUPPORTED_EXTENSIONS = {".wav", ".mp3"}


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio file (WAV or MP3) and return tensor and sample rate.

    Returns:
        Tuple of (waveform tensor of shape (channels, samples), sample_rate)
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".mp3":
        return _load_mp3(path)
    return _load_wav(path)


def _load_mp3(path: Path) -> tuple[torch.Tensor, int]:
    """Load MP3 file using librosa."""
    audio_np, sample_rate = librosa.load(str(path), sr=None, mono=False)
    waveform = torch.from_numpy(audio_np)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, int(sample_rate)


def _load_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Load WAV file using scipy."""
    sample_rate, audio_np = wavfile.read(str(path))

    if audio_np.dtype == np.int16:
        audio_float = audio_np.astype(np.float32) / 32767.0
    elif audio_np.dtype == np.int32:
        audio_float = audio_np.astype(np.float32) / 2147483647.0
    elif audio_np.dtype == np.float32:
        audio_float = audio_np
    else:
        audio_float = audio_np.astype(np.float32)

    waveform = torch.from_numpy(audio_float)
    waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform.T

    return waveform, sample_rate


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Load WAV file. Deprecated: use load_audio() instead."""
    return _load_wav(path)


def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save audio tensor to WAV file using scipy.

    Args:
        path: Output file path
        waveform: Audio tensor of shape (channels, samples) with values in [-1, 1]
        sample_rate: Sample rate in Hz
    """
    # Convert to numpy and transpose to (samples, channels)
    audio_np = waveform.numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # (channels, samples) -> (samples, channels)

    # Convert float [-1, 1] to int16
    audio_int16 = (audio_np * 32767).astype(np.int16)

    wavfile.write(str(path), sample_rate, audio_int16)

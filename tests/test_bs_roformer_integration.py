"""Integration tests for BS-RoFormer with real checkpoint.

These tests require the anvuew/BS-RoFormer checkpoint to be available.
They are marked as slow and will be skipped in CI without weights.
"""

import pytest
import torch

from harmony_trainer.model_loaders.bs_roformer import (
    CheckpointNotFoundError,
    get_bs_roformer_checkpoint_path,
    load_bs_roformer_model,
)


def checkpoint_available() -> bool:
    """Check if BS-RoFormer checkpoint is available."""
    try:
        get_bs_roformer_checkpoint_path()
        return True
    except CheckpointNotFoundError:
        return False


requires_checkpoint = pytest.mark.skipif(
    not checkpoint_available(),
    reason="BS-RoFormer checkpoint not available",
)


@pytest.mark.slow
@requires_checkpoint
class TestBSRoformerIntegration:
    """Integration tests with real BS-RoFormer checkpoint."""

    def test_load_real_checkpoint(self) -> None:
        """Model loads successfully from real anvuew checkpoint."""
        model = load_bs_roformer_model()
        assert model is not None
        assert not model.training  # Should be in eval mode

    def test_forward_pass_real_model(self) -> None:
        """Forward pass with real model produces valid output."""
        model = load_bs_roformer_model()
        # 1 second of stereo audio at 44100 Hz
        x = torch.randn(1, 2, 44100)
        with torch.no_grad():
            y = model(x)
        # Output shape: (batch, num_stems, channels, samples)
        # anvuew checkpoint uses num_stems=2
        assert y.shape == (1, 2, 2, 44100)

    def test_output_not_nan(self) -> None:
        """Model output contains no NaN values."""
        model = load_bs_roformer_model()
        x = torch.randn(1, 2, 44100)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any(), "Output contains NaN values"

    def test_output_not_inf(self) -> None:
        """Model output contains no infinite values."""
        model = load_bs_roformer_model()
        x = torch.randn(1, 2, 44100)
        with torch.no_grad():
            y = model(x)
        assert not torch.isinf(y).any(), "Output contains infinite values"

    def test_output_reasonable_magnitude(self) -> None:
        """Model output has reasonable magnitude (not exploding)."""
        model = load_bs_roformer_model()
        # Use normalized input
        x = torch.randn(1, 2, 44100) * 0.5
        with torch.no_grad():
            y = model(x)
        # Output should be within reasonable range for audio
        assert y.abs().max() < 100, f"Output magnitude too large: {y.abs().max()}"

    def test_longer_audio(self) -> None:
        """Model handles longer audio (3 seconds)."""
        model = load_bs_roformer_model()
        x = torch.randn(1, 2, 44100 * 3)
        with torch.no_grad():
            y = model(x)
        # anvuew checkpoint uses num_stems=2
        assert y.shape == (1, 2, 2, 44100 * 3)
        assert not torch.isnan(y).any()

    def test_batch_processing(self) -> None:
        """Model handles batch size > 1."""
        model = load_bs_roformer_model()
        x = torch.randn(2, 2, 44100)
        with torch.no_grad():
            y = model(x)
        # anvuew checkpoint uses num_stems=2
        assert y.shape == (2, 2, 2, 44100)
        assert not torch.isnan(y).any()

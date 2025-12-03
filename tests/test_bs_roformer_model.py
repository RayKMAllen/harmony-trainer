"""Smoke tests for BS-RoFormer model architecture."""

import torch

from harmony_trainer.models.bs_roformer import BSRoformerNet


class TestBSRoformerNet:
    """Tests for BSRoformerNet model."""

    def test_model_construction(self) -> None:
        """Model can be constructed with default hyperparameters."""
        model = BSRoformerNet()
        assert isinstance(model, torch.nn.Module)
        assert model.num_stems == 1
        assert model.n_fft == 2048
        assert model.hop_length == 512
        assert model.sample_rate == 44100
        assert model.stereo is True

    def test_forward_pass_stereo(self) -> None:
        """Forward pass works with stereo (batch, 2, samples) input."""
        model = BSRoformerNet()
        model.eval()
        # Use short audio for speed (1 second at 44100 Hz)
        x = torch.randn(1, 2, 44100)
        with torch.no_grad():
            y = model(x)
        # Output: (batch, stems, channels, samples)
        # Model returns 2 stems (Target + Residual)
        assert y.shape == (1, 2, 2, 44100)

    def test_forward_pass_mono(self) -> None:
        """Forward pass works with mono (batch, 1, samples) input."""
        model = BSRoformerNet(stereo=False)
        model.eval()
        x = torch.randn(1, 1, 44100)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 1, 44100)

    def test_forward_pass_batch(self) -> None:
        """Forward pass works with batch size > 1."""
        model = BSRoformerNet()
        model.eval()
        x = torch.randn(2, 2, 44100)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 2, 2, 44100)

    def test_model_imports_without_gpu(self) -> None:
        """Model can be imported and used on CPU."""
        model = BSRoformerNet()
        assert next(model.parameters()).device.type == "cpu"

    def test_freqs_per_band_sum(self) -> None:
        """Default freqs_per_band sums to n_fft // 2 + 1."""
        model = BSRoformerNet()
        expected_freqs = model.n_fft // 2 + 1
        assert sum(model.freqs_per_band) == expected_freqs

"""Smoke tests for MedleyVox model architecture."""

import torch

from harmony_trainer.models.medleyvox import (
    MedleyVoxNet,
    extract_online_model_state_dict,
)


class TestMedleyVoxNet:
    """Tests for MedleyVoxNet model."""

    def test_model_construction(self) -> None:
        """Model can be constructed with default hyperparameters."""
        model = MedleyVoxNet()
        assert isinstance(model, torch.nn.Module)
        assert model.num_sources == 2
        assert model.n_filters == 2050
        assert model.kernel_size == 2048

    def test_forward_pass_2d_input(self) -> None:
        """Forward pass works with (batch, samples) input."""
        model = MedleyVoxNet()
        model.eval()
        x = torch.randn(1, 16000)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 16000)

    def test_forward_pass_3d_input(self) -> None:
        """Forward pass works with (batch, 1, samples) input."""
        model = MedleyVoxNet()
        model.eval()
        x = torch.randn(1, 1, 16000)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 16000)

    def test_forward_pass_batch(self) -> None:
        """Forward pass works with batch size > 1."""
        model = MedleyVoxNet()
        model.eval()
        x = torch.randn(2, 16000)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 2, 16000)

    def test_model_imports_without_gpu(self) -> None:
        """Model can be imported and used on CPU."""
        model = MedleyVoxNet()
        assert next(model.parameters()).device.type == "cpu"


class TestExtractOnlineModelStateDict:
    """Tests for checkpoint extraction utility."""

    def test_extracts_online_model_keys(self) -> None:
        """Extracts keys with online_model.module. prefix."""
        checkpoint = {
            "online_model.module.encoder.weight": torch.randn(10),
            "online_model.module.decoder.weight": torch.randn(10),
            "ema_model.module.encoder.weight": torch.randn(10),
            "step": 100,
        }
        sd = extract_online_model_state_dict(checkpoint)
        assert "encoder.weight" in sd
        assert "decoder.weight" in sd
        assert "ema_model.module.encoder.weight" not in sd
        assert "step" not in sd

    def test_empty_checkpoint(self) -> None:
        """Returns empty dict for checkpoint without online_model keys."""
        checkpoint = {"step": 100, "other_key": "value"}
        sd = extract_online_model_state_dict(checkpoint)
        assert sd == {}

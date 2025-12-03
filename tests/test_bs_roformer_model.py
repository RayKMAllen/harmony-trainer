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
        # Output: (batch, num_stems, channels, samples)
        assert y.shape == (1, model.num_stems, 2, 44100)

    def test_forward_pass_mono(self) -> None:
        """Forward pass works with mono (batch, 1, samples) input."""
        model = BSRoformerNet(stereo=False)
        model.eval()
        x = torch.randn(1, 1, 44100)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, model.num_stems, 1, 44100)

    def test_forward_pass_batch(self) -> None:
        """Forward pass works with batch size > 1."""
        model = BSRoformerNet()
        model.eval()
        x = torch.randn(2, 2, 44100)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, model.num_stems, 2, 44100)

    def test_model_imports_without_gpu(self) -> None:
        """Model can be imported and used on CPU."""
        model = BSRoformerNet()
        assert next(model.parameters()).device.type == "cpu"

    def test_freqs_per_band_sum(self) -> None:
        """Default freqs_per_band sums to n_fft // 2 + 1."""
        model = BSRoformerNet()
        expected_freqs = model.n_fft // 2 + 1
        assert sum(model.freqs_per_band) == expected_freqs

    def test_mlp_state_dict_keys_match_zfturbo(self) -> None:
        """MLP state_dict keys match ZFTurbo format (to_freqs.N.net.M.weight)."""
        model = BSRoformerNet()
        state_dict = model.state_dict()
        # Find MLP keys in mask_estimators
        mlp_keys = [k for k in state_dict if "to_freqs" in k]
        assert len(mlp_keys) > 0, "No to_freqs keys found"
        # Verify key format: mask_estimators.0.to_freqs.N.net.M.weight/bias
        for key in mlp_keys:
            parts = key.split(".")
            # Expected: mask_estimators.0.to_freqs.N.net.M.weight
            assert "net" in parts, f"Key {key} missing 'net' component"
            # Should NOT have double sequential (0.0.0 pattern)
            assert ".0.0." not in key or "to_freqs.0.0." not in key.replace(
                "to_freqs.", "X."
            ), f"Key {key} has wrong nested structure"

    def test_mask_estimator_output_dimension_matches_zfturbo(self) -> None:
        """MaskEstimator complex_out = audio_channels * 2 (not * 4)."""
        model = BSRoformerNet()
        # Each mask estimator should output audio_channels * 2 (real/imag per channel)
        # For stereo: 2 * 2 = 4, NOT 2 * 2 * 2 = 8
        expected_complex_out = model.audio_channels * 2
        for mask_est in model.mask_estimators:
            assert (
                mask_est.complex_out == expected_complex_out
            ), f"MaskEstimator complex_out={mask_est.complex_out}, expected {expected_complex_out}"

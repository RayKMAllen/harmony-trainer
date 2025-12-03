"""Tests for model weight loader subsystem."""

import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from harmony_trainer.model_loaders.medleyvox import (
    ENV_VAR,
    CheckpointNotFoundError,
    get_medleyvox_checkpoint_path,
    load_medleyvox_model,
)


class TestGetMedleyvoxCheckpointPath:
    """Tests for get_medleyvox_checkpoint_path fallback hierarchy."""

    def test_env_var_override_valid_path(self, tmp_path: Path) -> None:
        """Environment variable takes precedence when path exists."""
        checkpoint = tmp_path / "custom_checkpoint.pth"
        checkpoint.write_bytes(b"fake checkpoint")

        with patch.dict(os.environ, {ENV_VAR: str(checkpoint)}):
            result = get_medleyvox_checkpoint_path()
            assert result == checkpoint

    def test_env_var_override_invalid_path(self) -> None:
        """Raises error when env var path doesn't exist."""
        with (
            patch.dict(os.environ, {ENV_VAR: "/nonexistent/path.pth"}),
            pytest.raises(CheckpointNotFoundError, match="does not exist"),
        ):
            get_medleyvox_checkpoint_path()

    def test_local_path_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to local project path when env var not set."""
        # Create a fake local checkpoint
        local_checkpoint = tmp_path / "models" / "medleyvox" / "vocals_100.pth"
        local_checkpoint.parent.mkdir(parents=True)
        local_checkpoint.write_bytes(b"fake checkpoint")

        # Patch LOCAL_PATH to point to our temp location
        monkeypatch.setattr(
            "harmony_trainer.model_loaders.medleyvox.LOCAL_PATH",
            local_checkpoint,
        )

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_VAR, None)
            result = get_medleyvox_checkpoint_path()
            assert result == local_checkpoint

    def test_hf_hub_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to hf_hub_download when local paths unavailable."""
        # Ensure env var not set and local path doesn't exist
        monkeypatch.setattr(
            "harmony_trainer.model_loaders.medleyvox.LOCAL_PATH",
            Path("/nonexistent/local/path.pth"),
        )

        mock_download = MagicMock(return_value="/cached/path/vocals 100.pth")

        # Create a fake huggingface_hub module
        fake_hf_hub = ModuleType("huggingface_hub")
        fake_hf_hub.hf_hub_download = mock_download  # type: ignore[attr-defined]

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_VAR, None)
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf_hub}):
                result = get_medleyvox_checkpoint_path()
                assert result == Path("/cached/path/vocals 100.pth")
                mock_download.assert_called_once_with(
                    repo_id="Cyru5/MedleyVox",
                    filename="vocals 100.pth",
                    revision="main",
                )

    def test_hf_hub_download_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises CheckpointNotFoundError when HF download fails."""
        monkeypatch.setattr(
            "harmony_trainer.model_loaders.medleyvox.LOCAL_PATH",
            Path("/nonexistent/local/path.pth"),
        )

        mock_download = MagicMock(side_effect=Exception("Network error"))

        # Create a fake huggingface_hub module
        fake_hf_hub = ModuleType("huggingface_hub")
        fake_hf_hub.hf_hub_download = mock_download  # type: ignore[attr-defined]

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_VAR, None)
            with (
                patch.dict(sys.modules, {"huggingface_hub": fake_hf_hub}),
                pytest.raises(CheckpointNotFoundError, match="Failed to download"),
            ):
                get_medleyvox_checkpoint_path()


class TestLoadMedleyvoxModel:
    """Tests for load_medleyvox_model function."""

    def test_load_with_online_model_format(self, tmp_path: Path) -> None:
        """Loads model from checkpoint with online_model.module. prefix."""
        import torch

        from harmony_trainer.models.medleyvox import MedleyVoxNet

        # Create a valid checkpoint with the expected format
        model = MedleyVoxNet()
        state_dict = {f"online_model.module.{k}": v for k, v in model.state_dict().items()}
        checkpoint = tmp_path / "checkpoint.pth"
        torch.save(state_dict, checkpoint)

        result = load_medleyvox_model(checkpoint)
        assert isinstance(result, MedleyVoxNet)

    def test_load_with_model_key(self, tmp_path: Path) -> None:
        """Loads model from checkpoint with 'model' key."""
        import torch

        from harmony_trainer.models.medleyvox import MedleyVoxNet

        model = MedleyVoxNet()
        checkpoint = tmp_path / "checkpoint.pth"
        torch.save({"model": model.state_dict()}, checkpoint)

        result = load_medleyvox_model(checkpoint)
        assert isinstance(result, MedleyVoxNet)

    def test_load_with_state_dict_key(self, tmp_path: Path) -> None:
        """Loads model from checkpoint with 'state_dict' key."""
        import torch

        from harmony_trainer.models.medleyvox import MedleyVoxNet

        model = MedleyVoxNet()
        checkpoint = tmp_path / "checkpoint.pth"
        torch.save({"state_dict": model.state_dict()}, checkpoint)

        result = load_medleyvox_model(checkpoint)
        assert isinstance(result, MedleyVoxNet)

    def test_load_raw_state_dict(self, tmp_path: Path) -> None:
        """Loads model from raw state_dict without wrapper."""
        import torch

        from harmony_trainer.models.medleyvox import MedleyVoxNet

        model = MedleyVoxNet()
        checkpoint = tmp_path / "checkpoint.pth"
        torch.save(model.state_dict(), checkpoint)

        result = load_medleyvox_model(checkpoint)
        assert isinstance(result, MedleyVoxNet)

    def test_load_nonexistent_path(self) -> None:
        """Raises error for nonexistent checkpoint."""
        with pytest.raises(CheckpointNotFoundError):
            load_medleyvox_model(Path("/nonexistent/checkpoint.pth"))

    def test_load_uses_fallback_when_no_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uses get_medleyvox_checkpoint_path when no path provided."""
        import torch

        from harmony_trainer.models.medleyvox import MedleyVoxNet

        model = MedleyVoxNet()
        checkpoint = tmp_path / "fallback.pth"
        torch.save(model.state_dict(), checkpoint)

        monkeypatch.setattr(
            "harmony_trainer.model_loaders.medleyvox.get_medleyvox_checkpoint_path",
            lambda: checkpoint,
        )

        result = load_medleyvox_model()
        assert isinstance(result, MedleyVoxNet)

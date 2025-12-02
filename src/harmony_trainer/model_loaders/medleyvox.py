"""MedleyVox model weight loader per ADR-0003."""

import os
from pathlib import Path
from typing import Any

import torch

# Canonical source configuration
HF_REPO_ID = "Cyru5/MedleyVox"
HF_FILENAME = "vocals 100.pth"
HF_REVISION = "main"
ENV_VAR = "MEDLEYVOX_WEIGHTS_PATH"
LOCAL_PATH = Path("models/medleyvox/vocals_100.pth")


class CheckpointNotFoundError(Exception):
    """Raised when MedleyVox checkpoint cannot be located."""


def get_medleyvox_checkpoint_path() -> Path:
    """Get MedleyVox checkpoint path using fallback hierarchy.

    Fallback order:
    1. MEDLEYVOX_WEIGHTS_PATH environment variable
    2. Local models/medleyvox/vocals_100.pth
    3. hf_hub_download() from Hugging Face

    Returns:
        Path to the checkpoint file.

    Raises:
        CheckpointNotFoundError: If checkpoint cannot be found or downloaded.
    """
    # 1. Environment variable override
    env_path = os.environ.get(ENV_VAR)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise CheckpointNotFoundError(f"Path from {ENV_VAR} does not exist: {path}")

    # 2. Local project path
    if LOCAL_PATH.exists():
        return LOCAL_PATH

    # 3. Hugging Face Hub download
    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            revision=HF_REVISION,
        )
        return Path(downloaded_path)
    except Exception as e:
        raise CheckpointNotFoundError(
            f"Failed to download MedleyVox checkpoint from {HF_REPO_ID}: {e}"
        ) from e


def load_medleyvox_model(checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Load MedleyVox model state dict from checkpoint.

    Args:
        checkpoint_path: Optional explicit path. If None, uses get_medleyvox_checkpoint_path().

    Returns:
        Model state dict.

    Raises:
        CheckpointNotFoundError: If checkpoint cannot be found.
        RuntimeError: If checkpoint loading fails.
    """
    if checkpoint_path is None:
        checkpoint_path = get_medleyvox_checkpoint_path()

    if not checkpoint_path.exists():
        raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        pkg = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    # Extract state_dict from various checkpoint formats
    if isinstance(pkg, dict):
        if "model" in pkg:
            return pkg["model"]  # type: ignore[no-any-return]
        if "state_dict" in pkg:
            return pkg["state_dict"]  # type: ignore[no-any-return]
        return pkg
    return {"model": pkg}

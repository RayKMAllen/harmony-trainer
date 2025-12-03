"""BS-RoFormer model weight loader per ADR-0006."""

import os
import re
from pathlib import Path

import torch

from harmony_trainer.models.bs_roformer import BSRoformerNet

# Canonical source configuration
HF_REPO_ID = "anvuew/BS-RoFormer"
HF_FILENAME = "bs_roformer_anvuew_sdr_12.45.ckpt"
HF_REVISION = "main"
ENV_VAR = "BS_ROFORMER_WEIGHTS_PATH"
LOCAL_PATH = Path("models/bs_roformer/bs_roformer_anvuew_sdr_12.45.ckpt")


class CheckpointNotFoundError(Exception):
    """Raised when BS-RoFormer checkpoint cannot be located."""


def get_bs_roformer_checkpoint_path() -> Path:
    """Get BS-RoFormer checkpoint path using fallback hierarchy.

    Fallback order:
    1. BS_ROFORMER_WEIGHTS_PATH environment variable
    2. Local models/bs_roformer/bs_roformer_anvuew_sdr_12.45.ckpt
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
            f"Failed to download BS-RoFormer checkpoint from {HF_REPO_ID}: {e}"
        ) from e


def _remap_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap checkpoint keys from anvuew format to our model format.

    anvuew checkpoint uses: mask_estimators.0.to_freqs.N.0.M.weight
    Our model expects:      mask_estimators.0.to_freqs.N.net.M.weight
    """
    remapped = {}
    # Pattern: to_freqs.N.0.M -> to_freqs.N.net.M
    pattern = re.compile(r"(to_freqs\.\d+)\.0\.(\d+)")
    for key, value in state_dict.items():
        new_key = pattern.sub(r"\1.net.\2", key)
        remapped[new_key] = value
    return remapped


def load_bs_roformer_model(checkpoint_path: Path | None = None) -> BSRoformerNet:
    """Load BS-RoFormer model from checkpoint.

    Args:
        checkpoint_path: Optional explicit path. If None, uses get_bs_roformer_checkpoint_path().

    Returns:
        Instantiated BSRoformerNet model with loaded weights in eval mode.

    Raises:
        CheckpointNotFoundError: If checkpoint cannot be found.
        RuntimeError: If checkpoint loading fails.
    """
    if checkpoint_path is None:
        checkpoint_path = get_bs_roformer_checkpoint_path()

    if not checkpoint_path.exists():
        raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        pkg = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    # Extract state_dict from various checkpoint formats
    if isinstance(pkg, dict):
        if "state_dict" in pkg:
            state_dict = pkg["state_dict"]
        elif "model" in pkg:
            state_dict = pkg["model"]
        else:
            state_dict = pkg
    else:
        raise RuntimeError("Unexpected checkpoint format")

    # Remap keys from anvuew checkpoint format to our model format
    # anvuew: mask_estimators.0.to_freqs.N.0.M.weight -> ours: mask_estimators.0.to_freqs.N.net.M.weight
    state_dict = _remap_checkpoint_keys(state_dict)

    # Instantiate model and load weights
    model = BSRoformerNet()
    model.load_state_dict(state_dict)
    model.eval()
    return model

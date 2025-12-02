"""MedleyVox model architecture for lead/backing vocal separation.

Vendored from the MedleyVox project (Cyru5/MedleyVox on HuggingFace).
This is a Conv-TasNet style architecture with STFT-based encoder/decoder.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional


class GlobalLayerNorm(nn.Module):
    """Global layer normalization with learnable gamma/beta (1D params)."""

    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        gamma = self.gamma.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        return gamma * (x - mean) / (var + self.eps).sqrt() + beta  # noqa: RUF100


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block for TCN."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        # Checkpoint uses single-element PReLU (num_parameters=1)
        self.shared_block = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.PReLU(num_parameters=1),
            GlobalLayerNorm(hidden_channels),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilation=dilation,
                groups=hidden_channels,
            ),
            nn.PReLU(num_parameters=1),
            GlobalLayerNorm(hidden_channels),
        )
        self.res_conv = nn.Conv1d(hidden_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(hidden_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared_block(x)
        res = self.res_conv(h)
        skip = self.skip_conv(h)
        return x + res, skip


class TCNMasker(nn.Module):
    """Temporal Convolutional Network masker."""

    def __init__(
        self,
        in_channels: int,
        bn_channels: int,
        hidden_channels: int,
        num_sources: int,
        num_blocks: int = 24,
        num_repeats: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.bottleneck = nn.Sequential(
            GlobalLayerNorm(in_channels),
            nn.Conv1d(in_channels, bn_channels, 1),
        )
        self.TCN = nn.ModuleList()
        for _ in range(num_repeats):
            for b in range(num_blocks // num_repeats):
                dilation = 2**b
                self.TCN.append(
                    DepthwiseSeparableConv(bn_channels, hidden_channels, kernel_size, dilation)
                )
        # Checkpoint uses single-element PReLU
        self.mask_net = nn.Sequential(
            nn.PReLU(num_parameters=1),
            nn.Conv1d(bn_channels, in_channels * num_sources, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time = x.shape
        h = self.bottleneck(x)
        skip_sum = torch.zeros_like(h)
        for block in self.TCN:
            h, skip = block(h)
            skip_sum = skip_sum + skip
        masks = self.mask_net(skip_sum)
        masks = masks.view(batch, self.num_sources, channels, time)
        return functional.relu(masks)


class Filterbank(nn.Module):
    """Learnable filterbank for encoder/decoder."""

    def __init__(self, n_filters: int, kernel_size: int):
        super().__init__()
        self._filters = nn.Parameter(torch.randn(n_filters, 1, kernel_size))
        # Checkpoint stores window as 1D tensor
        self.register_buffer("torch_window", torch.hann_window(kernel_size))


class STFTEncoder(nn.Module):
    """STFT-based encoder using learnable filterbank."""

    def __init__(self, n_filters: int, kernel_size: int):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = kernel_size // 2
        self.filterbank = Filterbank(n_filters, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return functional.conv1d(x, self.filterbank._filters, stride=self.stride)


class STFTDecoder(nn.Module):
    """STFT-based decoder using learnable filterbank."""

    def __init__(self, n_filters: int, kernel_size: int):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = kernel_size // 2
        self.filterbank = Filterbank(n_filters, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.conv_transpose1d(x, self.filterbank._filters, stride=self.stride)


class MedleyVoxNet(nn.Module):
    """MedleyVox network for lead/backing vocal separation.

    Architecture: Conv-TasNet with STFT-based encoder/decoder.
    Default hyperparameters match the vocals 100.pth checkpoint.
    """

    # Default hyperparameters for vocals 100.pth
    DEFAULT_N_FILTERS = 2050
    DEFAULT_KERNEL_SIZE = 2048
    DEFAULT_BN_CHANNELS = 256
    DEFAULT_HIDDEN_CHANNELS = 1024
    DEFAULT_NUM_SOURCES = 2
    DEFAULT_NUM_BLOCKS = 24
    DEFAULT_NUM_REPEATS = 3
    DEFAULT_TCN_KERNEL = 3

    def __init__(
        self,
        n_filters: int = DEFAULT_N_FILTERS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        bn_channels: int = DEFAULT_BN_CHANNELS,
        hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
        num_sources: int = DEFAULT_NUM_SOURCES,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        num_repeats: int = DEFAULT_NUM_REPEATS,
        tcn_kernel: int = DEFAULT_TCN_KERNEL,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.num_sources = num_sources
        self.encoder = STFTEncoder(n_filters, kernel_size)
        self.masker = TCNMasker(
            n_filters,
            bn_channels,
            hidden_channels,
            num_sources,
            num_blocks,
            num_repeats,
            tcn_kernel,
        )
        self.decoder = STFTDecoder(n_filters, kernel_size)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate mixture into sources.

        Args:
            mixture: (batch, samples) or (batch, 1, samples) mono audio

        Returns:
            (batch, num_sources, samples) separated sources
        """
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)
        original_length = mixture.shape[-1]
        encoded = self.encoder(mixture)
        masks = self.masker(encoded)
        masked = encoded.unsqueeze(1) * masks
        batch, sources, channels, time = masked.shape
        decoded: torch.Tensor = self.decoder(masked.view(batch * sources, channels, time))
        decoded = decoded.view(batch, sources, -1)
        if decoded.shape[-1] > original_length:
            decoded = decoded[..., :original_length]
        elif decoded.shape[-1] < original_length:
            decoded = functional.pad(decoded, (0, original_length - decoded.shape[-1]))
        return decoded


def extract_online_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize state dict from MedleyVox checkpoint.

    The checkpoint contains 'online_model.module.' prefixed keys.
    This function strips the prefix to match MedleyVoxNet structure.
    """
    prefix = "online_model.module."
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            state_dict[new_key] = value
    return state_dict

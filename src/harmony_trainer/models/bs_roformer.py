"""BS-RoFormer model architecture for music source separation.

Vendored from ZFTurbo/Music-Source-Separation-Training (MIT License).
https://github.com/ZFTurbo/Music-Source-Separation-Training

MIT License etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding  # type: ignore[import-untyped]

if TYPE_CHECKING:
    pass


def exists(val: object) -> bool:
    return val is not None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = functional.normalize(x, dim=-1) * self.scale * self.gamma
        return result


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


class Attention(nn.Module):
    """Multi-head attention with rotary embeddings."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotary_embed: Any = None,
        flash: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.rotary_embed = rotary_embed
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout))

        self.flash = flash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        if self.flash:
            out = functional.scaled_dot_product_attention(q, k, v)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        # Gate heads
        gates = self.to_gates(x)
        gates = rearrange(gates, "b n h -> b h n 1")
        gates = gates.sigmoid()
        out = out * gates

        out = rearrange(out, "b h n d -> b n (h d)")
        result: torch.Tensor = self.to_out(out)
        return result


class Transformer(nn.Module):
    """Transformer with rotary embeddings."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        rotary_embed: Any = None,
        flash_attn: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            flash=flash_attn,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            attn_layer: Attention = layer[0]  # type: ignore[index]
            ff_layer: FeedForward = layer[1]  # type: ignore[index]
            x = attn_layer(x) + x
            x = ff_layer(x) + x
        return x


class BandSplit(nn.Module):
    """Split frequency bands for band-split processing."""

    def __init__(self, dim: int, dim_inputs: tuple[int, ...], complex_as_channel: int = 2):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.complex_as_channel = complex_as_channel
        # Input features: freq_bins * complex_as_channel (real/imag per channel)
        self.to_features = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(dim_in * complex_as_channel),
                    nn.Linear(dim_in * complex_as_channel, dim),
                )
                for dim_in in dim_inputs
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, freq, time, complex_as_channel) - real/imag stacked per channel
        splits = x.split(list(self.dim_inputs), dim=1)  # type: ignore[no-untyped-call]
        outs = []
        for split_tensor, to_feat in zip(splits, self.to_features, strict=False):
            reshaped = rearrange(split_tensor, "b f t c -> b t (f c)")
            outs.append(to_feat(reshaped))
        return torch.stack(outs, dim=-2)


class MLP(nn.Module):
    """Simple MLP for mask estimation.

    Uses self.net to match ZFTurbo checkpoint keys (to_freqs.N.net.0.weight).
    """

    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int | None = None):
        super().__init__()
        hidden = dim_hidden or dim_in
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


class MaskEstimator(nn.Module):
    """Estimate complex masks for each band."""

    def __init__(
        self, dim: int, dim_outputs: tuple[int, ...], mlp_expansion: int = 4, complex_out: int = 2
    ):
        super().__init__()
        self.dim_outputs = dim_outputs
        self.complex_out = complex_out
        self.to_freqs = nn.ModuleList(
            [MLP(dim, dim_out * complex_out, dim * mlp_expansion) for dim_out in dim_outputs]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, bands, dim)
        bands = x.unbind(dim=-2)
        outs = []
        for band_feat, to_mask in zip(bands, self.to_freqs, strict=False):
            mask = to_mask(band_feat)
            outs.append(mask)
        # Stack and reshape to (batch, freq, time, complex_out)
        masks_list = [rearrange(m, "b t (f c) -> b f t c", c=self.complex_out) for m in outs]
        return torch.cat(masks_list, dim=1)


class BSRoformerNet(nn.Module):
    """Band-Split RoFormer for music source separation.

    Default hyperparameters match the anvuew/BS-RoFormer checkpoint config.yaml:
    - dim: 384
    - depth: 12
    - stereo: True
    - num_stems: 1
    - time_transformer_depth: 1
    - freq_transformer_depth: 1
    - n_fft: 2048
    - hop_length: 512
    - win_length: 2048
    - sample_rate: 44100
    """

    # Hardcoded config from anvuew/BS-RoFormer config.yaml
    DEFAULT_DIM = 256
    DEFAULT_DEPTH = 12
    DEFAULT_STEREO = True
    DEFAULT_NUM_STEMS = 1
    DEFAULT_TIME_TRANSFORMER_DEPTH = 1
    DEFAULT_FREQ_TRANSFORMER_DEPTH = 1
    DEFAULT_HEADS = 8
    DEFAULT_DIM_HEAD = 64
    DEFAULT_ATTN_DROPOUT = 0.0
    DEFAULT_FF_DROPOUT = 0.0
    DEFAULT_FF_MULT = 4
    DEFAULT_N_FFT = 2048
    DEFAULT_HOP_LENGTH = 512
    DEFAULT_WIN_LENGTH = 2048
    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_MLP_EXPANSION = 4
    DEFAULT_FLASH_ATTN = True

    # Band splits for 2048 FFT (1025 freq bins)
    # These are the frequency bin counts for each band
    DEFAULT_FREQS_PER_BAND: tuple[int, ...] = (
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        24,
        24,
        24,
        24,
        24,
        24,
        24,
        24,
        48,
        48,
        48,
        48,
        48,
        48,
        48,
        48,
        128,
        129,
    )

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        depth: int = DEFAULT_DEPTH,
        stereo: bool = DEFAULT_STEREO,
        num_stems: int = DEFAULT_NUM_STEMS,
        time_transformer_depth: int = DEFAULT_TIME_TRANSFORMER_DEPTH,
        freq_transformer_depth: int = DEFAULT_FREQ_TRANSFORMER_DEPTH,
        heads: int = DEFAULT_HEADS,
        dim_head: int = DEFAULT_DIM_HEAD,
        attn_dropout: float = DEFAULT_ATTN_DROPOUT,
        ff_dropout: float = DEFAULT_FF_DROPOUT,
        ff_mult: int = DEFAULT_FF_MULT,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        win_length: int = DEFAULT_WIN_LENGTH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        freqs_per_band: tuple[int, ...] = DEFAULT_FREQS_PER_BAND,
        mlp_expansion: int = DEFAULT_MLP_EXPANSION,
        flash_attn: bool = DEFAULT_FLASH_ATTN,
    ):
        super().__init__()
        self.stereo = stereo
        self.num_stems = num_stems
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.freqs_per_band = freqs_per_band

        self.audio_channels = 2 if stereo else 1
        num_freqs = n_fft // 2 + 1
        assert (
            sum(freqs_per_band) == num_freqs
        ), f"freqs_per_band must sum to {num_freqs}, got {sum(freqs_per_band)}"

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        # Band split module - input is freq * (channels * 2 for real/imag)
        # For stereo: 2 channels * 2 (real/imag) = 4
        # For mono: 1 channel * 2 (real/imag) = 2
        complex_as_channel = self.audio_channels * 2
        self.band_split = BandSplit(
            dim=dim, dim_inputs=freqs_per_band, complex_as_channel=complex_as_channel
        )

        # Rotary embeddings for time and frequency
        time_rotary = RotaryEmbedding(dim=dim_head)
        freq_rotary = RotaryEmbedding(dim=dim_head)

        # Main transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Transformer(
                            dim=dim,
                            depth=time_transformer_depth,
                            heads=heads,
                            dim_head=dim_head,
                            attn_dropout=attn_dropout,
                            ff_dropout=ff_dropout,
                            ff_mult=ff_mult,
                            rotary_embed=time_rotary,
                            flash_attn=flash_attn,
                        ),
                        Transformer(
                            dim=dim,
                            depth=freq_transformer_depth,
                            heads=heads,
                            dim_head=dim_head,
                            attn_dropout=attn_dropout,
                            ff_dropout=ff_dropout,
                            ff_mult=ff_mult,
                            rotary_embed=freq_rotary,
                            flash_attn=flash_attn,
                        ),
                    ]
                )
            )

        self.final_norm = RMSNorm(dim)

        # Mask estimator for each stem
        # Output: freq * (audio_channels * 2 for real/imag)
        # ZFTurbo uses num_stems separate mask estimators, each outputting one stem's mask
        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    dim=dim,
                    dim_outputs=freqs_per_band,
                    mlp_expansion=mlp_expansion,
                    complex_out=self.audio_channels * 2,
                )
                for _ in range(num_stems)
            ]
        )

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """Separate audio into stems.

        Args:
            raw_audio: (batch, channels, samples) or (batch, samples) for mono

        Returns:
            (batch, stems, channels, samples) separated audio
        """
        batch, channels, samples = self._normalize_input(raw_audio)

        # Ensure input has correct shape
        if raw_audio.dim() == 2:
            raw_audio = raw_audio.unsqueeze(1)

        # STFT for each channel
        stft_out = self._stft(raw_audio.view(batch * channels, samples))
        stft_out = stft_out.view(batch, channels, *stft_out.shape[1:])
        # stft_out: (batch, channels, freq, time)

        # Stack real and imaginary parts: (batch, channels, freq, time, 2)
        x = torch.stack([stft_out.real, stft_out.imag], dim=-1)

        # Flatten channels and real/imag: (batch, freq, time, channels * 2)
        x = rearrange(x, "b c f t ri -> b f t (c ri)")

        # Band split: (batch, time, bands, dim)
        x = self.band_split(x)

        # Get number of bands for reshaping
        num_bands = len(self.freqs_per_band)

        # Transformer layers
        for layer in self.layers:
            time_transformer: Transformer = layer[0]  # type: ignore[index]
            freq_transformer: Transformer = layer[1]  # type: ignore[index]
            # Time attention across time dimension
            x = rearrange(x, "b t f d -> (b f) t d")
            x = time_transformer(x)
            x = rearrange(x, "(b f) t d -> b t f d", f=num_bands)

            # Frequency attention across band dimension
            x = rearrange(x, "b t f d -> (b t) f d")
            x = freq_transformer(x)
            x = rearrange(x, "(b t) f d -> b t f d", b=batch)

        x = self.final_norm(x)

        # Estimate masks for each stem
        # Each mask_estimator outputs (batch, freq, time, audio_channels * 2)
        mask_list: list[torch.Tensor] = []
        for mask_estimator in self.mask_estimators:
            mask = mask_estimator(x)
            mask_list.append(mask)

        # Stack masks: (batch, num_stems, freq, time, audio_channels * 2)
        masks = torch.stack(mask_list, dim=1)

        # Reshape to (batch, num_stems, channels, freq, time, 2) for real/imag
        masks = rearrange(masks, "b s f t (c ri) -> b s c f t ri", c=channels, ri=2)

        # Apply masks to STFT using element-wise multiplication (not complex multiplication)
        # stft_out: (batch, channels, freq, time) -> expand for stems
        stft_real = stft_out.real.unsqueeze(1)  # (batch, 1, channels, freq, time)
        stft_imag = stft_out.imag.unsqueeze(1)  # (batch, 1, channels, freq, time)

        # Element-wise mask application per ADR-0008
        mask_real = masks[..., 0]  # (batch, num_stems, channels, freq, time)
        mask_imag = masks[..., 1]  # (batch, num_stems, channels, freq, time)
        out_real = stft_real * mask_real
        out_imag = stft_imag * mask_imag

        # Reconstruct complex STFT
        out_stft = torch.complex(out_real, out_imag)

        # iSTFT for each stem and channel
        num_stems = self.num_stems
        out_stft = rearrange(out_stft, "b s c f t -> (b s c) f t")
        out_audio = self._istft(out_stft, samples)
        # Reshape to (batch, num_stems, channels, samples)
        out_audio = out_audio.view(batch, num_stems, channels, samples)

        return out_audio

    def _normalize_input(self, raw_audio: torch.Tensor) -> tuple[int, int, int]:
        """Normalize input shape and return dimensions."""
        if raw_audio.dim() == 2:
            raw_audio = raw_audio.unsqueeze(1)
        batch, channels, samples = raw_audio.shape
        return batch, channels, samples

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT."""
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,  # type: ignore[arg-type]
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )

    def _istft(self, x: torch.Tensor, length: int) -> torch.Tensor:
        """Compute inverse STFT."""
        return torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,  # type: ignore[arg-type]
            length=length,
            center=True,
        )

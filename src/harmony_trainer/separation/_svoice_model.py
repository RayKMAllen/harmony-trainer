"""Vendored SVoice model code from facebookresearch/svoice.

Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
Licensed under CC-BY-NC 4.0 license.

Authors: Eliya Nachmani (enk100), Yossi Adi (adiyoss), Lior Wolf
"""

from __future__ import annotations

import functools
import inspect
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable

# ============ Utility functions from svoice/utils.py ============


def capture_init(init: Any) -> Any:
    """Decorator to capture __init__ args/kwargs for serialization."""

    @functools.wraps(init)
    def _wrapper_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return _wrapper_init


def deserialize_model(package: dict[str, Any], strict: bool = False) -> nn.Module:
    """Deserialize a model from a checkpoint package."""
    klass = package["class"]
    if strict:
        model: nn.Module = klass(*package["args"], **package["kwargs"])
    else:
        sig = inspect.signature(klass)
        kw = package["kwargs"].copy()
        for key in list(kw):
            if key not in sig.parameters:
                del kw[key]
        model = klass(*package["args"], **kw)
    model.load_state_dict(package["state"])
    return model


def overlap_and_add(signal: torch.Tensor, frame_step: int) -> torch.Tensor:
    """Reconstruct signal from framed representation.

    Args:
        signal: [..., frames, frame_length] tensor
        frame_step: overlap offset, must be <= frame_length

    Returns:
        [..., output_size] tensor where output_size = (frames-1)*frame_step + frame_length
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.clone().detach().long().to(signal.device)
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


# ============ Model components from svoice/models/swave.py ============


class MulCatBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.rnn_proj = nn.Linear(hidden_size * self.num_direction, input_size)

        self.gate_rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.gate_rnn_proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.block_projection = nn.Linear(input_size * 2, input_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = (
            self.rnn_proj(rnn_output.contiguous().view(-1, rnn_output.shape[2]))
            .view(output.shape)
            .contiguous()
        )

        gate_rnn_output, _ = self.gate_rnn(output)
        gate_rnn_output = (
            self.gate_rnn_proj(gate_rnn_output.contiguous().view(-1, gate_rnn_output.shape[2]))
            .view(output.shape)
            .contiguous()
        )

        gated_output = torch.mul(rnn_output, gate_rnn_output)
        gated_output = torch.cat([gated_output, output], 2)
        result: torch.Tensor = self.block_projection(
            gated_output.contiguous().view(-1, gated_output.shape[2])
        ).view(output.shape)
        return result


class ByPass(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class DPMulCat(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_spk: int,
        dropout: float = 0,
        num_layers: int = 1,
        bidirectional: bool = True,
        input_normalize: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.in_norm = input_normalize
        self.num_layers = num_layers

        self.rows_grnn = nn.ModuleList([])
        self.cols_grnn = nn.ModuleList([])
        self.rows_normalization = nn.ModuleList([])
        self.cols_normalization = nn.ModuleList([])

        for _ in range(num_layers):
            self.rows_grnn.append(
                MulCatBlock(input_size, hidden_size, dropout, bidirectional=bidirectional)
            )
            self.cols_grnn.append(
                MulCatBlock(input_size, hidden_size, dropout, bidirectional=bidirectional)
            )
            if self.in_norm:
                self.rows_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
                self.cols_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
            else:
                self.rows_normalization.append(ByPass())
                self.cols_normalization.append(ByPass())

        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size * num_spk, 1))

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        batch_size, _, d1, d2 = input.shape
        output = input
        output_all = []

        for i in range(self.num_layers):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * d2, d1, -1)
            row_output = self.rows_grnn[i](row_input)
            row_output = row_output.view(batch_size, d2, d1, -1).permute(0, 3, 2, 1).contiguous()
            row_output = self.rows_normalization[i](row_output)
            if self.training:
                output = output + row_output
            else:
                output += row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * d1, d2, -1)
            col_output = self.cols_grnn[i](col_input)
            col_output = col_output.view(batch_size, d1, d2, -1).permute(0, 3, 1, 2).contiguous()
            col_output = self.cols_normalization[i](col_output).contiguous()
            if self.training:
                output = output + col_output
            else:
                output += col_output

            output_i = self.output(output)
            if self.training or i == (self.num_layers - 1):
                output_all.append(output_i)

        return output_all


class Separator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_spk: int = 2,
        layer: int = 4,
        segment_size: int = 100,
        input_normalize: bool = False,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.input_normalize = input_normalize

        self.rnn_model = DPMulCat(
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim,
            self.num_spk,
            num_layers=layer,
            bidirectional=bidirectional,
            input_normalize=input_normalize,
        )

    def pad_segment(self, input: torch.Tensor, segment_size: int) -> tuple[torch.Tensor, int]:
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    def create_chuncks(self, input: torch.Tensor, segment_size: int) -> tuple[torch.Tensor, int]:
        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = (
            input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        )
        segments2 = (
            input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        )
        segments = (
            torch.cat([segments1, segments2], 3)
            .view(batch_size, dim, -1, segment_size)
            .transpose(2, 3)
        )
        return segments.contiguous(), rest

    def merge_chuncks(self, input: torch.Tensor, rest: int) -> torch.Tensor:
        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)

        input1 = (
            input[:, :, :, :segment_size]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, segment_stride:]
        )
        input2 = (
            input[:, :, :, segment_size:]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, :-segment_stride]
        )

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]
        return output.contiguous()

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        enc_segments, enc_rest = self.create_chuncks(input, self.segment_size)
        output_all = self.rnn_model(enc_segments)

        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_chuncks(output_all[ii], enc_rest)
            output_all_wav.append(output_ii)
        return output_all_wav


class Encoder(nn.Module):
    def __init__(self, enc_l: int, enc_n: int):
        super().__init__()
        self.enc_l, self.enc_n = enc_l, enc_n
        self.conv = nn.Conv1d(1, enc_n, kernel_size=enc_l, stride=enc_l // 2, bias=False)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = functional.relu(self.conv(mixture))
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, dec_l: int):
        super().__init__()
        self.dec_l = dec_l

    def forward(self, est_source: torch.Tensor) -> torch.Tensor:
        est_source = torch.transpose(est_source, 2, 3)
        est_source = nn.AvgPool2d((1, self.dec_l))(est_source)
        est_source = overlap_and_add(est_source, self.dec_l // 2)
        return est_source


class SWave(nn.Module):
    @capture_init
    def __init__(  # noqa: PLR0913
        self,
        n_filters: int,
        filter_len: int,
        hidden_dim: int,
        num_layers: int,
        num_speakers: int,
        sr: int,
        segment: float,
        input_normalize: bool,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.filter_len = filter_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_speakers = num_speakers
        self.sr = sr
        self.segment = segment
        self.input_normalize = input_normalize
        self.context_len = 2 * self.sr / 1000
        self.context = int(self.sr * self.context_len / 1000)
        self.layer = self.num_layers
        self.filter_dim = self.context * 2 + 1
        self.num_spk = self.num_speakers
        self.segment_size = int(np.sqrt(2 * self.sr * self.segment / (self.filter_len / 2)))

        self.encoder = Encoder(filter_len, n_filters)
        self.decoder = Decoder(filter_len)
        self.separator = Separator(
            self.filter_dim + self.n_filters,
            self.n_filters,
            self.hidden_dim,
            self.filter_dim,
            self.num_spk,
            self.layer,
            self.segment_size,
            self.input_normalize,
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mixture_w = self.encoder(mixture)
        output_all = self.separator(mixture_w)

        t_mix = mixture.size(-1)
        outputs = []
        for ii in range(len(output_all)):
            output_ii = output_all[ii].view(
                mixture.shape[0], self.num_speakers, self.n_filters, mixture_w.shape[2]
            )
            output_ii = self.decoder(output_ii)
            t_est = output_ii.size(-1)
            output_ii = functional.pad(output_ii, (0, t_mix - t_est))
            outputs.append(output_ii)
        return torch.stack(outputs)

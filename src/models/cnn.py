# src/models/cnn.py
"""
CNN architecture for binary Parkinson's disease classification from
mel spectrograms.

Input : (batch_size, 1, N_MELS, n_frames) — normalized log-mel spectrograms
        produced by dataset.py.  Concretely (B, 1, 128, 63) for the 2 s
        windows this pipeline generates.
Output: (batch_size,) raw logits — one per window.  No sigmoid is applied;
        BCEWithLogitsLoss applies it internally during training, and it is
        applied explicitly at inference time.

Design rationale
----------------
Depth (3 conv blocks)
    Each block halves both spatial axes, so 128x63 becomes 16x7 by the end
    of block 3.  A fourth block would reduce the time axis to ~3 frames,
    discarding the temporal structure that distinguishes sustained phonation
    from connected speech — and that the CNN-LSTM variant will later exploit.
    Three blocks reach a useful receptive field without over-compressing.

Modest channel growth (1 -> 32 -> 64 -> 128)
    With only 119 training subjects, capacity is the primary overfitting
    risk.  Doubling channels per block is enough to build a feature
    hierarchy without letting the network memorise individual speakers.

Global average pooling before the classifier
    Flattening the final (128, 16, 7) feature map would produce 14,336
    features and a classifier head with ~14k parameters — larger than the
    rest of the network combined, and a direct invitation to overfit.  GAP
    collapses each channel to its spatial mean, giving 128 features and a
    129-parameter head.  It also makes the model robust to small shifts in
    where a discriminative pattern occurs within the window.

Dropout before the classifier
    A final regularisation step on the pooled feature vector, where the
    model is most prone to fitting speaker-specific quirks.

BatchNorm in every block
    Stabilises and speeds up training, and contributes its own mild
    regularisation effect.

Total parameters: ~93k — deliberately small for a dataset of this size.

Usage
-----
    from models.cnn import ParkinsonCNN
    model = ParkinsonCNN()
    logits = model(batch)          # (B, 1, 128, 63) -> (B,)
    print(model.count_parameters())
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import config as C


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build one Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d block.

    The 3x3 kernel with padding=1 preserves spatial dimensions through the
    convolution; the 2x2 max-pool then halves both axes.  Bias is disabled
    on the convolution because the following BatchNorm has its own learnable
    shift, making the conv bias redundant.

    Parameters
    ----------
    in_channels : int
        Channels entering the block.
    out_channels : int
        Channels leaving the block.

    Returns
    -------
    nn.Sequential
        The assembled block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class ParkinsonCNN(nn.Module):
    """CNN classifier for Parkinson's detection from mel spectrogram windows.

    Three convolutional blocks followed by global average pooling, dropout,
    and a single-unit linear classifier.  See the module docstring for the
    full design rationale.

    Attributes
    ----------
    features : nn.Sequential
        The three convolutional blocks.
    global_pool : nn.AdaptiveAvgPool2d
        Collapses each channel's feature map to a single value.
    dropout : nn.Dropout
        Regularisation applied to the pooled feature vector.
    classifier : nn.Linear
        Final layer producing one raw logit per sample.
    """

    def __init__(
        self,
        in_channels: int = 1,
        dropout: float = 0.4,
    ) -> None:
        """Build the model.

        Parameters
        ----------
        in_channels : int, optional
            Channels in the input.  Defaults to 1 (mel spectrograms are
            single-channel).
        dropout : float, optional
            Dropout probability before the classifier.  Defaults to 0.4,
            mid-range for a small dataset where overfitting is the main risk.

        Raises
        ------
        ValueError
            If dropout is not in [0, 1).
        """
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        # ── Feature extractor: 1 -> 32 -> 64 -> 128 channels ──────────────
        # Spatial dims for a (128, 63) input:
        #   block 1 -> (64, 31)
        #   block 2 -> (32, 15)
        #   block 3 -> (16,  7)
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )

        # ── Head ──────────────────────────────────────────────────────────
        # GAP over the remaining spatial extent -> (B, 128, 1, 1).
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        # Single output unit: one raw logit, consumed by BCEWithLogitsLoss.
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(batch_size, in_channels, n_mels,
            n_frames)`` — e.g. ``(B, 1, 128, 63)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(batch_size,)`` — one per window, no
            sigmoid applied.

            The trailing singleton dimension from the classifier is squeezed
            so the output shape matches the ``(batch_size,)`` label tensor
            that BCEWithLogitsLoss expects.  Returning ``(batch_size, 1)``
            against ``(batch_size,)`` targets would broadcast into a
            ``(B, B)`` loss matrix — a silent and serious bug.

            Note for the training loop: labels arrive as ``torch.long`` and
            must be cast with ``.float()`` before being passed to
            BCEWithLogitsLoss.

        Raises
        ------
        ValueError
            If the input is not a 4-D tensor.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (batch, channels, mels, frames), "
                f"got shape {tuple(x.shape)}."
            )

        x = self.features(x)             # (B, 128, H', W')
        x = self.global_pool(x)          # (B, 128, 1, 1)
        x = torch.flatten(x, start_dim=1)  # (B, 128)
        x = self.dropout(x)
        x = self.classifier(x)           # (B, 1)
        return x.squeeze(-1)             # (B,)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the model's parameter count.

        Parameters
        ----------
        trainable_only : bool, optional
            If True (default), count only parameters with
            ``requires_grad=True``.

        Returns
        -------
        int
            Number of parameters.
        """
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def summary(self, input_shape: Tuple[int, int, int] = (1, C.N_MELS, 63)) -> str:
        """Return a short human-readable summary of the model.

        Parameters
        ----------
        input_shape : Tuple[int, int, int], optional
            ``(channels, n_mels, n_frames)`` for a single sample, used to
            report the shape the model expects.  Defaults to
            ``(1, config.N_MELS, 63)``.

        Returns
        -------
        str
            A formatted summary including parameter count.
        """
        total     = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return (
            f"ParkinsonCNN\n"
            f"  input shape (per sample) : {input_shape}\n"
            f"  output                   : (batch,) raw logits\n"
            f"  total parameters         : {total:,}\n"
            f"  trainable parameters     : {trainable:,}\n"
            f"  dropout                  : {self.dropout.p}"
        )
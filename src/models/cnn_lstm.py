# src/models/cnn_lstm.py
"""
CNN-LSTM architecture for binary Parkinson's disease classification from
mel spectrograms.

Input : (batch_size, 1, N_MELS, n_frames) — normalized log-mel spectrograms
        produced by dataset.py.  Concretely (B, 1, 128, 63).
Output: (batch_size,) raw logits — one per window.  No sigmoid applied;
        BCEWithLogitsLoss applies it internally during training.

This satisfies the identical interface as ParkinsonCNN, so it is a drop-in
replacement for the architecture-agnostic Trainer.

Design rationale
----------------
Comparison basis against ParkinsonCNN
    Fairness between the two models is enforced by an identical TRAINING
    PROTOCOL — same subject-level splits, seed, optimizer, LR schedule,
    batch size, early stopping, and loss — all of which live in trainer.py
    and train_cnn_lstm.py, not here.  The conv backbones are deliberately
    NOT identical: each model's feature extractor is free to suit its own
    modelling approach.  This is therefore a comparison of two complete
    architectures, not an ablation isolating the LSTM.

Asymmetric pooling: pool frequency hard, preserve time
    The two axes of a mel spectrogram carry different kinds of information.
    Frequency is a "what" axis — which bands are active, i.e. the spectral
    envelope.  The conv channels already encode those spectral patterns, so
    reducing the frequency resolution costs relatively little.  Time is the
    "when" axis, and it is where the Parkinsonian markers live: reduced
    prosodic variation (monotony), imprecise articulatory timing, altered
    speech rate, and tremor are all temporal phenomena.

    ParkinsonCNN pools both axes by 2x in every block, which is coherent for
    that model — it averages the whole feature map away anyway.  But an LSTM
    exists precisely to model the time axis, so pooling time away before the
    LSTM sees it destroys the signal the LSTM was added to capture.  The
    earlier version of this file mirrored ParkinsonCNN's symmetric pooling
    and handed the LSTM only 7 timesteps — too short a sequence for
    recurrence to contribute much beyond what pooling already provided.

    This design therefore pools frequency by 2x in all three blocks
    (128 -> 64 -> 32 -> 16, the same total frequency reduction as before)
    but pools time by 2x only in block 1 (63 -> 31), leaving blocks 2 and 3
    to operate at full temporal resolution via MaxPool2d((2, 1)).  The LSTM
    receives 31 timesteps instead of 7 — a sequence long enough for temporal
    dynamics to actually be modelled.

    Block 1 still pools time once: at 63 frames the immediate neighbours are
    highly redundant (the mel hop is 512 samples, ~32 ms), so one reduction
    costs little and meaningfully cuts the compute of the two wider blocks
    that follow.

Conv output -> LSTM input reshaping
    The backbone emits (B, 128, 16, 31) = (batch, channels, freq, time).
    The LSTM requires (batch, seq_len, features).  Using channels x freq
    (128 x 16 = 2048) as the per-timestep feature vector would give the LSTM
    ~541k parameters — far beyond what 119 training subjects can support.
    Instead the frequency axis is collapsed by averaging, as in the previous
    design:

        (B, 128, 16, 31)  --mean over freq-->  (B, 128, 31)
                          --permute-->         (B, 31, 128)

    Each of the 31 timesteps is a 128-dimensional vector of channel
    activations.  Frequency information is not discarded — it is encoded in
    the channel dimension by the convolutions — only its explicit spatial
    layout is averaged out.

LSTM hidden size (64)
    Unchanged from the previous design.  With input_size=128, a single layer
    costs 4 * (64*128 + 64*64 + 2*64) = 49,664 parameters.  Larger hidden
    sizes inflate capacity without a corresponding gain on a dataset of this
    size, where overfitting is the dominant risk.

Final hidden state for classification
    The LSTM's last hidden state h_n summarises the whole sequence and is
    the standard choice for sequence classification.  Consistent with the
    previous design.  At 31 timesteps a single LSTM layer carries context
    across the sequence without vanishing-gradient concerns that would
    justify mean-pooling the outputs instead.

Dropout (0.4) before the classifier
    Same rate and placement as ParkinsonCNN, regularising the sequence
    summary at the point most prone to fitting speaker-specific quirks.

Parameter count: 142,625 — unchanged from the previous 7-timestep design,
because pooling layers have no parameters.  The redesign buys a 4.4x longer
sequence at zero parameter cost.  It does cost compute: blocks 2 and 3
convolve over wider feature maps, and the LSTM runs 31 sequential steps
instead of 7, so expect noticeably slower epochs than ParkinsonCNN.

Usage
-----
    from models.cnn_lstm import ParkinsonCNNLSTM
    model = ParkinsonCNNLSTM()
    logits = model(batch)          # (B, 1, 128, 63) -> (B,)
    print(model.count_parameters())
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import config as C


def _conv_block(
    in_channels: int,
    out_channels: int,
    pool_size: Tuple[int, int],
) -> nn.Sequential:
    """Build one Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d block.

    Structurally identical to ParkinsonCNN's block except that the pooling
    kernel is a parameter, which is what allows this model's asymmetric
    (frequency-heavy, time-light) pooling schedule.  Defined locally rather
    than imported from cnn.py because the two backbones are intentionally
    independent now.

    The 3x3 kernel with padding=1 preserves spatial dimensions through the
    convolution; the pool then reduces them.  Bias is disabled on the
    convolution because the following BatchNorm has its own learnable shift,
    making the conv bias redundant.

    Parameters
    ----------
    in_channels : int
        Channels entering the block.
    out_channels : int
        Channels leaving the block.
    pool_size : Tuple[int, int]
        MaxPool2d kernel as ``(freq_factor, time_factor)``.  Stride defaults
        to the kernel size, so each axis is reduced by its factor.  Use
        ``(2, 1)`` to pool frequency while leaving time untouched.

    Returns
    -------
    nn.Sequential
        The assembled block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=pool_size),
    )


class ParkinsonCNNLSTM(nn.Module):
    """CNN-LSTM classifier for Parkinson's detection from mel spectrogram windows.

    Three convolutional blocks extract local time-frequency features using an
    asymmetric pooling schedule that reduces the frequency axis aggressively
    while largely preserving the time axis.  The frequency axis is then
    averaged away and the resulting 31-step sequence is passed through a
    single LSTM layer.  The LSTM's final hidden state feeds dropout and a
    single-unit linear classifier.

    See the module docstring for the full design rationale.

    Attributes
    ----------
    features : nn.Sequential
        The three convolutional blocks.
    lstm : nn.LSTM
        Single-layer LSTM over the time axis, batch_first.
    dropout : nn.Dropout
        Regularisation applied to the LSTM's final hidden state.
    classifier : nn.Linear
        Final layer producing one raw logit per sample.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 64,
        dropout: float = 0.4,
    ) -> None:
        """Build the model.

        Parameters
        ----------
        in_channels : int, optional
            Channels in the input.  Defaults to 1 (mel spectrograms are
            single-channel).
        hidden_size : int, optional
            LSTM hidden dimension.  Defaults to 64 — enough to summarise a
            31-step sequence without inflating capacity on a dataset of 119
            training subjects.
        dropout : float, optional
            Dropout probability before the classifier.  Defaults to 0.4,
            matching ParkinsonCNN.

        Raises
        ------
        ValueError
            If dropout is not in [0, 1), or hidden_size is not positive.
        """
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")

        # ── Feature extractor: 1 -> 32 -> 64 -> 128 channels ──────────────
        # Asymmetric pooling — (freq, time) reduction per block:
        #   block 1: (2, 2)  128 x 63 -> 64 x 31   (time pooled once)
        #   block 2: (2, 1)   64 x 31 -> 32 x 31   (time preserved)
        #   block 3: (2, 1)   32 x 31 -> 16 x 31   (time preserved)
        # Net: frequency 128 -> 16 (as in ParkinsonCNN), time 63 -> 31.
        self.features = nn.Sequential(
            _conv_block(in_channels, 32, pool_size=(2, 2)),
            _conv_block(32, 64, pool_size=(2, 1)),
            _conv_block(64, 128, pool_size=(2, 1)),
        )

        # ── Sequence model ────────────────────────────────────────────────
        # Per-timestep feature size is the channel count (128), because the
        # frequency axis is averaged out in forward() before this runs.
        self._conv_out_channels = 128
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=self._conv_out_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # ── Head ──────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout)
        # Single output unit: one raw logit, consumed by BCEWithLogitsLoss.
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Shape flow for a (B, 1, 128, 63) input::

            (B,   1, 128, 63)   input
            (B,  32,  64, 31)   after block 1  — pool (2, 2)
            (B,  64,  32, 31)   after block 2  — pool (2, 1), time preserved
            (B, 128,  16, 31)   after block 3  — pool (2, 1), time preserved
            (B, 128,      31)   mean over the frequency axis
            (B,  31,     128)   permuted to (batch, seq_len, features)
            (B,  31,      64)   LSTM outputs   (h_n is (1, B, 64))
            (B,           64)   final hidden state
            (B,            1)   classifier
            (B,)                squeezed

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

            The trailing singleton dimension is squeezed so the output
            matches the ``(batch_size,)`` label tensor BCEWithLogitsLoss
            expects.  Returning ``(batch_size, 1)`` against ``(batch_size,)``
            targets would broadcast into a ``(B, B)`` loss matrix — a silent
            and serious bug.

            Note for the training loop: labels arrive as ``torch.long`` and
            must be cast with ``.float()`` before BCEWithLogitsLoss.

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

        x = self.features(x)              # (B, 128, F', T')

        # Collapse the frequency axis by averaging, keeping time intact.
        # Frequency content survives in the channel dimension; only its
        # explicit spatial layout is averaged away.
        x = x.mean(dim=2)                 # (B, 128, T')

        # (batch, channels, time) -> (batch, time, channels) for batch_first.
        x = x.permute(0, 2, 1)            # (B, T', 128)

        # h_n has shape (num_layers, B, hidden_size); take the last layer.
        _outputs, (h_n, _c_n) = self.lstm(x)
        x = h_n[-1]                       # (B, hidden_size)

        x = self.dropout(x)
        x = self.classifier(x)            # (B, 1)
        return x.squeeze(-1)              # (B,)

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
            ``(channels, n_mels, n_frames)`` for a single sample.  Defaults
            to ``(1, config.N_MELS, 63)``.

        Returns
        -------
        str
            A formatted summary including parameter count, for direct
            comparison against ``ParkinsonCNN.summary()``.
        """
        total     = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return (
            f"ParkinsonCNNLSTM\n"
            f"  input shape (per sample) : {input_shape}\n"
            f"  output                   : (batch,) raw logits\n"
            f"  LSTM hidden size         : {self.hidden_size}\n"
            f"  LSTM sequence length     : 31 timesteps\n"
            f"  total parameters         : {total:,}\n"
            f"  trainable parameters     : {trainable:,}\n"
            f"  dropout                  : {self.dropout.p}"
        )
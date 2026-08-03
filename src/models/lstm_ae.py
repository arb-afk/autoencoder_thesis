"""
Baseline LSTM Autoencoder (Section 3.4.1).

Encoder: LSTM → last hidden state → latent bottleneck
Decoder: RepeatVector → LSTM → TimeDistributed Dense → reconstruction
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        seq_len: int = 50,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        return self.to_latent(h_n[-1])

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        repeated = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(repeated)
        return self.output_layer(dec_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
